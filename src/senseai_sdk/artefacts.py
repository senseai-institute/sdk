# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Locate, verify, cache and load the signed proprietary artefacts.

An **artefact** is a directory named by its id (``world-v1.2``, ``reflex-v1.0``,
``profile-panther-v0``) that holds:

- ``manifest.json``: ``{"id", "kind", "version", "files": {relpath: sha256}, "meta": {...}}``
- ``manifest.sig``: hex Ed25519 signature over the canonical manifest bytes
- the files the manifest lists

The SDK ships SenseAI's release public key. A manifest signed by any other key, a file whose
hash does not match, a file the manifest does not list, or an id that does not match the
directory: none of these load. Verification runs on every load, including from the cache,
so a file edited after download is caught too.

The store is a local directory or an HTTPS base URL with the same layout. Profile bundles
additionally carry ``senseai-profiles``' own HMAC sidecar, which the registry checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from senseai_sdk.errors import ArtefactError

__all__ = [
    "Artefact",
    "ArtefactError",
    "ArtefactStore",
    "Keyring",
    "make_manifest",
    "parse_artefact_id",
    "sign_manifest",
]

MANIFEST = "manifest.json"
SIGNATURE = "manifest.sig"

#: SenseAI's release public keys (hex, Ed25519). A new release key is appended, never
#: replaced, so artefacts signed with an older key keep loading. The private halves live in
#: the release pipeline only. This placeholder key is replaced at the first signed release;
#: until then every real deployment sets ``SENSEAI_RELEASE_KEYS``.
RELEASE_PUBLIC_KEYS_HEX: tuple[str, ...] = (
    "3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29",
)

_ID_RE = re.compile(r"^(?P<kind>[a-z][a-z0-9-]*?)-v(?P<version>\d+(?:\.\d+)*)$")


def profile_artefact_prefix(name: str) -> str:
    """``profile-<name>-v`` with ``_`` spelled ``-`` (artefact ids allow ``[a-z0-9-]`` only)."""
    return f"profile-{name.replace('_', '-')}-v"


def profile_artefact_id(name: str, version: int | str) -> str:
    """The artefact id of profile ``name`` version ``version``, e.g. ``profile-panther-vision-v0``."""
    return f"{profile_artefact_prefix(name)}{version}"


def parse_artefact_id(artefact_id: str) -> tuple[str, str]:
    """Split ``world-v1.2`` into ``("world", "1.2")``. The kind may contain dashes."""
    m = _ID_RE.match(artefact_id)
    if m is None:
        raise ArtefactError(f"{artefact_id!r} is not a valid artefact id (expected <kind>-v<version>)")
    return m["kind"], m["version"]


def _canonical(manifest: Mapping[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- keys ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Keyring:
    """The public keys an artefact manifest may be signed with."""

    keys: tuple[Ed25519PublicKey, ...]

    @classmethod
    def from_hex(cls, hex_keys: list[str] | tuple[str, ...]) -> Keyring:
        """Build from raw 32-byte public keys in hex."""
        return cls(tuple(Ed25519PublicKey.from_public_bytes(bytes.fromhex(h)) for h in hex_keys))

    @classmethod
    def from_private(cls, key: Ed25519PrivateKey) -> Keyring:
        """A keyring trusting only ``key``'s public half (tests and local signing)."""
        return cls((key.public_key(),))

    @classmethod
    def default(cls) -> Keyring:
        """The shipped release keys, plus any in ``SENSEAI_RELEASE_KEYS`` (comma-separated hex)."""
        extra = [k for k in os.environ.get("SENSEAI_RELEASE_KEYS", "").split(",") if k.strip()]
        return cls.from_hex(list(RELEASE_PUBLIC_KEYS_HEX) + extra)

    def public_hex(self) -> list[str]:
        """The keys as hex, in order."""
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        return [k.public_bytes(Encoding.Raw, PublicFormat.Raw).hex() for k in self.keys]

    def verify(self, data: bytes, signature: bytes) -> bool:
        """True if any key verifies ``signature`` over ``data``."""
        for k in self.keys:
            try:
                k.verify(signature, data)
                return True
            except InvalidSignature:
                continue
        return False


# -- building and signing (release tooling, tests) ------------------------------------------


def make_manifest(root: Path, artefact_id: str, *, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write ``manifest.json`` for every file under ``root`` (except the manifest and signature)."""
    kind, version = parse_artefact_id(artefact_id)
    files = {
        p.relative_to(root).as_posix(): _sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in (MANIFEST, SIGNATURE)
    }
    manifest = {"id": artefact_id, "kind": kind, "version": version, "files": files, "meta": dict(meta or {})}
    (root / MANIFEST).write_bytes(_canonical(manifest))
    return manifest


def sign_manifest(root: Path, key: Ed25519PrivateKey) -> None:
    """Write ``manifest.sig`` over the manifest bytes as they are on disk."""
    sig = key.sign((root / MANIFEST).read_bytes())
    (root / SIGNATURE).write_text(sig.hex())


# -- the verified artefact ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Artefact:
    """A verified artefact on local disk."""

    id: str
    kind: str
    version: str
    root: Path
    files: Mapping[str, str]  # relpath → sha256
    meta: Mapping[str, Any] = field(default_factory=dict)

    def path(self, relpath: str) -> Path:
        """The path of a file the manifest lists.

        Raises:
            ArtefactError: ``relpath`` is not in the manifest (so it was never verified).
        """
        if relpath not in self.files:
            raise ArtefactError(f"{self.id}: {relpath}: not in the manifest")
        return self.root / relpath


def verify_dir(root: Path, artefact_id: str, keyring: Keyring) -> Artefact:
    """Verify a bundle directory in place and return it as an :class:`Artefact`."""
    mpath, spath = root / MANIFEST, root / SIGNATURE
    if not mpath.is_file():
        raise ArtefactError(f"{artefact_id}: no {MANIFEST} in {root}")
    if not spath.is_file():
        raise ArtefactError(f"{artefact_id}: no {SIGNATURE} in {root}; unsigned artefacts never load")
    raw = mpath.read_bytes()
    try:
        sig = bytes.fromhex(spath.read_text().strip())
    except ValueError:
        raise ArtefactError(f"{artefact_id}: {SIGNATURE} is not hex") from None
    if not keyring.verify(raw, sig):
        raise ArtefactError(
            f"{artefact_id}: manifest signature is invalid for every trusted release key; "
            "the artefact was tampered with or signed by an unknown key"
        )
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ArtefactError(f"{artefact_id}: {MANIFEST} is not JSON: {e}") from e
    if manifest.get("id") != artefact_id:
        raise ArtefactError(f"{artefact_id}: manifest is for {manifest.get('id')!r}")
    files: dict[str, str] = dict(manifest.get("files", {}))
    for rel, expected in files.items():
        p = root / rel
        if not p.is_file():
            raise ArtefactError(f"{artefact_id}: {rel}: listed in the manifest but missing")
        got = _sha256(p)
        if got != expected:
            raise ArtefactError(
                f"{artefact_id}: {rel}: sha256 mismatch (manifest {expected[:12]}…, file {got[:12]}…)"
            )
    return Artefact(
        id=artefact_id,
        kind=str(manifest["kind"]),
        version=str(manifest["version"]),
        root=root,
        files=files,
        meta=dict(manifest.get("meta", {})),
    )


# -- the store --------------------------------------------------------------------------------

Fetcher = Callable[[str], bytes]


def _https_fetch(url: str) -> bytes:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data: bytes = r.read()
            return data
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(url) from e
        raise


class ArtefactStore:
    """Fetches artefacts from a local directory or an HTTPS base URL into the cache, verified.

    Args:
        store: Directory or ``https://`` base URL holding ``<artefact_id>/manifest.json`` …
        cache: Local cache root; artefacts land in ``<cache>/artefacts/<artefact_id>/``.
        keyring: Trusted release keys. Default: :meth:`Keyring.default`.
        fetcher: ``url -> bytes`` for HTTPS stores (tests inject one).
    """

    def __init__(
        self,
        store: Path | str,
        cache: Path,
        keyring: Keyring | None = None,
        *,
        fetcher: Fetcher | None = None,
    ) -> None:
        self.store = store
        self.cache = Path(cache).expanduser()
        self.keyring = keyring or Keyring.default()
        self._fetch = fetcher or _https_fetch
        self._loaded: dict[str, Artefact] = {}

    @property
    def remote(self) -> bool:
        """True for an HTTPS store."""
        return isinstance(self.store, str) and self.store.startswith(("https://", "http://"))

    def local_dir(self, artefact_id: str) -> Path:
        """Where ``artefact_id`` lives in the cache (whether or not it is there yet)."""
        return self.cache / "artefacts" / artefact_id

    def fetch(self, artefact_id: str) -> Artefact:
        """Return the verified artefact, downloading or copying it into the cache if needed.

        Raises:
            ArtefactError: Not found, unsigned, wrong key, hash mismatch, or id mismatch.
        """
        parse_artefact_id(artefact_id)
        dest = self.local_dir(artefact_id)
        if not (dest / MANIFEST).is_file():
            self._materialise(artefact_id, dest)
        a = verify_dir(dest, artefact_id, self.keyring)
        self._loaded[artefact_id] = a
        return a

    def loaded(self) -> Mapping[str, Artefact]:
        """Every artefact this store has verified so far, by id."""
        return dict(self._loaded)

    # -- transport --------------------------------------------------------------------------

    def _materialise(self, artefact_id: str, dest: Path) -> None:
        tmp = dest.with_name(dest.name + ".partial")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            if self.remote:
                self._download(artefact_id, tmp)
            else:
                src = Path(str(self.store)).expanduser() / artefact_id
                if not (src / MANIFEST).is_file():
                    raise ArtefactError(f"{artefact_id}: not found in {self.store}")
                shutil.rmtree(tmp)
                shutil.copytree(src, tmp)
            if dest.exists():
                shutil.rmtree(dest)
            tmp.rename(dest)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

    def _download(self, artefact_id: str, into: Path) -> None:
        base = f"{str(self.store).rstrip('/')}/{artefact_id}"
        try:
            raw = self._fetch(f"{base}/{MANIFEST}")
        except FileNotFoundError:
            raise ArtefactError(f"{artefact_id}: not found in {self.store}") from None
        (into / MANIFEST).write_bytes(raw)
        try:
            (into / SIGNATURE).write_bytes(self._fetch(f"{base}/{SIGNATURE}"))
        except FileNotFoundError:
            raise ArtefactError(
                f"{artefact_id}: no {SIGNATURE} in {self.store}; unsigned artefacts never load"
            ) from None
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ArtefactError(f"{artefact_id}: {MANIFEST} is not JSON: {e}") from e
        for rel in manifest.get("files", {}):
            p = into / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                p.write_bytes(self._fetch(f"{base}/{rel}"))
            except FileNotFoundError:
                raise ArtefactError(
                    f"{artefact_id}: {rel}: listed in the manifest but missing from {self.store}"
                ) from None
