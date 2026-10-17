from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from senseai_sdk.artefacts import (
    ArtefactError,
    ArtefactStore,
    Keyring,
    make_manifest,
    parse_artefact_id,
    sign_manifest,
)


@pytest.fixture
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture
def keyring(key: Ed25519PrivateKey) -> Keyring:
    return Keyring.from_private(key)


def build_bundle(root: Path, artefact_id: str, key: Ed25519PrivateKey, files: dict[str, bytes]) -> Path:
    d = root / artefact_id
    d.mkdir(parents=True)
    for name, data in files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(data)
    make_manifest(d, artefact_id, meta={"tick_ms": 10})
    sign_manifest(d, key)
    return d


def test_parse_artefact_id() -> None:
    assert parse_artefact_id("world-v1.2") == ("world", "1.2")
    assert parse_artefact_id("profile-panther-v0") == ("profile-panther", "0")
    with pytest.raises(ArtefactError, match="not a valid artefact id"):
        parse_artefact_id("world")


def test_fetch_verifies_and_caches(tmp_path: Path, key: Ed25519PrivateKey, keyring: Keyring) -> None:
    store = tmp_path / "store"
    build_bundle(store, "world-v1.2", key, {"checkpoint.ckpt": b"weights", "tokens.yaml": b"d_model: 32\n"})
    cache = tmp_path / "cache"
    a = ArtefactStore(store, cache, keyring).fetch("world-v1.2")
    assert a.id == "world-v1.2"
    assert a.kind == "world"
    assert a.version == "1.2"
    assert a.meta["tick_ms"] == 10
    assert a.path("checkpoint.ckpt").read_bytes() == b"weights"
    assert a.root == cache / "artefacts" / "world-v1.2"
    # a second fetch uses the cache and still verifies
    b = ArtefactStore(store, cache, keyring).fetch("world-v1.2")
    assert b.root == a.root


def test_tampered_file_refuses(tmp_path: Path, key: Ed25519PrivateKey, keyring: Keyring) -> None:
    store = tmp_path / "store"
    d = build_bundle(store, "world-v1.2", key, {"checkpoint.ckpt": b"weights"})
    (d / "checkpoint.ckpt").write_bytes(b"evil")
    with pytest.raises(ArtefactError, match=r"world-v1.2: checkpoint.ckpt: sha256 mismatch"):
        ArtefactStore(store, tmp_path / "cache", keyring).fetch("world-v1.2")


def test_tampered_manifest_refuses(tmp_path: Path, key: Ed25519PrivateKey, keyring: Keyring) -> None:
    store = tmp_path / "store"
    d = build_bundle(store, "world-v1.2", key, {"checkpoint.ckpt": b"weights"})
    m = json.loads((d / "manifest.json").read_text())
    m["meta"]["tick_ms"] = 20
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ArtefactError, match=r"world-v1.2: manifest signature is invalid"):
        ArtefactStore(store, tmp_path / "cache", keyring).fetch("world-v1.2")


def test_unsigned_refuses(tmp_path: Path, key: Ed25519PrivateKey, keyring: Keyring) -> None:
    store = tmp_path / "store"
    d = build_bundle(store, "world-v1.2", key, {"checkpoint.ckpt": b"weights"})
    (d / "manifest.sig").unlink()
    with pytest.raises(ArtefactError, match=r"world-v1.2: no manifest.sig"):
        ArtefactStore(store, tmp_path / "cache", keyring).fetch("world-v1.2")


def test_wrong_key_refuses(tmp_path: Path, key: Ed25519PrivateKey) -> None:
    store = tmp_path / "store"
    build_bundle(store, "world-v1.2", key, {"checkpoint.ckpt": b"weights"})
    other = Keyring.from_private(Ed25519PrivateKey.generate())
    with pytest.raises(ArtefactError, match="signature is invalid"):
        ArtefactStore(store, tmp_path / "cache", other).fetch("world-v1.2")


def test_id_mismatch_refuses(tmp_path: Path, key: Ed25519PrivateKey, keyring: Keyring) -> None:
    store = tmp_path / "store"
    d = build_bundle(store, "world-v1.2", key, {"checkpoint.ckpt": b"weights"})
    (store / "world-v1.3").mkdir()
    for f in d.iterdir():
        (store / "world-v1.3" / f.name).write_bytes(f.read_bytes())
    with pytest.raises(ArtefactError, match=r"world-v1.3: manifest is for 'world-v1.2'"):
        ArtefactStore(store, tmp_path / "cache", keyring).fetch("world-v1.3")


def test_missing_refuses_with_store_named(tmp_path: Path, keyring: Keyring) -> None:
    with pytest.raises(ArtefactError, match=r"world-v9.9: not found in .*store"):
        ArtefactStore(tmp_path / "store", tmp_path / "cache", keyring).fetch("world-v9.9")


def test_cached_tamper_is_caught_on_reload(tmp_path: Path, key: Ed25519PrivateKey, keyring: Keyring) -> None:
    store = tmp_path / "store"
    build_bundle(store, "reflex-v1.0", key, {"reflex.yaml": b"a: 1\n"})
    s = ArtefactStore(store, tmp_path / "cache", keyring)
    a = s.fetch("reflex-v1.0")
    a.path("reflex.yaml").write_bytes(b"a: 2\n")
    with pytest.raises(ArtefactError, match="sha256 mismatch"):
        s.fetch("reflex-v1.0")


def test_extra_file_in_bundle_is_ignored_but_listed(
    tmp_path: Path, key: Ed25519PrivateKey, keyring: Keyring
) -> None:
    store = tmp_path / "store"
    d = build_bundle(store, "reflex-v1.0", key, {"reflex.yaml": b"a: 1\n"})
    (d / "notes.txt").write_text("not in manifest")
    a = ArtefactStore(store, tmp_path / "cache", keyring).fetch("reflex-v1.0")
    assert "notes.txt" not in a.files
    with pytest.raises(ArtefactError, match=r"notes\.txt: not in the manifest"):
        a.path("notes.txt")


def test_keyring_from_hex_and_env(monkeypatch: pytest.MonkeyPatch, key: Ed25519PrivateKey) -> None:
    kr = Keyring.from_private(key)
    hexkey = kr.public_hex()[0]
    assert Keyring.from_hex([hexkey]).public_hex() == [hexkey]
    monkeypatch.setenv("SENSEAI_RELEASE_KEYS", hexkey)
    assert hexkey in Keyring.default().public_hex()
    monkeypatch.delenv("SENSEAI_RELEASE_KEYS")
    assert Keyring.default().public_hex()  # the built-in release key


def test_https_store_uses_fetcher(tmp_path: Path, key: Ed25519PrivateKey, keyring: Keyring) -> None:
    src = tmp_path / "remote"
    d = build_bundle(src, "world-v1.2", key, {"checkpoint.ckpt": b"w"})
    served = {f"https://example.test/artefacts/world-v1.2/{p.name}": p.read_bytes() for p in d.iterdir()}

    def fetch(url: str) -> bytes:
        if url not in served:
            raise FileNotFoundError(url)
        return served[url]

    s = ArtefactStore("https://example.test/artefacts", tmp_path / "cache", keyring, fetcher=fetch)
    a = s.fetch("world-v1.2")
    assert a.path("checkpoint.ckpt").read_bytes() == b"w"
    with pytest.raises(ArtefactError, match="not found"):
        s.fetch("world-v2.0")
