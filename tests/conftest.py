"""Test fixtures: a signed artefact store built from real upstream objects.

- ``world-v0.0``: a tiny real ``senseai_world`` checkpoint + the ``tokens.yaml`` it was built
  against (d_model 32, 128 tokens/tick), consistent tokenizer_id.
- ``reflex-v0.0``: a ``reflex.yaml`` (reflex itself is not built yet).
- ``profile-null-v0`` / ``profile-panther-v0``: the profiles repo's own library, specs,
  priors and sensor specs, as self-contained artefacts.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from senseai_sdk.artefacts import Keyring, make_manifest, sign_manifest
from senseai_sdk.config import Config

PROFILES_REPO = Path(os.environ.get("SENSEAI_PROFILES_REPO", "/home/user/profiles"))

D_MODEL = 32
BUDGET = 128
TOKENS_YAML: dict[str, Any] = {
    "d_model": D_MODEL,
    "device": "cpu",
    "seed": 0,
    "sources": ["dvs0", "rgb0", "mic0", "imu0"],
    "enabled": ["event", "frame"],
    "global": {"budget": BUDGET},
    "head": {"event": {"budget": 64}, "frame": {"budget": 32}},
}
WORLD_CFG: dict[str, Any] = {
    "encoder": {"n_latents": 8, "d_latent": 32, "n_heads": 4, "n_cross_layers": 2, "n_self_layers": 1},
    "predictor": {"n_layers": 1, "n_heads": 4},
    "runtime": {"device": "cpu"},
}


def _sign(d: Path, artefact_id: str, key: Ed25519PrivateKey, meta: dict[str, Any] | None = None) -> None:
    make_manifest(d, artefact_id, meta=meta)
    sign_manifest(d, key)


@pytest.fixture(scope="session")
def release_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture(scope="session")
def store(tmp_path_factory: pytest.TempPathFactory, release_key: Ed25519PrivateKey) -> Path:
    """A signed local artefact store with world, reflex and two profiles."""
    from senseai_tokens.config import TokenizerConfig
    from senseai_tokens.tokenizer import Tokenizer
    from senseai_world.checkpoint import save_checkpoint
    from senseai_world.config import WorldConfig
    from senseai_world.model.net import WorldNet

    root = tmp_path_factory.mktemp("store")
    # world (with its tokenizer)
    w = root / "world-v0.0"
    w.mkdir()
    (w / "tokens.yaml").write_text(yaml.safe_dump(TOKENS_YAML))
    tok = Tokenizer(TokenizerConfig.from_mapping(TOKENS_YAML), device="cpu")
    wcfg = WorldConfig.from_dict(WORLD_CFG).resolve(
        d_model=D_MODEL, tokens_per_tick=BUDGET, tick_ns=10_000_000
    )
    save_checkpoint(w / "checkpoint.ckpt", WorldNet(wcfg), provenance={"tokenizer_id": tok.tokenizer_id})
    _sign(w, "world-v0.0", release_key, {"tick_ms": 10, "d_model": D_MODEL})
    # reflex
    r = root / "reflex-v0.0"
    r.mkdir()
    (r / "reflex.yaml").write_text("horizon_s: 1.5\n")
    _sign(r, "reflex-v0.0", release_key)
    # profiles
    pkg = PROFILES_REPO / "packages" / "senseai-profiles"
    for name in ("null", "panther"):
        d = root / f"profile-{name}-v0"
        (d / "profiles").mkdir(parents=True)
        (d / "priors").mkdir()
        shutil.copy(pkg / "profiles" / f"{name}.profile.yaml", d / "profiles")
        for f in (pkg / "priors").glob(f"{name}.v0.*"):
            shutil.copy(f, d / "priors")
        shutil.copytree(PROFILES_REPO / "library", d / "library")
        shutil.copytree(PROFILES_REPO / "sensors", d / "sensors")
        _sign(d, f"profile-{name}-v0", release_key)
    return root


@pytest.fixture(scope="session")
def licence_key(release_key: Ed25519PrivateKey) -> str:
    """A licence for the test endpoint, all profiles and senses, valid for a year from 2026-10-17."""
    from senseai_sdk.licence import issue_licence

    return issue_licence(
        release_key,
        licence_id="TEST-LICENCE",
        endpoint_class="amr",
        issued=dt.date(2026, 10, 17),
        expires=dt.date(2027, 10, 17),
    )


@pytest.fixture
def keyring(release_key: Ed25519PrivateKey) -> Keyring:
    return Keyring.from_private(release_key)


@pytest.fixture
def trust_release_key(monkeypatch: pytest.MonkeyPatch, keyring: Keyring) -> None:
    monkeypatch.setenv("SENSEAI_RELEASE_KEYS", keyring.public_hex()[0])


def base_config(store: Path, cache: Path, **over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "endpoint_id": "test-0001",
        "licence_key": "TEST",
        "profile": {"name": "null"},
        "artefacts": {
            "world": "world-v0.0",
            "reflex": "reflex-v0.0",
            "cache": str(cache),
            "store": str(store),
        },
        "sensors": [
            {
                "id": "dvs0",
                "adapter": "synthetic",
                "modality": "event",
                "events_per_s": 20000,
                "duration_s": 0.5,
                "spec": "generic_event_camera",
            },
            {
                "id": "rgb0",
                "adapter": "synthetic",
                "modality": "frame",
                "rate_hz": 20,
                "duration_s": 0.5,
                "spec": "generic_rgb_camera",
            },
            {"id": "imu0", "adapter": "synthetic", "modality": "imu", "rate_hz": 200, "duration_s": 0.5},
        ],
        "tick_ms": 10,
    }
    d.update(over)
    return d


@pytest.fixture
def config(store: Path, tmp_path: Path, trust_release_key: None, licence_key: str) -> Config:
    return Config.from_mapping(
        base_config(store, tmp_path / "cache", licence_key=licence_key), env={}, base=tmp_path
    )


@pytest.fixture
def config_path(store: Path, tmp_path: Path, trust_release_key: None, licence_key: str) -> Iterator[Path]:
    p = tmp_path / "senseai.yaml"
    p.write_text(yaml.safe_dump(base_config(store, tmp_path / "cache", licence_key=licence_key)))
    yield p
