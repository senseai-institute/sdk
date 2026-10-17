# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""A signed fixture store for tests, examples and benchmarks: real upstream objects, tiny sizes.

:func:`build_fixture_store` writes, signed with the key you pass:

- ``world-v0.0``: a tiny real ``senseai_world`` checkpoint plus the ``tokens.yaml`` it was
  built against (d_model 32, 128 tokens per tick, 10 ms ticks), with a consistent tokenizer id;
- ``reflex-v0.0``: a ``reflex.yaml`` (reflex itself is not built yet);
- ``profile-null-v0`` and ``profile-panther-v0``: the ``senseai-profiles`` library, specs,
  priors and sensor specs as self-contained artefacts (needs a checkout of the profiles
  repository; ``SENSEAI_PROFILES_REPO`` overrides the default path).

Nothing here is a production artefact. The world model is untrained.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from senseai_sdk.artefacts import make_manifest, sign_manifest
from senseai_sdk.licence import issue_licence

__all__ = [
    "BUDGET",
    "D_MODEL",
    "PROFILES_REPO",
    "build_fixture_store",
    "fixture_config",
    "issue_test_licence",
]

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


def build_fixture_store(root: Path, key: Ed25519PrivateKey, *, profiles_repo: Path | None = None) -> Path:
    """Write the signed fixture artefacts under ``root`` and return it."""
    from senseai_tokens.config import TokenizerConfig
    from senseai_tokens.tokenizer import Tokenizer
    from senseai_world.checkpoint import save_checkpoint
    from senseai_world.config import WorldConfig
    from senseai_world.model.net import WorldNet

    root.mkdir(parents=True, exist_ok=True)
    w = root / "world-v0.0"
    w.mkdir()
    (w / "tokens.yaml").write_text(yaml.safe_dump(TOKENS_YAML))
    tok = Tokenizer(TokenizerConfig.from_mapping(TOKENS_YAML), device="cpu")
    wcfg = WorldConfig.from_dict(WORLD_CFG).resolve(
        d_model=D_MODEL, tokens_per_tick=BUDGET, tick_ns=10_000_000
    )
    save_checkpoint(w / "checkpoint.ckpt", WorldNet(wcfg), provenance={"tokenizer_id": tok.tokenizer_id})
    _sign(w, "world-v0.0", key, {"tick_ms": 10, "d_model": D_MODEL})
    r = root / "reflex-v0.0"
    r.mkdir()
    (r / "reflex.yaml").write_text("horizon_s: 1.5\n")
    _sign(r, "reflex-v0.0", key)
    repo = profiles_repo or PROFILES_REPO
    pkg = repo / "packages" / "senseai-profiles"
    if not pkg.is_dir():
        raise FileNotFoundError(f"profiles repository not found at {repo}; set SENSEAI_PROFILES_REPO")
    for name in ("null", "panther"):
        d = root / f"profile-{name}-v0"
        (d / "profiles").mkdir(parents=True)
        (d / "priors").mkdir()
        shutil.copy(pkg / "profiles" / f"{name}.profile.yaml", d / "profiles")
        for f in (pkg / "priors").glob(f"{name}.v0.*"):
            shutil.copy(f, d / "priors")
        shutil.copytree(repo / "library", d / "library")
        shutil.copytree(repo / "sensors", d / "sensors")
        _sign(d, f"profile-{name}-v0", key)
    return root


def issue_test_licence(key: Ed25519PrivateKey) -> str:
    """A licence for any endpoint, all profiles and senses, valid for a year from 2026-10-17."""
    return issue_licence(
        key,
        licence_id="TEST-LICENCE",
        endpoint_class="amr",
        issued=dt.date(2026, 10, 17),
        expires=dt.date(2027, 10, 17),
    )


def fixture_config(
    store: Path, cache: Path, *, licence_key: str = "TEST", seconds: float = 0.5, **over: Any
) -> dict[str, Any]:
    """A ``senseai.yaml`` mapping over the fixture store with synthetic replay sources."""
    d: dict[str, Any] = {
        "endpoint_id": "test-0001",
        "licence_key": licence_key,
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
                "duration_s": seconds,
                "spec": "generic_event_camera",
            },
            {
                "id": "rgb0",
                "adapter": "synthetic",
                "modality": "frame",
                "rate_hz": 20,
                "duration_s": seconds,
                "spec": "generic_rgb_camera",
            },
            {"id": "imu0", "adapter": "synthetic", "modality": "imu", "rate_hz": 200, "duration_s": seconds},
        ],
        "tick_ms": 10,
    }
    d.update(over)
    return d
