from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from senseai_sdk.artefacts import ArtefactError
from senseai_sdk.config import Config, ConfigError
from senseai_sdk.pipeline import Pipeline
from senseai_sdk.stages import FallbackCommand
from senseai_sdk.testing import build_fixture_store
from tests.conftest import base_config
from tests.test_pipeline import run_all


def test_config_world_section(tmp_path: Path, store: Path) -> None:
    d = base_config(store, tmp_path / "cache")
    cfg = Config.from_mapping(d, env={}, base=tmp_path)
    assert cfg.world.mode == "learned" and cfg.world.library is None
    d["world"] = {"mode": "prior", "library": "lib/felidae.yaml"}
    cfg = Config.from_mapping(d, env={}, base=tmp_path)
    assert cfg.world.mode == "prior" and cfg.world.library == tmp_path / "lib" / "felidae.yaml"
    d["world"] = {"mode": "magic"}
    with pytest.raises(ConfigError, match=r"world\.mode: 'magic' is not one of learned, prior, fused"):
        Config.from_mapping(d, env={}, base=tmp_path)
    d["world"] = {"mdoe": "prior"}
    with pytest.raises(ConfigError, match=r"world\.mdoe: unknown key.*did you mean 'mode'"):
        Config.from_mapping(d, env={}, base=tmp_path)


@pytest.fixture(scope="module")
def prior_store(tmp_path_factory: pytest.TempPathFactory, release_key: Any) -> Path:
    return build_fixture_store(tmp_path_factory.mktemp("prior-store"), release_key, world_mode="prior")


@pytest.fixture(scope="module")
def fused_store(tmp_path_factory: pytest.TempPathFactory, release_key: Any) -> Path:
    return build_fixture_store(tmp_path_factory.mktemp("fused-store"), release_key, world_mode="fused")


def test_prior_artefact_has_no_checkpoint(prior_store: Path) -> None:
    files = {p.name for p in (prior_store / "world-v0.0").iterdir()}
    assert {"tokens.yaml", "world.yaml", "library.yaml", "manifest.json", "manifest.sig"} == files


def test_prior_mode_runs_without_weights(
    prior_store: Path, tmp_path: Path, trust_release_key: None, licence_key: str
) -> None:
    d = base_config(prior_store, tmp_path / "cache", licence_key=licence_key, world={"mode": "prior"})
    p = Pipeline.from_config(Config.from_mapping(d, env={}, base=tmp_path))
    assert p.world.mode == "prior" and p.world.model_id == "prior-v0"
    assert p.world.library is not None and p.world.library.name == "library.yaml"
    results = run_all(p)
    assert len(results) == 50
    assert all(r.world_mode == "prior" and r.world_model_id == "prior-v0" for r in results)
    assert all(r.surprise is not None and r.surprise.per_cell.shape == (32, 32) for r in results)
    assert all(r.surprise is not None and r.surprise.has_bin_stats for r in results)
    h = p.health()
    assert h.artefacts["world_mode"] == "prior" and h.artefacts["world_prior"] == "prior-v0"
    assert h.artefacts["world_model"] == "prior-v0" and h.artefacts["world_library"].endswith("library.yaml")
    assert h.to_json()["artefacts"]["world_mode"] == "prior"


def test_learned_mode_refuses_prior_artefact(
    prior_store: Path, tmp_path: Path, trust_release_key: None, licence_key: str
) -> None:
    d = base_config(prior_store, tmp_path / "cache", licence_key=licence_key)
    with pytest.raises(ArtefactError, match=r"has no checkpoint\.ckpt \(world\.mode: learned\)"):
        Pipeline.from_config(Config.from_mapping(d, env={}, base=tmp_path))


def test_fused_mode_and_state_round_trip(
    fused_store: Path, tmp_path: Path, trust_release_key: None, licence_key: str
) -> None:
    d = base_config(fused_store, tmp_path / "cache", licence_key=licence_key, world={"mode": "fused"})
    cfg = Config.from_mapping(d, env={}, base=tmp_path)
    p = Pipeline.from_config(cfg)
    assert p.world.mode == "fused" and p.world.model_id.endswith("+prior-v0")
    assert p.world.prior_model_id == "prior-v0"
    for _ in range(15):
        r = p.step()
    assert r.world_mode == "fused" and r.surprise is not None
    assert set(r.surprise.scale) >= {"learned.warming_up", "prior.warming_up"}
    st = p.state()
    q = Pipeline.from_config(cfg)
    q.restore(st)
    r2 = q.step()
    assert r2.surprise is not None and not r2.surprise.warming_up
    p.close()
    q.close()


def test_prior_mode_library_from_config_and_missing(
    prior_store: Path, tmp_path: Path, trust_release_key: None, licence_key: str
) -> None:
    import os

    lib = Path(os.environ.get("SENSEAI_PROFILES_REPO", "/home/user/profiles")) / "library" / "felidae.yaml"
    if not lib.is_file():
        pytest.skip("profiles repo not available")
    d = base_config(
        prior_store, tmp_path / "cache", licence_key=licence_key, world={"mode": "prior", "library": str(lib)}
    )
    p = Pipeline.from_config(Config.from_mapping(d, env={}, base=tmp_path))
    assert p.world.library == lib
    p.close()
    d["world"] = {"mode": "prior", "library": str(tmp_path / "nope.yaml")}
    from senseai_sdk.errors import PipelineError

    with pytest.raises(PipelineError, match="does not exist"):
        Pipeline.from_config(Config.from_mapping(d, env={}, base=tmp_path))


def test_prior_mode_dsec_panther_vision_reflex_every_tick(
    tmp_path: Path, release_key: Any, licence_key: str, trust_release_key: None
) -> None:
    """The Zurich-night POC path: synthetic DSEC, panther_vision, world in prior mode, no training.

    After the world's warm-up every command must come from the reflex (never the SDK watchdog).
    """
    pytest.importorskip("PIL")
    from senseai_ingest.testing import make_dsec_fixture

    seq = make_dsec_fixture(tmp_path / "seq", duration_s=0.5, events_per_s=50_000)
    store = build_fixture_store(
        tmp_path / "store", release_key, sources=["events_left", "rgb_left"], world_mode="prior"
    )
    d = base_config(store, tmp_path / "cache", licence_key=licence_key, world={"mode": "prior"})
    d["profile"] = {"name": "panther_vision"}
    d["sensors"] = [
        {"id": "events_left", "adapter": "dsec", "path": str(seq), "spec": "generic_event_camera"},
        {"id": "rgb_left", "adapter": "dsec", "path": str(seq), "spec": "generic_rgb_camera"},
    ]
    p = Pipeline.from_config(Config.from_mapping(d, env={}, base=tmp_path))
    results = run_all(p)
    assert len(results) >= 40
    assert all(r.world_mode == "prior" for r in results)
    warm = [r for r in results if r.surprise is not None and r.surprise.warming_up]
    after = [r for r in results if r.surprise is not None and not r.surprise.warming_up]
    assert warm and after
    assert not any(isinstance(r.command, FallbackCommand) for r in results)
    assert not any(r.command.provenance.degraded for r in after)
    assert {r.command.provenance.profile_name for r in results} == {"panther_vision"}
    assert p.health().artefacts["world_mode"] == "prior"
