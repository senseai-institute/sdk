from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from senseai_sdk.artefacts import ArtefactError
from senseai_sdk.bridges import CallbackBridge, PlannerCommand
from senseai_sdk.config import Config
from senseai_sdk.pipeline import Pipeline, StepResult
from senseai_sdk.stages import FallbackCommand
from tests.conftest import base_config


def run_all(p: Pipeline, limit: int = 10_000) -> list[StepResult]:
    out: list[StepResult] = []
    try:
        for _ in range(limit):
            out.append(p.step())
    except StopIteration:
        pass
    p.close()
    return out


def digest(results: list[StepResult]) -> str:
    h = hashlib.sha256()
    for r in results:
        s = r.surprise
        h.update(
            json.dumps(
                [r.tick_index, r.t_master_ns, r.command.action, r.command.speed_scale, r.command.stop]
            ).encode()
        )
        if s is not None:
            h.update(s.per_cell.tobytes())
            h.update(s.per_bearing.tobytes())
    return h.hexdigest()


def test_build_verifies_artefacts(config: Config) -> None:
    p = Pipeline.from_config(config)
    assert p.artefacts["world"].id == "world-v0.0"
    assert p.artefacts["reflex"].id == "reflex-v0.0"
    assert p.surprise.tokenizer_id == p.tokens.tokenizer_id and p.world is p.surprise
    assert p.reflex.available is True  # senseai-reflex is a dependency
    assert p.cortex.available is False


def test_tick_mismatch_is_refused(store: Path, tmp_path: Path, trust_release_key: None) -> None:
    cfg = Config.from_mapping(base_config(store, tmp_path / "cache", tick_ms=20), env={}, base=tmp_path)
    with pytest.raises(ArtefactError, match="trained at 10 ms"):
        Pipeline.from_config(cfg)


def test_replay_runs_to_end_and_commands_every_tick(config: Config) -> None:
    seen: list[Any] = []
    actuated: list[PlannerCommand] = []
    bridge = CallbackBridge(actuated.append, planner=lambda: PlannerCommand(1.0, 0.0))
    p = Pipeline.from_config(config, bridge=bridge)
    p.on_command(seen.append)
    results = run_all(p)
    assert len(results) == 50  # 0.5 s at 10 ms
    assert len(seen) == 50 and len(actuated) == 51  # + the final hold
    assert all(r.surprise is not None for r in results)
    assert all(r.surprise is not None and r.surprise.per_cell.shape == (32, 32) for r in results)
    # the real reflex decides every tick; no SDK watchdog fallback on a healthy replay
    assert not any(isinstance(r.command, FallbackCommand) for r in results)
    assert all(not r.command.provenance.degraded for r in results)
    assert [r.command.seq for r in results] == list(range(1, 51))  # snapshots, not one reused object
    assert all(r.t_sensor_ns is not None and r.t_sensor_ns <= r.t_master_ns for r in results)
    assert all(r.command.provenance.latency.pipeline_ns > 0 for r in results)
    assert all(r.t_emit_ns > 0 for r in results)
    h = p.health()
    # degraded only because senseai-cortex is not built yet; nothing about reflex
    assert all("cortex" in x for x in h.reasons), h.reasons
    assert h.ticks == 50
    assert [s.name for s in h.stages] == [
        "ingest",
        "circuits",
        "tokens",
        "surprise",
        "reflex",
        "cortex",
        "sdk",
    ]
    assert h.stages[2].p99_ms > 0
    assert h.to_json()["stages"][2]["name"] == "tokens"
    assert "degraded" in h.summary()


def test_single_loop_is_deterministic(config: Config) -> None:
    a = digest(run_all(Pipeline.from_config(config)))
    b = digest(run_all(Pipeline.from_config(config)))
    assert a == b


def test_matches_running_the_packages_by_hand(config: Config) -> None:
    """The SDK adds nothing: its surprise series equals ingest → circuits + tokens → world called directly."""
    import numpy as np
    from senseai_circuits.grid import GridGeometry
    from senseai_circuits.stage import CircuitStage as L1Circuits
    from senseai_circuits.stage import SurpriseFromCircuits
    from senseai_ingest.sync import Aligner, AlignerConfig
    from senseai_tokens.config import TokenizerConfig
    from senseai_tokens.tokenizer import Tokenizer
    from senseai_world.inference import WorldModel

    from senseai_sdk import adapters

    p = Pipeline.from_config(config)
    results = run_all(p)
    world_art = p.artefacts["world"]
    tok = Tokenizer(TokenizerConfig.from_yaml(world_art.path("tokens.yaml")), device="cpu")
    wm = WorldModel.load(world_art.path("checkpoint.ckpt"), device="cpu")
    ads = [adapters.build(s.adapter, s.id, s.options) for s in config.sensors]
    al = Aligner(ads, AlignerConfig(tick_ns=config.tick_ns, speed=0.0))
    al.open()
    tok.set_specs(al.specs())
    head = tok.heads["event"].cfg
    l1 = L1Circuits(tick_ns=config.tick_ns)
    fov, px = head.default_fov_deg, head.voxel_px  # type: ignore[attr-defined]
    l1.add_source("dvs0", GridGeometry.from_camera(tok.frame.camera("dvs0", fov), px))
    l1.set_specs(al.specs())
    sfc = SurpriseFromCircuits()
    by_hand = []
    for tick in al:
        maps = l1.step(tick)
        batch = tok(tick)
        s_c = sfc.step(maps, batch)
        s_l, _ = wm.step(batch)
        by_hand.append((s_l, s_c))
    al.close()
    assert len(by_hand) == len(results)
    for r, (s_l, s_c) in zip(results, by_hand, strict=True):
        got = r.surprise
        assert got is not None
        assert got.t_master_ns == s_c.t_master_ns
        assert got.global_surprise == max(s_l.global_surprise, s_c.global_surprise)
        assert (got.per_token == np.maximum(s_l.per_token, s_c.per_token)).all()
        assert (got.per_bearing == np.maximum(s_l.per_bearing, s_c.per_bearing)).all()
        assert got.provenance.model_id == f"{wm.model_id}+{sfc.model_id}"


def test_stage_failure_degrades_and_watchdog_brakes(config: Config) -> None:
    p = Pipeline.from_config(config)
    results = run_all(p, limit=5)[:5]
    p = Pipeline.from_config(config)
    p.step()

    def boom(_maps: Any, _batch: Any) -> Any:
        raise RuntimeError("world exploded")

    p.surprise.step = boom  # type: ignore[method-assign, assignment]
    r = p.step()
    assert r.degraded and r.surprise is None
    assert any("surprise stage failed: RuntimeError: world exploded" in x for x in r.reasons)
    h = p.health()
    assert h.state == "degraded"
    assert h.stages[3].name == "surprise"
    assert h.stages[3].errors == 1 and "world exploded" in (h.stages[3].last_error or "")
    # reflex's envelope watchdog takes over: slow after 50 ms without a map, brake after 250 ms
    later = [p.step() for _ in range(30)]
    assert any(x.command.provenance.degraded and x.command.action == "slow" for x in later)
    assert later[-1].command.action == "brake" and later[-1].command.stop
    assert later[-1].command.provenance.degraded
    assert not any(isinstance(x.command, FallbackCommand) for x in later)  # reflex itself did not fail
    assert all(not x.command.provenance.degraded for x in results)
    p.close()


def test_callback_exception_does_not_stop_pipeline(config: Config) -> None:
    p = Pipeline.from_config(config)

    def bad(_cmd: Any) -> None:
        raise ValueError("customer bug")

    p.on_command(bad)
    r = p.step()
    assert r.command is not None
    assert any("on_command callback raised ValueError" in x for x in r.reasons)
    p.close()


def test_profile_swap_between_ticks(config: Config) -> None:
    p = Pipeline.from_config(config)
    p.step()
    assert p.profiles.active_ref == ("null", 0)
    p.load_profile("panther", 0)
    assert p.profiles.active_ref == ("null", 0)  # queued, not yet applied
    p.step()
    assert p.profiles.active_ref == ("panther", 0)
    assert p.artefacts["profile"].id == "profile-panther-v0"
    # panther's priors depend on the frame head's lux map, which arrives with the next frame
    # (20 Hz against a 100 Hz tick), so the priors resolve within one frame period
    for _ in range(5):
        p.step()
    pri = p.tokens.tokenizer.profile()
    assert pri["frame"].budget_multiplier == pytest.approx(0.75)
    p.load_profile("null")
    p.step()
    assert p.profiles.active_ref == ("null", 0)
    assert p.tokens.tokenizer.profile()["frame"].budget_multiplier == 1.0
    p.close()


def test_profile_from_config_and_newest_version(store: Path, tmp_path: Path, trust_release_key: None) -> None:
    cfg = Config.from_mapping(
        base_config(store, tmp_path / "cache", profile={"name": "panther"}), env={}, base=tmp_path
    )
    p = Pipeline.from_config(cfg)
    assert p.profiles.active_ref == ("panther", 0)
    assert p.health().profile == ("panther", 0)
    with pytest.raises(ArtefactError, match="no profile-hawk-v"):
        p.load_profile("hawk")


def test_update_context_and_bind(config: Config) -> None:
    from senseai_ingest.adapters.replay.synthetic import SyntheticReplay
    from senseai_ingest.stream import Modality

    p = Pipeline.from_config(config)
    with pytest.raises(KeyError, match="unknown context"):
        p.update_context(lumens=3)
    p.update_context(speed=1.5, yaw_rate=0.1, heading=0.2, speed_cap=2.0, mode="aisle", lux=0.5)
    p.step()
    assert p.motion.speed_mps == 1.5 and p.motion.mode == "aisle" and p.motion.speed_cap_mps == 2.0
    with pytest.raises(ValueError, match="not in the tokenizer vocabulary"):
        p.bind_sensor("lidar9", SyntheticReplay("lidar9", Modality.LIDAR))
    mic = SyntheticReplay("mic0", Modality.AUDIO, duration_s=0.5, speed=0.0)
    with pytest.raises(ValueError, match="reports source_id"):
        p.ingest.bind("rgb0", mic)
    p.bind_sensor("mic0", mic)
    p.step()  # re-opens the aligner with the new source
    assert "mic0" in p.ingest.specs()
    p.close()


def test_state_round_trip(config: Config) -> None:
    p = Pipeline.from_config(config)
    for _ in range(15):
        p.step()
    st = p.state()
    assert st["ticks"] == 15 and st["artefacts"]["world"] == "world-v0.0"
    q = Pipeline.from_config(config)
    q.restore(st)
    r = q.step()
    assert q.ticks == 16
    assert r.surprise is not None and not r.surprise.warming_up  # restored: no new warm-up
    bad = dict(st)
    bad["artefacts"] = {"world": "world-v9.9", "reflex": "reflex-v0.0"}
    with pytest.raises(ArtefactError, match="saved with artefacts"):
        q.restore(bad)
    p.close()
    q.close()


def test_close_is_idempotent_and_publishes_hold(config: Config) -> None:
    actuated: list[PlannerCommand] = []
    p = Pipeline.from_config(config, bridge=CallbackBridge(actuated.append))
    p.step()
    p.close()
    p.close()
    assert actuated[-1] == PlannerCommand(0.0, 0.0)


def test_dsec_replay_end_to_end_with_profiles(
    tmp_path: Path, release_key: Any, licence_key: str, trust_release_key: None
) -> None:
    """A DSEC sequence directory through every stage, for each shipped profile."""
    pytest.importorskip("PIL")
    from senseai_ingest.testing import make_dsec_fixture

    from senseai_sdk.testing import build_fixture_store

    seq = make_dsec_fixture(tmp_path / "seq", duration_s=0.3, events_per_s=50_000)
    store = build_fixture_store(tmp_path / "store", release_key, sources=["events_left", "rgb_left"])
    for name in ("null", "panther"):
        d = base_config(store, tmp_path / f"cache-{name}", licence_key=licence_key)
        d["profile"] = {"name": name}
        d["sensors"] = [
            {"id": "events_left", "adapter": "dsec", "path": str(seq), "spec": "generic_event_camera"},
            {"id": "rgb_left", "adapter": "dsec", "path": str(seq), "spec": "generic_rgb_camera"},
        ]
        p = Pipeline.from_config(Config.from_mapping(d, env={}, base=tmp_path))
        results = run_all(p)
        assert len(results) >= 25
        assert not any(isinstance(r.command, FallbackCommand) for r in results)
        assert {r.command.provenance.profile_name for r in results} == {name}
        assert all(r.surprise is not None for r in results)


def test_world_null_runs_the_l1_circuits_end_to_end(
    store: Path, tmp_path: Path, trust_release_key: None, licence_key: str
) -> None:
    """``artefacts.world: null``: the fixture end to end on the circuits alone, no learned model."""
    from senseai_circuits import __version__ as circuits_version

    d = base_config(store, tmp_path / "cache", licence_key=licence_key, world=None)
    assert d["artefacts"]["world"] is None and d["artefacts"]["tokens"] == "tokens-v0.0"
    p = Pipeline.from_config(Config.from_mapping(d, env={}, base=tmp_path))
    assert p.surprise.model is None and set(p.artefacts) == {"tokens", "reflex"}
    results = run_all(p)
    assert len(results) == 50
    assert all(r.surprise is not None for r in results)
    maps = [r.surprise for r in results if r.surprise is not None]
    assert {m.provenance.model_id for m in maps} == {f"circuits-{circuits_version}"}
    assert any(m.warming_up for m in maps[:10]) and not maps[-1].warming_up
    assert maps[-1].has_bin_stats and maps[-1].per_bearing.shape == (72,)
    assert not any(isinstance(r.command, FallbackCommand) for r in results)
    assert {"circuits", "tokens", "surprise", "reflex"} <= set(results[-1].latency_ns)
    assert results[-1].command.provenance.surprise_source_id == f"circuits-{circuits_version}"
    st = p.state()
    assert st["artefacts"] == {"tokens": "tokens-v0.0", "reflex": "reflex-v0.0"} and st["circuits"]
    p2 = Pipeline.from_config(Config.from_mapping(d, env={}, base=tmp_path))
    p2.restore(st)


def test_world_null_needs_a_tokens_artefact(store: Path, tmp_path: Path, licence_key: str) -> None:
    from senseai_sdk.errors import ConfigError

    d = base_config(store, tmp_path / "cache", licence_key=licence_key, world=None)
    del d["artefacts"]["tokens"]
    with pytest.raises(ConfigError, match=r"tokens: required when artefacts\.world is null"):
        Config.from_mapping(d, env={}, base=tmp_path)
    d["artefacts"].pop("world")
    with pytest.raises(ConfigError, match="world: required"):
        Config.from_mapping(d, env={}, base=tmp_path)
