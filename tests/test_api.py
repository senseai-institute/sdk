from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from senseai_sdk import SenseAI
from senseai_sdk.bridges import CallbackBridge, PlannerCommand
from senseai_sdk.errors import ArtefactError, LicenceError, PipelineError
from senseai_sdk.licence import issue_licence
from senseai_sdk.stages import FallbackCommand
from tests.conftest import base_config

TODAY = dt.date(2026, 10, 17)


def test_quickstart_shape(config_path: Path) -> None:
    cmds: list[Any] = []
    sai = SenseAI(config_path, today=TODAY)
    sai.on_command(cmds.append)
    sai.on_event(lambda ev: None)
    sai.run()  # threaded: 50 ticks of data plus a few wall-clock watchdog ticks around them
    assert len(cmds) >= 50
    h = sai.health()
    assert h.ticks == len(cmds) and h.licence == "valid"
    assert h.artefacts == {"world": "world-v0.0", "reflex": "reflex-v0.0"}
    assert sai.meter.reports and sai.meter.reports[0]["ticks"] == {"null": len(cmds)}
    # senses are metered on ticks that carried data; the threaded runner adds wall-clock watchdog
    # ticks (FallbackCommand) between data ticks, and how many depends on machine speed
    sensed = sai.meter.reports[0]["senses"]["null"]["event"]
    assert 45 <= sensed <= len(cmds)
    assert not any(isinstance(c, FallbackCommand) for c in cmds)  # reflex decided every tick


def test_start_stop(config_path: Path) -> None:
    sai = SenseAI(config_path, today=TODAY)
    sai.start()
    sai.start()  # idempotent
    time.sleep(0.3)
    with pytest.raises(PipelineError, match="cannot be mixed"):
        sai.step()
    sai.stop()
    sai.stop()
    assert sai.health().ticks > 0


def test_run_is_threaded_and_telemetry_spools(
    store: Path, tmp_path: Path, trust_release_key: None, licence_key: str
) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(
        yaml.safe_dump(
            base_config(store, tmp_path / "cache", licence_key=licence_key, telemetry={"enabled": True})
        )
    )
    sai = SenseAI(p, today=TODAY)
    t0 = time.monotonic()
    sai.run()
    assert time.monotonic() - t0 >= 0.4  # replayed at real time: 0.5 s of data
    assert sai.health().ticks >= 50
    assert list((tmp_path / "cache" / "telemetry").glob("telemetry-*.jsonl"))


def test_step_and_state(config_path: Path, tmp_path: Path) -> None:
    sai = SenseAI(config_path, today=TODAY)
    for _ in range(12):
        r = sai.step()
    assert r.tick_index == 11
    sai.save_state(tmp_path / "state.bin")
    other = SenseAI(config_path, today=TODAY)
    other.load_state(tmp_path / "state.bin")
    r2 = other.step()
    assert r2.surprise is not None and not r2.surprise.warming_up
    sai.stop()
    other.stop()


def test_licence_gates(
    store: Path, tmp_path: Path, trust_release_key: None, release_key: Ed25519PrivateKey
) -> None:
    def cfg_with(key: str) -> Path:
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump(base_config(store, tmp_path / "cache", licence_key=key)))
        return p

    with pytest.raises(LicenceError, match="SAI1"):
        SenseAI(cfg_with("TEST"), today=TODAY)
    expired = issue_licence(
        release_key, licence_id="X", endpoint_class="amr", expires=TODAY - dt.timedelta(days=40)
    )
    with pytest.raises(LicenceError, match="grace period has ended"):
        SenseAI(cfg_with(expired), today=TODAY)
    grace = issue_licence(
        release_key, licence_id="G", endpoint_class="amr", expires=TODAY - dt.timedelta(days=5)
    )
    sai = SenseAI(cfg_with(grace), today=TODAY)
    assert sai.health().licence == "grace" and sai.health().licence_days_left == 25
    wrong_ep = issue_licence(
        release_key, licence_id="E", endpoint_class="amr", expires=TODAY, endpoint_ids=["other"]
    )
    with pytest.raises(LicenceError, match="does not cover endpoint 'test-0001'"):
        SenseAI(cfg_with(wrong_ep), today=TODAY)
    no_panther = issue_licence(
        release_key, licence_id="P", endpoint_class="amr", expires=TODAY, profiles=["hawk"]
    )
    sai = SenseAI(cfg_with(no_panther), today=TODAY)
    with pytest.raises(LicenceError, match="does not allow profile 'panther'"):
        sai.load_profile("panther")
    frames_only = issue_licence(
        release_key, licence_id="S", endpoint_class="amr", expires=TODAY, senses=["frame"]
    )
    sai = SenseAI(cfg_with(frames_only), today=TODAY)
    with pytest.raises(LicenceError, match="does not allow senses"):
        sai.step()


def test_bridge_and_context(config_path: Path) -> None:
    acts: list[PlannerCommand] = []
    sai = SenseAI(
        config_path, today=TODAY, bridge=CallbackBridge(acts.append, planner=lambda: PlannerCommand(1.0, 0.1))
    )
    sai.update_context(speed=1.0)
    with pytest.raises(KeyError):
        sai.update_context(lumens=1)
    r = sai.step()
    # quiet fixture: reflex passes the planner command through (speed × 1.0)
    assert acts and acts[-1] == PlannerCommand(1.0, 0.1)
    sai.bind_actuator(CallbackBridge(lambda c: acts.append(PlannerCommand(9.0, 9.0))))
    sai.step()
    assert acts[-1] == PlannerCommand(9.0, 9.0)
    assert "profile null v0" in sai.explain(r.t_master_ns)
    with pytest.raises(KeyError):
        sai.explain(r.t_master_ns - 1)  # before the first command in the audit log
    sai.stop()


def test_profile_swap_through_api(config_path: Path) -> None:
    sai = SenseAI(config_path, today=TODAY)
    sai.step()
    sai.load_profile("panther")
    sai.step()
    assert sai.health().profile == ("panther", 0)
    with pytest.raises(ArtefactError):
        sai.load_profile("hawk")
    sai.stop()
