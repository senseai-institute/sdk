from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from senseai_sdk.config import Config, ConfigError, template

MINIMAL = {
    "endpoint_id": "amr-0042",
    "licence_key": "KEY",
    "artefacts": {"world": "world-v1.2", "reflex": "reflex-v1.0"},
    "sensors": [
        {"id": "dvs0", "adapter": "synthetic", "modality": "event"},
        {"id": "rgb0", "adapter": "synthetic", "modality": "frame"},
    ],
}


def write(tmp_path: Path, data: Mapping[str, Any]) -> Path:
    p = tmp_path / "senseai.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_minimal_loads_with_defaults(tmp_path: Path) -> None:
    cfg = Config.load(write(tmp_path, MINIMAL))
    assert cfg.endpoint_id == "amr-0042"
    assert cfg.profile.name == "null"
    assert cfg.profile.version is None
    assert cfg.tick_ms == 10
    assert cfg.reflex.bridge == "callback"
    assert cfg.cortex.sinks == ("callback",)
    assert cfg.telemetry.enabled is False
    assert cfg.artefacts.cache == Path("~/.senseai").expanduser()
    assert cfg.artefacts.store == cfg.artefacts.cache / "store"
    assert cfg.robot.footprint_radius_m > 0
    assert [s.id for s in cfg.sensors] == ["dvs0", "rgb0"]
    assert cfg.sensors[0].options == {"modality": "event"}


def test_env_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENSEAI_LICENCE", "from-env")
    cfg = Config.load(write(tmp_path, {**MINIMAL, "licence_key": "${SENSEAI_LICENCE}"}))
    assert cfg.licence_key == "from-env"


def test_env_missing_names_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENSEAI_NOPE", raising=False)
    with pytest.raises(ConfigError, match=r"licence_key: .*SENSEAI_NOPE.* is not set"):
        Config.load(write(tmp_path, {**MINIMAL, "licence_key": "${SENSEAI_NOPE}"}))


def test_missing_required_key(tmp_path: Path) -> None:
    d = dict(MINIMAL)
    del d["endpoint_id"]
    with pytest.raises(ConfigError, match=r"endpoint_id: required"):
        Config.load(write(tmp_path, d))


def test_unknown_key_suggests(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"tick_sm: unknown key.*did you mean 'tick_ms'"):
        Config.load(write(tmp_path, {**MINIMAL, "tick_sm": 10}))


def test_unknown_adapter_suggests(tmp_path: Path) -> None:
    d = {**MINIMAL, "sensors": [{"id": "mic0", "adapter": "respeker"}]}
    with pytest.raises(ConfigError, match=r"sensors\[0\]\.adapter: 'respeker'.*did you mean 'respeaker'"):
        Config.load(write(tmp_path, d))


def test_wrong_type_names_expected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"tick_ms: got 'ten' \(str\), expected int"):
        Config.load(write(tmp_path, {**MINIMAL, "tick_ms": "ten"}))


def test_duplicate_sensor_id(tmp_path: Path) -> None:
    d = {
        **MINIMAL,
        "sensors": [{"id": "a", "adapter": "synthetic"}, {"id": "a", "adapter": "synthetic"}],
    }
    with pytest.raises(ConfigError, match=r"sensors\[1\]\.id: 'a' is already used"):
        Config.load(write(tmp_path, d))


def test_lateral_above_envelope_is_error(tmp_path: Path) -> None:
    d = {**MINIMAL, "reflex": {"max_lateral_m": 5.0}}
    with pytest.raises(ConfigError, match=r"reflex\.max_lateral_m: 5.0 exceeds the safety envelope"):
        Config.load(write(tmp_path, d))


def test_jsonl_sink_requires_path(tmp_path: Path) -> None:
    d = {**MINIMAL, "cortex": {"sinks": ["jsonl"]}}
    with pytest.raises(ConfigError, match=r"cortex\.jsonl_path: required when 'jsonl'"):
        Config.load(write(tmp_path, d))


def test_unknown_sink_and_bridge(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"cortex\.sinks\[0\]: 'kafka' is not one of"):
        Config.load(write(tmp_path, {**MINIMAL, "cortex": {"sinks": ["kafka"]}}))
    with pytest.raises(ConfigError, match=r"reflex\.bridge: 'serial' is not one of"):
        Config.load(write(tmp_path, {**MINIMAL, "reflex": {"bridge": "serial"}}))


def test_relative_paths_resolve_against_yaml(tmp_path: Path) -> None:
    d = {**MINIMAL, "sensors": [{"id": "rgb0", "adapter": "v4l2", "calib": "calib/rgb0.yaml"}]}
    cfg = Config.load(write(tmp_path, d))
    assert cfg.sensors[0].calib == tmp_path / "calib" / "rgb0.yaml"


def test_template_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "senseai.yaml"
    p.write_text(template())
    cfg = Config.load(p, env={"SENSEAI_LICENCE": "x"})
    assert cfg.endpoint_id
    assert "#" in template()  # it is commented


def test_template_replay_variant(tmp_path: Path) -> None:
    p = tmp_path / "senseai.yaml"
    p.write_text(template(replay="fixture"))
    cfg = Config.load(p, env={"SENSEAI_LICENCE": "x"})
    assert all(s.adapter == "synthetic" for s in cfg.sensors)


def test_from_mapping_and_to_dict_round_trip() -> None:
    cfg = Config.from_mapping(MINIMAL, env={}, base=Path("/x"))
    again = Config.from_mapping(cfg.to_dict(), env={}, base=Path("/x"))
    assert again == cfg


def test_not_a_mapping(tmp_path: Path) -> None:
    p = tmp_path / "senseai.yaml"
    p.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        Config.load(p)
