from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from senseai_sdk import calibration as cal
from senseai_sdk.bridges import PlannerCommand
from senseai_sdk.cli import main
from senseai_sdk.config import Config
from senseai_sdk.ros2.nodes import Ros2CmdVelBridge, _event_json
from senseai_sdk.stages import FallbackCommand, Provenance


def test_camera_from_fov() -> None:
    c = cal.camera_from_fov(640, 480, 90.0)
    assert c["intrinsics"][0][0] == pytest.approx(320.0)
    assert c["intrinsics"][1][1] == pytest.approx(320.0)
    assert c["intrinsics"][0][2] == 320 and c["intrinsics"][1][2] == 240
    c2 = cal.camera_from_fov(640, 480, 90.0, 60.0)
    assert c2["intrinsics"][1][1] == pytest.approx(240 / math.tan(math.radians(30)))


def test_checkerboard_needs_opencv_or_frames() -> None:
    pytest.importorskip("cv2", reason="checkerboard path is exercised when OpenCV is installed")


def test_checkerboard_without_opencv_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_cv2(name: str, *a: object, **k: object) -> object:
        if name == "cv2":
            raise ImportError("no cv2")
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_cv2)
    with pytest.raises(RuntimeError, match="needs opencv"):
        cal.camera_from_checkerboard([np.zeros((48, 64), np.uint8)])


def test_mic_geometry() -> None:
    assert len(cal.mic_array_from_geometry("respeaker_4")["channel_positions"]) == 4
    assert cal.mic_array_from_geometry([[0, 0, 0], [0.1, 0, 0]])["channel_positions"][1] == [0.1, 0.0, 0.0]
    with pytest.raises(KeyError, match="unknown mic geometry"):
        cal.mic_array_from_geometry("nope")


def test_imu_at_rest_recovers_tilt() -> None:
    pitch = math.radians(10.0)
    g = 9.80665
    acc = np.array([-g * math.sin(pitch), 0.0, g * math.cos(pitch)])  # sensor pitched 10° nose-down
    s = np.tile(np.concatenate([acc, [0.01, -0.02, 0.0]]), (200, 1)) + np.random.default_rng(0).normal(
        0, 1e-3, (200, 6)
    )
    e = cal.imu_at_rest(s)
    t = np.asarray(e["T_body_sensor"])
    assert t.shape == (4, 4)
    recovered_pitch = math.atan2(-t[2, 0], math.hypot(t[0, 0], t[1, 0]))
    assert recovered_pitch == pytest.approx(-pitch, abs=1e-2) or recovered_pitch == pytest.approx(
        pitch, abs=1e-2
    )
    assert e["gyro_bias"] == pytest.approx([0.01, -0.02, 0.0], abs=1e-3)
    with pytest.raises(ValueError, match="not at rest"):
        cal.imu_at_rest(np.zeros((20, 6)))
    with pytest.raises(ValueError, match="need an"):
        cal.imu_at_rest(np.zeros((3, 6)))


def test_write_and_merge(tmp_path: Path) -> None:
    p = tmp_path / "calib" / "rgb0.yaml"
    cal.write_calibration(p, "rgb0", {"a": 1})
    cal.write_calibration(p, "rgb1", {"b": 2})
    doc = yaml.safe_load(p.read_text())
    assert doc["frame"]["convention"] == "x_fwd_y_left_z_up"
    assert set(doc["sources"]) == {"rgb0", "rgb1"}


def test_guided_and_tokens_can_read_it(config: Config, tmp_path: Path) -> None:
    answers = {"width": "64", "height": "48", "field of view": "90", "geometry": "respeaker_4"}

    def ask(prompt: str, default: str) -> str:
        return next((v for k, v in answers.items() if k in prompt), default)

    said: list[str] = []
    files = cal.guided(config, tmp_path / "calib", ask=ask, say=said.append)
    assert [f.name for f in files] == ["dvs0.yaml", "rgb0.yaml", "imu0.yaml"]
    from senseai_tokens.frame import BodyFrame

    BodyFrame.from_yaml(files[1])  # tokens accepts the file
    doc = yaml.safe_load(files[1].read_text())
    assert doc["sources"]["rgb0"]["intrinsics"][0][2] == 32


def test_calibrate_cli(config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "calib"
    assert (
        main(["calibrate", "--config", str(config_path), "--out", str(out), "--yes", "--sensor", "rgb0"]) == 0
    )
    assert (out / "rgb0.yaml").is_file()
    csv = tmp_path / "rest.csv"
    g = 9.80665
    np.savetxt(csv, np.tile([0.0, 0.0, g, 0.0, 0.0, 0.0], (50, 1)), delimiter=",")
    assert (
        main(
            [
                "calibrate",
                "--config",
                str(config_path),
                "--out",
                str(out),
                "--sensor",
                "imu0",
                "--imu-rest",
                str(csv),
            ]
        )
        == 0
    )
    assert "50 rest samples" in capsys.readouterr().out
    doc = yaml.safe_load((out / "imu0.yaml").read_text())
    assert doc["sources"]["imu0"]["accel_scale"] == pytest.approx(1.0)


# -- ROS 2 bridge logic, without ROS ------------------------------------------------------


def test_ros2_bridge_publishes_only_while_healthy() -> None:
    out: list[PlannerCommand] = []
    healthy = [True]
    clock = [0.0]
    b = Ros2CmdVelBridge(out.append, lambda: healthy[0], max_lateral_m=0.3, now=lambda: clock[0])
    b.on_planner(1.0, 0.0)
    cmd = FallbackCommand(0, "slow", 0.5, 1.0, False, False, False, 0.0, 0.0, Provenance())
    assert b.apply(cmd, b.planner_cmd()) == PlannerCommand(0.5, 0.3)  # lateral clamped to 0.3
    assert b.published == 1
    healthy[0] = False
    b.apply(cmd, b.planner_cmd())
    assert b.published == 1 and b.suppressed == 1  # client's own stack stays authoritative
    clock[0] = 2.0  # planner command is stale: treated as zero
    assert b.planner_cmd() == PlannerCommand()
    stop = FallbackCommand(0, "brake", 0.0, 0.0, True, False, False, 0.0, 0.0, Provenance())
    assert b.apply(stop, PlannerCommand(1.0, 1.0)) == PlannerCommand(0.0, 0.0)
    assert _event_json(stop)["text"].startswith("FallbackCommand")
