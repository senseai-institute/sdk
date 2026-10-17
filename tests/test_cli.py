from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from senseai_sdk.cli import main
from tests.conftest import base_config


def test_init_writes_template_and_refuses_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "senseai.yaml"
    assert main(["init", "--out", str(out), "--replay", "fixture"]) == 0
    assert "adapter: synthetic" in out.read_text()
    assert main(["init", "--out", str(out)]) == 2
    assert main(["init", "--out", str(out), "--force"]) == 0
    assert "adapter: prophesee" in out.read_text()
    assert "wrote" in capsys.readouterr().out


def test_doctor_on_fixture(config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["doctor", "--config", str(config_path), "--probe-seconds", "0.2"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "[ OK ] config" in out and "[ OK ] licence" in out
    assert "[ OK ] artefact world: world-v0.0 verified" in out
    assert "[WARN] package senseai_reflex" in out
    assert "[ OK ] tick: 10 ms matches world-v0.0" in out
    assert "[ OK ] sensor dvs0: synthetic (event) produced" in out
    assert "[WARN] calibration dvs0" in out
    assert "[ OK ] clock" in out
    assert "passed with warnings" in out


def test_doctor_failures(
    store: Path, tmp_path: Path, trust_release_key: None, licence_key: str, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("endpoint_id: x\n")
    assert main(["doctor", "--config", str(p)]) == 1
    assert "[FAIL] config" in capsys.readouterr().out

    d = base_config(store, tmp_path / "cache", licence_key="TEST")
    p.write_text(yaml.safe_dump(d))
    assert main(["doctor", "--config", str(p)]) == 1
    assert "[FAIL] licence" in capsys.readouterr().out

    d = base_config(
        store,
        tmp_path / "cache",
        licence_key=licence_key,
        tick_ms=20,
        artefacts={
            "world": "world-v0.0",
            "reflex": "reflex-v9.9",
            "cache": str(tmp_path / "c"),
            "store": str(store),
        },
    )
    d["sensors"].append(
        {"id": "cam9", "adapter": "v4l2", "device": "/dev/video99", "calib": str(tmp_path / "missing.yaml")}
    )
    d["profile"] = {"name": "hawk"}
    p.write_text(yaml.safe_dump(d))
    assert main(["doctor", "--config", str(p)]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] artefact reflex: reflex-v9.9: not found" in out
    assert "[FAIL] artefact profile" in out
    assert "[FAIL] tick: config 20 ms, world-v0.0 trained at 10 ms -> set tick_ms: 10" in out
    assert "[FAIL] sensor cam9" in out
    assert "FAILED" in out


def test_run_ticks_and_stats(config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        ["run", "--config", str(config_path), "--ticks", "7", "--stats", "--verbose", "--health-port", "0"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "7 commands" in out and '"stages"' in out and "brake" in out and "health at http" in out


def test_run_to_end(config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--config", str(config_path)]) == 0
    assert "50 commands" in capsys.readouterr().out


def test_licence_show_activate_report(
    config_path: Path, tmp_path: Path, licence_key: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["licence", "show", "--config", str(config_path)]) == 0
    assert json.loads(capsys.readouterr().out)["licence_id"] == "TEST-LICENCE"
    dest = tmp_path / "k" / "licence.key"
    assert main(["licence", "activate", licence_key, "--store-to", str(dest)]) == 0
    assert dest.read_text().strip() == licence_key
    assert main(["licence", "activate", "--store-to", str(dest)]) == 2
    assert main(["licence", "activate", "nope", "--store-to", str(dest)]) == 1
    main(["run", "--config", str(config_path), "--ticks", "3"])
    capsys.readouterr()
    out_file = tmp_path / "usage.json"
    assert main(["licence", "report", "--config", str(config_path), "--out", str(out_file)]) == 0
    reps = json.loads(out_file.read_text())
    assert reps and reps[-1]["ticks"] == {"null": 3}
    assert main(["licence", "report", "--config", str(config_path)]) == 0
    assert "TEST-LICENCE" in capsys.readouterr().out


def test_profiles_list_show(config_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profiles", "list", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "panther" in out and "null" in out
    assert main(["profiles", "show", "panther", "--config", str(config_path)]) == 0
    assert "profile-panther-v0" in capsys.readouterr().out
    assert main(["profiles", "show", "hawk", "--config", str(config_path)]) == 1


def test_not_yet_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench"]) == 3
    assert main(["calibrate"]) == 3
    assert main(["logs", "tail"]) == 3
    assert "not implemented" in capsys.readouterr().err
