from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from senseai_sdk.cli import main
from senseai_sdk.config import Config
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


def test_bench(config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_file = tmp_path / "bench.json"
    rc = main(
        ["bench", "--config", str(config_path), "--warmup", "5", "--device", "cpu", "--out", str(out_file)]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "end-to-end" in out and "sdk" in out and "45 ticks" in out
    assert "senseai-reflex not installed" in out
    data = json.loads(out_file.read_text())
    assert data["stages"]["world"]["p99_ms"] > 0 and data["sdk_overhead_p99_ms"] >= 0
    assert main(["bench", "--config", str(config_path), "--max-ticks", "10", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ticks"] == 0  # 10 ticks all inside the 20-tick warm-up


def test_bench_on_tick_file(config: Config, tmp_path: Path) -> None:
    """``--ticks t.h5``: an ingest tick file replaces the sensors."""
    from senseai_ingest.sync import Aligner, AlignerConfig
    from senseai_ingest.writer import write_ticks

    from senseai_sdk import adapters
    from senseai_sdk.bench import run_bench

    ads = [adapters.build(s.adapter, s.id, s.options) for s in config.sensors]
    al = Aligner(ads, AlignerConfig(tick_ns=config.tick_ns, speed=0.0))
    al.open()
    ticks = list(al)
    tf = tmp_path / "ticks.h5"
    write_ticks(tf, ticks, al.specs(), tick_ns=config.tick_ns)
    al.close()
    res = run_bench(config, ticks_file=tf, warmup=5, device="cpu")
    assert res.ticks == 45 and "world" in res.stages


def test_logs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "events.jsonl"
    f.write_text('{"text": "one"}\n{"text": "two"}\n')
    assert main(["logs", "tail", str(f)]) == 0
    assert capsys.readouterr().out.count("text") == 2
    assert main(["logs", "tail", str(tmp_path / "missing.jsonl")]) == 1
    assert main(["logs", "explain", "--at", "6.21"]) == 3
    assert "senseai-reflex" in capsys.readouterr().err
