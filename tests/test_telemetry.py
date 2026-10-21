from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from senseai_sdk.config import Config, TelemetryConfig
from senseai_sdk.pipeline import Pipeline
from senseai_sdk.telemetry import Telemetry, privacy_filter, resolve_upload_hook

UPLOADED: list[Path] = []


def upload_ok(path: Path) -> bool:
    UPLOADED.append(path)
    return True


def test_privacy_filter() -> None:
    rec = {"kind": "tick", "raw_samples": 3, "surprise": 1}
    assert privacy_filter(rec, raw_allowed=False) == {"kind": "tick", "surprise": 1}
    assert privacy_filter(rec, raw_allowed=True) == rec
    assert resolve_upload_hook(None) is None
    assert resolve_upload_hook("tests.test_telemetry:upload_ok") is upload_ok


def test_disabled_records_nothing(config: Config, tmp_path: Path) -> None:
    p = Pipeline.from_config(config)
    t = Telemetry(TelemetryConfig(enabled=False), "e", tmp_path / "spool")
    t.attach(p)
    p.step()
    t.close()
    assert not (tmp_path / "spool").exists() and t.records == 0
    p.close()


def test_spool_contents_and_upload(config: Config, tmp_path: Path) -> None:
    UPLOADED.clear()
    p = Pipeline.from_config(config)
    cfg = TelemetryConfig(enabled=True, raw_frames=False, upload="tests.test_telemetry:upload_ok")
    t = Telemetry(cfg, "e", tmp_path / "spool", sample_every=5, map_every=10, hist_every=20)
    t.attach(p)
    real_step = p.surprise.step

    def boom(_maps: Any, _batch: Any) -> Any:
        raise RuntimeError("world down for one tick")

    p.surprise.step = boom  # type: ignore[method-assign, assignment]
    p.step()  # a degraded first tick is always recorded as a command transition
    p.surprise.step = real_step  # type: ignore[method-assign]
    for _ in range(24):
        p.step()
    spool = tmp_path / "spool"
    files = list(spool.glob("telemetry-*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(line) for line in files[0].read_text().splitlines()]
    kinds = [r["kind"] for r in lines]
    assert kinds[0] == "start" and "tick" in kinds
    ticks = [r for r in lines if r["kind"] == "tick"]
    assert ticks[0]["command"]["degraded"] is True
    assert any("world down for one tick" in x for x in ticks[0]["command"]["reasons"])
    assert any("per_cell" in r.get("surprise", {}) for r in ticks)
    assert any("latency" in r for r in ticks)
    assert all("raw_samples" not in r for r in ticks)
    t.close()
    assert [files[0]] == UPLOADED and not files[0].exists()  # hook accepted it: deleted
    p.close()


def test_raw_frames_opt_in(config: Config, tmp_path: Path) -> None:
    p = Pipeline.from_config(config)
    t = Telemetry(TelemetryConfig(enabled=True, raw_frames=True), "e", tmp_path / "spool", sample_every=1)
    t.attach(p)
    assert p.keep_ticks
    p.step()
    p.step()
    t.close()
    lines = [json.loads(line) for line in next((tmp_path / "spool").glob("*.jsonl")).read_text().splitlines()]
    assert any("raw_samples" in r for r in lines)
    p.close()


def test_spool_cap_deletes_oldest(config: Config, tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    spool.mkdir()
    old = spool / "telemetry-00000000T000000000000.jsonl"
    old.write_text("x" * (2 << 20))
    p = Pipeline.from_config(config)
    t = Telemetry(TelemetryConfig(enabled=True, spool_max_mb=1), "e", spool)
    t.attach(p)
    p.step()
    t.close()
    assert not old.exists()
    p.close()


def test_failing_hook_is_logged_not_raised(config: Config, tmp_path: Path) -> None:
    p = Pipeline.from_config(config)

    def bad(_p: Path) -> bool:
        raise OSError("network down")

    t = Telemetry(TelemetryConfig(enabled=True), "e", tmp_path / "spool", upload=bad)
    t.attach(p)
    p.step()
    t.close()
    assert list((tmp_path / "spool").glob("*.jsonl"))  # kept for later
    p.close()
