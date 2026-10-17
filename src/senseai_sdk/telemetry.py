# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Opt-in telemetry: sampled surprise, every reflex transition, cortex events, latency histograms.

Posture:

- **Off** unless ``telemetry.enabled: true``.
- **No raw sensor data** (frames, audio, events) unless ``raw_frames: true``; by default the
  privacy filter drops any record key starting with ``raw_``.
- **Local spool** of JSON lines under ``<cache>/telemetry/``, rotated at 16 MB, capped at
  ``spool_max_mb`` (oldest files deleted first).
- **Upload only through the customer's hook** (``telemetry.upload: module:function``), called
  with the path of each closed spool file; it returns True to have the file deleted. With no
  hook, files stay on disk. The SDK never sends data anywhere on its own.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from senseai_sdk.config import TelemetryConfig
from senseai_sdk.pipeline import Pipeline, StepResult

__all__ = ["Telemetry", "privacy_filter", "resolve_upload_hook"]

log = logging.getLogger("senseai.telemetry")

ROTATE_BYTES = 16 << 20
UploadHook = Callable[[Path], bool]


def privacy_filter(record: dict[str, Any], *, raw_allowed: bool) -> dict[str, Any]:
    """Drop ``raw_*`` keys unless raw data is explicitly allowed."""
    if raw_allowed:
        return record
    return {k: v for k, v in record.items() if not k.startswith("raw_")}


def resolve_upload_hook(spec: str | None) -> UploadHook | None:
    """``"module:function"`` → callable, or None."""
    if not spec:
        return None
    mod, _, fn = spec.partition(":")
    hook: UploadHook = getattr(importlib.import_module(mod), fn)
    return hook


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    return v


class Telemetry:
    """Attach to a pipeline with :meth:`attach`; detach with :meth:`close`.

    Args:
        cfg: The ``telemetry:`` section.
        endpoint_id: From config.
        spool_dir: Where spool files go (default from config, else ``cache/telemetry``).
        sample_every: Record a surprise summary every N ticks (default 10 → 10 Hz).
        map_every: Record the full per-cell map every N ticks (default 100 → 1 Hz).
        hist_every: Record latency percentiles every N ticks (default 1000).
        upload: Override the hook from config (tests).
    """

    def __init__(
        self,
        cfg: TelemetryConfig,
        endpoint_id: str,
        spool_dir: Path,
        *,
        sample_every: int = 10,
        map_every: int = 100,
        hist_every: int = 1000,
        upload: UploadHook | None = None,
    ) -> None:
        self.cfg = cfg
        self.endpoint_id = endpoint_id
        self.spool_dir = Path(cfg.spool_dir or spool_dir)
        self.sample_every, self.map_every, self.hist_every = sample_every, map_every, hist_every
        self.upload = upload if upload is not None else resolve_upload_hook(cfg.upload)
        self._pipeline: Pipeline | None = None
        self._file: Any = None
        self._path: Path | None = None
        self._bytes = 0
        self._last_action = "none"
        self.records = 0

    # -- lifecycle ----------------------------------------------------------------------

    def attach(self, pipeline: Pipeline) -> None:
        """Start recording from ``pipeline`` (no-op unless enabled)."""
        if not self.cfg.enabled:
            return
        self._pipeline = pipeline
        pipeline.keep_ticks = pipeline.keep_ticks or self.cfg.raw_frames
        pipeline.taps.append(self._tap)
        pipeline.on_event(self._event)
        self._open()
        self._write(
            {
                "kind": "start",
                "endpoint_id": self.endpoint_id,
                "raw_frames": self.cfg.raw_frames,
                "artefacts": {k: a.id for k, a in pipeline.artefacts.items()},
            }
        )

    def close(self) -> None:
        """Flush, close the spool file and offer it to the upload hook."""
        if self._pipeline is not None and self._tap in self._pipeline.taps:
            self._pipeline.taps.remove(self._tap)
        if self._file is not None:
            self._write({"kind": "stop"})
            self._rotate(final=True)
        self._pipeline = None

    # -- recording ----------------------------------------------------------------------

    def _tap(self, res: StepResult) -> None:
        p = self._pipeline
        assert p is not None
        cmd = res.command
        action = str(getattr(cmd, "action", "none"))
        rec: dict[str, Any] = {"kind": "tick", "i": res.tick_index, "t_ns": res.t_master_ns}
        transition = action != self._last_action or res.degraded
        self._last_action = action
        if transition:
            rec["command"] = {
                "action": action,
                "speed_scale": float(getattr(cmd, "speed_scale", 1.0)),
                "stop": bool(getattr(cmd, "stop", False)),
                "degraded": res.degraded,
                "reasons": list(res.reasons),
            }
        s = res.surprise
        if s is not None and (transition or res.tick_index % self.sample_every == 0):
            rec["surprise"] = {
                "global": float(s.global_surprise),
                "per_head": _jsonable(s.per_head),
                "cell_max": float(s.per_cell.max()),
                "bearing_max": float(s.per_bearing.max()),
                "warming_up": bool(s.warming_up),
                "latency_ns": int(s.latency_ns),
            }
            if transition or res.tick_index % self.map_every == 0:
                rec["surprise"]["per_cell"] = s.per_cell.round(3).tolist()
        if res.tick is not None and self.cfg.raw_frames:
            rec["raw_samples"] = {sid: len(v) for sid, v in res.tick.samples.items()}
        if res.tick_index and res.tick_index % self.hist_every == 0:
            h = p.health()
            rec["latency"] = {st.name: {"p50_ms": st.p50_ms, "p99_ms": st.p99_ms} for st in h.stages}
            rec["latency"]["end_to_end"] = {"p99_ms": h.end_to_end_p99_ms}
        if len(rec) > 3:
            self._write(rec)

    def _event(self, ev: Any) -> None:
        self._write(
            {"kind": "event", "text": getattr(ev, "text", str(ev)), "type": str(getattr(ev, "type", ""))}
        )

    # -- spool ------------------------------------------------------------------------------

    def _open(self) -> None:
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%f")
        self._path = self.spool_dir / f"telemetry-{stamp}.jsonl"
        self._file = self._path.open("a", buffering=1)  # line-buffered: a crash loses at most one record
        self._bytes = 0

    def _write(self, rec: dict[str, Any]) -> None:
        if self._file is None:
            return
        rec = privacy_filter(rec, raw_allowed=self.cfg.raw_frames)
        rec.setdefault("ts", dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"))
        line = json.dumps(_jsonable(rec), separators=(",", ":")) + "\n"
        self._file.write(line)
        self._bytes += len(line)
        self.records += 1
        if self._bytes >= ROTATE_BYTES:
            self._rotate()

    def _rotate(self, *, final: bool = False) -> None:
        if self._file is None:
            return
        self._file.close()
        closed = self._path
        self._file = self._path = None
        if closed is not None and self.upload is not None:
            try:
                if self.upload(closed):
                    closed.unlink(missing_ok=True)
            except Exception as e:  # a failing hook never stops the pipeline
                log.warning("telemetry upload hook failed for %s: %s", closed, e)
        self._enforce_cap()
        if not final:
            self._open()

    def _enforce_cap(self) -> None:
        cap = self.cfg.spool_max_mb << 20
        files = sorted(self.spool_dir.glob("telemetry-*.jsonl"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while files and total > cap:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
