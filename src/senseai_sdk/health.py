# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Pipeline health: per-stage latency windows, drops, errors, and the ``/health`` document.

Latency is recorded as nanoseconds into a fixed ring per stage (default: the last 1000 ticks,
10 s at 100 Hz). Percentiles are computed on demand with NumPy; recording is one array write.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Literal

import numpy as np

__all__ = ["Health", "HealthServer", "LatencyRing", "StageHealth", "StageStats"]

HealthState = Literal["ok", "degraded", "failed"]

#: Per-stage p99 budgets in milliseconds, from the README's latency table.
BUDGET_MS: Mapping[str, float] = {
    "ingest": 1.0,
    "tokens": 2.0,
    "world": 6.0,
    "reflex": 1.0,
    "bridge": 3.0,
    "cortex": 5.0,
    "sdk": 1.0,
    "total": 15.0,
}


class LatencyRing:
    """A fixed-size ring of nanosecond samples."""

    __slots__ = ("_buf", "_i", "_n")

    def __init__(self, size: int = 1000) -> None:
        self._buf = np.zeros(size, dtype=np.int64)
        self._i = 0
        self._n = 0

    def add(self, ns: int) -> None:
        """Record one sample."""
        self._buf[self._i] = ns
        self._i = (self._i + 1) % len(self._buf)
        self._n = min(self._n + 1, len(self._buf))

    def __len__(self) -> int:
        return self._n

    def percentiles_ms(self, q: tuple[float, ...] = (50.0, 99.0)) -> tuple[float, ...]:
        """Percentiles in milliseconds; zeros when empty."""
        if self._n == 0:
            return tuple(0.0 for _ in q)
        vals = self._buf[: self._n]
        return tuple(float(x) / 1e6 for x in np.percentile(vals, q))

    def last_ms(self) -> float:
        """The most recent sample in milliseconds."""
        if self._n == 0:
            return 0.0
        return float(self._buf[(self._i - 1) % len(self._buf)]) / 1e6


class StageStats:
    """Live counters for one stage. Owned by the pipeline; read by :class:`Health`."""

    def __init__(self, name: str, window: int = 1000) -> None:
        self.name = name
        self.latency = LatencyRing(window)
        self.drops = 0
        self.errors = 0
        self.overruns = 0
        self.last_error: str | None = None
        self.available = True
        self.reason: str | None = None

    def record(self, ns: int) -> None:
        """Record a successful run."""
        self.latency.add(ns)
        budget = BUDGET_MS.get(self.name)
        if budget is not None and ns > budget * 1e6:
            self.overruns += 1

    def fail(self, err: BaseException) -> None:
        """Record a failure. Keeps the latest message."""
        self.errors += 1
        self.last_error = f"{type(err).__name__}: {err}"

    def snapshot(self) -> StageHealth:
        """Freeze the counters."""
        p50, p99 = self.latency.percentiles_ms()
        return StageHealth(
            name=self.name,
            p50_ms=p50,
            p99_ms=p99,
            budget_ms=BUDGET_MS.get(self.name, 0.0),
            drops=self.drops,
            errors=self.errors,
            overruns=self.overruns,
            last_error=self.last_error,
            available=self.available,
            reason=self.reason,
        )


@dataclass(slots=True)
class StageHealth:
    """Health of one pipeline stage over the recent window."""

    name: str
    p50_ms: float
    p99_ms: float
    budget_ms: float
    drops: int
    errors: int
    overruns: int = 0
    last_error: str | None = None
    available: bool = True
    reason: str | None = None


@dataclass(slots=True)
class Health:
    """Pipeline health, returned by ``SenseAI.health()`` and served at ``/health``.

    Attributes:
        state: ``ok``; ``degraded`` (a stage failed, overran or is missing, so the watchdog is
            driving the bridge); or ``failed`` (the pipeline cannot run).
        endpoint_id: From config.
        stages: Per-stage latency, drops and errors, in pipeline order.
        end_to_end_p99_ms: Sensor-to-actuator p99 over the window.
        watchdog: Reflex watchdog state: ``idle``, ``slow`` or ``brake``.
        licence: ``valid``, ``grace`` or ``expired``, with ``licence_days_left``.
        artefacts: Loaded artefact ids by kind.
        profile: Active profile name and version.
        ticks: Ticks processed since start.
        reasons: Human-readable reasons for any state other than ``ok``.
    """

    state: HealthState
    endpoint_id: str
    stages: list[StageHealth]
    end_to_end_p99_ms: float
    watchdog: Literal["idle", "slow", "brake"]
    licence: Literal["valid", "grace", "expired"]
    licence_days_left: int
    artefacts: Mapping[str, str]
    profile: tuple[str, int]
    ticks: int = 0
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """The JSON-serialisable form served at ``/health``."""
        return {
            "state": self.state,
            "endpoint_id": self.endpoint_id,
            "ticks": self.ticks,
            "end_to_end_p99_ms": round(self.end_to_end_p99_ms, 3),
            "watchdog": self.watchdog,
            "licence": {"state": self.licence, "days_left": self.licence_days_left},
            "artefacts": dict(self.artefacts),
            "profile": {"name": self.profile[0], "version": self.profile[1]},
            "reasons": list(self.reasons),
            "stages": [
                {
                    "name": s.name,
                    "p50_ms": round(s.p50_ms, 3),
                    "p99_ms": round(s.p99_ms, 3),
                    "budget_ms": s.budget_ms,
                    "drops": s.drops,
                    "errors": s.errors,
                    "overruns": s.overruns,
                    "last_error": s.last_error,
                    "available": s.available,
                    "reason": s.reason,
                }
                for s in self.stages
            ],
        }

    def summary(self) -> str:
        """One line for logs and the CLI."""
        worst = max(self.stages, key=lambda s: s.p99_ms) if self.stages else None
        tail = f", slowest {worst.name} p99 {worst.p99_ms:.2f} ms" if worst else ""
        why = f" ({'; '.join(self.reasons)})" if self.reasons else ""
        return (
            f"{self.state}{why}: {self.ticks} ticks, end-to-end p99 {self.end_to_end_p99_ms:.2f} ms, "
            f"watchdog {self.watchdog}, licence {self.licence} ({self.licence_days_left} d), "
            f"profile {self.profile[0]} v{self.profile[1]}{tail}"
        )


class HealthServer:
    """A tiny HTTP server that answers ``GET /health`` with the JSON document. Optional."""

    def __init__(self, provider: Any, port: int, host: str = "127.0.0.1") -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path.rstrip("/") not in ("", "/health"):
                    self.send_error(404)
                    return
                body = json.dumps(server._provider().to_json()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_: Any) -> None:
                return

        self._provider = provider
        self._httpd = HTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="senseai-health", daemon=True)

    @property
    def port(self) -> int:
        """The bound port (useful when 0 was requested)."""
        return int(self._httpd.server_address[1])

    def start(self) -> None:
        """Serve in a daemon thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop serving."""
        self._httpd.shutdown()
        self._httpd.server_close()
