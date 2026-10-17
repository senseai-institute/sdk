# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""``senseai bench``: sensor-to-actuator latency on this machine, per stage and end to end.

Runs the single-loop pipeline (so every stage is measured back to back, with no queueing)
over a recording or the configured sensors, and reports p50/p99 per stage against the
README's budget table. This is the number that goes on the Gate 3 slide; it is only meaningful
on the target hardware with the production artefacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from senseai_sdk.config import Config
from senseai_sdk.health import BUDGET_MS, LatencyRing
from senseai_sdk.pipeline import Pipeline
from senseai_sdk.stages import IngestStage

__all__ = ["BenchResult", "run_bench"]


@dataclass(slots=True)
class BenchResult:
    """Per-stage and end-to-end percentiles in milliseconds."""

    ticks: int
    stages: dict[str, tuple[float, float]]  # name → (p50, p99)
    end_to_end_p99_ms: float
    sdk_overhead_p99_ms: float
    device: str
    reflex_installed: bool
    cortex_installed: bool
    warmup_ticks: int = 0
    notes: list[str] = field(default_factory=list)

    def table(self) -> str:
        """The latency table as text."""
        rows = [f"{'stage':<10} {'p50 ms':>8} {'p99 ms':>8} {'budget':>8}  verdict"]
        for name, (p50, p99) in self.stages.items():
            budget = BUDGET_MS.get(name)
            verdict = "" if budget is None else ("ok" if p99 <= budget else "OVER")
            rows.append(
                f"{name:<10} {p50:8.3f} {p99:8.3f} {budget if budget is not None else '':>8}  {verdict}"
            )
        rows.append(
            f"{'end-to-end':<10} {'':>8} {self.end_to_end_p99_ms:8.3f} {BUDGET_MS['total']:>8}  "
            f"{'ok' if self.end_to_end_p99_ms <= BUDGET_MS['total'] else 'OVER'}"
        )
        rows.append(
            f"{self.ticks} ticks on {self.device}; reflex {'yes' if self.reflex_installed else 'NO'}, "
            f"cortex {'yes' if self.cortex_installed else 'NO'}; {self.warmup_ticks} warm-up ticks excluded"
        )
        rows.extend(f"note: {n}" for n in self.notes)
        return "\n".join(rows)

    def to_json(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "ticks": self.ticks,
            "device": self.device,
            "stages": {
                k: {"p50_ms": v[0], "p99_ms": v[1], "budget_ms": BUDGET_MS.get(k)}
                for k, v in self.stages.items()
            },
            "end_to_end_p99_ms": self.end_to_end_p99_ms,
            "sdk_overhead_p99_ms": self.sdk_overhead_p99_ms,
            "reflex_installed": self.reflex_installed,
            "cortex_installed": self.cortex_installed,
            "warmup_ticks": self.warmup_ticks,
            "notes": self.notes,
        }


def run_bench(
    cfg: Config,
    *,
    ticks_file: Path | None = None,
    max_ticks: int | None = None,
    warmup: int = 20,
    device: str = "auto",
    pipeline: Pipeline | None = None,
) -> BenchResult:
    """Measure the pipeline. ``ticks_file`` replays an ingest ``ticks.h5`` instead of the sensors."""
    p = pipeline or Pipeline.from_config(cfg, device=device)
    if ticks_file is not None:
        p.ingest = IngestStage(cfg, tick_file=ticks_file)
    rings: dict[str, LatencyRing] = {
        k: LatencyRing(100_000)
        for k in ("ingest", "tokens", "world", "reflex", "cortex", "bridge", "sdk", "total")
    }
    n = 0
    try:
        while max_ticks is None or n < max_ticks:
            res = p.step()
            n += 1
            if n <= warmup:
                continue
            for k, ring in rings.items():
                v = res.latency_ns.get(k)
                if v is not None:
                    ring.add(int(v))
    except StopIteration:
        pass
    finally:
        p.close()
    stages: dict[str, tuple[float, float]] = {}
    for k, r in rings.items():
        if len(r) and k != "total":
            p50, p99 = r.percentiles_ms()
            stages[k] = (p50, p99)
    total_p99 = rings["total"].percentiles_ms((99.0,))[0] if len(rings["total"]) else 0.0
    notes = []
    if not p.reflex.available:
        notes.append("senseai-reflex not installed: the reflex row is the SDK watchdog, not a reflex")
    if not p.cortex.available:
        notes.append("senseai-cortex not installed: no cortex row")
    if p.world.model.device.type == "cpu":
        notes.append("world ran on CPU; the ≤ 6 ms world budget assumes CUDA")
    return BenchResult(
        ticks=max(0, n - warmup),
        stages=stages,
        end_to_end_p99_ms=total_p99,
        sdk_overhead_p99_ms=stages.get("sdk", (0.0, 0.0))[1],
        device=str(p.world.model.device),
        reflex_installed=p.reflex.available,
        cortex_installed=p.cortex.available,
        warmup_ticks=min(warmup, n),
        notes=notes,
    )


def main_json(result: BenchResult) -> str:
    """Pretty JSON for ``--json``."""
    return json.dumps(result.to_json(), indent=2)
