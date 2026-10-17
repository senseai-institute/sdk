from __future__ import annotations

import time
from typing import Any

from senseai_sdk.bridges import CallbackBridge, PlannerCommand
from senseai_sdk.config import Config
from senseai_sdk.pipeline import Pipeline
from senseai_sdk.stages import FallbackCommand
from senseai_sdk.threaded import ThreadedRunner


def _starved(c: Any) -> bool:
    """A command produced without a surprise map: reflex's envelope watchdog or the SDK fallback."""
    return isinstance(c, FallbackCommand) or bool(c.provenance.degraded)


def wait_until(pred: Any, timeout: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_no_drops_at_100_ticks_per_second(config: Config) -> None:
    cmds: list[Any] = []
    p = Pipeline.from_config(config)
    p.on_command(cmds.append)
    r = ThreadedRunner(p)
    r.start()
    assert wait_until(lambda: r.finished, timeout=10.0)
    r.stop()
    drops = r.drops()
    assert drops == {"tokens": 0, "world": 0, "reflex": 0, "cortex": 0}
    assert p.ticks >= 50  # 50 ticks of data; watchdog ticks on the wall clock may add a few
    with_surprise = [
        c for c in cmds if not isinstance(c, FallbackCommand) and c.provenance.surprise_tick_index
    ]
    assert len(with_surprise) >= 45
    h = p.health()
    assert h.ticks == p.ticks and h.end_to_end_p99_ms > 0


def test_stalled_world_degrades_to_brake_within_budget(config: Config) -> None:
    """A world stall never freezes the last command: the wall-clock watchdog keeps the bridge fed."""
    cmds: list[tuple[float, Any]] = []
    acts: list[tuple[float, PlannerCommand]] = []
    bridge = CallbackBridge(
        lambda c: acts.append((time.monotonic(), c)), planner=lambda: PlannerCommand(1.0, 0.0)
    )
    p = Pipeline.from_config(config, bridge=bridge)
    p.on_command(lambda c: cmds.append((time.monotonic(), c)))
    real_step = p.world.step
    stall_from: list[float | None] = [None]

    def slow(batch: Any) -> Any:
        if stall_from[0] is not None:
            time.sleep(0.4)
        return real_step(batch)

    p.world.step = slow  # type: ignore[method-assign]
    r = ThreadedRunner(p)
    r.start()
    assert wait_until(lambda: p.ticks >= 10, timeout=5.0)
    t0 = time.monotonic()
    stall_from[0] = t0

    def starved() -> list[tuple[float, Any]]:
        return [(t, c) for t, c in cmds if t > t0 and _starved(c)]

    def braking() -> list[tuple[float, Any]]:
        return [(t, c) for t, c in starved() if c.action == "brake" and c.stop]

    assert wait_until(lambda: len(braking()) >= 3, timeout=3.0)
    r.stop()
    t_first, _ = braking()[0]
    assert t_first - t0 < 0.7  # reflex watchdog: slow at 50 ms, brake at 250 ms, + one stalled world call
    assert any(t > t0 and a.linear == 0.0 for t, a in acts)
    assert p.health().state == "degraded"


def test_profile_swap_while_running(config: Config) -> None:
    p = Pipeline.from_config(config)
    r = ThreadedRunner(p)
    r.start()
    assert wait_until(lambda: p.ticks >= 5, timeout=5.0)
    p.load_profile("panther", 0)
    assert wait_until(lambda: p.profiles.active_ref == ("panther", 0), timeout=5.0)
    r.stop()
