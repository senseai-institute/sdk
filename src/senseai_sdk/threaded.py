# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Threaded mode: one thread per stage, bounded drop-oldest queues, a wall-clock reflex thread.

Threads (in pipeline order):

1. **ingest**: pulls ticks from the aligner (paced to real time for replay) → ``q_ticks``
2. **tokens**: applies queued between-tick hooks, tokenizes, updates profiles → ``q_batches``
3. **world**: surprise → ``q_surprise``
4. **reflex**: every ``tick_ms`` on the wall clock, takes the newest surprise (or none) and runs
   reflex → bridge → command callbacks; hands the result to ``q_cortex``
5. **cortex**: events → event callbacks

Queues are bounded (ticks and batches four deep, surprise one deep): a stage that falls
behind drops the oldest item and counts the drop. The reflex thread never waits for the
world: if no surprise arrived within a tick, it runs the watchdog, so a stalled stage
degrades the robot to ``slow`` then ``brake`` on schedule instead of freezing the last
command.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from senseai_sdk.pipeline import Pipeline, StepResult, oldest_sample_ns

__all__ = ["ThreadedRunner"]

log = logging.getLogger("senseai.threaded")


_END = object()


class _Slot:
    """A one-item mailbox: put drops the previous item (and counts it); get waits with a timeout.

    :meth:`close` marks the end of the stream; :meth:`get` returns ``_END`` once the slot is
    drained, so closing never discards the last item.
    """

    def __init__(self, depth: int = 1) -> None:
        self._q: deque[Any] = deque(maxlen=depth)
        self._cv = threading.Condition()
        self._closed = False
        self.drops = 0

    def put(self, item: Any) -> None:
        with self._cv:
            if len(self._q) == self._q.maxlen:
                self.drops += 1
            self._q.append(item)
            self._cv.notify()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def get(self, timeout: float) -> Any | None:
        with self._cv:
            if not self._q and not self._closed:
                self._cv.wait(timeout)
            if self._q:
                return self._q.popleft()
            return _END if self._closed else None


class ThreadedRunner:
    """Runs a :class:`Pipeline` in threaded mode. :meth:`start`, :meth:`stop`, :meth:`join`."""

    def __init__(self, pipeline: Pipeline, *, torch_threads: int | None = None) -> None:
        self.p = pipeline
        self.torch_threads = torch_threads
        # ticks and batches may queue a few deep to absorb start-up jitter (torch's first calls are
        # slow); surprise is one deep because the reflex must act on the newest map, never a stale one
        self.q_ticks, self.q_batches = _Slot(depth=4), _Slot(depth=4)
        self.q_surprise, self.q_cortex = _Slot(depth=1), _Slot(depth=1)
        self._stop = threading.Event()
        self._ended = threading.Event()  # replay finished
        self._threads: list[threading.Thread] = []
        self.results: deque[StepResult] = deque(maxlen=1)

    # -- lifecycle ------------------------------------------------------------------------

    def start(self) -> None:
        """Open devices and start the five threads."""
        self._cap_torch_threads()
        self.p.open(paced=True)
        self._stop.clear()
        self._ended.clear()
        for name, fn in (
            ("ingest", self._ingest),
            ("tokens", self._tokens),
            ("world", self._world),
            ("reflex", self._reflex),
            ("cortex", self._cortex),
        ):
            t = threading.Thread(target=fn, name=f"senseai-{name}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        """Stop the threads, publish the final hold and close devices."""
        self._stop.set()
        self.join(timeout=5.0)
        self.p.close()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the threads (they end on :meth:`stop` or at the end of a replay)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        for t in self._threads:
            t.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
        self._threads = [t for t in self._threads if t.is_alive()]

    @property
    def finished(self) -> bool:
        """True once the replay has ended and the reflex thread has drained."""
        return self._ended.is_set() and not any(t.is_alive() for t in self._threads)

    def _cap_torch_threads(self) -> None:
        """Leave cores for the pipeline threads when the models run on CPU.

        torch's OpenMP workers spin-wait after every op; with one worker per core they starve
        the ingest, reflex and cortex threads and the world stage runs 3-4× slower than in
        single-loop mode (measured: 12 ms vs 3 ms on 4 cores, with dropped ticks). The rule
        ``max(1, cpu_count // 2 - 1)`` gives 1 thread on 4 cores and 7 on 16. ``torch_threads``
        overrides it; on CUDA the cap is irrelevant and skipped.
        """
        import os

        import torch

        if self.p.world.model.device.type != "cpu":
            return
        n = self.torch_threads if self.torch_threads is not None else max(1, (os.cpu_count() or 2) // 2 - 1)
        torch.set_num_threads(n)
        log.info("threaded mode: torch intra-op threads capped at %d", n)

    # -- threads --------------------------------------------------------------------------

    def _ingest(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    tick = self.p.ingest.step()
                except StopIteration:
                    break
                self.q_ticks.put(tick)
        finally:
            self.q_ticks.close()

    def _tokens(self) -> None:
        p = self.p
        try:
            while not self._stop.is_set():
                tick = self.q_ticks.get(0.05)
                if tick is None:
                    continue
                if tick is _END:
                    break
                p._apply_hooks()
                self.q_batches.put(self._tokenize(tick))
        finally:
            self.q_batches.close()

    def _tokenize(self, tick: Any) -> tuple[Any, Any, list[str]]:
        p = self.p
        reasons: list[str] = []
        batch = p._guard(p.tokens.stats, lambda: p.tokens.step(tick), reasons, "tokens")
        if batch is not None:
            p._guard(
                p.profiles.stats, lambda: p.profiles.per_tick(p.tokens.tokenizer, batch), reasons, "profiles"
            )
        return tick, batch, reasons

    def _world(self) -> None:
        p = self.p
        try:
            while not self._stop.is_set():
                item = self.q_batches.get(0.05)
                if item is None:
                    continue
                if item is _END:
                    break
                tick, batch, reasons = item
                surprise = world_state = None
                if batch is not None:
                    b = batch
                    out = p._guard(p.world.stats, lambda: p.world.step(b), reasons, "world")  # noqa: B023
                    if out is not None:
                        surprise, world_state = out
                self.q_surprise.put((tick, batch, surprise, world_state, reasons))
        finally:
            self.q_surprise.close()

    def _reflex(self) -> None:
        """Wall-clock loop: one command per tick, whether or not the world delivered."""
        p = self.p
        period = p.cfg.tick_ns / 1e9
        next_due = time.perf_counter()
        ended = False
        try:
            while not self._stop.is_set():
                # wait until the tick is due (a floor keeps this from spinning on the GIL)
                item = self.q_surprise.get(max(0.0005, next_due - time.perf_counter()))
                if item is _END:
                    ended = True
                    break
                now = time.perf_counter()
                if item is None and now < next_due:
                    continue
                next_due = max(next_due + period, now)  # never schedule into the past after a stall
                t_start = time.perf_counter_ns()
                lat: dict[str, int] = {}
                if item is not None:
                    tick, batch, surprise, world_state, reasons = item
                    t_ns = int(tick.t_master_ns)
                    lat["ingest"] = p._lat(p.ingest.stats)
                    if batch is not None:
                        lat["tokens"] = p._lat(p.tokens.stats)
                    if surprise is not None:
                        lat["world"] = p._lat(p.world.stats)
                else:  # nothing arrived this tick: the watchdog decides
                    tick = batch = surprise = world_state = None
                    reasons = ["no SurpriseMap within the tick"]
                    t_ns = time.time_ns()
                t_sensor = oldest_sample_ns(tick) if tick is not None else None
                cmd, actuator, t_emit = p.reflex_path(surprise, t_ns, lat, reasons, t_sensor_ns=t_sensor)
                res = StepResult(
                    tick_index=int(tick.index) if tick is not None else -1,
                    t_master_ns=t_ns,
                    command=cmd,
                    events=[],
                    surprise=surprise,
                    latency_ns=lat,
                    actuator=actuator,
                    reasons=reasons,
                    tick=tick if p.keep_ticks else None,
                    t_sensor_ns=t_sensor,
                    t_emit_ns=t_emit,
                    world_mode=p.world.mode,
                    world_model_id=p.world.model_id,
                )
                p.finish_tick(res, t_start)
                self.results.append(res)
                if tick is not None:
                    self.q_cortex.put((tick, world_state, surprise, cmd, batch))
        finally:
            self.q_cortex.close()
            if ended:
                self._ended.set()

    def _cortex(self) -> None:
        p = self.p
        while not self._stop.is_set():
            item = self.q_cortex.get(0.05)
            if item is None:
                continue
            if item is _END:
                break
            tick, world_state, surprise, cmd, batch = item
            p.cortex_path(int(tick.index), world_state, surprise, cmd, batch, {})

    # -- counters ---------------------------------------------------------------------------

    def drops(self) -> dict[str, int]:
        """Drops per queue, also mirrored into the stage stats."""
        d = {
            "tokens": self.q_ticks.drops,
            "world": self.q_batches.drops,
            "reflex": self.q_surprise.drops,
            "cortex": self.q_cortex.drops,
        }
        self.p.tokens.stats.drops = d["tokens"]
        self.p.world.stats.drops = d["world"]
        self.p.reflex.stats.drops = d["reflex"]
        self.p.cortex.stats.drops = d["cortex"]
        return d
