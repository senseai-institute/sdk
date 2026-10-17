# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Wires ingest → tokens → profiles → world → reflex → bridge → cortex.

Single-loop mode: :meth:`Pipeline.step` runs one tick through every stage, in order, on the
calling thread. It is deterministic. Threaded mode (:mod:`senseai_sdk.threaded`) runs the
same stage objects on one thread each with bounded queues.

Fail-safe: every stage call is guarded. A stage that raises marks itself failed for this
tick, the pipeline keeps going with what it has (no surprise map → the reflex watchdog;
no command → the watchdog), and health reports ``degraded`` with the reason. The bridge is
called on every tick no matter what, so the controller always has a fresh command.

Profile swaps, sensor binds and context updates are queued and applied at the top of the
next tick, never during one.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from senseai_sdk.artefacts import Artefact, ArtefactStore
from senseai_sdk.bridges import BridgeLike, PlannerCommand, make_bridge
from senseai_sdk.config import Config
from senseai_sdk.errors import ArtefactError, PipelineError
from senseai_sdk.health import Health, LatencyRing, StageStats
from senseai_sdk.stages import (
    CortexStage,
    IngestStage,
    MotionContext,
    ProfilesStage,
    ReflexStage,
    TokensStage,
    WorldStage,
    load_calibration,
)

__all__ = ["Pipeline", "StepResult"]

log = logging.getLogger("senseai")

CommandCallback = Callable[[Any], None]
EventCallback = Callable[[Any], None]
LicenceState = Callable[[], tuple[str, int]]


@dataclass(slots=True)
class StepResult:
    """What one tick produced. See ``SenseAI.step``."""

    tick_index: int
    t_master_ns: int
    command: Any
    events: list[Any]
    surprise: Any | None
    latency_ns: Mapping[str, int]
    degraded: bool = False
    actuator: PlannerCommand | None = None
    reasons: list[str] = field(default_factory=list)


class Pipeline:
    """The composed stages for one robot. Build with :meth:`from_config`."""

    def __init__(
        self,
        cfg: Config,
        *,
        ingest: IngestStage,
        tokens: TokensStage,
        profiles: ProfilesStage,
        world: WorldStage,
        reflex: ReflexStage,
        cortex: CortexStage,
        bridge: BridgeLike,
        artefacts: Mapping[str, Artefact],
        store: ArtefactStore,
        licence_state: LicenceState | None = None,
    ) -> None:
        self.cfg = cfg
        self.ingest, self.tokens, self.profiles = ingest, tokens, profiles
        self.world, self.reflex, self.cortex = world, reflex, cortex
        self.bridge = bridge
        self.artefacts = dict(artefacts)
        self.store = store
        self._licence_state = licence_state or (lambda: ("valid", 0))
        self.motion = MotionContext(
            speed_cap_mps=cfg.robot.speed_cap_mps,
            footprint_radius_m=cfg.robot.footprint_radius_m,
            max_decel_mps2=cfg.robot.max_decel_mps2,
            kind=cfg.robot.kind,
        )
        self.sdk_stats = StageStats("sdk")
        self.total = LatencyRing()
        self.e2e = LatencyRing()
        self.ticks = 0
        self._cmd_cbs: list[CommandCallback] = []
        self._event_cbs: list[EventCallback] = []
        self._hooks: list[Callable[[Pipeline], None]] = []  # applied between ticks
        self._lock = threading.Lock()
        self._opened = False
        self._stopped = False
        self._last_reasons: list[str] = []
        self._world_ok_ticks = 0
        self.on_tick: Callable[[StepResult], None] | None = None  # telemetry / audit tap

    # -- construction -----------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        store: ArtefactStore | None = None,
        bridge: BridgeLike | None = None,
        licence_state: LicenceState | None = None,
        device: str = "auto",
        priors_key: bytes | None = None,
    ) -> Pipeline:
        """Verify artefacts and build every stage. Opens no devices.

        Raises:
            ArtefactError: Any artefact fails verification, or the world artefact's tick does
                not match ``tick_ms``, or its tokenizer does not match the checkpoint.
            PipelineError: A sensor adapter rejects its options, a calibration file is missing.
        """
        store = store or ArtefactStore(cfg.artefacts.store, cfg.artefacts.cache)
        world_art = store.fetch(cfg.artefacts.world)
        reflex_art = store.fetch(cfg.artefacts.reflex)
        for name in ("checkpoint.ckpt", "tokens.yaml"):
            if name not in world_art.files:
                raise ArtefactError(f"{world_art.id}: world artefact has no {name}")
        tokens = TokensStage(
            world_art, load_calibration(cfg.sensors), device="cpu" if device == "auto" else device
        )
        world = WorldStage(world_art, device=device)
        if world.tick_ns != cfg.tick_ns:
            raise ArtefactError(
                f"tick_ms: config says {cfg.tick_ms} ms but {world_art.id} was trained at "
                f"{world.tick_ns // 1_000_000} ms; they must match"
            )
        if world.tokenizer_id and world.tokenizer_id != tokens.tokenizer_id:
            raise ArtefactError(
                f"{world_art.id}: checkpoint expects tokenizer {world.tokenizer_id!r} but its tokens.yaml "
                f"builds {tokens.tokenizer_id!r}; the artefact is inconsistent"
            )
        ingest = IngestStage(cfg)
        profiles = ProfilesStage([s.spec for s in cfg.sensors if s.spec], priors_key=priors_key)
        reflex = ReflexStage(reflex_art, cfg.robot, cfg.reflex.max_lateral_m)
        cortex = CortexStage(tick_ms=cfg.tick_ms)
        if bridge is None:
            bridge = make_bridge(cfg.reflex.bridge, max_lateral_m=cfg.reflex.max_lateral_m)
        p = cls(
            cfg,
            ingest=ingest,
            tokens=tokens,
            profiles=profiles,
            world=world,
            reflex=reflex,
            cortex=cortex,
            bridge=bridge,
            artefacts={"world": world_art, "reflex": reflex_art},
            store=store,
            licence_state=licence_state,
        )
        if cfg.profile.name != "null" or cfg.profile.version is not None:
            p.load_profile(cfg.profile.name, cfg.profile.version)
            p._apply_hooks()
        return p

    # -- between-tick requests ---------------------------------------------------------------

    def load_profile(self, name: str, version: int | None = None) -> Artefact | None:
        """Fetch, verify and queue a profile swap for the next tick boundary.

        ``null`` with no version needs no artefact and clears the priors.
        """
        if name == "null" and version is None:
            self._queue(lambda p: p._set_profile(None))
            return None
        art = self.store.fetch(f"profile-{name}-v{version if version is not None else self._newest(name)}")
        prof = self.profiles.load(art, name, version)
        self.artefacts["profile"] = art
        self._queue(lambda p: p._set_profile(prof))
        return art

    def _newest(self, name: str) -> int:
        """The newest profile version in the store. Only local stores can be listed."""
        if self.store.remote:
            raise ArtefactError(f"profile {name!r}: pin a version in config when the store is remote")
        from pathlib import Path

        prefix = f"profile-{name}-v"
        root = Path(str(self.store.store)).expanduser()
        versions = sorted(
            int(p.name[len(prefix) :]) for p in root.glob(prefix + "*") if p.name[len(prefix) :].isdigit()
        )
        if not versions:
            raise ArtefactError(f"profile {name!r}: no profile-{name}-v* in {root}")
        return versions[-1]

    def _set_profile(self, profile: Any | None) -> None:
        if profile is None:
            self.tokens.tokenizer.set_profile(None)
            self.profiles.active = None
            self.profiles.active_ref = ("null", 0)
            return
        self.profiles.activate(profile, self.tokens.tokenizer, self.reflex)

    def bind_sensor(self, source_id: str, adapter: Any) -> None:
        """Queue an adapter add/replace for the next tick boundary."""
        if source_id not in self.tokens.tokenizer.cfg.sources:
            known = list(self.tokens.tokenizer.cfg.sources)
            raise ValueError(f"source {source_id!r} is not in the tokenizer vocabulary {known}")
        self._queue(lambda p: p.ingest.bind(source_id, adapter))

    def bind_actuator(self, bridge: BridgeLike) -> None:
        """Queue a bridge replacement for the next tick boundary."""
        self._queue(lambda p: setattr(p, "bridge", bridge))

    def update_context(self, **ctx: Any) -> None:
        """Queue scene and motion context. Unknown keys raise immediately."""
        known = {"lux", "lux_map", "speed", "yaw_rate", "heading", "speed_cap", "mode"}
        bad = set(ctx) - known
        if bad:
            raise KeyError(f"unknown context {sorted(bad)}; known: {sorted(known)}")

        def apply(p: Pipeline) -> None:
            m = p.motion
            if "speed" in ctx:
                m.speed_mps = float(ctx["speed"])
            if "yaw_rate" in ctx:
                m.yaw_rate_rps = float(ctx["yaw_rate"])
            if "heading" in ctx:
                m.heading_rad = float(ctx["heading"])
            if "speed_cap" in ctx:
                m.speed_cap_mps = float(ctx["speed_cap"])
            if "mode" in ctx:
                m.mode = str(ctx["mode"])
            scene = {k: ctx[k] for k in ("lux", "lux_map", "speed") if k in ctx}
            if scene:
                p.profiles.update_context(**scene)

        self._queue(apply)

    def on_command(self, cb: CommandCallback) -> None:
        """Register a per-tick command callback."""
        self._cmd_cbs.append(cb)

    def on_event(self, cb: EventCallback) -> None:
        """Register a cortex event callback."""
        self._event_cbs.append(cb)

    def _queue(self, fn: Callable[[Pipeline], None]) -> None:
        with self._lock:
            self._hooks.append(fn)

    def _apply_hooks(self) -> None:
        with self._lock:
            hooks, self._hooks = self._hooks, []
        for h in hooks:
            h(self)

    # -- lifecycle ----------------------------------------------------------------------------

    def open(self) -> None:
        """Open devices and register their specs with the tokenizer."""
        if self._opened:
            return
        self.ingest.open()
        self.tokens.set_specs(self.ingest.specs())
        self._opened = True

    def close(self) -> None:
        """Publish a final hold, then close devices. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        if self._opened:
            try:
                self.bridge.apply(
                    self.reflex._watchdog_cmd(0, self.profiles.active_ref, True), PlannerCommand()
                )
            except Exception as e:
                log.warning("final hold failed: %s", e)
            self.ingest.close()
        self._opened = False
        self._stopped = False

    # -- the tick ----------------------------------------------------------------------------

    def step(self) -> StepResult:
        """One tick through every stage. Raises ``StopIteration`` at the end of a replay."""
        if not self._opened:
            self.open()
        self._apply_hooks()
        t_start = time.perf_counter_ns()
        reasons: list[str] = []
        lat: dict[str, int] = {}

        tick = self.ingest.step()  # StopIteration propagates: end of replay
        lat["ingest"] = self.ingest.stats.latency._buf[(self.ingest.stats.latency._i - 1) % 1000]
        t_ns = int(tick.t_master_ns)

        batch = self._guard(self.tokens.stats, lambda: self.tokens.step(tick), reasons, "tokens")
        if batch is not None:
            lat["tokens"] = self.tokens.stats.latency._buf[(self.tokens.stats.latency._i - 1) % 1000]
            self._guard(
                self.profiles.stats,
                lambda: self.profiles.per_tick(self.tokens.tokenizer, batch),
                reasons,
                "profiles",
            )

        surprise = world_state = None
        if batch is not None:
            out = self._guard(self.world.stats, lambda: self.world.step(batch), reasons, "world")
            if out is not None:
                surprise, world_state = out
                lat["world"] = self.world.stats.latency._buf[(self.world.stats.latency._i - 1) % 1000]

        cmd = self.reflex.step(surprise, self.motion, t_ns, self.profiles.active_ref)
        if self.reflex.available:
            lat["reflex"] = self.reflex.stats.latency._buf[(self.reflex.stats.latency._i - 1) % 1000]
        if getattr(cmd, "degraded", False):
            reasons.append(cmd.provenance.reason)

        t_b = time.perf_counter_ns()
        planner = self.bridge.planner_cmd() if hasattr(self.bridge, "planner_cmd") else PlannerCommand()
        actuator = self._guard(self.sdk_stats, lambda: self.bridge.apply(cmd, planner), reasons, "bridge")
        lat["bridge"] = time.perf_counter_ns() - t_b

        for cb in self._cmd_cbs:
            try:
                cb(cmd)
            except Exception as e:
                self.sdk_stats.fail(e)
                reasons.append(f"on_command callback raised {type(e).__name__}")

        events = self.cortex.step(tick.index, world_state, surprise, cmd, batch)
        if self.cortex.available:
            lat["cortex"] = self.cortex.stats.latency._buf[(self.cortex.stats.latency._i - 1) % 1000]
        for ev in events:
            for ecb in self._event_cbs:
                try:
                    ecb(ev)
                except Exception as e:
                    self.sdk_stats.fail(e)

        total = time.perf_counter_ns() - t_start
        lat["total"] = total
        lat["sdk"] = max(0, total - sum(v for k, v in lat.items() if k not in ("total", "sdk")))
        self.sdk_stats.record(lat["sdk"])
        self.total.add(total)
        if not self.ingest.live:
            self.e2e.add(total)  # replay: sensor time is not wall time, so the loop is the e2e
        else:
            self.e2e.add(max(0, time.time_ns() - t_ns))
        self.ticks += 1
        self._last_reasons = reasons
        res = StepResult(
            tick_index=int(tick.index),
            t_master_ns=t_ns,
            command=cmd,
            events=events,
            surprise=surprise,
            latency_ns=lat,
            degraded=bool(reasons),
            actuator=actuator if isinstance(actuator, PlannerCommand) else None,
            reasons=reasons,
        )
        if self.on_tick is not None:
            self.on_tick(res)
        return res

    @staticmethod
    def _guard(stats: StageStats, fn: Callable[[], Any], reasons: list[str], name: str) -> Any:
        try:
            return fn()
        except Exception as e:
            stats.fail(e)
            reasons.append(f"{name} stage failed: {type(e).__name__}: {e}")
            log.exception("%s stage failed", name)
            return None

    # -- state and health ---------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Every stage's checkpointable state plus the active profile."""
        return {
            "format": "senseai-sdk/state",
            "version": 1,
            "artefacts": {k: a.id for k, a in self.artefacts.items()},
            "profile": list(self.profiles.active_ref),
            "ticks": self.ticks,
            "tokens": self.tokens.state(),
            "world": self.world.state(),
            "reflex": self.reflex.state(),
            "cortex": self.cortex.state(),
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        """Restore :meth:`state` between ticks."""
        mine = {k: a.id for k, a in self.artefacts.items() if k in ("world", "reflex")}
        theirs = {k: v for k, v in state.get("artefacts", {}).items() if k in ("world", "reflex")}
        if theirs != mine:
            raise ArtefactError(f"state was saved with artefacts {theirs}, this pipeline runs {mine}")

        def apply(p: Pipeline) -> None:
            p.tokens.restore(state["tokens"])
            p.world.restore(state["world"])
            p.reflex.restore(state.get("reflex", b""))
            p.cortex.restore(state.get("cortex", b""))
            p.ticks = int(state.get("ticks", 0))

        self._queue(apply)

    def health(self) -> Health:
        """Current health. Cheap."""
        stages = [
            s.stats.snapshot() for s in (self.ingest, self.tokens, self.world, self.reflex, self.cortex)
        ]
        stages.append(self.sdk_stats.snapshot())
        reasons = list(self._last_reasons)
        for s in stages:
            if not s.available and s.reason:
                reasons.append(s.reason)
        lic, days = self._licence_state()
        state: Any = "ok"
        if reasons or self.reflex.watchdog != "idle":
            state = "degraded"
        if lic == "expired":
            state = "failed"
            reasons.append("licence expired")
        return Health(
            state=state,
            endpoint_id=self.cfg.endpoint_id,
            stages=stages,
            end_to_end_p99_ms=self.e2e.percentiles_ms((99.0,))[0],
            watchdog=self.reflex.watchdog,
            licence=lic,  # type: ignore[arg-type]
            licence_days_left=days,
            artefacts={k: a.id for k, a in self.artefacts.items()},
            profile=self.profiles.active_ref,
            ticks=self.ticks,
            reasons=reasons,
        )


def _unused() -> None:  # keeps PipelineError imported for callers that catch it from here
    raise PipelineError
