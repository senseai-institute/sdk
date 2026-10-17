# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""The pipeline's stages, each wrapping one upstream package behind a small, uniform surface.

A stage owns one upstream object, converts between the SDK's config and that package's
constructors, and records its own latency. Stages add no behaviour; the one exception is
:class:`ReflexStage`, which carries the SDK's fail-safe command when ``senseai-reflex`` is
missing or failing, because the SDK must never stop publishing commands.

``senseai-reflex`` and ``senseai-cortex`` are not started yet. Their stages import them
lazily, report ``available=False`` when absent, and the pipeline runs in the degraded state
with the watchdog driving the bridge. When they land, only the two ``_import`` bodies and the
argument mapping change.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from senseai_sdk import adapters as adapter_table
from senseai_sdk.artefacts import Artefact
from senseai_sdk.config import Config, RobotConfig, SensorConfig
from senseai_sdk.errors import ArtefactError, PipelineError
from senseai_sdk.health import StageStats

__all__ = [
    "CortexStage",
    "FallbackCommand",
    "IngestStage",
    "MotionContext",
    "ProfilesStage",
    "Provenance",
    "ReflexStage",
    "TokensStage",
    "WorldStage",
]

Action = Literal["none", "slow", "brake", "swerve", "retract", "hold"]


# -- the SDK's own small types at the stage boundaries ---------------------------------------


@dataclass(slots=True)
class MotionContext:
    """What the reflex motion model needs from the robot, updated by ``update_context``.

    Fields map onto ``senseai_reflex.motion.RobotState`` when reflex ships.
    """

    speed_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    heading_rad: float = 0.0
    speed_cap_mps: float = 1.5
    footprint_radius_m: float = 0.35
    max_decel_mps2: float = 2.0
    kind: str = "amr"
    mode: str = ""


@dataclass(slots=True)
class Provenance:
    """Provenance of a :class:`FallbackCommand`, shaped like reflex's ``Provenance``."""

    cells: tuple[int, ...] = ()
    heads: Mapping[str, float] = field(default_factory=dict)
    profile: str = "null"
    profile_version: int = 0
    surprise_scale: Mapping[str, float] = field(default_factory=dict)
    latency_ns: Mapping[str, int] = field(default_factory=dict)
    reason: str = ""


@dataclass(slots=True)
class FallbackCommand:
    """The command the SDK publishes when the reflex stage cannot: the watchdog's ``slow``/``brake``.

    Same fields as ``senseai_reflex.command.ReflexCommand`` so bridges and callbacks see one
    shape. ``provenance.reason`` says why it is a fallback.
    """

    t_master_ns: int
    action: Action
    speed_scale: float
    lateral_offset_m: float
    stop: bool
    hold: bool
    latched: bool
    risk: float
    threshold: float
    provenance: Provenance
    degraded: bool = True


def _timed(stats: StageStats, fn: Callable[[], Any]) -> Any:
    t0 = time.perf_counter_ns()
    out = fn()
    stats.record(time.perf_counter_ns() - t0)
    return out


# -- ingest -----------------------------------------------------------------------------------


class IngestStage:
    """Builds the configured adapters and the ``Aligner``; yields one ``AlignedTick`` per step.

    With ``tick_file`` it reads an ingest ``ticks.h5`` instead (``senseai bench --ticks``).
    """

    name = "ingest"

    def __init__(self, cfg: Config, tick_file: Path | None = None) -> None:
        self.cfg = cfg
        self.stats = StageStats(self.name)
        self._adapters: dict[str, Any] = {}
        self._aligner: Any = None
        self._reader: Any = None
        self._iter: Iterator[Any] | None = None
        self.live = False
        self.paced = False
        self._t0_wall_ns = 0
        self._t0_master_ns: int | None = None
        self.tick_file = tick_file
        if tick_file is None:
            for s in cfg.sensors:
                self._adapters[s.id] = self._build(s)

    @staticmethod
    def _build(s: SensorConfig) -> Any:
        try:
            return adapter_table.build(s.adapter, s.id, s.options)
        except TypeError as e:
            raise PipelineError(f"sensors[{s.id}]: {e}") from e

    def bind(self, source_id: str, adapter: Any) -> None:
        """Add or replace a source. Takes effect at the next :meth:`open`."""
        spec_id = adapter.spec().source_id
        if spec_id != source_id:
            raise ValueError(f"adapter reports source_id {spec_id!r}, not {source_id!r}")
        self._adapters[source_id] = adapter
        if self._aligner is not None:
            self.close()

    def open(self, *, paced: bool = False) -> None:
        """Open every adapter and start the aligner (or the tick-file reader)."""
        self.paced = paced
        self._t0_wall_ns = time.time_ns()
        self._t0_master_ns = None
        if self.tick_file is not None:
            from senseai_ingest.writer import TickReader

            self._reader = TickReader(self.tick_file)
            if int(self._reader.tick_ns) != self.cfg.tick_ns:
                file_ms = int(self._reader.tick_ns) // 1_000_000
                raise PipelineError(f"{self.tick_file}: tick {file_ms} ms, config {self.cfg.tick_ms} ms")
            self._iter = self._paced(iter(self._reader)) if paced else iter(self._reader)
            return
        from senseai_ingest.sync import Aligner, AlignerConfig

        ads = list(self._adapters.values())
        self.live = any(getattr(a, "is_live", False) for a in ads)
        speed = None if self.live else (1.0 if paced else 0.0)
        self._aligner = Aligner(ads, AlignerConfig(tick_ns=self.cfg.tick_ns, speed=speed))
        self._aligner.open()
        self._iter = iter(self._aligner)

    def _paced(self, it: Iterator[Any]) -> Iterator[Any]:
        """Replay a tick file at real time."""
        t0: int | None = None
        wall0 = time.perf_counter_ns()
        for tick in it:
            if t0 is None:
                t0 = int(tick.t_master_ns)
            due = wall0 + (int(tick.t_master_ns) - t0)
            while time.perf_counter_ns() < due:
                time.sleep(min(0.001, (due - time.perf_counter_ns()) / 1e9))
            yield tick

    def wall_ns(self, t_master_ns: int) -> int:
        """Map a master-clock time onto the wall clock (identity for live sources)."""
        if self.live:
            return t_master_ns
        if self._t0_master_ns is None:
            self._t0_master_ns = t_master_ns
        return self._t0_wall_ns + (t_master_ns - self._t0_master_ns)

    def specs(self) -> dict[str, Any]:
        """Source specs, available after :meth:`open`."""
        if self._aligner is None and self._reader is None:
            self.open()
        src = self._reader if self._reader is not None else self._aligner
        return dict(src.specs())

    def step(self) -> Any:
        """The next ``AlignedTick``. Raises ``StopIteration`` at the end of a replay."""
        if self._iter is None:
            self.open()
        assert self._iter is not None
        return _timed(self.stats, lambda: next(self._iter))  # type: ignore[arg-type]

    def ingest_stats(self) -> dict[str, Any]:
        """The aligner's own statistics (drops, late samples)."""
        return dict(self._aligner.stats()) if self._aligner is not None else {}

    def close(self) -> None:
        """Close adapters."""
        if self._aligner is not None:
            self._aligner.close()
        if self._reader is not None:
            self._reader.close()
        self._aligner = self._reader = None
        self._iter = None


# -- tokens -----------------------------------------------------------------------------------


class TokensStage:
    """The ``senseai_tokens.Tokenizer`` built from the world artefact's ``tokens.yaml``."""

    name = "tokens"

    def __init__(self, world: Artefact, calibration: Mapping[str, Any] | None, device: str = "cpu") -> None:
        from senseai_tokens.config import TokenizerConfig
        from senseai_tokens.frame import BodyFrame
        from senseai_tokens.tokenizer import Tokenizer

        self.stats = StageStats(self.name)
        tcfg = TokenizerConfig.from_yaml(world.path("tokens.yaml"))
        frame = BodyFrame.from_dict(calibration) if calibration else None
        self.tokenizer = Tokenizer(tcfg, calibration=frame, device=device)
        self.tokenizer_id: str = self.tokenizer.tokenizer_id

    def set_specs(self, specs: Mapping[str, Any]) -> None:
        """Register ingest source specs (calibration from their meta)."""
        self.tokenizer.set_specs(specs)

    def step(self, tick: Any) -> Any:
        """``AlignedTick`` → ``TokenBatch``."""
        return _timed(self.stats, lambda: self.tokenizer(tick))

    def state(self) -> dict[str, Any]:
        """Checkpointable running stats."""
        return dict(self.tokenizer.state_dict())

    def restore(self, state: Mapping[str, Any]) -> None:
        """Restore :meth:`state`."""
        self.tokenizer.load_state_dict(state)


# -- profiles ---------------------------------------------------------------------------------


class ProfilesStage:
    """The ``senseai_profiles.Registry`` over verified profile artefacts.

    A profile artefact (``profile-<name>-v<N>``) carries ``profiles/``, ``priors/``,
    ``library/`` and ``sensors/`` so the registry is self-contained. Profiles are applied to
    the tokenizer and the reflex between ticks only.
    """

    name = "profiles"

    def __init__(self, sensor_spec_ids: Sequence[str], priors_key: bytes | None = None) -> None:
        self.stats = StageStats(self.name)
        self._spec_ids = tuple(sensor_spec_ids)
        self._key = priors_key
        self._registries: dict[str, Any] = {}
        self.active: Any = None
        self.active_ref: tuple[str, int] = ("null", 0)
        self._pending_scene: dict[str, Any] = {}

    def load(self, artefact: Artefact, name: str, version: int | None) -> Any:
        """Compose profile ``name`` from a verified artefact."""
        from senseai_profiles.registry import Registry, RegistryError
        from senseai_profiles.sensors import load_sensors

        root = artefact.root
        sensors_dir = root / "sensors"
        all_specs = load_sensors(sensors_dir) if sensors_dir.is_dir() else {}
        wanted = {k: v for k, v in all_specs.items() if k in self._spec_ids} if self._spec_ids else all_specs
        try:
            reg = Registry(
                library_dir=root / "library",
                profiles_dir=root / "profiles",
                priors_dir=root / "priors",
                sensors=wanted,
                key=self._key,
            )
            prof = reg.get(name, version)
        except RegistryError as e:
            raise ArtefactError(f"{artefact.id}: {e}") from e
        self._registries[artefact.id] = reg
        return prof

    def activate(self, profile: Any, tokenizer: Any, reflex: Any | None) -> None:
        """Install ``profile`` on the tokenizer and reflex. Between ticks only."""
        from senseai_profiles.profile import SceneError

        self.active = profile
        self.active_ref = (profile.name, int(profile.version))
        with contextlib.suppress(SceneError):  # resolved on the first tick that carries the lux map
            profile.apply(tokenizer, self._pending_scene or None)
        if reflex is not None and getattr(reflex, "available", False):
            profile.apply(reflex.engine)

    def update_context(self, **scene: Any) -> None:
        """Scene variables for the next tick (``lux``, ``lux_map``, ``speed``)."""
        self._pending_scene.update(scene)

    def per_tick(self, tokenizer: Any, batch: Any) -> None:
        """Re-resolve scene-dependent priors from the batch just produced, when they change."""
        if self.active is None:
            return
        from senseai_profiles import runtime
        from senseai_profiles.profile import SceneError

        try:
            if self._pending_scene:
                runtime.update_tokenizer(self.active, tokenizer, **self._pending_scene)
                self._pending_scene.clear()
            runtime.update_from_batch(self.active, tokenizer, batch)
        except SceneError:
            pass


# -- world ------------------------------------------------------------------------------------


class WorldStage:
    """The ``senseai_world.WorldModel`` loaded from the world artefact's checkpoint."""

    name = "world"

    def __init__(self, world: Artefact, device: str = "auto") -> None:
        from senseai_world.inference import WorldModel

        self.stats = StageStats(self.name)
        self.model = WorldModel.load(world.path("checkpoint.ckpt"), device=device)
        self.tick_ns: int = int(self.model.cfg.tick_ns or 0)
        self.tokenizer_id: str | None = self.model.tokenizer_id

    def step(self, batch: Any) -> tuple[Any, Any]:
        """``TokenBatch`` → ``(SurpriseMap, WorldState | None)``."""
        return _timed(self.stats, lambda: self.model.step(batch))  # type: ignore[no-any-return]

    def state(self) -> bytes:
        """Checkpointable window and normaliser state."""
        return bytes(self.model.checkpoint_state())

    def restore(self, blob: bytes) -> None:
        """Restore :meth:`state`."""
        self.model.restore_state(blob)


# -- reflex -----------------------------------------------------------------------------------


class ReflexStage:
    """``senseai_reflex.ReflexEngine`` when installed; the SDK watchdog's fallback when not.

    The watchdog mirrors the reflex envelope's rule so the fail-safe exists before reflex
    does: ``slow`` (speed × 0.5) after ``watchdog_ms`` without a surprise map, ``brake`` after
    ``watchdog_stop_ms``. When reflex ships, its own envelope watchdog is the primary and this
    one only covers the case where the reflex stage itself raises.
    """

    name = "reflex"
    WATCHDOG_MS = 50
    WATCHDOG_STOP_MS = 250

    def __init__(self, reflex: Artefact | None, robot: RobotConfig, max_lateral_m: float) -> None:
        self.stats = StageStats(self.name)
        self.robot = robot
        self.max_lateral_m = max_lateral_m
        self.engine: Any = None
        self.available = False
        self.config: dict[str, Any] = {}
        if reflex is not None and "reflex.yaml" in reflex.files:
            self.config = dict(yaml.safe_load(reflex.path("reflex.yaml").read_text()) or {})
        self._import(reflex)
        self._last_ok_ns: int | None = None
        self.watchdog: Literal["idle", "slow", "brake"] = "idle"

    def _import(self, reflex: Artefact | None) -> None:
        try:
            from senseai_reflex.config import ReflexConfig
            from senseai_reflex.engine import ReflexEngine
        except ImportError as e:
            self.stats.available = False
            self.stats.reason = f"senseai-reflex not installed ({e.name}); watchdog drives the bridge"
            return
        self.engine = ReflexEngine(ReflexConfig.from_mapping(self.config), None)
        self.available = True

    def step(
        self, surprise: Any | None, motion: MotionContext, t_ns: int, profile_ref: tuple[str, int]
    ) -> Any:
        """One tick. Always returns a command; never raises past the pipeline."""
        if self.available and surprise is not None:
            try:
                cmd = _timed(self.stats, lambda: self.engine.step(surprise, self._robot_state(motion), t_ns))
            except Exception as e:
                self.stats.fail(e)
            else:
                self._last_ok_ns = t_ns
                self.watchdog = "idle"
                return cmd
        if surprise is not None and not self.available:
            self.stats.latency.add(0)
        return self._watchdog_cmd(t_ns, profile_ref, surprise is None)

    def _robot_state(self, m: MotionContext) -> Any:
        from senseai_reflex.motion import RobotState

        return RobotState(
            v=m.speed_mps,
            omega=m.yaw_rate_rps,
            heading=m.heading_rad,
            speed_cap=m.speed_cap_mps,
            footprint_radius_m=m.footprint_radius_m,
        )

    def _watchdog_cmd(self, t_ns: int, profile_ref: tuple[str, int], no_surprise: bool) -> FallbackCommand:
        since_ms = (t_ns - self._last_ok_ns) / 1e6 if self._last_ok_ns is not None else float("inf")
        if since_ms >= self.WATCHDOG_STOP_MS:
            self.watchdog = "brake"
            action: Action = "brake"
            scale, stop = 0.0, True
        elif since_ms >= self.WATCHDOG_MS:
            self.watchdog = "slow"
            action, scale, stop = "slow", 0.5, False
        else:
            action, scale, stop = "none", 1.0, False
        if no_surprise:
            reason = "no SurpriseMap this tick"
        elif not self.available:
            reason = self.stats.reason or "reflex unavailable"
        else:
            reason = f"reflex stage failed: {self.stats.last_error}"
        return FallbackCommand(
            t_master_ns=t_ns,
            action=action,
            speed_scale=scale,
            lateral_offset_m=0.0,
            stop=stop,
            hold=False,
            latched=False,
            risk=0.0,
            threshold=0.0,
            provenance=Provenance(profile=profile_ref[0], profile_version=profile_ref[1], reason=reason),
        )

    def state(self) -> bytes:
        """Latch and ramp state (empty without reflex)."""
        return bytes(self.engine.checkpoint_state()) if self.available else b""

    def restore(self, blob: bytes) -> None:
        """Restore :meth:`state`."""
        if self.available and blob:
            self.engine.restore_state(blob)


# -- cortex -----------------------------------------------------------------------------------


class CortexStage:
    """``senseai_cortex.CortexEngine`` when installed; produces no events when not."""

    name = "cortex"

    def __init__(self, cycle_ms: int = 100, tick_ms: int = 10) -> None:
        self.stats = StageStats(self.name)
        self.engine: Any = None
        self.available = False
        self.every = max(1, cycle_ms // tick_ms)
        try:
            from senseai_cortex.config import CortexConfig
            from senseai_cortex.engine import CortexEngine
        except ImportError as e:
            self.stats.available = False
            self.stats.reason = f"senseai-cortex not installed ({e.name}); no semantic events"
            return
        self.engine = CortexEngine(CortexConfig(cycle_ms=cycle_ms))
        self.available = True

    def step(self, tick_index: int, world_state: Any, surprise: Any, cmd: Any, batch: Any) -> list[Any]:
        """Runs every ``every`` ticks; returns the events it emitted (none when unavailable)."""
        if not self.available or tick_index % self.every:
            return []
        try:
            out: list[Any] = _timed(
                self.stats, lambda: list(self.engine.step(world_state, surprise, cmd, batch))
            )
        except Exception as e:
            self.stats.fail(e)
            return []
        return out

    def state(self) -> bytes:
        """Track state (empty without cortex)."""
        return bytes(self.engine.checkpoint_state()) if self.available else b""

    def restore(self, blob: bytes) -> None:
        """Restore :meth:`state`."""
        if self.available and blob:
            self.engine.restore_state(blob)


def load_calibration(sensors: Sequence[SensorConfig]) -> dict[str, Any] | None:
    """Merge the per-sensor ``calib`` files into one tokens ``calibration.yaml`` mapping."""
    sources: dict[str, Any] = {}
    for s in sensors:
        if s.calib is None:
            continue
        p = Path(s.calib)
        if not p.is_file():
            raise PipelineError(f"sensors[{s.id}].calib: {p} does not exist")
        doc = yaml.safe_load(p.read_text()) or {}
        sources[s.id] = doc.get("sources", {}).get(s.id, doc)
    if not sources:
        return None
    return {"frame": {"convention": "x_fwd_y_left_z_up", "origin": "base_link"}, "sources": sources}
