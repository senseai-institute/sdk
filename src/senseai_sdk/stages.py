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
import json
import pickle
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from senseai_sdk import adapters as adapter_table
from senseai_sdk.artefacts import Artefact
from senseai_sdk.config import Config, RobotConfig, SensorConfig
from senseai_sdk.errors import ArtefactError, PipelineError
from senseai_sdk.health import StageStats

__all__ = [
    "CircuitStage",
    "CortexStage",
    "FallbackCommand",
    "IngestStage",
    "MotionContext",
    "ProfilesStage",
    "Provenance",
    "ReflexStage",
    "SurpriseStage",
    "TokensStage",
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
    """The ``senseai_tokens.Tokenizer`` built from an artefact's ``tokens.yaml``.

    That artefact is the world bundle when one is configured, else ``artefacts.tokens``.
    """

    name = "tokens"

    def __init__(self, source: Artefact, calibration: Mapping[str, Any] | None, device: str = "cpu") -> None:
        from senseai_tokens.config import TokenizerConfig
        from senseai_tokens.frame import BodyFrame
        from senseai_tokens.tokenizer import Tokenizer

        self.stats = StageStats(self.name)
        tcfg = TokenizerConfig.from_yaml(source.path("tokens.yaml"))
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
        self._roots: dict[str, Path] = {}  # profile name -> the artefact it was loaded from
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
        self._roots[name] = root
        return prof

    def circuit_params(self, profile: Any | None) -> Any:
        """The species' circuit knobs, from where its profile keeps them (what senseai-studio shows).

        The profile spec's ``circuits:`` section wins over the newest prior bundle's
        ``meta["circuits"]``; anything unset keeps the registry default. ``None``: all defaults.
        """
        from senseai_circuits.params import CircuitParams

        if profile is None or profile.name not in self._roots:
            return CircuitParams()
        root = self._roots[profile.name]
        spec_file = root / "profiles" / f"{profile.name}.profile.yaml"
        spec = yaml.safe_load(spec_file.read_text()) if spec_file.is_file() else None
        meta = None
        for b in sorted((root / "priors").glob(f"{profile.name}.v*.json"), reverse=True):
            with contextlib.suppress(OSError, ValueError):
                meta = json.loads(b.read_text())
                break
        return CircuitParams.from_profile(
            spec if isinstance(spec, Mapping) else None, meta if isinstance(meta, Mapping) else None
        )

    def activate(self, profile: Any, tokenizer: Any, reflex: Any | None) -> None:
        """Install ``profile`` on the tokenizer and reflex. Between ticks only."""
        from senseai_profiles.profile import SceneError

        self.active = profile
        self.active_ref = (profile.name, int(profile.version))
        with contextlib.suppress(SceneError):  # resolved on the first tick that carries the lux map
            profile.apply(tokenizer, self._pending_scene or None)
        if reflex is not None and getattr(reflex, "available", False):
            reflex.set_profile(
                profile
            )  # not profile.apply(): that routes engines with set_profile to the tokenizer path

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


# -- circuits (L1) ---------------------------------------------------------------------------


class CircuitStage:
    """``senseai_circuits.CircuitStage``: every event camera through the L1 circuits, before tokens.

    Camera geometry is the tokenizer's own (its ``BodyFrame`` camera at the event head's voxel
    size), so an event token's ray passes through the circuit cell it is scored by. The ego yaw
    rate comes from the tick's IMU samples, or from ``update_context(yaw_rate=...)`` without IMU.
    """

    name = "circuits"

    def __init__(self, params: Any = None, *, tick_ns: int, n_bearing: int = 72) -> None:
        from senseai_circuits.blas import limit_blas_threads
        from senseai_circuits.stage import CircuitStage as L1Circuits

        # the circuits' matrices are small; multi-threaded OpenBLAS workers would spin and starve the
        # other stages (measured: the learned model's step doubled, 5 -> 10 ms, on 4 cores)
        limit_blas_threads(1)
        self.stats = StageStats(self.name)
        self._tick_ns, self._n_bearing = tick_ns, n_bearing
        self._specs: Mapping[str, Any] | None = None
        self._tokenizer: Any = None
        self.stage = L1Circuits(params, tick_ns=tick_ns, n_bearing=n_bearing)

    @property
    def params(self) -> Any:
        """The ``CircuitParams`` the circuits run with."""
        return self.stage.params

    def set_params(self, params: Any) -> None:
        """New knob values (a profile switch): rebuild the circuits, which restarts their state."""
        from senseai_circuits.stage import CircuitStage as L1Circuits

        if params == self.stage.params:
            return
        self.stage = L1Circuits(params, tick_ns=self._tick_ns, n_bearing=self._n_bearing)
        if self._specs is not None:
            self.set_specs(self._specs, self._tokenizer)

    def set_specs(self, specs: Mapping[str, Any], tokenizer: Any | None = None) -> None:
        """One circuit chain per EVENT source; geometry from the tokenizer's calibration when it has one."""
        from senseai_circuits.grid import GridGeometry

        self._specs, self._tokenizer = specs, tokenizer

        head = getattr(tokenizer, "heads", {}).get("event") if tokenizer is not None else None
        cell_px = int(getattr(getattr(head, "cfg", None), "voxel_px", self.stage.cell_px))
        fov = getattr(getattr(head, "cfg", None), "default_fov_deg", None)
        self.stage.cell_px, self.stage.default_fov_deg = cell_px, fov
        for sid, spec in specs.items():
            if str(getattr(spec, "modality", "")) != "event" or sid in self.stage.chains:
                continue
            if tokenizer is not None:
                with contextlib.suppress(Exception):  # no calibration: fall back to the spec's meta
                    cam = tokenizer.frame.camera(sid, default_fov_deg=fov)
                    self.stage.add_source(sid, GridGeometry.from_camera(cam, cell_px))
                    continue
        self.stage.set_specs(specs)

    def step(self, tick: Any, yaw_rate_rps: float = 0.0) -> Any:
        """``AlignedTick`` → ``CircuitMaps``."""
        yaw = None if self.stage.imu_sources else yaw_rate_rps
        return _timed(self.stats, lambda: self.stage.step(tick, ego_yaw_rps=yaw))

    def state(self) -> bytes:
        """Circuit state (adaptation, delays, last grid)."""
        return bytes(self.stage.state())

    def restore(self, blob: bytes) -> None:
        """Restore :meth:`state`."""
        if blob:
            self.stage.restore(blob)


# -- surprise ---------------------------------------------------------------------------------


class SurpriseStage:
    """The ``SurpriseMap`` producer: the L1 circuits always, the learned world model when configured.

    With no world artefact the map is ``senseai_circuits.SurpriseFromCircuits`` alone
    (``provenance.model_id == "circuits-<version>"``). With one, both run and are fused the way
    ``senseai_world.prior.fused.FusedWorldModel`` fuses prior and learned model: element-wise
    ``max`` of the per-token z-scores and of the per-bin maps, ``max`` of the globals, both
    normalisers in ``scale`` under ``learned.`` and ``circuits.``, and ``WorldState`` from the
    learned model.
    """

    name = "surprise"

    def __init__(
        self,
        world: Artefact | None,
        params: Any = None,
        *,
        tokens_per_tick: int | None = None,
        device: str = "auto",
    ) -> None:
        from senseai_circuits.stage import SurpriseFromCircuits

        self.stats = StageStats(self.name)
        self._tokens_per_tick = tokens_per_tick
        self.l1 = SurpriseFromCircuits(params, tokens_per_tick=tokens_per_tick)
        self.model: Any = None
        self.tick_ns: int | None = None
        self.tokenizer_id: str | None = None
        if world is not None:
            from senseai_world.inference import WorldModel

            self.model = WorldModel.load(world.path("checkpoint.ckpt"), device=device)
            self.tick_ns = int(self.model.cfg.tick_ns or 0)
            self.tokenizer_id = self.model.tokenizer_id
        self.model_id: str = (
            self.l1.model_id if self.model is None else f"{self.model.model_id}+{self.l1.model_id}"
        )

    def set_params(self, params: Any) -> None:
        """New knob values (a profile switch): a fresh L1 producer, which warms up again."""
        from senseai_circuits.stage import SurpriseFromCircuits

        if params != self.l1.params:
            self.l1 = SurpriseFromCircuits(params, tokens_per_tick=self._tokens_per_tick)

    @property
    def device(self) -> str:
        """Where the learned model runs (``cpu`` when there is none: the circuits are host NumPy)."""
        return "cpu" if self.model is None else str(self.model.device)

    def step(self, maps: Any, batch: Any) -> tuple[Any, Any]:
        """``(CircuitMaps, TokenBatch)`` → ``(SurpriseMap, WorldState | None)``."""
        return _timed(self.stats, lambda: self._step(maps, batch))  # type: ignore[no-any-return]

    def _step(self, maps: Any, batch: Any) -> tuple[Any, Any]:
        if maps is None:
            if self.model is None or batch is None:
                raise PipelineError("no circuit maps this tick")
            learned: tuple[Any, Any] = self.model.step(batch)
            return learned
        s_c = self.l1.step(maps, batch)
        if self.model is None or batch is None:
            return s_c, None
        s_l, state = self.model.step(batch)
        return self._fuse(s_l, s_c), state

    def _fuse(self, s_l: Any, s_c: Any) -> Any:
        from senseai_circuits.contract import Provenance, SurpriseMap

        scale = {f"learned.{k}": v for k, v in s_l.scale.items()}
        scale.update({f"circuits.{k}": v for k, v in s_c.scale.items()})
        warming = bool(s_l.warming_up or s_c.warming_up)
        scale["warming_up"] = 1.0 if warming else 0.0
        heads = dict(s_c.per_head)
        for h, v in s_l.per_head.items():
            heads[h] = max(v, heads.get(h, 0.0))
        return SurpriseMap(
            t_master_ns=s_c.t_master_ns,
            per_token=np.maximum(s_l.per_token, s_c.per_token).astype(np.float32),
            per_cell=np.maximum(s_l.per_cell, s_c.per_cell),
            per_bearing=np.maximum(s_l.per_bearing, s_c.per_bearing),
            per_head=heads,
            global_surprise=max(float(s_l.global_surprise), float(s_c.global_surprise)),
            scale=scale,
            latency_ns=int(s_l.latency_ns) + int(s_c.latency_ns),
            warming_up=warming,
            tick_index=s_c.tick_index,
            grid_half_extent_m=s_c.grid_half_extent_m,
            provenance=Provenance(model_id=self.model_id, tokenizer_id=s_c.provenance.tokenizer_id),
            cell_count=np.maximum(s_l.cell_count, s_c.cell_count),
            cell_max=np.maximum(s_l.cell_max, s_c.cell_max),
            bearing_count=np.maximum(s_l.bearing_count, s_c.bearing_count),
            bearing_max=np.maximum(s_l.bearing_max, s_c.bearing_max),
        )

    def state(self) -> bytes:
        """Checkpointable normaliser state of both producers."""
        learned = b"" if self.model is None else bytes(self.model.checkpoint_state())
        return pickle.dumps({"circuits": self.l1.checkpoint_state(), "learned": learned})

    def restore(self, blob: bytes) -> None:
        """Restore :meth:`state` (our own checkpoint, never external input)."""
        st = pickle.loads(blob)
        self.l1.restore_state(st["circuits"])
        if self.model is not None and st["learned"]:
            self.model.restore_state(st["learned"])


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
            from senseai_reflex.motion import Footprint
        except ImportError as e:
            self.stats.available = False
            self.stats.reason = f"senseai-reflex not installed ({e.name}); watchdog drives the bridge"
            return
        cfg = ReflexConfig.from_dict(self.config)
        self._footprint = Footprint(radius_m=self.robot.footprint_radius_m)
        self.engine = ReflexEngine(cfg, footprint=self._footprint, warm=False)
        self.available = True
        self.set_profile(None)

    def set_profile(self, profile: Any | None) -> None:
        """Install a profile's ``ReflexPrior`` (with its name and version); ``None`` is the null profile."""
        if not self.available:
            return
        if profile is None:
            from senseai_reflex.profile_io import NULL_PRIOR

            self.engine.set_prior(NULL_PRIOR, name="null", version=0)
        else:
            self.engine.set_profile(profile)

    def step(
        self,
        surprise: Any | None,
        motion: MotionContext,
        t_ns: int,
        profile_ref: tuple[str, int],
        t_sensor_ns: int | None = None,
    ) -> Any:
        """One tick. Always returns a command; never raises past the pipeline.

        ``t_sensor_ns`` is the oldest sample time in the tick; reflex uses it for
        ``provenance.latency.pipeline_ns`` (sensor → command). With reflex installed, a tick
        without a surprise map still goes to the engine: its envelope watchdog is the primary
        (``slow`` then ``brake``, ``provenance.degraded``). The SDK watchdog only covers a reflex
        stage that raises, or a missing reflex.
        """
        if self.available:
            try:
                state = self._robot_state(motion, t_sensor_ns)
                # ReflexEngine reuses one command object; hand out a snapshot callers can keep
                cmd = _timed(self.stats, lambda: self.engine.step(surprise, state, t_ns).copy())
            except Exception as e:
                self.stats.fail(e)
            else:
                if surprise is not None:
                    self._last_ok_ns = t_ns
                self.watchdog = (
                    "brake"
                    if cmd.stop and cmd.provenance.degraded
                    else ("slow" if cmd.provenance.degraded else "idle")
                )
                return cmd
        elif surprise is not None:
            self.stats.latency.add(0)
        return self._watchdog_cmd(t_ns, profile_ref, surprise is None)

    def _robot_state(self, m: MotionContext, t_sensor_ns: int | None = None) -> Any:
        from senseai_reflex.motion import RobotState

        return RobotState(
            v_mps=m.speed_mps,
            omega_rps=m.yaw_rate_rps,
            speed_cap_mps=m.speed_cap_mps,
            footprint=self._footprint,
            t_sensor_ns=t_sensor_ns,
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
        return pickle.dumps(self.engine.checkpoint_state()) if self.available else b""

    def restore(self, blob: bytes) -> None:
        """Restore :meth:`state`."""
        if self.available and blob:
            self.engine.restore_state(pickle.loads(blob))  # our own checkpoint, never external input


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
