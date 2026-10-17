# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""The whole public Python API of the SenseAI SDK: one class, :class:`SenseAI`.

Design stub (step 1): every signature and docstring here is the contract under review;
method bodies raise ``NotImplementedError`` until the pipeline lands.

A customer should be able to integrate from this file alone::

    from senseai_sdk import SenseAI

    sai = SenseAI("senseai.yaml")
    sai.on_command(lambda cmd: controller.apply(cmd))
    sai.on_event(lambda ev: planner.notify(ev))
    sai.run()
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from senseai_cortex.events import Event
    from senseai_ingest.adapters.base import Adapter
    from senseai_reflex.bridge.base import ControllerBridge
    from senseai_reflex.command import ReflexCommand
    from senseai_world.outputs import SurpriseMap

    from senseai_sdk.config import Config

HealthState = Literal["ok", "degraded", "failed"]


@dataclass(slots=True)
class StepResult:
    """What one tick produced, returned by :meth:`SenseAI.step`.

    Attributes:
        tick_index: Ingest tick index, 0-based and contiguous.
        t_master_ns: End of the tick window on the master clock.
        command: The reflex command for this tick. Never ``None``: a quiet tick is
            ``action="none"`` with ``speed_scale=1.0``.
        events: Cortex events emitted on this tick. Usually empty: cortex runs on its own
            ~100 ms cycle and only emits on change.
        surprise: The world model's surprise map for this tick, for inspection and sim scoring.
        latency_ns: Wall time per stage (``ingest``, ``tokens``, ``world``, ``reflex``,
            ``cortex``, ``bridge``, ``sdk``) and ``total``.
        degraded: True if any stage failed or overran on this tick and the reflex watchdog
            produced the command.
    """

    tick_index: int
    t_master_ns: int
    command: ReflexCommand
    events: list[Event]
    surprise: SurpriseMap | None
    latency_ns: Mapping[str, int]
    degraded: bool = False


@dataclass(slots=True)
class StageHealth:
    """Health of one pipeline stage over the recent window (default: last 10 s)."""

    name: str
    p50_ms: float
    p99_ms: float
    budget_ms: float
    drops: int
    errors: int
    last_error: str | None = None


@dataclass(slots=True)
class Health:
    """Pipeline health, returned by :meth:`SenseAI.health` and served at ``/health``.

    Attributes:
        state: ``ok``; ``degraded`` (a stage failed or overran, so the reflex watchdog is
            driving the bridge); or ``failed`` (the pipeline cannot run, for example no
            licence or no valid artefact).
        endpoint_id: From config.
        stages: Per-stage latency, drops and errors, in pipeline order.
        end_to_end_p99_ms: Sensor-to-actuator p99 over the window.
        watchdog: Reflex watchdog state: ``idle``, ``slow`` or ``brake``.
        licence: ``valid``, ``grace`` or ``expired``, with ``licence_days_left``.
        artefacts: Loaded artefact ids, for example ``{"world": "world-v1.2", "reflex": ...,
            "profile": "panther.v1"}``.
        profile: Active profile name and version.
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
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        """Return the JSON-serialisable form served at ``/health``."""
        raise NotImplementedError


class SenseAI:
    """The SenseAI perception pipeline for one robot.

    It composes ingest → tokens → profiles → world → reflex → cortex from one
    ``senseai.yaml``. Reflex commands go to your controller through a bridge. Semantic
    events go to your planner, fleet manager or logs through callbacks and sinks.

    Construction does the following, and opens no devices:

    - validates the config;
    - verifies the licence;
    - fetches and verifies every artefact (Ed25519 manifest + sha256);
    - builds the stages.

    Devices open on :meth:`run`, :meth:`start` or the first :meth:`step`.

    Thread safety: every public method may be called from any thread. Changes to profile,
    sensors, actuator and context are queued and applied between ticks, never mid-tick.

    Raises (from ``__init__``):
        ConfigError: The YAML is invalid. The message names the key path, the value and
            what was expected.
        LicenceError: The key is missing, invalid, for another endpoint class, or past its
            30-day grace period.
        ArtefactError: An artefact is missing, unsigned, tampered with, or does not match
            its pinned version.
    """

    def __init__(self, config: str | Path | Config) -> None:
        """Build the pipeline from a ``senseai.yaml`` path or a :class:`Config` object.

        Args:
            config: Path to ``senseai.yaml``, or an already-validated ``Config``
                (``senseai_sdk.config.Config.load(path)``).
        """
        raise NotImplementedError

    # -- configuration at runtime -------------------------------------------------------

    def load_profile(self, name: str, version: int | None = None) -> None:
        """Switch the active profile. The swap is applied at the next tick boundary.

        The profile's head priors go to the tokenizer and its ``ReflexPrior`` goes to the
        reflex engine, both between ticks. No tick is dropped and none is split.

        Cost: under 1 ms on the tick that applies it.

        Args:
            name: Profile name, for example ``"panther"``. ``"null"`` (no priors) is always
                allowed.
            version: Bundle version. ``None`` means the newest version the licence allows.

        Raises:
            LicenceError: The licence does not allow this profile.
            ArtefactError: The bundle fails signature or spec-hash verification.
            KeyError: No such profile or version.
        """
        raise NotImplementedError

    def bind_sensor(self, source_id: str, adapter: Adapter) -> None:
        """Add a sensor source at runtime, or replace an existing one.

        The adapter is any ``senseai_ingest`` adapter, including your own (see
        ``examples/custom_adapter.py``). Its ``spec().source_id`` must equal ``source_id``.
        Replacing a source keeps its tokenizer slot, so the world model sees the same
        source index. A brand-new source must be one the world artefact's tokenizer
        vocabulary knows.

        Raises:
            ValueError: ``source_id`` does not match the adapter, or the tokenizer
                vocabulary does not know the source.
            LicenceError: The licence does not allow this sense (modality).
        """
        raise NotImplementedError

    def bind_actuator(self, bridge: ControllerBridge) -> None:
        """Set the bridge that applies each reflex command to your controller.

        This replaces the bridge named in ``reflex.bridge``. The bridge multiplies the
        planner's speed by ``speed_scale``, nudges heading for ``lateral_offset_m``, and
        zeroes speed on ``stop``. It never plans. The SDK keeps calling the bridge while
        degraded, so the watchdog's ``slow`` / ``brake`` reach the robot.
        """
        raise NotImplementedError

    def on_command(self, cb: Callable[[ReflexCommand], None]) -> None:
        """Register a callback for every reflex command, one per tick (~100 Hz).

        Every tick emits a command, including ``action="none"``, so your code always has a
        fresh value. The callback runs on the reflex thread and must return quickly (well
        under 1 ms). Hand work to your own thread if you need more. A callback that raises
        is logged and counted in health; it does not stop the pipeline.

        You may register more than one callback. They are called in registration order.
        """
        raise NotImplementedError

    def on_event(self, cb: Callable[[Event], None]) -> None:
        """Register a callback for cortex events.

        Events are emitted on change: a new or lost track, a reflex fire or release, a
        sensor degrading or recovering, a profile change. A heartbeat is sent once per
        second. A quiet scene produces only heartbeats. Each event carries a one-sentence
        ``text`` and full attribution. The callback runs on the cortex thread.
        """
        raise NotImplementedError

    def update_context(self, **ctx: float | object) -> None:
        """Pass scene and motion context to the pipeline. Applied at the next tick boundary.

        Recognised keys:
            lux: Scene illuminance (float). Resolves profile priors that depend on lux.
            lux_map: ``[rows, cols]`` lux over the camera image. Resolves per-region priors.
            speed: Current forward speed in m/s. Used by profiles and by the reflex motion
                model.
            yaw_rate: Angular rate in rad/s, for the reflex motion model.
            heading: Heading in rad, for the reflex motion model.
            speed_cap: The planner's current speed cap in m/s.
            mode: Free-form operating mode string, recorded in telemetry and events.

        If the ``ros2_cmd_vel`` bridge is bound, ``speed`` and ``yaw_rate`` are taken from
        the planner's command, and you do not need to send them.

        Raises:
            KeyError: Unknown key. Misspelt context is an error, not a silent no-op.
        """
        raise NotImplementedError

    # -- running ------------------------------------------------------------------------

    def run(self) -> None:
        """Run the pipeline in the foreground until :meth:`stop`, SIGINT or end of replay.

        Uses threaded mode: one thread per stage, bounded drop-oldest queues.
        """
        raise NotImplementedError

    def start(self) -> None:
        """Start the pipeline on background threads and return immediately."""
        raise NotImplementedError

    def stop(self) -> None:
        """Stop the pipeline.

        Sequence:

        1. The bridge publishes a final ``hold``.
        2. Sinks flush.
        3. Usage is metered.
        4. Devices close.

        Idempotent.
        """
        raise NotImplementedError

    def step(self) -> StepResult:
        """Run exactly one tick through every stage, in order, on the calling thread.

        This is single-loop mode, for sim, tests and replay. It is deterministic: the same
        recording, config and artefacts give the same command and event hashes as calling
        the packages by hand. Do not mix with :meth:`start`.

        Raises:
            StopIteration: The replay recording has ended.
        """
        raise NotImplementedError

    # -- observability ------------------------------------------------------------------

    def health(self) -> Health:
        """Return current pipeline health. Cheap; safe to poll at any rate."""
        raise NotImplementedError

    def explain(self, t_ns: int) -> str:
        """Explain the reflex command in force at master time ``t_ns``, in one sentence.

        Example: ``"slowed to 40% at 6.21 s: path risk 3.4 > 2.6 from 7 cells ahead-left,
        heads event 71% / audio 22%, profile panther v1, latency 11.2 ms"``.
        This passes through to the reflex audit log, which holds a rolling window.

        Raises:
            KeyError: ``t_ns`` is outside the audit window.
        """
        raise NotImplementedError

    def save_state(self, path: str | Path) -> None:
        """Checkpoint all stage state to ``path``.

        The state covers tokenizer running stats, the world window and normalisers,
        reflex latch and ramp, cortex tracks, and the active profile. A restart from it
        skips the world model's warm-up.
        """
        raise NotImplementedError

    def load_state(self, path: str | Path) -> None:
        """Restore state written by :meth:`save_state`. Applied between ticks.

        Raises:
            ArtefactError: The state was written with different artefact versions.
        """
        raise NotImplementedError
