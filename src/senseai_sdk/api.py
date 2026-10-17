# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""The whole public Python API of the SenseAI SDK: one class, :class:`SenseAI`.

A customer should be able to integrate from this file alone::

    from senseai_sdk import SenseAI

    sai = SenseAI("senseai.yaml")
    sai.on_command(lambda cmd: controller.apply(cmd))
    sai.on_event(lambda ev: planner.notify(ev))
    sai.run()
"""

from __future__ import annotations

import datetime as dt
import logging
import pickle
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from senseai_sdk.artefacts import ArtefactStore, Keyring
from senseai_sdk.config import Config
from senseai_sdk.errors import LicenceError, PipelineError
from senseai_sdk.health import Health, StageHealth
from senseai_sdk.licence import Licence, Meter
from senseai_sdk.pipeline import Pipeline, StepResult

if TYPE_CHECKING:
    from senseai_sdk.bridges import BridgeLike

__all__ = ["Health", "SenseAI", "StageHealth", "StepResult"]

log = logging.getLogger("senseai")


class SenseAI:
    """The SenseAI perception pipeline for one robot.

    It composes ingest → tokens → profiles → world → reflex → cortex from one
    ``senseai.yaml``. Reflex commands go to your controller through a bridge. Semantic
    events go to your planner, fleet manager or logs through callbacks and sinks.

    Construction does the following, and opens no devices:

    - validates the config;
    - verifies the licence (offline) and that it covers this endpoint and its senses;
    - fetches and verifies every artefact (Ed25519 manifest + sha256);
    - builds the stages.

    Devices open on :meth:`run`, :meth:`start` or the first :meth:`step`.

    Thread safety: every public method may be called from any thread. Changes to profile,
    sensors, actuator and context are queued and applied between ticks, never mid-tick.

    Raises (from ``__init__``):
        ConfigError: The YAML is invalid. The message names the key path, the value and
            what was expected.
        LicenceError: The key is missing, invalid, for another endpoint, or past its
            30-day grace period.
        ArtefactError: An artefact is missing, unsigned, tampered with, or does not match
            its pinned version.
    """

    def __init__(
        self,
        config: str | Path | Config,
        *,
        keyring: Keyring | None = None,
        bridge: BridgeLike | None = None,
        device: str = "auto",
        today: dt.date | None = None,
    ) -> None:
        """Build the pipeline from a ``senseai.yaml`` path or a :class:`Config` object.

        Args:
            config: Path to ``senseai.yaml``, or an already-validated ``Config``
                (``senseai_sdk.config.Config.load(path)``).
            keyring: Trusted release keys; default: the shipped keys plus
                ``SENSEAI_RELEASE_KEYS``.
            bridge: A controller bridge to use instead of the one named in ``reflex.bridge``.
            device: ``auto`` (CUDA when present), ``cpu`` or ``cuda[:n]`` for the models.
            today: Overrides the date the licence is checked against (tests).
        """
        self.config: Config = config if isinstance(config, Config) else Config.load(config)
        cfg = self.config
        self._keyring = keyring or Keyring.default()
        self._today = today
        self.licence: Licence = Licence.parse(cfg.licence_key, self._keyring)
        state, days = self.licence.state(today)
        if state == "expired":
            raise LicenceError(
                f"licence {self.licence.licence_id} expired on {self.licence.expires} and its "
                f"{self._grace()}-day grace period has ended; run `senseai licence activate <new key>`"
            )
        if state == "grace":
            log.warning(
                "licence %s expired on %s; %d days of grace left",
                self.licence.licence_id,
                self.licence.expires,
                days,
            )
        self.licence.check_endpoint(cfg.endpoint_id)
        self.licence.require_profile(cfg.profile.name)
        store = ArtefactStore(cfg.artefacts.store, cfg.artefacts.cache, self._keyring)
        self.meter = Meter(self.licence, cfg.endpoint_id, spool=cfg.artefacts.cache / "usage")
        self._pipeline = Pipeline.from_config(
            cfg,
            store=store,
            bridge=bridge,
            licence_state=lambda: self.licence.state(self._today),
            device=device,
        )
        self._pipeline.on_tick = self._meter_tick
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._senses_checked = False

    @staticmethod
    def _grace() -> int:
        from senseai_sdk.licence import GRACE_DAYS

        return GRACE_DAYS

    # -- configuration at runtime -------------------------------------------------------

    def load_profile(self, name: str, version: int | None = None) -> None:
        """Switch the active profile. The swap is applied at the next tick boundary.

        The profile's head priors go to the tokenizer and its ``ReflexPrior`` goes to the
        reflex engine, both between ticks. No tick is dropped and none is split.

        Cost: under 1 ms on the tick that applies it.

        Args:
            name: Profile name, for example ``"panther"``. ``"null"`` (no priors) is always
                allowed.
            version: Bundle version. ``None`` means the newest version in the store.

        Raises:
            LicenceError: The licence does not allow this profile.
            ArtefactError: The bundle fails signature or spec-hash verification, or no such
                profile or version exists in the store.
        """
        self.licence.require_profile(name)
        self._pipeline.load_profile(name, version)

    def bind_sensor(self, source_id: str, adapter: Any) -> None:
        """Add a sensor source at runtime, or replace an existing one.

        The adapter is any ``senseai_ingest`` adapter, including your own (see
        ``examples/custom_adapter.py``). Its ``spec().source_id`` must equal ``source_id``.
        Replacing a source keeps its tokenizer slot, so the world model sees the same
        source index. A brand-new source must be one the world artefact's tokenizer
        vocabulary knows. The change takes effect at the next tick boundary.

        Raises:
            ValueError: ``source_id`` does not match the adapter, or the tokenizer
                vocabulary does not know the source.
            LicenceError: The licence does not allow this sense (modality).
        """
        self.licence.require_senses([str(adapter.spec().modality)])
        self._pipeline.bind_sensor(source_id, adapter)

    def bind_actuator(self, bridge: BridgeLike) -> None:
        """Set the bridge that applies each reflex command to your controller.

        This replaces the bridge named in ``reflex.bridge`` at the next tick boundary. The
        bridge multiplies the planner's speed by ``speed_scale``, nudges heading for
        ``lateral_offset_m``, and zeroes speed on ``stop``. It never plans. The SDK keeps
        calling the bridge while degraded, so the watchdog's ``slow`` / ``brake`` reach the
        robot.
        """
        self._pipeline.bind_actuator(bridge)

    def on_command(self, cb: Callable[[Any], None]) -> None:
        """Register a callback for every reflex command, one per tick (~100 Hz).

        Every tick emits a command, including ``action="none"``, so your code always has a
        fresh value. The callback runs on the reflex thread and must return quickly (well
        under 1 ms). Hand work to your own thread if you need more. A callback that raises
        is logged and counted in health; it does not stop the pipeline.

        You may register more than one callback. They are called in registration order.
        """
        self._pipeline.on_command(cb)

    def on_event(self, cb: Callable[[Any], None]) -> None:
        """Register a callback for cortex events.

        Events are emitted on change: a new or lost track, a reflex fire or release, a
        sensor degrading or recovering, a profile change. A heartbeat is sent once per
        second. A quiet scene produces only heartbeats. Each event carries a one-sentence
        ``text`` and full attribution. The callback runs on the cortex thread.
        """
        self._pipeline.on_event(cb)

    def update_context(self, **ctx: Any) -> None:
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
        self._pipeline.update_context(**ctx)

    # -- running ------------------------------------------------------------------------

    def run(self) -> None:
        """Run the pipeline in the foreground until :meth:`stop`, SIGINT or end of replay."""
        self._stop.clear()
        try:
            while not self._stop.is_set():
                try:
                    self.step()
                except StopIteration:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def start(self) -> None:
        """Start the pipeline on a background thread and return immediately."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run, name="senseai-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the pipeline: final ``hold`` on the bridge, sinks flushed, usage metered, devices closed.

        Idempotent. Returns once the pipeline thread has finished.
        """
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join()
        self._thread = None
        self._shutdown()

    def _shutdown(self) -> None:
        self._pipeline.close()
        if self.meter.counts()["ticks"]:
            self.meter.report()

    def step(self) -> StepResult:
        """Run exactly one tick through every stage, in order, on the calling thread.

        This is single-loop mode, for sim, tests and replay. It is deterministic: the same
        recording, config and artefacts give the same command and event hashes as calling
        the packages by hand. Do not mix with :meth:`start`.

        Raises:
            StopIteration: The replay recording has ended.
            LicenceError: A configured sensor's modality is not licensed (checked when
                devices open, since replay adapters only report their modality then).
        """
        if not self._senses_checked:
            self._pipeline.open()
            self.licence.require_senses(str(s.modality) for s in self._pipeline.ingest.specs().values())
            self._senses_checked = True
        return self._pipeline.step()

    # -- observability ------------------------------------------------------------------

    def health(self) -> Health:
        """Return current pipeline health. Cheap; safe to poll at any rate."""
        return self._pipeline.health()

    def explain(self, t_ns: int) -> str:
        """Explain the reflex command in force at master time ``t_ns``, in one sentence.

        Example: ``"slowed to 40% at 6.21 s: path risk 3.4 > 2.6 from 7 cells ahead-left,
        heads event 71% / audio 22%, profile panther v1, latency 11.2 ms"``.
        This passes through to the reflex audit log, which holds a rolling window.

        Raises:
            KeyError: ``t_ns`` is outside the audit window.
            PipelineError: ``senseai-reflex`` (which owns the audit log) is not installed.
        """
        reflex = self._pipeline.reflex
        if not reflex.available:
            raise PipelineError("explain needs senseai-reflex, which is not installed")
        text: str = reflex.engine.audit.explain(t_ns)
        return text

    def save_state(self, path: str | Path) -> None:
        """Checkpoint all stage state to ``path``.

        The state covers tokenizer running stats, the world window and normalisers,
        reflex latch and ramp, cortex tracks, and the active profile. A restart from it
        skips the world model's warm-up.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self._pipeline.state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, path: str | Path) -> None:
        """Restore state written by :meth:`save_state`. Applied between ticks.

        Raises:
            ArtefactError: The state was written with different artefact versions.
        """
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        self._pipeline.restore(state)

    # -- internals ----------------------------------------------------------------------

    def _meter_tick(self, res: StepResult) -> None:
        active = [k for k, v in res.latency_ns.items() if k in ("tokens",) and v] and self._active_senses()
        self.meter.tick(self._pipeline.profiles.active_ref[0], active)

    def _active_senses(self) -> list[str]:
        return sorted({str(s.modality) for s in self._pipeline.ingest.specs().values()})
