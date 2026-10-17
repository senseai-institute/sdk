# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""``senseai.yaml``: the one file a customer edits.

:class:`Config` is the validated form. :func:`Config.load` reads a YAML file, expands
``${ENV}`` references, resolves relative paths against the file, checks every key and
raises :class:`ConfigError` with the key path, the offending value and what was expected.
:func:`template` writes the commented file that ``senseai init`` produces.

The key set is the contract in the README. Adding a key is a design change, not a patch.
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

from senseai_sdk.adapters import ADAPTERS
from senseai_sdk.errors import ConfigError

__all__ = [
    "ArtefactsConfig",
    "Config",
    "ConfigError",
    "CortexConfig",
    "ProfileRef",
    "ReflexConfig",
    "RobotConfig",
    "SensorConfig",
    "TelemetryConfig",
    "template",
]

BRIDGES = ("callback", "ros2_cmd_vel")
SINKS = ("callback", "jsonl", "ros2", "mqtt")

#: Mirror of the reflex safety envelope's lateral limit. ``reflex.max_lateral_m`` may only
#: lower it. When ``senseai_reflex.envelope`` ships, this constant is imported from there
#: so the two can never disagree.
ENVELOPE_MAX_LATERAL_M = 0.5

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_T = TypeVar("_T")


# -- validated sections ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileRef:
    """``profile:`` — which species prior to load at start."""

    name: str = "null"
    version: int | None = None


@dataclass(frozen=True, slots=True)
class ArtefactsConfig:
    """``artefacts:`` — the signed proprietary bundles and where they live."""

    world: str
    reflex: str
    cache: Path = field(default_factory=lambda: Path("~/.senseai").expanduser())
    store: Path | str = field(  # a directory, or an HTTPS base URL
        default_factory=lambda: Path("~/.senseai/store").expanduser()
    )


@dataclass(frozen=True, slots=True)
class RobotConfig:
    """``robot:`` — the geometry and braking the reflex motion model needs.

    Attributes:
        footprint_radius_m: Radius of the circle that contains the robot, in metres.
        max_decel_mps2: Deceleration a controlled stop may use, in m/s². The reflex
            envelope caps it; a larger value here is clamped and logged, never trusted.
        speed_cap_mps: The planner's nominal top speed, used until ``update_context`` sends
            a live one.
        kind: ``amr`` (wheeled base) or ``manipulator``. Enables ``retract`` / ``hold``.
    """

    footprint_radius_m: float = 0.35
    max_decel_mps2: float = 2.0
    speed_cap_mps: float = 1.5
    kind: Literal["amr", "manipulator"] = "amr"


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """One ``sensors[]`` entry.

    ``options`` is every key that is not ``id``, ``adapter``, ``calib`` or ``spec``; it goes
    to the adapter's constructor unchanged.
    """

    id: str
    adapter: str
    calib: Path | None = None
    spec: str | None = None  # a senseai-profiles sensor spec id (sensors/<id>.sensor.yaml)
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """``live`` or ``replay``."""
        return ADAPTERS[self.adapter].kind


@dataclass(frozen=True, slots=True)
class ReflexConfig:
    """``reflex:`` — how commands reach the controller."""

    bridge: str = "callback"
    planner_topic: str = "/cmd_vel"
    out_topic: str = "/senseai/cmd_vel"
    max_lateral_m: float = ENVELOPE_MAX_LATERAL_M


@dataclass(frozen=True, slots=True)
class CortexConfig:
    """``cortex:`` — where events go."""

    sinks: tuple[str, ...] = ("callback",)
    jsonl_path: Path | None = None
    mqtt_url: str | None = None


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """``telemetry:`` — opt-in sampling. Off by default; never uploads without a hook."""

    enabled: bool = False
    raw_frames: bool = False
    upload: str | None = None  # "module:function"
    spool_dir: Path | None = None  # None → artefacts.cache / "telemetry"
    spool_max_mb: int = 256


@dataclass(frozen=True, slots=True)
class Config:
    """The validated ``senseai.yaml``. Build it with :meth:`load` or :meth:`from_mapping`."""

    endpoint_id: str
    licence_key: str
    artefacts: ArtefactsConfig
    sensors: tuple[SensorConfig, ...]
    profile: ProfileRef = ProfileRef()
    robot: RobotConfig = RobotConfig()
    tick_ms: int = 10
    reflex: ReflexConfig = ReflexConfig()
    cortex: CortexConfig = CortexConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    source: Path | None = None  # the file this came from, for messages

    @property
    def tick_ns(self) -> int:
        """Tick length in nanoseconds, the unit ingest and world use."""
        return self.tick_ms * 1_000_000

    @classmethod
    def load(cls, path: str | Path, *, env: Mapping[str, str] | None = None) -> Config:
        """Read and validate ``path``. Relative paths inside resolve against its directory."""
        p = Path(path).expanduser()
        try:
            raw = yaml.safe_load(p.read_text())
        except FileNotFoundError:
            raise ConfigError(f"{p}: no such file; run `senseai init` to create one") from None
        except yaml.YAMLError as e:
            raise ConfigError(f"{p}: not valid YAML: {e}") from e
        return cls.from_mapping(raw, env=env, base=p.parent, source=p)

    @classmethod
    def from_mapping(
        cls,
        raw: Any,
        *,
        env: Mapping[str, str] | None = None,
        base: Path | None = None,
        source: Path | None = None,
    ) -> Config:
        """Validate an already-parsed mapping. ``base`` resolves relative paths."""
        if not isinstance(raw, Mapping):
            raise ConfigError("top level must be a mapping of keys, not a list or a scalar")
        n = _Node(_expand(raw, env if env is not None else os.environ, ""), "", base or Path())
        n.check_keys(_TOP_KEYS)
        sensors = _sensors(n.items("sensors", required=True), n.base)
        cfg = cls(
            endpoint_id=n.text("endpoint_id", required=True),
            licence_key=n.text("licence_key", required=True),
            artefacts=_artefacts(n.section("artefacts", required=True)),
            sensors=sensors,
            profile=_profile(n.section("profile")),
            robot=_robot(n.section("robot")),
            tick_ms=n.integer("tick_ms", default=10, min=1),
            reflex=_reflex(n.section("reflex")),
            cortex=_cortex(n.section("cortex")),
            telemetry=_telemetry(n.section("telemetry")),
            source=source,
        )
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """A plain mapping that :meth:`from_mapping` accepts again (paths as strings)."""
        d = asdict(self)
        d.pop("source")
        d["sensors"] = [
            {
                "id": s.id,
                "adapter": s.adapter,
                **({"calib": str(s.calib)} if s.calib else {}),
                **({"spec": s.spec} if s.spec else {}),
                **dict(s.options),
            }
            for s in self.sensors
        ]
        d["cortex"]["sinks"] = list(self.cortex.sinks)
        out: dict[str, Any] = _stringify_paths(d)
        return out


# -- validation helpers ------------------------------------------------------------------


def _expand(v: Any, env: Mapping[str, str], path: str) -> Any:
    """Expand ``${VAR}`` in every string, recursively. A missing variable is an error."""
    if isinstance(v, str):

        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in env:
                raise ConfigError(f"{path}: environment variable ${{{name}}} is not set")
            return env[name]

        return _ENV_RE.sub(sub, v)
    if isinstance(v, Mapping):
        return {k: _expand(x, env, f"{path}.{k}" if path else str(k)) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x, env, f"{path}[{i}]") for i, x in enumerate(v)]
    return v


def _stringify_paths(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: _stringify_paths(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_stringify_paths(x) for x in v]
    return v


def _suggest(name: str, choices: Sequence[str]) -> str:
    m = difflib.get_close_matches(name, list(choices), n=1, cutoff=0.6)
    return f"; did you mean '{m[0]}'?" if m else ""


class _Node:
    """A mapping under validation, with its key path for messages."""

    def __init__(self, data: Mapping[str, Any], prefix: str, base: Path) -> None:
        self.data = data
        self.prefix = prefix
        self.base = base

    def _p(self, key: str) -> str:
        return f"{self.prefix}.{key}" if self.prefix else key

    def check_keys(self, known: Sequence[str]) -> None:
        for k in self.data:
            if k not in known:
                raise ConfigError(
                    f"{self._p(str(k))}: unknown key{_suggest(str(k), known)} (known: {', '.join(known)})"
                )

    def _get(
        self,
        key: str,
        typ: type[_T] | tuple[type, ...],
        default: Any,
        required: bool,
        expected: str,
    ) -> Any:
        if key not in self.data or self.data[key] is None:
            if required:
                raise ConfigError(f"{self._p(key)}: required")
            return default
        v = self.data[key]
        if isinstance(v, bool) and bool not in (typ if isinstance(typ, tuple) else (typ,)):
            v_ok = False
        else:
            v_ok = isinstance(v, typ)
        if not v_ok:
            raise ConfigError(f"{self._p(key)}: got {v!r} ({type(v).__name__}), expected {expected}")
        return v

    def text(
        self,
        key: str,
        *,
        default: str | None = None,
        required: bool = False,
        one_of: Sequence[str] | None = None,
    ) -> Any:
        v = self._get(key, str, default, required, "str")
        if one_of is not None and v is not None and v not in one_of:
            raise ConfigError(f"{self._p(key)}: {v!r} is not one of {', '.join(one_of)}{_suggest(v, one_of)}")
        return v

    def integer(
        self,
        key: str,
        *,
        default: int | None = None,
        required: bool = False,
        min: int | None = None,
    ) -> Any:
        v = self._get(key, int, default, required, "int")
        if v is not None and min is not None and v < min:
            raise ConfigError(f"{self._p(key)}: {v} must be >= {min}")
        return v

    def number(
        self,
        key: str,
        *,
        default: float | None = None,
        min: float | None = None,
        max: float | None = None,
    ) -> Any:
        v = self._get(key, (int, float), default, False, "number")
        if v is None:
            return None
        v = float(v)
        if min is not None and v < min:
            raise ConfigError(f"{self._p(key)}: {v} must be >= {min}")
        if max is not None and v > max:
            raise ConfigError(f"{self._p(key)}: {v} must be <= {max}")
        return v

    def flag(self, key: str, *, default: bool) -> Any:
        return self._get(key, bool, default, False, "true/false")

    def path(self, key: str, *, default: Path | None = None, required: bool = False) -> Any:
        v = self._get(key, str, None, required, "path")
        if v is None:
            return default
        p = Path(v).expanduser()
        return p if p.is_absolute() else (self.base / p).resolve()

    def items(self, key: str, *, required: bool = False) -> Any:
        return self._get(key, list, [], required, "list")

    def section(self, key: str, *, required: bool = False) -> _Node:
        v = self._get(key, Mapping, {}, required, "mapping")
        return _Node(v, self._p(key), self.base)


_TOP_KEYS = (
    "endpoint_id",
    "licence_key",
    "profile",
    "artefacts",
    "robot",
    "sensors",
    "tick_ms",
    "reflex",
    "cortex",
    "telemetry",
)


def _profile(n: _Node) -> ProfileRef:
    n.check_keys(("name", "version"))
    return ProfileRef(name=n.text("name", default="null"), version=n.integer("version", min=0))


def _artefacts(n: _Node) -> ArtefactsConfig:
    n.check_keys(("world", "reflex", "cache", "store"))
    cache = n.path("cache", default=Path("~/.senseai").expanduser())
    store_raw = n.data.get("store")
    store: Path | str
    if store_raw is None:
        store = cache / "store"
    elif isinstance(store_raw, str) and store_raw.startswith(("https://", "http://")):
        store = store_raw
    else:
        store = n.path("store")
    return ArtefactsConfig(
        world=n.text("world", required=True),
        reflex=n.text("reflex", required=True),
        cache=cache,
        store=store,
    )


def _robot(n: _Node) -> RobotConfig:
    n.check_keys(("footprint_radius_m", "max_decel_mps2", "speed_cap_mps", "kind"))
    return RobotConfig(
        footprint_radius_m=n.number("footprint_radius_m", default=0.35, min=0.01),
        max_decel_mps2=n.number("max_decel_mps2", default=2.0, min=0.1),
        speed_cap_mps=n.number("speed_cap_mps", default=1.5, min=0.0),
        kind=n.text("kind", default="amr", one_of=("amr", "manipulator")),
    )


def _sensors(items: list[Any], base: Path) -> tuple[SensorConfig, ...]:
    out: list[SensorConfig] = []
    seen: set[str] = set()
    if not items:
        raise ConfigError("sensors: at least one sensor is required")
    for i, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ConfigError(f"sensors[{i}]: got {item!r}, expected a mapping with id and adapter")
        n = _Node(item, f"sensors[{i}]", base)
        sid = n.text("id", required=True)
        if sid in seen:
            raise ConfigError(f"sensors[{i}].id: {sid!r} is already used by another sensor")
        seen.add(sid)
        adapter = n.text("adapter", required=True)
        if adapter not in ADAPTERS:
            raise ConfigError(
                f"sensors[{i}].adapter: {adapter!r} is not a known adapter"
                f"{_suggest(adapter, sorted(ADAPTERS))} (known: {', '.join(sorted(ADAPTERS))})"
            )
        options = {k: v for k, v in item.items() if k not in ("id", "adapter", "calib", "spec")}
        out.append(
            SensorConfig(id=sid, adapter=adapter, calib=n.path("calib"), spec=n.text("spec"), options=options)
        )
    return tuple(out)


def _reflex(n: _Node) -> ReflexConfig:
    n.check_keys(("bridge", "planner_topic", "out_topic", "max_lateral_m"))
    lat = n.number("max_lateral_m", default=ENVELOPE_MAX_LATERAL_M, min=0.0)
    if lat > ENVELOPE_MAX_LATERAL_M:
        raise ConfigError(
            f"reflex.max_lateral_m: {lat} exceeds the safety envelope ({ENVELOPE_MAX_LATERAL_M} m); "
            "a profile or config can only lower it"
        )
    return ReflexConfig(
        bridge=n.text("bridge", default="callback", one_of=BRIDGES),
        planner_topic=n.text("planner_topic", default="/cmd_vel"),
        out_topic=n.text("out_topic", default="/senseai/cmd_vel"),
        max_lateral_m=lat,
    )


def _cortex(n: _Node) -> CortexConfig:
    n.check_keys(("sinks", "jsonl_path", "mqtt_url"))
    sinks = n.items("sinks") or ["callback"]
    for i, s in enumerate(sinks):
        if s not in SINKS:
            raise ConfigError(
                f"cortex.sinks[{i}]: {s!r} is not one of {', '.join(SINKS)}{_suggest(str(s), SINKS)}"
            )
    jsonl = n.path("jsonl_path")
    if "jsonl" in sinks and jsonl is None:
        raise ConfigError("cortex.jsonl_path: required when 'jsonl' is a sink")
    return CortexConfig(sinks=tuple(sinks), jsonl_path=jsonl, mqtt_url=n.text("mqtt_url"))


def _telemetry(n: _Node) -> TelemetryConfig:
    n.check_keys(("enabled", "raw_frames", "upload", "spool_dir", "spool_max_mb"))
    upload = n.text("upload")
    if upload is not None and ":" not in upload:
        raise ConfigError(f"telemetry.upload: {upload!r} must be 'module:function'")
    return TelemetryConfig(
        enabled=n.flag("enabled", default=False),
        raw_frames=n.flag("raw_frames", default=False),
        upload=upload,
        spool_dir=n.path("spool_dir"),
        spool_max_mb=n.integer("spool_max_mb", default=256, min=1),
    )


# -- the template `senseai init` writes -----------------------------------------------------

_LIVE_SENSORS = """\
sensors:
  # One entry per sensor. `id` is the name every log and event uses for it.
  # Keys other than id / adapter / calib / spec go straight to the adapter.
  - {id: dvs0, adapter: prophesee, sensor: evk4, calib: calib/dvs0.yaml, spec: generic_event_camera}
  - {id: rgb0, adapter: v4l2, device: /dev/video0, calib: calib/rgb0.yaml, spec: generic_rgb_camera}
  - {id: mic0, adapter: respeaker, channels: 4, calib: calib/mic0.yaml, spec: generic_mic_array}
  - {id: imu0, adapter: serial, port: /dev/ttyUSB0}
"""

_FIXTURE_SENSORS = """\
sensors:
  # Replay: generated samples, so the pipeline runs without hardware.
  # Replace with your sensors (`senseai init` without --replay lists them).
  - {id: dvs0, adapter: synthetic, modality: event, duration_s: 10, spec: generic_event_camera}
  - {id: rgb0, adapter: synthetic, modality: frame, rate_hz: 20, duration_s: 10, spec: generic_rgb_camera}
  - {id: mic0, adapter: synthetic, modality: audio, duration_s: 10, spec: generic_mic_array}
  - {id: imu0, adapter: synthetic, modality: imu, rate_hz: 200, duration_s: 10}
"""


def template(*, replay: str | None = None, endpoint_id: str = "amr-0042") -> str:
    """The commented ``senseai.yaml`` that ``senseai init`` writes.

    Args:
        replay: ``None`` for live sensors, ``"fixture"`` for the generated replay sources,
            or a path to a DSEC / M3ED / MVSEC recording.
        endpoint_id: The robot identifier to write.
    """
    if replay is None:
        sensors = _LIVE_SENSORS
    elif replay == "fixture":
        sensors = _FIXTURE_SENSORS
    else:
        p = Path(replay).expanduser()
        kind = "dsec" if p.is_dir() else "hdf5"
        sensors = (
            "sensors:\n"
            f"  # Replay of {p}. One entry per source in the recording.\n"
            f"  - {{id: events_left, adapter: {kind}, path: {p}, spec: generic_event_camera}}\n"
            f"  - {{id: rgb_left, adapter: {kind}, path: {p}, spec: generic_rgb_camera}}\n"
        )
    adapters = "\n".join(
        f"#   {e.name:<10} {e.description}" for e in sorted(ADAPTERS.values(), key=lambda e: e.name)
    )
    return f"""\
# senseai.yaml — the whole SenseAI configuration for one robot.
# Every key is documented in the README's YAML reference. Unknown keys are errors.

endpoint_id: {endpoint_id}            # this robot; used in licence metering, telemetry and health
licence_key: ${{SENSEAI_LICENCE}}     # ${{ENV}} is expanded; never commit a literal key

profile: {{name: panther, version: null}}   # version null = newest allowed; name null = no priors

artefacts:
  world: world-v1.2                 # signed world-model bundle (carries its tokenizer)
  reflex: reflex-v1.0               # signed reflex bundle
  cache: ~/.senseai                 # local cache; every load re-verifies signatures
  # store: https://artefacts.senseai.example/   # where to fetch from; default is <cache>/store

robot:
  footprint_radius_m: 0.35          # circle that contains the robot
  max_decel_mps2: 2.0               # braking the reflex may request (the envelope caps it)
  speed_cap_mps: 1.5                # planner's top speed until update_context() sends a live one
  kind: amr                         # amr | manipulator

{sensors}
# Adapters:
{adapters}

tick_ms: 10                         # must equal the tick the world artefact was trained on

reflex:
  bridge: callback                  # callback | ros2_cmd_vel
  planner_topic: /cmd_vel           # ros2_cmd_vel: the planner's command topic
  out_topic: /senseai/cmd_vel       # ros2_cmd_vel: point your controller here
  max_lateral_m: 0.3                # may only lower the safety envelope's limit

cortex:
  sinks: [callback]                 # callback | jsonl | ros2 | mqtt
  # jsonl_path: /var/log/senseai/events.jsonl

telemetry:
  enabled: false                    # opt-in
  raw_frames: false                 # raw sensor data is never included unless true
  upload: null                      # module:function hook; null = spool locally, never upload
"""
