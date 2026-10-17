# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Controller bridges: how a reflex command modifies the planner's command.

``senseai-reflex`` owns ``ControllerBridge`` and ships the ``callback`` and ``ros2_cmd_vel``
bridges. The SDK's ``callback`` bridge is its own (:class:`CallbackBridge`): it is part of
the SDK's public API (``on_actuator`` receives an SDK :class:`PlannerCommand`) and must work
on the watchdog path even when the reflex stage fails. It implements the reflex rule: speed ×
``speed_scale``, lateral offset as a bounded angular nudge, zero on ``stop``.
``ros2_cmd_vel`` wraps reflex's ``CmdVelBridge`` in :class:`ReflexBridgeAdapter`, which
converts between the SDK's and reflex's planner/actuator types.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = ["BridgeLike", "CallbackBridge", "PlannerCommand", "ReflexBridgeAdapter", "make_bridge"]


@dataclass(frozen=True, slots=True)
class PlannerCommand:
    """The planner's velocity command: linear m/s and angular rad/s (a ROS ``Twist`` subset)."""

    linear: float = 0.0
    angular: float = 0.0


@runtime_checkable
class BridgeLike(Protocol):
    """What the pipeline needs from a bridge. ``senseai_reflex.bridge.base.ControllerBridge`` satisfies it."""

    def apply(self, cmd: Any, planner_cmd: Any) -> Any:
        """Combine a reflex command with the planner's command into the actuator command."""
        ...


class CallbackBridge:
    """Applies reflex commands to a planner command and hands the result to a Python callback.

    Args:
        on_actuator: Called with the actuator :class:`PlannerCommand` every tick.
        planner: Returns the planner's current command; default: stand still.
        max_lateral_m: The lateral offset bound from config (never above the envelope).
        lateral_gain: Angular nudge per metre of lateral offset, rad/s per m.
    """

    def __init__(
        self,
        on_actuator: Callable[[PlannerCommand], None] | None = None,
        *,
        planner: Callable[[], PlannerCommand] | None = None,
        max_lateral_m: float = 0.5,
        lateral_gain: float = 1.0,
    ) -> None:
        self._on_actuator = on_actuator
        self._planner = planner or (lambda: PlannerCommand())
        self.max_lateral_m = max_lateral_m
        self.lateral_gain = lateral_gain
        self.last: PlannerCommand = PlannerCommand()
        self.applied = 0

    def planner_cmd(self) -> PlannerCommand:
        """The planner's current command."""
        return self._planner()

    def apply(self, cmd: Any, planner_cmd: Any) -> PlannerCommand:
        """The documented rule: speed × scale; lateral → angular nudge; stop → zero."""
        p = planner_cmd if isinstance(planner_cmd, PlannerCommand) else PlannerCommand()
        if bool(getattr(cmd, "stop", False)):
            out = PlannerCommand(0.0, 0.0)
        else:
            scale = min(1.0, max(0.0, float(getattr(cmd, "speed_scale", 1.0))))
            lat = float(getattr(cmd, "lateral_offset_m", 0.0))
            lat = math.copysign(min(abs(lat), self.max_lateral_m), lat)
            out = PlannerCommand(p.linear * scale, p.angular + self.lateral_gain * lat)
        self.last = out
        self.applied += 1
        if self._on_actuator is not None:
            self._on_actuator(out)
        return out


class ReflexBridgeAdapter:
    """Presents a ``senseai_reflex.bridge.base.ControllerBridge`` with the SDK's bridge shape.

    The pipeline hands it an SDK :class:`PlannerCommand` (``linear``/``angular``); reflex wants
    its own ``PlannerCommand`` (``linear_mps``/``angular_rps``) and returns an
    ``ActuatorCommand`` (or ``None`` when unhealthy), which is converted back.
    """

    def __init__(self, inner: Any) -> None:
        from senseai_reflex.bridge.base import PlannerCommand as ReflexPlanner

        self.inner = inner
        self._planner = ReflexPlanner()
        self.last: PlannerCommand = PlannerCommand()

    def apply(self, cmd: Any, planner_cmd: Any) -> PlannerCommand | None:
        """Apply through the reflex bridge; ``None`` when it is unhealthy (nothing delivered)."""
        p = planner_cmd if isinstance(planner_cmd, PlannerCommand) else PlannerCommand()
        self._planner.linear_mps, self._planner.angular_rps = p.linear, p.angular
        out = self.inner.apply(cmd, self._planner)
        if out is None:
            return None
        self.last = PlannerCommand(out.linear_mps, out.angular_rps)
        return self.last


def make_bridge(name: str, **kw: Any) -> Any:
    """Build the bridge ``name`` from config.

    ``callback`` is the SDK's :class:`CallbackBridge`. ``ros2_cmd_vel`` is reflex's
    ``CmdVelBridge`` behind :class:`ReflexBridgeAdapter`.

    Raises:
        PipelineError: ``ros2_cmd_vel`` without ``senseai_reflex``, or an unknown name.
    """
    from senseai_sdk.errors import PipelineError

    if name == "callback":
        return CallbackBridge(**kw)
    if name != "ros2_cmd_vel":
        raise PipelineError(f"reflex.bridge: unknown bridge {name!r}; use callback or ros2_cmd_vel")
    try:
        from senseai_reflex.bridge.ros2_cmd_vel import CmdVelBridge
    except ImportError:
        raise PipelineError(
            "reflex.bridge: ros2_cmd_vel needs senseai-reflex, which is not installed; use bridge: callback"
        ) from None
    kw.pop("max_lateral_m", None)  # reflex's envelope bounds the lateral offset itself
    return ReflexBridgeAdapter(CmdVelBridge(**kw))
