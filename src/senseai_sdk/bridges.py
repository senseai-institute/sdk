# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Controller bridges: how a reflex command modifies the planner's command.

``senseai-reflex`` owns ``ControllerBridge`` and ships the ``callback`` and ``ros2_cmd_vel``
bridges. Until it does, the SDK carries this minimal callback bridge so the fail-safe path
(watchdog → bridge → controller) is real today. It implements exactly the rule the reflex
README states: speed × ``speed_scale``, lateral offset as a bounded angular nudge, zero on
``stop``. When ``senseai_reflex.bridge`` ships, :func:`make_bridge` returns its classes and
this file keeps only the protocol.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = ["BridgeLike", "CallbackBridge", "PlannerCommand", "make_bridge"]


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


def make_bridge(name: str, **kw: Any) -> Any:
    """Build the bridge ``name`` from config: ``senseai_reflex``'s when installed, else the SDK's.

    Raises:
        PipelineError: ``ros2_cmd_vel`` without ``senseai_reflex`` (it needs the reflex ROS node).
    """
    from senseai_sdk.errors import PipelineError

    try:
        import importlib

        mod = importlib.import_module(
            "senseai_reflex.bridge." + ("ros2_cmd_vel" if name == "ros2_cmd_vel" else "callback")
        )
    except ImportError:
        if name == "ros2_cmd_vel":
            raise PipelineError(
                "reflex.bridge: ros2_cmd_vel needs senseai-reflex[ros2], which is not installed; "
                "use bridge: callback until it ships"
            ) from None
        return CallbackBridge(**kw)
    cls = getattr(mod, "Ros2CmdVelBridge" if name == "ros2_cmd_vel" else "CallbackBridge")
    return cls(**kw)
