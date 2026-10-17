# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""The ROS 2 node that runs the whole pipeline from one ``senseai.yaml``.

Topics (all standard message types; the SDK defines none of its own):

- subscribes ``reflex.planner_topic`` (``geometry_msgs/Twist``): the planner's command
- publishes ``reflex.out_topic`` (``geometry_msgs/Twist``): the reflex-modified command
- publishes ``/senseai/health`` (``std_msgs/String``, JSON) at 1 Hz
- publishes ``/senseai/events`` (``std_msgs/String``, JSON) per cortex event, until
  ``senseai_cortex`` ships its own ``Event.msg``

``/senseai/cmd_vel`` is published only while the pipeline is ``ok`` or ``degraded`` and has
produced a tick in the last second. When the node is unhealthy (``failed``, or no ticks),
nothing is published, so the client's own safety stack stays authoritative. This mirrors the
``senseai_reflex`` ROS bridge contract and is replaced by it when that package ships.

The ROS parts are thin and kept apart from the logic (:class:`Ros2CmdVelBridge` takes its
publisher and clock as arguments) so they can be tested without a ROS install.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Any

from senseai_sdk.bridges import PlannerCommand

__all__ = ["Ros2CmdVelBridge", "main", "run_node"]

STALE_S = 1.0


class Ros2CmdVelBridge:
    """Applies reflex commands to the planner's Twist and publishes the result.

    Args:
        publish: Called with the actuator :class:`PlannerCommand` (the node wraps it in a Twist).
        healthy: Returns True while the node may publish.
        max_lateral_m: Bound from config.
        lateral_gain: rad/s per metre of lateral offset.
        now: Monotonic clock (tests inject one).
    """

    def __init__(
        self,
        publish: Callable[[PlannerCommand], None],
        healthy: Callable[[], bool],
        *,
        max_lateral_m: float = 0.5,
        lateral_gain: float = 1.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._publish = publish
        self._healthy = healthy
        self._now = now
        self.max_lateral_m = max_lateral_m
        self.lateral_gain = lateral_gain
        self.planner = PlannerCommand()
        self.planner_at: float | None = None
        self.published = 0
        self.suppressed = 0

    def on_planner(self, linear: float, angular: float) -> None:
        """The planner's latest command (from the ``/cmd_vel`` subscription)."""
        self.planner = PlannerCommand(float(linear), float(angular))
        self.planner_at = self._now()

    def planner_cmd(self) -> PlannerCommand:
        """What the pipeline multiplies: the latest planner command, or zero if stale."""
        if self.planner_at is None or self._now() - self.planner_at > STALE_S:
            return PlannerCommand()
        return self.planner

    def apply(self, cmd: Any, planner_cmd: Any) -> PlannerCommand:
        """Speed × scale, lateral → angular nudge, zero on stop; publish only while healthy."""
        import math

        p = planner_cmd if isinstance(planner_cmd, PlannerCommand) else PlannerCommand()
        if bool(getattr(cmd, "stop", False)):
            out = PlannerCommand(0.0, 0.0)
        else:
            scale = min(1.0, max(0.0, float(getattr(cmd, "speed_scale", 1.0))))
            lat = float(getattr(cmd, "lateral_offset_m", 0.0))
            lat = math.copysign(min(abs(lat), self.max_lateral_m), lat)
            out = PlannerCommand(p.linear * scale, p.angular + self.lateral_gain * lat)
        if self._healthy():
            self._publish(out)
            self.published += 1
        else:
            self.suppressed += 1
        return out


def run_node(
    config_path: str, *, args: Sequence[str] | None = None
) -> None:  # pragma: no cover - needs ROS 2
    """Spin the node until shutdown."""
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import String

    from senseai_sdk.api import SenseAI

    rclpy.init(args=list(args) if args else None)
    node = Node("senseai")
    sai = SenseAI(config_path)
    cfg = sai.config
    pub_cmd = node.create_publisher(Twist, cfg.reflex.out_topic, 10)
    pub_health = node.create_publisher(String, "/senseai/health", 10)
    pub_events = node.create_publisher(String, "/senseai/events", 10)
    last_tick = [time.monotonic()]

    def publish(out: PlannerCommand) -> None:
        msg = Twist()
        msg.linear.x = out.linear
        msg.angular.z = out.angular
        pub_cmd.publish(msg)

    def healthy() -> bool:
        return sai.health().state != "failed" and time.monotonic() - last_tick[0] < STALE_S

    bridge = Ros2CmdVelBridge(publish, healthy, max_lateral_m=cfg.reflex.max_lateral_m)
    node.create_subscription(
        Twist, cfg.reflex.planner_topic, lambda m: bridge.on_planner(m.linear.x, m.angular.z), 10
    )
    sai.bind_actuator(bridge)
    sai.on_command(lambda _c: last_tick.__setitem__(0, time.monotonic()))
    sai.on_event(lambda ev: pub_events.publish(String(data=json.dumps(_event_json(ev)))))
    node.create_timer(1.0, lambda: pub_health.publish(String(data=json.dumps(sai.health().to_json()))))
    sai.start()
    try:
        rclpy.spin(node)
    finally:
        sai.stop()
        node.destroy_node()
        rclpy.shutdown()


def _event_json(ev: Any) -> dict[str, Any]:
    """A cortex ``Event`` as JSON (until ``senseai_cortex`` ships ``Event.msg``)."""
    return {
        "id": getattr(ev, "id", None),
        "t_master_ns": getattr(ev, "t_master_ns", None),
        "type": str(getattr(ev, "type", "")),
        "text": getattr(ev, "text", str(ev)),
        "confidence": getattr(ev, "confidence", None),
        "track_id": getattr(ev, "track_id", None),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs ROS 2
    """``senseai-ros2 --config senseai.yaml``."""
    import argparse

    ap = argparse.ArgumentParser(prog="senseai-ros2")
    ap.add_argument("--config", default="senseai.yaml")
    a, rest = ap.parse_known_args(argv)
    run_node(a.config, args=rest)
    return 0
