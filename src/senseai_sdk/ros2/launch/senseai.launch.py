# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""``ros2 launch senseai_sdk senseai.launch.py config:=senseai.yaml``

Brings up the whole pipeline as one node from the same YAML the Python API reads. Every
parameter lives in that file; the launch argument is only its path.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config = LaunchConfiguration("config")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value="senseai.yaml", description="path to senseai.yaml"),
            Node(
                package="senseai_sdk",
                executable="senseai-ros2",
                name="senseai",
                output="screen",
                arguments=["--config", config],
            ),
        ]
    )
