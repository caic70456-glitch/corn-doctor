#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""玉米医生一键启动：相机采集 + ONNX 病害识别。

用法（colcon build && source 之后）:
  ros2 launch corn_doctor_ros2 corn_doctor.launch.py

可带参数覆盖默认值:
  ros2 launch corn_doctor_ros2 corn_doctor.launch.py camera_index:=1 infer_interval:=3.0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE_NAME = "corn_doctor_ros2"


def generate_launch_description():
    pkg_share = get_package_share_directory(PACKAGE_NAME)
    default_model = os.path.join(pkg_share, "models", "corn_doctor.onnx")

    args = [
        DeclareLaunchArgument("camera_index", default_value="0",
                              description="摄像头编号"),
        DeclareLaunchArgument("model_path", default_value=default_model,
                              description="ONNX 模型路径"),
        DeclareLaunchArgument("infer_interval", default_value="2.0",
                              description="识别间隔(秒)"),
        DeclareLaunchArgument("publish_hz", default_value="5.0",
                              description="相机发布频率(Hz)"),
    ]

    camera = Node(
        package=PACKAGE_NAME,
        executable="camera_node",
        name="camera",
        output="screen",
        parameters=[{
            "camera_index": LaunchConfiguration("camera_index"),
            "width": 640,
            "height": 480,
            "publish_hz": LaunchConfiguration("publish_hz"),
        }],
    )

    detector = Node(
        package=PACKAGE_NAME,
        executable="detector_node",
        name="detector",
        output="screen",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "infer_interval": LaunchConfiguration("infer_interval"),
            "min_conf": 0.60,
        }],
    )

    return LaunchDescription(args + [camera, detector])
