#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键启动：RealSense D415(深度对齐) + 病害检测 + 串口桥接。

用法:
  ros2 launch corn_detect detect.launch.py
  # 指定串口/波特率:
  ros2 launch corn_detect detect.launch.py serial_port:=/dev/ttyACM0 serial_baud:=9600

说明:
  Humble 版 realsense2_camera 默认话题前缀为 /camera/camera
  （即 /camera/camera/color/image_raw），detect_node 内部按 /color/... 订阅，
  通过下面的 topic_prefix 参数自动映射。若你的驱动版本话题是 /camera/color/...，
  启动时传 topic_prefix:=/camera 即可。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch_ros.substitutions import FindPackageShare

PACKAGE_NAME = "corn_detect"


def generate_launch_description():
    pkg_share = get_package_share_directory(PACKAGE_NAME)
    default_model = os.path.join(pkg_share, "data", "best.onnx")

    args = [
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyUSB0",
                              description="串口设备"),
        DeclareLaunchArgument("serial_baud", default_value="115200",
                              description="串口波特率"),
        DeclareLaunchArgument("model_path", default_value=default_model,
                              description="ONNX 模型路径"),
        DeclareLaunchArgument("infer_interval", default_value="0.5",
                              description="推理节流间隔(秒)"),
        DeclareLaunchArgument("topic_prefix", default_value="/camera/camera",
                              description="RealSense 话题前缀"),
    ]

    # 1) RealSense D415 驱动，必须开深度对齐
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare("realsense2_camera"), "/launch/rs_launch.py"]),
        launch_arguments={
            "align_depth": "true",
            "camera_name": "camera",
            "camera_namespace": "camera",
        }.items(),
    )

    # 2) 病害检测节点（彩色图推理 + 深度反投影 -> /target_3d）
    prefix = LaunchConfiguration("topic_prefix")
    detect = Node(
        package=PACKAGE_NAME,
        executable="detect_node",
        name="detect_node",
        output="screen",
        remappings=[
            ("/color/image_raw", [prefix, "/color/image_raw"]),
            ("/aligned_depth_to_color/image_raw",
             [prefix, "/aligned_depth_to_color/image_raw"]),
            ("/color/camera_info", [prefix, "/color/camera_info"]),
        ],
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "infer_interval": LaunchConfiguration("infer_interval"),
        }],
    )

    # 3) 串口桥接节点（/target_3d -> 单片机）
    serial_bridge = Node(
        package=PACKAGE_NAME,
        executable="serial_bridge_node",
        name="serial_bridge_node",
        output="screen",
        parameters=[{
            "port": LaunchConfiguration("serial_port"),
            "baudrate": LaunchConfiguration("serial_baud"),
        }],
    )

    return LaunchDescription(args + [realsense, detect, serial_bridge])
