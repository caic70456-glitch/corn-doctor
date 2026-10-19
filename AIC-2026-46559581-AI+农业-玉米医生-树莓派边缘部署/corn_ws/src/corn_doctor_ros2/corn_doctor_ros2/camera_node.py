#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相机采集节点：OpenCV 打开摄像头，按固定频率发布 sensor_msgs/Image。

话题:
  发布 /image_raw  (sensor_msgs/Image, bgr8)

参数:
  camera_index  摄像头编号 (默认 0；多个摄像头时换 1/2/...)
  width/height  采集分辨率 (默认 640x480)
  publish_hz    发布频率 Hz (默认 5.0)

任何能发布 /image_raw 的 ROS2 相机节点（usb_cam、realsense2_camera 等）
都可以替代本节点，识别节点只认话题不认设备。
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraNode(Node):

    def __init__(self):
        super().__init__("camera_node")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("publish_hz", 5.0)

        index = int(self.get_parameter("camera_index").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        hz = float(self.get_parameter("publish_hz").value)

        import cv2
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"无法打开摄像头 {index}。排查: ls /dev/video* 看设备；"
                f"换 --ros-args -p camera_index:=1 试其他编号；"
                f"检查当前用户是否在 video 组(groups 命令)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.get_logger().info(f"摄像头 {index} 已打开 ({width}x{height})，"
                               f"发布 /image_raw @ {hz}Hz")

        self.pub = self.create_publisher(Image, "/image_raw", qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / max(hz, 0.1), self.on_timer)

    def on_timer(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("读取画面失败，重试中...", throttle_duration_sec=3.0)
            return
        h, w = frame.shape[:2]
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height = h
        msg.width = w
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = 3 * w
        msg.data = frame.tobytes()
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
