#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""串口桥接节点：/target_3d 三维坐标 -> 串口文本输出给下位单片机。

话题:
  订阅 /target_3d (geometry_msgs/PointStamped)

串口输出格式（每行一条，方便单片机逐行解析）:
  X:%.3f,Y:%.3f,Z:%.3f\r\n
  单位: 米（相机坐标系）

参数:
  port      串口设备（默认 /dev/ttyUSB0）
  baudrate  波特率（默认 115200）

设计要点:
  - 回调只把坐标放进内存队列，串口 IO 全部在独立线程完成，不阻塞 ROS 回调
  - 串口打开失败/中途断开只打印警告并持续重连，节点不崩溃
  - 队列满时丢弃最旧坐标，保证单片机收到的是较新的目标位置
"""

import queue
import threading

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node

import serial


class SerialBridgeNode(Node):

    def __init__(self):
        super().__init__("serial_bridge_node")
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.port = str(self.get_parameter("port").value)
        self.baud = int(self.get_parameter("baudrate").value)

        self.ser = None
        self.q: queue.Queue = queue.Queue(maxsize=20)

        self.sub = self.create_subscription(PointStamped, "/target_3d", self.on_target, 10)
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self.get_logger().info(f"串口桥接已启动: {self.port} @ {self.baud}，"
                               f"等待 /target_3d ...")

    # ---------------- ROS 回调（不做任何串口 IO）----------------
    def on_target(self, msg: PointStamped):
        p = (float(msg.point.x), float(msg.point.y), float(msg.point.z))
        try:
            self.q.put_nowait(p)
        except queue.Full:
            try:                       # 丢最旧，保最新
                self.q.get_nowait()
                self.q.put_nowait(p)
            except queue.Empty:
                pass

    # ---------------- 串口线程 ----------------
    def _ensure_open(self):
        """确保串口已打开；失败只告警，返回 False。"""
        if self.ser is not None and self.ser.is_open:
            return True
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self.get_logger().info(f"串口已打开: {self.port} @ {self.baud}")
            return True
        except Exception as e:
            self.get_logger().warn(
                f"串口打开失败({self.port}): {e}，将持续重试..."
                f"(检查设备/权限: sudo usermod -aG dialout $USER)",
                throttle_duration_sec=5.0)
            return False

    def _worker(self):
        self._ensure_open()            # 启动时先试一次，尽早暴露问题
        while True:
            p = self.q.get()           # 阻塞等新坐标
            if not self._ensure_open():
                continue
            try:
                self.ser.write(("X:%.3f,Y:%.3f,Z:%.3f\r\n" % p).encode("ascii"))
                self.get_logger().info("串口输出: X:%.3f,Y:%.3f,Z:%.3f" % p,
                                       throttle_duration_sec=1.0)
            except Exception as e:
                self.get_logger().warn(f"串口写入失败，关闭等待重连: {e}",
                                       throttle_duration_sec=5.0)
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

    def destroy_node(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
