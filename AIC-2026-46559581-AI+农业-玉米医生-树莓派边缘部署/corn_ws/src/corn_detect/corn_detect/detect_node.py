#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""病害识别节点（分类模型 + 画面中心点方案）。

背景:
  现有数据集标注为图片级（无病斑框），因此采用:
    1. 分类模型(best.onnx, MobileNetV2)对整张彩色图判断 生病/健康 + 病害类别;
    2. 三维坐标取"画面正中心"像素在对齐深度图上的有效深度, 按内参反投影。

话题:
  订阅 /color/image_raw                  (sensor_msgs/Image, 经 launch 重映射)
  订阅 /aligned_depth_to_color/image_raw (sensor_msgs/Image, mono16, 对齐深度)
  订阅 /color/camera_info               (sensor_msgs/CameraInfo, 内参)
  发布 /target_3d                        (geometry_msgs/PointStamped, 相机坐标系)

  说明: /target_3d 发布的是画面中心点的三维坐标(单位米), 健康时同样发布,
        串口节点按该话题持续转发给下位单片机。

参数:
  model_path     模型路径(默认 包内 data/best.onnx)
  img_size       模型输入边长(默认 224)
  infer_interval 推理节流间隔秒(默认 0.5)
  depth_unit     深度图原始值->米的换算(默认 0.001, 即毫米)

预处理与训练端完全一致:
  Resize(224,224) -> /255 -> ImageNet 归一化 -> RGB; ONNX 输出已含 softmax。
"""

import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_CN = {
    "Blight": "叶斑病",
    "Common_Rust": "普通锈病",
    "Gray_Leaf_Spot": "灰斑病",
    "Healthy": "健康",
}


def default_model_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("corn_detect"), "data", "best.onnx")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    return "data/best.onnx"


class DetectNode(Node):

    def __init__(self):
        super().__init__("detect_node")
        self.declare_parameter("model_path", default_model_path())
        self.declare_parameter("img_size", 224)
        self.declare_parameter("infer_interval", 0.5)
        self.declare_parameter("depth_unit", 0.001)

        self.img_size = int(self.get_parameter("img_size").value)
        self.interval = float(self.get_parameter("infer_interval").value)
        self.depth_unit = float(self.get_parameter("depth_unit").value)

        # ---- 模型加载 ----
        import onnxruntime as ort
        model_path = str(self.get_parameter("model_path").value)
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"未找到模型: {model_path}\n"
                f"请把 best.onnx 放入包内 data/ 目录后重新 colcon build --symlink-install")
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.class_names = self._read_class_names()
        self.get_logger().info(f"模型加载成功: {model_path}")
        self.get_logger().info(f"输入 {self.sess.get_inputs()[0].shape}, "
                               f"类别: {self.class_names}")

        self.bridge = CvBridge()
        self._K = None            # 内参 [fx, fy, cx, cy]
        self._frame_id = ""       # PointStamped 参考系(取 camera_info 的 frame)
        self._depth = None        # 最新对齐深度图(uint16)
        self._last_infer = 0.0
        self._last_log = ""

        self.sub_color = self.create_subscription(
            Image, "/color/image_raw", self.on_color, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(
            Image, "/aligned_depth_to_color/image_raw", self.on_depth, qos_profile_sensor_data)
        self.sub_info = self.create_subscription(
            CameraInfo, "/color/camera_info", self.on_info, qos_profile_sensor_data)
        self.pub = self.create_publisher(PointStamped, "/target_3d", 10)
        self.get_logger().info("订阅 /color/image_raw + /aligned_depth_to_color/image_raw, "
                               "发布 /target_3d")

    # ---------------- 回调 ----------------
    def on_info(self, msg: CameraInfo):
        k = msg.k
        self._K = (k[0], k[4], k[2], k[5])          # fx, fy, cx, cy
        self._frame_id = msg.header.frame_id

    def on_depth(self, msg: Image):
        self._depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")   # uint16

    def on_color(self, msg: Image):
        now = time.monotonic()
        if now - self._last_infer < self.interval:
            return
        self._last_infer = now

        if self._K is None or self._depth is None:
            self.get_logger().warn("等待深度图/相机内参...", throttle_duration_sec=3.0)
            return
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w = frame.shape[:2]

        # 1) 整图分类
        cls, conf, probs = self._classify(frame)
        name = self.class_names[cls]
        healthy = (name == "Healthy")
        verdict = "健康" if healthy else "生病"

        # 2) 画面中心像素 -> 有效深度 -> 反投影
        u, v = w // 2, h // 2
        z = self._sample_depth(u, v, w, h)
        if z is None:
            self.get_logger().warn(
                f"画面中心({u},{v})深度无效，未发布三维坐标", throttle_duration_sec=3.0)
            return
        fx, fy, cx, cy = self._K
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # 3) 发布
        out = PointStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._frame_id or "camera_color_optical_frame"
        out.point.x = float(x)
        out.point.y = float(y)
        out.point.z = float(z)
        self.pub.publish(out)

        log = (f"结论: {verdict} | {CLASS_CN.get(name, name)} conf={conf:.2f} -> "
               f"X={x:.3f} Y={y:.3f} Z={z:.3f} m (中心像素 {u},{v})")
        if log != self._last_log:
            self.get_logger().info(log)
            self._last_log = log

    # ---------------- 深度采样 ----------------
    def _sample_depth(self, u, v, w, h, radius=4):
        """取 (u,v) 处深度(米)；无效值时在邻域找最近有效像素，全无效返回 None。"""
        depth = self._depth
        for r in range(radius + 1):
            y0, y1 = max(v - r, 0), min(v + r + 1, h)
            x0, x1 = max(u - r, 0), min(u + r + 1, w)
            patch = depth[y0:y1, x0:x1].astype(np.float32)
            valid = patch[patch > 0]
            if valid.size:
                return float(valid[0]) * self.depth_unit
        return None

    # ---------------- 分类推理 ----------------
    def _read_class_names(self):
        """从 ONNX 元数据读类别名(训练导出时写入 class_0..class_3)。"""
        try:
            meta = dict(self.sess.get_modelmeta().custom_metadata_map)
            names = [meta[f"class_{i}"] for i in range(len(meta)) if f"class_{i}" in meta]
            if names:
                return names
        except Exception:
            pass
        return ["class0", "class1", "class2", "class3"]

    def _classify(self, frame_bgr):
        """整图分类，返回 (类别索引, 置信度, 概率向量)。"""
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        x = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])   # (1,3,224,224)
        probs = self.sess.run(None, {self.input_name: x})[0][0]
        cls = int(np.argmax(probs))
        return cls, float(probs[cls]), probs


def main(args=None):
    rclpy.init(args=args)
    node = DetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
