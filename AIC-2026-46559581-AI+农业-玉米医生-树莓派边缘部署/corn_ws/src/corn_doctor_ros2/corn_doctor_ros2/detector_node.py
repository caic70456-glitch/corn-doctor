#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""病害识别节点：订阅相机图像 -> ONNX 模型推理 -> 发布"是否生病"结论。

与 PC 端训练产出的 ONNX 模型（MobileNetV2 分类模型）严格配套：
  输入 (batch,3,224,224)，输出 (batch,4) 四类概率；
  预处理与训练完全一致：BGR转RGB -> 缩放224 -> /255 -> 标准化 -> CHW。

话题:
  订阅 /image_raw          (sensor_msgs/Image)
  发布 /corn_doctor/result (std_msgs/String, JSON)

参数:
  model_path     ONNX 模型路径（默认自动取包内 models/ 下的模型）
  infer_interval 识别间隔秒 (默认 2.0，降低 CPU 占用)
  min_conf       最低置信度 (默认 0.6，低于则报"不确定")
  dark_mean      画面平均亮度低于该值视为光照不足 (默认 45.0)
  leaf_min_ratio 绿色(叶色)像素占比低于该值视为画面无叶片 (默认 0.02)

结论 JSON 示例:
  {"verdict": "生病", "class": "Common_Rust", "class_cn": "普通锈病",
   "confidence": 0.97, "infer_ms": 210.5, "stamp": "2026-10-19 10:23:01"}
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

# 与 PC 端训练预处理严格一致，不要改动
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 类别兜底（正常情况下从 ONNX 元数据自动读取，顺序与训练一致）
FALLBACK_CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
CLASS_CN = {
    "Blight": "叶斑病",
    "Common_Rust": "普通锈病",
    "Gray_Leaf_Spot": "灰斑病",
    "Healthy": "健康",
}


class DetectorNode(Node):

    def __init__(self):
        super().__init__("detector_node")
        # ---- 参数 ----
        self.declare_parameter("model_path", self._default_model_path())
        self.declare_parameter("infer_interval", 2.0)
        self.declare_parameter("min_conf", 0.60)
        self.declare_parameter("dark_mean", 45.0)
        self.declare_parameter("leaf_min_ratio", 0.02)

        model_path = str(self.get_parameter("model_path").value)
        self.interval = float(self.get_parameter("infer_interval").value)
        self.min_conf = float(self.get_parameter("min_conf").value)
        self.dark_mean = float(self.get_parameter("dark_mean").value)
        self.leaf_min_ratio = float(self.get_parameter("leaf_min_ratio").value)

        # ---- 模型加载 ----
        import onnxruntime as ort
        if not os.path.exists(model_path):
            raise RuntimeError(f"未找到模型: {model_path}（检查包是否 colcon build 过）")
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.class_names = self._read_class_names()
        self.get_logger().info(f"模型加载成功: {model_path}")
        self.get_logger().info(f"类别顺序: {self.class_names}")
        self.get_logger().info(f"每 {self.interval}s 识别一帧，结论发布到 /corn_doctor/result")

        self.sub = self.create_subscription(
            Image, "/image_raw", self.on_image, qos_profile_sensor_data)
        self.pub = self.create_publisher(String, "/corn_doctor/result", 10)

        self._last_infer = 0.0        # 上次推理时间（节流用）
        self._prev_sig = None         # 上次打印的结论（结论变化才打印，防刷屏）

    # ---------------- 模型相关 ----------------
    @staticmethod
    def _default_model_path():
        """默认取安装后包内 models/ 目录的模型（colcon build 时已随包安装）。"""
        try:
            from ament_index_python.packages import get_package_share_directory
            d = os.path.join(get_package_share_directory("corn_doctor_ros2"), "models")
            onnx = [f for f in os.listdir(d) if f.endswith(".onnx")] if os.path.isdir(d) else []
            if onnx:
                return os.path.join(d, onnx[0])
        except Exception:
            pass
        return "models/corn_doctor.onnx"

    def _read_class_names(self):
        """类别名优先读 ONNX 元数据（class_0..class_n），保证与训练顺序一致。"""
        try:
            meta = dict(self.sess.get_modelmeta().custom_metadata_map)
            names = [meta[f"class_{i}"] for i in range(len(meta)) if f"class_{i}" in meta]
            if names:
                return names
        except Exception:
            pass
        return list(FALLBACK_CLASSES)

    def preprocess(self, frame_bgr):
        rgb = self._cvtColor(frame_bgr)
        x = rgb.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])   # (1,3,224,224)

    @staticmethod
    def _cvtColor(frame_bgr):
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def green_ratio(frame_bgr):
        """画面中绿色(玉米叶色)像素占比，用于预检画面里是否有叶片。"""
        import cv2
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
        return float(cv2.countNonZero(mask)) / mask.size

    # ---------------- 主回调 ----------------
    def on_image(self, msg: Image):
        # 节流：infer_interval 秒内只识别一帧，其余直接丢弃（省 CPU）
        now = time.time()
        if now - self._last_infer < self.interval:
            return
        self._last_infer = now

        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

        # ---- 叶片预检：太暗/没叶片时提示并跳过（防把墙/手判成病叶）----
        if float(frame.mean()) < self.dark_mean:
            self._report_status("光照不足，请对准玉米叶片")
            return
        gr = self.green_ratio(frame)
        if gr < self.leaf_min_ratio:
            self._report_status(f"未检测到玉米叶片(绿色占比 {gr * 100:.1f}%)，请对准叶片")
            return

        # ---- 推理 ----
        t0 = time.perf_counter()
        probs = self.sess.run(None, {self.input_name: self.preprocess(frame)})[0][0]
        infer_ms = (time.perf_counter() - t0) * 1000

        top = self.class_names[int(probs.argmax())]
        conf = float(probs.max())
        if conf < self.min_conf:
            verdict = "不确定(置信度低)"
        else:
            verdict = "健康" if top == "Healthy" else "生病"

        result = {
            "verdict": verdict,
            "class": top,
            "class_cn": CLASS_CN.get(top, top),
            "confidence": round(conf, 4),
            "infer_ms": round(infer_ms, 1),
            "stamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        out = String()
        out.data = json.dumps(result, ensure_ascii=False)
        self.pub.publish(out)

        # 结论变化才打印
        sig = (verdict, top, round(conf, 2))
        if sig != self._prev_sig:
            self.get_logger().info(
                f"结论: {verdict} | {result['class_cn']} 置信度={conf:.2f} "
                f"| 推理 {infer_ms:.0f} ms")
            self._prev_sig = sig

    def _report_status(self, text):
        sig = ("status", text)
        if sig != self._prev_sig:
            self.get_logger().warn(text)
            self._prev_sig = sig


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
