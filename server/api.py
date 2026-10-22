# -*- coding: utf-8 -*-
"""玉米医生 · 小程序识别后端服务（ONNX Runtime，无需 PyTorch）。

功能：接收小程序上传的玉米叶片图片，用 MobileNetV2 ONNX 模型推理，
返回病害类别、置信度、是否生病及各类别概率。

启动: python server/api.py
接口: POST /predict  (multipart/form-data, 字段 file=图片文件)
       GET  /health
依赖: flask, pillow, numpy, onnxruntime
"""
import io
import logging
import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

# ---------- 日志（后台自启运行时输出到 server/backend.log） ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../server
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "backend.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("corn-doctor")

# ---------- 路径与常量 ----------
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # 项目根

# ONNX 模型（与 PC 端 06 脚本导出的一致）
_MODEL_SUBDIR = os.path.join(
    "AIC-2026-46559581-AI+农业-玉米医生-PC端训练与识别代码",
    "AIC-2026-46559581-AI+农业-玉米医生-输出结果",
    "AIC-2026-46559581-AI+农业-玉米医生-最终模型",
)
ONNX_NAME = "AIC-2026-46559581-AI+农业-玉米医生-最终模型最优.onnx"
MODEL_PATH = os.path.join(PROJECT_ROOT, _MODEL_SUBDIR, ONNX_NAME)

# 类别（顺序与训练端一致，ONNX 元数据 class_0..3 为 ["Blight","Common_Rust","Gray_Leaf_Spot","Healthy"]）
CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
CLASS_CN = {
    "Blight": "叶斑病",
    "Common_Rust": "普通锈病",
    "Gray_Leaf_Spot": "灰斑病",
    "Healthy": "健康",
}
ADVICE = {
    "Blight": "玉米叶斑病：及时摘除下部重病叶并带出田外销毁；发病初期可用苯醚甲环唑、吡唑醚菌酯等药剂喷雾防治，间隔7~10天连喷2次；合理密植、增施钾肥提高抗性。",
    "Common_Rust": "玉米普通锈病：选用抗病品种；发病初期喷施三唑酮、戊唑醇或苯醚甲环唑，间隔7~10天连喷2~3次；注意田间排水、降低湿度。",
    "Gray_Leaf_Spot": "玉米灰斑病：收获后清除病残体减少越冬菌源；发病初期用嘧菌酯、吡唑醚菌酯或苯醚甲环唑喷雾，重点保护果穗上下叶片。",
    "Healthy": "玉米长势健康，请继续保持良好水肥管理，定期巡查即可。",
}
# ImageNet 归一化（与训练端严格一致）
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224

# ---------- 场景预检（绿色叶色检测，与 07_摄像头实时检测.py 的 green_ratio 一致） ----------
DARK_MEAN = 45.0        # 画面平均亮度低于该值 -> 光照不足，不做病害分析
LEAF_MIN_RATIO = 0.02   # 绿色(叶色)像素占比低于该值 -> 视为未对准叶片
# OpenCV 绿色域 H∈[35,90]（H 为角度/2）、S≥60、V≥40；
# PIL 的 HSV 模式 H 量程为 0~255（角度*255/360），故换算为 [50,128]
LEAF_H_MIN, LEAF_H_MAX = 50, 128
LEAF_S_MIN, LEAF_V_MIN = 60, 40


def analyze_leaf(img_rgb):
    """判断画面中是否为玉米绿叶（HSV 绿色占比 + 亮度）。

    返回 dict: is_leaf / green_ratio(%) / brightness / message
    """
    hsv = np.asarray(img_rgb.convert("HSV"), dtype=np.int16)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= LEAF_H_MIN) & (h <= LEAF_H_MAX) & (s >= LEAF_S_MIN) & (v >= LEAF_V_MIN)
    green = float(mask.sum()) / mask.size
    brightness = float(v.mean())

    if brightness < DARK_MEAN:
        return {"is_leaf": False, "green_ratio": round(green * 100, 2),
                "brightness": round(brightness, 1),
                "message": "光照不足，请对准玉米叶片并保证光线充足"}
    if green < LEAF_MIN_RATIO:
        return {"is_leaf": False, "green_ratio": round(green * 100, 2),
                "brightness": round(brightness, 1),
                "message": "未检测到玉米叶片（绿色占比 %.1f%%），请对准叶片后重拍" % (green * 100)}
    return {"is_leaf": True, "green_ratio": round(green * 100, 2),
            "brightness": round(brightness, 1),
            "message": "已检测到玉米叶片"}

# ---------- Flask 应用 ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB


def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.after_request
def add_cors(resp):
    return _cors(resp)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": os.path.exists(MODEL_PATH)})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "请先选择一张图片"}), 400

    try:
        # 1) 读取并预处理
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        # 1.5) 场景预检：先判断画面是否为玉米绿叶，不是则不做病害分析
        leaf = analyze_leaf(img)
        if not leaf["is_leaf"]:
            return jsonify({"leaf": leaf, "error": leaf["message"]})

        x = np.asarray(img, dtype=np.float32) / 255.0          # (224,224,3) 0~1
        x = (x - MEAN) / STD                                    # 归一化
        x = x.transpose(2, 0, 1)[None, ...]                    # (1,3,224,224) NCHW
        x = np.ascontiguousarray(x, dtype=np.float32)

        # 2) ONNX 推理（输出已是 softmax 概率）
        outputs = sess.run(None, {"images": x})[0][0]          # (4,)
        probs = outputs.astype(float).tolist()

        ranked = sorted(zip(CLASSES, probs), key=lambda t: t[1], reverse=True)
        top = ranked[0]
        return jsonify({
            "class": top[0],
            "class_cn": CLASS_CN[top[0]],
            "confidence": round(top[1] * 100, 2),
            "has_disease": top[0] != "Healthy",
            "advice": ADVICE[top[0]],
            "leaf": leaf,
            "probs": [{"class": c, "class_cn": CLASS_CN[c], "prob": round(p * 100, 2)}
                      for c, p in ranked],
        })
    except Exception as e:
        logger.exception("识别失败")
        return jsonify({"error": f"识别失败: {e}"}), 500


# ---------- 加载模型 ----------
def _load_session():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"[错误] 未找到 ONNX 模型: {MODEL_PATH}\n"
            "请先运行 PC端训练与识别代码/...-源代码/06_ONNX模型导出.py 生成模型。")
    import onnxruntime as ort
    s = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    # 预热
    s.run(None, {"images": np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)})
    logger.info("模型加载完成: %s", MODEL_PATH)
    return s


sess = _load_session()


if __name__ == "__main__":
    # 监听 0.0.0.0，同一局域网内的手机可直接访问
    logger.info("识别服务启动: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
