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
import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

# ---------- 路径与常量 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../server
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
# ImageNet 归一化（与训练端严格一致）
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224

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
            "probs": [{"class": c, "class_cn": CLASS_CN[c], "prob": round(p * 100, 2)}
                      for c, p in ranked],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    print(f"[模型加载完成] {MODEL_PATH}")
    return s


sess = _load_session()


if __name__ == "__main__":
    # 监听 0.0.0.0，同一局域网内的手机可直接访问
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
