# -*- coding: utf-8 -*-
"""玉米医生 - 拍照识别后端 (Flask + ONNX Runtime)。

复用树莓派 corn_detect 包同款 best.onnx 模型（MobileNetV2，4 分类），
与 detect_node 的预处理完全一致：Resize(224,224) -> /255 -> ImageNet 归一化 -> RGB。

启动:
  python server.py            # 默认 0.0.0.0:8000，供网页端与微信小程序共用
  # 微信小程序要求 HTTPS，正式部署时用 nginx 反代或 cloudflared 内网穿透
"""
import io
import os

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型查找：优先本目录 model/best.onnx；否则用仓库 docs/models/best.onnx（同内容，避免重复提交 9MB 文件）
_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "model", "best.onnx"),
    os.path.join(BASE_DIR, "..", "..", "docs", "models", "best.onnx"),
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)), _MODEL_CANDIDATES[0])

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 场景预检（与 07_摄像头实时检测.py 的 green_ratio 一致）
DARK_MEAN = 45.0        # 平均亮度低于该值 -> 光照不足
LEAF_MIN_RATIO = 0.02   # 绿色(叶色)像素占比低于该值 -> 未对准叶片
# OpenCV 绿色域 H∈[35,90]、S≥60、V≥40；PIL HSV 的 H 量程 0~255 -> [50,128]
LEAF_H_MIN, LEAF_H_MAX = 50, 128
LEAF_S_MIN, LEAF_V_MIN = 60, 40

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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name
meta = dict(sess.get_modelmeta().custom_metadata_map)
CLASS_NAMES = [meta[f"class_{i}"] for i in range(4) if f"class_{i}" in meta]
print("模型加载完成:", INPUT_NAME, CLASS_NAMES)


def analyze_leaf(img_rgb) -> dict:
    """判断画面是否为玉米绿叶（HSV 绿色占比 + 亮度），不是则不做病害判定。"""
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
            "brightness": round(brightness, 1), "message": "已检测到玉米叶片"}


def predict_bytes(img_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)

    leaf = analyze_leaf(img)
    if not leaf["is_leaf"]:
        return {"leaf": leaf, "error": leaf["message"]}

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])  # (1,3,224,224)
    probs = sess.run(None, {INPUT_NAME: x})[0][0]
    ranked = sorted(zip(CLASS_NAMES, probs), key=lambda t: t[1], reverse=True)
    top_cls, top_p = ranked[0]
    return {
        "class": top_cls,
        "class_cn": CLASS_CN.get(top_cls, top_cls),
        "confidence": round(float(top_p) * 100, 2),
        "has_disease": top_cls != "Healthy",
        "advice": ADVICE.get(top_cls, ""),
        "leaf": leaf,
        "probs": [
            {"class": c, "class_cn": CLASS_CN.get(c, c), "prob": round(float(p) * 100, 2)}
            for c, p in ranked
        ],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "未收到图片，请先拍照或选择图片"}), 400
    try:
        return jsonify(predict_bytes(f.read()))
    except Exception as e:
        return jsonify({"error": f"识别失败: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))   # 云端发布时由平台注入 PORT
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
