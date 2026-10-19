# -*- coding: utf-8 -*-
"""前端服务：Flask + 线程池异步推理。

上传玉米叶片图片 -> 自动识别有无病虫害及病害种类。

启动: python ...-源代码/05_网页识别应用.py  然后浏览器打开 http://127.0.0.1:5000
依赖: ...-输出结果/...-最终模型/...-最终模型最优权重.pth（先运行 03_最终模型训练.py 生成）
"""
import importlib
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 文件名以数字开头，需通过 importlib 导入公共模块
_du = importlib.import_module("AIC-2026-46559581-AI+农业-玉米医生-01_数据集加载与公共工具")
CLASS_CN = _du.CLASS_CN
FINAL_DIR = _du.FINAL_DIR
MEAN = _du.MEAN
STD = _du.STD

DEVICE = torch.device("cpu")
MODEL_PATH = os.path.join(FINAL_DIR, _du.MATERIAL_PREFIX + "最终模型最优权重.pth")
CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ...-网页模板/ 位于项目根目录（...-源代码 的上一级）
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", _du.MATERIAL_PREFIX + "网页模板")
app = Flask(__name__, template_folder=os.path.abspath(TEMPLATE_DIR))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB 上限
executor = ThreadPoolExecutor(max_workers=4)         # 线程池：并发/异步推理


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"[错误] 未找到最终模型: {MODEL_PATH}\n"
            f"请先运行: python ...-源代码/03_最终模型训练.py 生成最终模型后再启动本服务。")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes", CLASSES)
    if ckpt.get("arch") == "mobilenet_v2":
        m = models.mobilenet_v2()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(classes))
    else:
        _te = importlib.import_module(_du.MATERIAL_PREFIX + "02_对比实验训练")
        cfg = ckpt["cfg"]
        m = _te.ConfigurableCNN(cfg["blocks"], cfg["dropout"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, classes


model, CLASS_NAMES = load_model()
model = model.to(DEVICE)
with torch.no_grad():  # 预热，加快首次请求
    model(torch.zeros(1, 3, 224, 224))
print("模型加载完成，可以开始识别。")


def predict_bytes(img_bytes: bytes):
    """在后台线程中执行推理。"""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = PREPROCESS(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    probs = probs.tolist()
    ranked = sorted(zip(CLASS_NAMES, probs), key=lambda t: t[1], reverse=True)
    top = ranked[0]
    healthy_prob = dict(ranked).get("Healthy", 0.0)
    return {
        "class": top[0],
        "class_cn": CLASS_CN.get(top[0], top[0]),
        "confidence": round(top[1] * 100, 2),
        "has_disease": top[0] != "Healthy",
        "healthy_prob": round(healthy_prob * 100, 2),
        "probs": [{"class": c, "class_cn": CLASS_CN.get(c, c),
                   "prob": round(p * 100, 2)} for c, p in ranked],
    }


@app.route("/")
def index():
    return render_template(_du.MATERIAL_PREFIX + "识别页面.html")


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "请先选择一张图片"}), 400
    img_bytes = f.read()
    # 提交到线程池异步执行，避免阻塞 Flask 主线程，可并发处理多个上传
    future = executor.submit(predict_bytes, img_bytes)
    try:
        result = future.result(timeout=60)
    except Exception as e:
        return jsonify({"error": f"识别失败: {e}"}), 500
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
