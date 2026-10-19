# -*- coding: utf-8 -*-
"""ONNX 模型导出：将最终模型最优权重(.pth)导出为 ONNX 格式。

用途:
  1. PC 端 USB 摄像头实时检测（07_摄像头实时检测.py，无需安装 PyTorch）
  2. 树莓派边缘部署：把导出的 .onnx 拷贝到树莓派的 模型/ 文件夹，
     用 ONNX Runtime 加载（树莓派不需要装 PyTorch）

运行: python ...-源代码/06_ONNX模型导出.py
输入: ...-输出结果/...-最终模型/...-最终模型最优权重.pth（先运行 03_最终模型训练.py 生成）
输出: ...-输出结果/...-最终模型/...-最终模型最优.onnx
      （网络内部已含 softmax，输出直接是 4 类概率；类别名写入 ONNX 元数据）

导出后自动用 onnxruntime 做一致性校验：随机输入下 PyTorch 与 ONNX
输出的最大绝对误差应 < 1e-4。
"""
import importlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 文件名以数字开头，需通过 importlib 导入公共模块
_du = importlib.import_module("AIC-2026-46559581-AI+农业-玉米医生-01_数据集加载与公共工具")
FINAL_DIR = _du.FINAL_DIR
MATERIAL_PREFIX = _du.MATERIAL_PREFIX
CLASSES = _du.CLASSES

PTH_PATH = os.path.join(FINAL_DIR, MATERIAL_PREFIX + "最终模型最优权重.pth")
ONNX_PATH = os.path.join(FINAL_DIR, MATERIAL_PREFIX + "最终模型最优.onnx")


class ProbModel(nn.Module):
    """MobileNetV2 + softmax：ONNX 输出直接为各类别概率（和为1）。"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


def build_model():
    """按 05_网页识别应用.py 相同方式从 checkpoint 重建模型并加载权重。"""
    if not os.path.exists(PTH_PATH):
        raise SystemExit(
            f"[错误] 未找到最终模型: {PTH_PATH}\n"
            f"请先运行: python ...-源代码/03_最终模型训练.py 生成最终模型。")
    ckpt = torch.load(PTH_PATH, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes", CLASSES)
    assert ckpt.get("arch") == "mobilenet_v2", f"不支持的模型结构: {ckpt.get('arch')}"
    m = models.mobilenet_v2()
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(classes))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    wrapper = ProbModel(m)
    wrapper.eval()
    return wrapper, classes


def main():
    model, classes = build_model()
    dummy = torch.zeros(1, 3, 224, 224)

    print(f"[1/3] 导出 ONNX: {ONNX_PATH}")
    # 动态 batch：滑窗定位会批量送入多个窗口图块（07_摄像头实时检测.py 需要）
    dyn = {"images": {0: "batch"}, "probs": {0: "batch"}}
    try:
        # torch>=2.9 默认使用新的 torch.export 导出器（精度更好）
        torch.onnx.export(
            model, dummy, ONNX_PATH,
            input_names=["images"], output_names=["probs"],
            opset_version=18, dynamic_axes=dyn)
    except TypeError:                       # 兼容不支持新导出参数的旧版 torch
        torch.onnx.export(
            model, dummy, ONNX_PATH,
            input_names=["images"], output_names=["probs"],
            opset_version=17, dynamo=False, dynamic_axes=dyn)

    print(f"[2/3] 写入类别名到 ONNX 元数据: {classes}")
    import onnx
    mo = onnx.load(ONNX_PATH)
    for i, name in enumerate(classes):
        kv = mo.metadata_props.add()
        kv.key, kv.value = f"class_{i}", name
    onnx.save(mo, ONNX_PATH)

    print("[3/3] onnxruntime 一致性校验...")
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    max_err = 0.0
    onnx_out = None
    for b in (1, 9):        # batch=1 与滑窗批量 batch=9 两种尺寸都校验
        x = np.random.rand(b, 3, 224, 224).astype(np.float32)
        onnx_out = sess.run(None, {"images": x})[0]
        with torch.no_grad():
            pt_out = model(torch.from_numpy(x)).numpy()
        max_err = max(max_err, float(np.abs(onnx_out - pt_out).max()))
    print(f"  输入 (batch,3,224,224) 动态batch -> 输出 {onnx_out.shape}")
    print(f"  PyTorch vs ONNX 最大绝对误差: {max_err:.2e}")
    assert max_err < 1e-4, "导出结果与原模型不一致，请检查!"

    size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
    print(f"\n导出完成: {ONNX_PATH}  ({size_mb:.1f} MB)")
    print("后续步骤:")
    print("  1) PC 摄像头实时检测: python ...-源代码/07_摄像头实时检测.py")
    print("  2) 树莓派部署: 将本 onnx 拷贝到树莓派 模型/ 文件夹")


if __name__ == "__main__":
    main()
