# -*- coding: utf-8 -*-
"""验证导出的扁平权重与 meta 是否可直接用于推理（抽样评估准确率）。

小程序 utils/inference.js 用的就是同一份 model_weights.bin + model_meta.json，
这里用等价的向量化卷积复现，确认权重切片偏移、通道顺序、预处理都正确。
"""
import json
import os
import random

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS = os.path.abspath(os.path.join(HERE, "..", "utils"))
DATA_ROOT = r"E:\作业\AIC-2026-46559581-AI+农业-玉米医生-代码\AIC-2026-46559581-AI+农业-玉米医生-玉米病害数据集"
SAMPLE = 240

with open(os.path.join(UTILS, "model_meta.json"), encoding="utf-8") as f:
    META = json.load(f)
W = np.fromfile(os.path.join(UTILS, "model_weights.bin"), dtype=np.float32)
assert W.size == META["totalFloats"], f"权重长度不符 {W.size} vs {META['totalFloats']}"

SIZE = META["input"]
CLASSES_SRC = META["classesSrc"]
print(f"权重 {W.size} 个 float32；输入 {SIZE}x{SIZE}；类别 {META['classes']}")


def conv_relu(x, w, b, stride, pad):
    """x: (C,H,W) -> (OC,OH,OW)，等价于 inference.js 的 convRelu3x3。"""
    c, h, wd = x.shape
    oc, _, kh, kw = w.shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (wd + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad))).astype(np.float32)
    out = np.zeros((oc, oh, ow), dtype=np.float32)
    for ky in range(kh):
        for kx in range(kw):
            patch = xp[:, ky:ky + stride * (oh - 1) + 1:stride, kx:kx + stride * (ow - 1) + 1:stride]
            out += np.tensordot(w[:, :, ky, kx], patch, axes=([1], [0]))
    out += b[:, None, None]
    return np.maximum(out, 0)


def infer(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    x = (arr / 127.5 - 1.0).transpose(2, 0, 1)      # 与 JS 的 chw 一致

    for L in META["layers"]:
        wc = L["w_flat"]
        w = W[L["w_offset"]:L["w_offset"] + wc].reshape(L["w"])
        b = W[L["b_offset"]:L["b_offset"] + L["b_flat"]]
        x = conv_relu(x, w, b, L["stride"], L["pad"])

    gap = x.mean(axis=(1, 2))
    FC = META["fc"]
    fw = W[FC["w_offset"]:FC["w_offset"] + FC["w_flat"]].reshape(FC["w"])
    fb = W[FC["b_offset"]:FC["b_offset"] + FC["b_flat"]]
    logits = fw @ gap + fb
    z = np.exp(logits - logits.max())
    return z / z.sum()


# 抽样评估
d = os.path.join(DATA_ROOT, "test", "images")
files = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
by_cls = {}
for f in files:
    name = f.rsplit(".", 1)[0].rsplit("_", 1)[0]
    if name in CLASSES_SRC:
        by_cls.setdefault(name, []).append(os.path.join(d, f))

random.seed(0)
picks = []
for name, lst in by_cls.items():
    random.shuffle(lst)
    picks += [(p, CLASSES_SRC.index(name)) for p in lst[:SAMPLE // len(CLASSES_SRC)]]

correct, conf_sum = 0, 0.0
per_cls_hit, per_cls_tot = {}, {}
for path, y in picks:
    p = infer(path)
    pred = int(p.argmax())
    correct += int(pred == y)
    conf_sum += float(p[pred])
    key = CLASSES_SRC[y]
    per_cls_tot[key] = per_cls_tot.get(key, 0) + 1
    per_cls_hit[key] = per_cls_hit.get(key, 0) + int(pred == y)

n = len(picks)
print(f"\n抽样 {n} 张（每类 {SAMPLE // len(CLASSES_SRC)} 张）")
print(f"准确率: {correct / n * 100:.2f}%   平均置信度: {conf_sum / n * 100:.2f}%")
for k in CLASSES_SRC:
    print(f"  {k:16s} {per_cls_hit.get(k,0)}/{per_cls_tot.get(k,0)}")
print(f"\n训练记录的最高测试准确率: {META['testAcc']}%")
