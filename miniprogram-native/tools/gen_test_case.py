# -*- coding: utf-8 -*-
"""生成 Node 端自测所需的输入数据与 Python 基准结果。

流程与小程序一致：图片 -> 96x96 RGBA ->（JS)buildTensor/forward。
这里用同一份权重在 numpy 上复算，作为对照基准。
"""
import base64
import json
import os

import numpy as np
from PIL import Image

D = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
IMG_SRC = r"C:\Users\ASUS\OneDrive\文档\Trae-CN\temp_20261019_01\AIC-2026-46559581-AI+农业-玉米医生-代码\docs\test_leaf.jpg"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_meta():
    s = open(os.path.join(D, "model_meta.js"), encoding="utf-8").read()
    key = "module.exports = "
    s = s[s.index(key) + len(key):].strip().rstrip(";")
    return json.loads(s)


def load_weights():
    s = open(os.path.join(D, "model_weights_b64.js"), encoding="utf-8").read()
    key = 'module.exports = "'
    s = s[s.index(key) + len(key):]
    s = s[:s.index('"')]
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)


META = load_meta()
W = load_weights()


def conv_relu(x, w, b, stride, pad):
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


def infer_from_rgba(rgba, size):
    rgb = rgba[..., :3].astype(np.float32)
    chw = np.empty((3, size, size), dtype=np.float32)
    for i in range(3):
        chw[i] = rgb[:, :, i] / 127.5 - 1.0

    x = chw
    for L in META["layers"]:
        w = W[L["w_offset"]:L["w_offset"] + L["w_flat"]].reshape(L["w"])
        b = W[L["b_offset"]:L["b_offset"] + L["b_flat"]]
        x = conv_relu(x, w, b, L["stride"], L["pad"])
    gap = x.mean(axis=(1, 2))
    FC = META["fc"]
    fw = W[FC["w_offset"]:FC["w_offset"] + FC["w_flat"]].reshape(FC["w"])
    fb = W[FC["b_offset"]:FC["b_offset"] + FC["b_flat"]]
    logits = fw @ gap + fb
    z = np.exp(logits - logits.max())
    return z / z.sum()


def green_check(rgba):
    """与 inference.js buildTensor 中的 HSV 判定一致。"""
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    d = mx - mn
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    h = np.zeros_like(mx)
    d_safe = np.where(d > 1e-6, d, 1)
    cond = d > 1e-6
    h = np.where(cond & (mx == r), 60 * (((g - b) / d_safe) % 6), h)
    h = np.where(cond & (mx == g) & (mx != r), 60 * ((b - r) / d_safe + 2), h)
    h = np.where(cond & (mx == b) & (mx != r) & (mx != g), 60 * ((r - g) / d_safe + 4), h)
    h = np.where(h < 0, h + 360, h)
    s = np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx))
    mask = (h >= 70) & (h <= 180) & (s >= 60 / 255) & (mx >= 40 / 255)
    return float(mask.mean()), float(mx.mean() * 255)


def main():
    size = META["input"]
    img = Image.open(IMG_SRC).convert("RGB").resize((size, size), Image.BILINEAR)
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.asarray(img)
    rgba[:, :, 3] = 255
    rgba.tofile(os.path.join(OUT_DIR, "test_rgba.bin"))

    probs = infer_from_rgba(rgba, size)
    gr, br = green_check(rgba)
    baseline = {
        "input": size,
        "classes": META["classes"],
        "probs": [round(float(p) * 100, 4) for p in probs],
        "green_ratio": round(gr * 100, 2),
        "brightness": round(br, 1),
    }
    with open(os.path.join(OUT_DIR, "test_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=1)
    print("Python 基准:", json.dumps(baseline, ensure_ascii=False))


if __name__ == "__main__":
    main()
