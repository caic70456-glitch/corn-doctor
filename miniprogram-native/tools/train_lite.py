# -*- coding: utf-8 -*-
"""训练小程序端轻量 CNN（纯 JS 可推理，无需后端/插件）。

为什么单独训练：
  原 MobileNetV2(224x224, 9MB) 依赖 onnxruntime，微信小程序无法直接加载。
  这里训练一个 4 层卷积的极小网络（约 60K 参数 / 240KB），导出为扁平
  float32 权重 + 元数据，小程序用几十行 JS 即可完成前向计算，完全离线。

结构: Conv3x3/s2 通道 3->16->32->64->64 + BN + ReLU -> GAP -> FC(64->4)
输入: 96x96 RGB，预处理 x/255 -> (x-0.5)/0.5
输出: utils/model_weights.bin（层权重依次拼接）、utils/model_meta.json

运行: python tools/train_lite.py
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DATA_ROOT = r"E:\作业\AIC-2026-46559581-AI+农业-玉米医生-代码\AIC-2026-46559581-AI+农业-玉米医生-玉米病害数据集"
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

# 顺序必须与后端/网页端一致
CLASSES = ["corn_blight", "common_rust", "gray_leaf_spot", "healthy"]
CLASS_LABELS = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
IMG = 96
EPOCHS = 14
BATCH = 64
LR = 3e-3


class CornDataset(Dataset):
    """文件名形如 corn_blight_0001.jpg，前缀即类别。"""

    def __init__(self, split, tf):
        self.items, self.tf = [], tf
        d = os.path.join(DATA_ROOT, split, "images")
        for fn in os.listdir(d):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            name = fn.rsplit(".", 1)[0].rsplit("_", 1)[0]
            if name in CLASSES:
                self.items.append((os.path.join(d, fn), CLASSES.index(name)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), y


class LiteNet(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o, s):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(blk(3, 16, 2), blk(16, 32, 2), blk(32, 64, 2), blk(64, 64, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, len(CLASSES))

    def forward(self, x):
        x = self.features(x)
        return self.fc(self.pool(x).flatten(1))


def fuse_bn(conv, bn):
    """BN 融合进卷积：推理时只需 y = x*W' + b'。"""
    w = conv.weight.detach().clone()
    gamma, beta = bn.weight.detach(), bn.bias.detach()
    mean, var, eps = bn.running_mean, bn.running_var, bn.eps
    std = torch.sqrt(var + eps)
    scale = gamma / std
    w = w * scale.view(-1, 1, 1, 1)
    b = beta - mean * scale
    return w.numpy().astype(np.float32), b.numpy().astype(np.float32)


def main():
    train_tf = T.Compose([
        T.RandomResizedCrop(IMG, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_tf = T.Compose([
        T.Resize((IMG, IMG)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    tr = CornDataset("train", train_tf)
    te = CornDataset("test", test_tf)
    print(f"训练集 {len(tr)} 张，测试集 {len(te)} 张，类别 {CLASSES}")

    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(te, batch_size=128, shuffle=False, num_workers=0)

    model = LiteNet()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot = 0.0
        for x, y in tl:
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        sched.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in vl:
                correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / len(te)
        best = max(best, acc)
        print(f"epoch {ep:2d}/{EPOCHS}  loss {tot/len(tr):.4f}  test_acc {acc*100:.2f}%")

    # ---------- 导出（BN 融合 -> 扁平 float32） ----------
    model.eval()
    blobs, meta_layers = [], []
    for i, seq in enumerate(model.features):
        conv, bn = seq[0], seq[1]
        w, b = fuse_bn(conv, bn)
        meta_layers.append({"w": list(w.shape), "b": list(b.shape),
                            "stride": conv.stride[0], "pad": conv.padding[0]})
        blobs.append(w.ravel())
        blobs.append(b)

    fw = model.fc.weight.detach().numpy().astype(np.float32)
    fb = model.fc.bias.detach().numpy().astype(np.float32)
    blobs.append(fw.ravel())
    blobs.append(fb)
    meta_layers.append({"w": list(fw.shape), "b": list(fb.shape)})

    os.makedirs(OUT_DIR, exist_ok=True)
    flat = np.concatenate(blobs)
    flat.tofile(os.path.join(OUT_DIR, "model_weights.bin"))

    # 记录每层在 bin 中的偏移，小程序按偏移切片即可
    offset = 0
    for L in meta_layers:
        wc = int(np.prod(L["w"]))
        bc = int(np.prod(L["b"]))
        L["w_flat"], L["b_flat"] = wc, bc
        L["w_offset"], L["b_offset"] = offset, offset + wc
        offset += wc + bc

    fc_meta = meta_layers[-1]
    meta = {
        "input": IMG,
        "classes": CLASS_LABELS,
        "classesSrc": CLASSES,
        "layers": meta_layers[:-1],
        "fc": fc_meta,
        "totalFloats": int(flat.size),
        "testAcc": round(best * 100, 2),
    }
    with open(os.path.join(OUT_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    kb = os.path.getsize(os.path.join(OUT_DIR, "model_weights.bin")) / 1024
    print(f"\n导出完成: utils/model_weights.bin ({kb:.1f} KB, {flat.size} 个 float32)")
    print(f"最高测试准确率: {best*100:.2f}%")

    # ---------- 用 numpy 复现 JS 推理，确认导出权重正确 ----------
    def conv_np(x, w, b, stride, pad):
        n, c, h, wd = x.shape
        oc = w.shape[0]
        oh = (h + 2 * pad - 3) // stride + 1
        ow = (wd + 2 * pad - 3) // stride + 1
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        out = np.zeros((n, oc, oh, ow), dtype=np.float32)
        for oy in range(oh):
            for ox in range(ow):
                patch = xp[:, :, oy * stride:oy * stride + 3, ox * stride:ox * stride + 3]
                out[:, :, oy, ox] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3])) + b
        return out

    x, y = next(iter(vl))
    x = x.numpy()
    feat = x
    for k in range(len(model.features)):
        L = meta["layers"][k]
        wc = int(np.prod(L["w"]))
        wmat = flat[L["w_offset"]:L["w_offset"] + wc].reshape(L["w"])
        bvec = flat[L["b_offset"]:L["b_offset"] + int(np.prod(L["b"]))]
        feat = np.maximum(conv_np(feat, wmat, bvec, L["stride"], L["pad"]), 0)
    gap = feat.mean(axis=(2, 3))
    logits = gap @ fw.T + fb
    js_pred = logits.argmax(1)

    with torch.no_grad():
        pt_pred = model(x).argmax(1).numpy()
    agree = float((js_pred == pt_pred).mean())

    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    print(f"numpy 复现 vs PyTorch 预测一致率: {agree*100:.1f}%")
    print(f"示例样本概率: {np.round(p[0]*100, 2).tolist()}  (类别顺序 {CLASS_LABELS})")
    print(f"真实标签: {CLASS_LABELS[y[0]]}")


if __name__ == "__main__":
    main()
