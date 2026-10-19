# -*- coding: utf-8 -*-
"""最终模型训练：MobileNetV2 迁移学习（ImageNet 预训练）。

运行: python ...-源代码/03_最终模型训练.py

两阶段训练（CPU 友好）：
  阶段1: 冻结特征提取器，只训练分类头 (2 epochs, lr=1e-3)
  阶段2: 解冻全部层，小学习率微调 (8 epochs, lr=1e-4)

输出（统一存放在 ...-输出结果/...-最终模型/ 目录）：
  ...-最终模型最优权重.pth       最优最终模型（含类别信息）
  ...-最终模型训练历史曲线.png   训练损失/MSE/准确率曲线
"""
import importlib
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 文件名以数字开头，需通过 importlib 导入公共模块
_du = importlib.import_module("AIC-2026-46559581-AI+农业-玉米医生-01_数据集加载与公共工具")
CLASSES = _du.CLASSES
FINAL_DIR = _du.FINAL_DIR
PLOT_DIR = _du.PLOT_DIR
BEST_PATH = os.path.join(FINAL_DIR, _du.MATERIAL_PREFIX + "最终模型最优权重.pth")
NUM_CLASSES = _du.NUM_CLASSES
build_datasets = _du.build_datasets
make_loader = _du.make_loader

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = torch.device("cpu")
IMG_SIZE = 224
EARLY_STOP = 2  # 全量微调阶段: 验证准确率连续 N 轮无提升则提前结束
BATCH = 16    # 32 会触发内存耗尽（OOM），降到 16 更稳
WORKERS = 2   # Windows 下每个 worker 是独立进程，4 个太占内存


def build_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m


def evaluate(model, loader):
    model.eval()
    ce_sum, mse_sum, correct, n = 0.0, 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            onehot = F.one_hot(y, NUM_CLASSES).float()
            ce_sum += F.cross_entropy(logits, y, reduction="sum").item()
            mse_sum += torch.sum((probs - onehot) ** 2).item() / NUM_CLASSES
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            all_preds.append(preds)
            all_labels.append(y)
            n += y.size(0)
    f1 = f1_score(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(), average="macro")
    return ce_sum / n, mse_sum / n, correct / n, f1


def train_phase(model, train_ds, val_ds, epochs, lr, freeze_backbone, hist, tag):
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    else:
        for p in model.features.parameters():
            p.requires_grad = True
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    tr_loader = make_loader(train_ds, BATCH, shuffle=True, workers=WORKERS)
    va_loader = make_loader(val_ds, BATCH, shuffle=False, workers=WORKERS)

    best_acc, best_state = hist.get("best_val_acc", 0.0), hist.get("best_state")
    bad = 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        ce_sum, mse_sum, n = 0.0, 0.0, 0
        for x, y in tr_loader:
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                onehot = F.one_hot(y, NUM_CLASSES).float()
                ce_sum += F.cross_entropy(logits, y, reduction="sum").item()
                mse_sum += torch.sum((probs - onehot) ** 2).item() / NUM_CLASSES
                n += y.size(0)
        va_loss, va_mse, va_acc, va_f1 = evaluate(model, va_loader)
        for k, v in zip(("train_loss", "train_mse", "val_loss", "val_mse", "val_acc", "val_f1"),
                        (ce_sum / n, mse_sum / n, va_loss, va_mse, va_acc, va_f1)):
            hist[k].append(v)
        mark = ""
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            # 增量保存：最优即落盘，训练被中断也不丢已得的最优权重
            torch.save({"state_dict": best_state, "classes": CLASSES,
                        "img_size": IMG_SIZE, "arch": "mobilenet_v2",
                        "val_acc": best_acc}, BEST_PATH)
            mark = "  <-- 最优(已保存)"
        print(f"  [{tag}] epoch {ep}/{epochs} train_CE={ce_sum/n:.4f} MSE={mse_sum/n:.4f} | "
              f"val_CE={va_loss:.4f} MSE={va_mse:.4f} acc={va_acc:.4f} f1={va_f1:.4f} ({time.time()-t0:.0f}s){mark}",
              flush=True)
        # 每个 epoch 落盘历史指标，中断后曲线数据也不丢
        with open(os.path.join(FINAL_DIR, _du.MATERIAL_PREFIX + "最终模型训练历史数据.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in hist.items() if k != "best_state"}, f, ensure_ascii=False, indent=2)
        # 早停：全量微调阶段验证准确率连续 EARLY_STOP 轮无提升则提前结束
        if not freeze_backbone:
            if va_acc <= best_acc:
                bad += 1
                if bad >= EARLY_STOP:
                    print(f"  [{tag}] 早停: 验证准确率连续 {EARLY_STOP} 轮无提升, 提前结束微调", flush=True)
                    break
            else:
                bad = 0
    hist["best_val_acc"] = best_acc
    hist["best_state"] = best_state


def main():
    torch.manual_seed(42)
    train_ds, val_ds, test_ds, names = build_datasets(img_size=IMG_SIZE)
    print(f"数据集: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    model = build_model()
    hist = {"train_loss": [], "train_mse": [], "val_loss": [], "val_mse": [],
            "val_acc": [], "val_f1": [], "best_val_acc": 0.0, "best_state": None}

    print("\n=== 阶段1: 冻结主干, 训练分类头 ===", flush=True)
    train_phase(model, train_ds, val_ds, epochs=2, lr=1e-3,
                freeze_backbone=True, hist=hist, tag="冻结微调")
    print("\n=== 阶段2: 全模型微调 ===", flush=True)
    train_phase(model, train_ds, val_ds, epochs=8, lr=1e-4,
                freeze_backbone=False, hist=hist, tag="全量微调")

    torch.save({"state_dict": hist["best_state"], "classes": CLASSES,
                "img_size": IMG_SIZE, "arch": "mobilenet_v2",
                "val_acc": hist["best_val_acc"]}, BEST_PATH)
    print(f"\n最终模型已保存: {BEST_PATH} (val_acc={hist['best_val_acc']*100:.2f}%)", flush=True)

    # 绘制训练历史曲线（2x2 子图：CE损失 / MSE / 准确率 / F1）
    eps = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(eps, hist["train_loss"], "o-", label="训练 CE")
    axes[0, 0].plot(eps, hist["val_loss"], "s-", label="验证 CE")
    axes[0, 0].set_title("交叉熵损失变化"); axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(eps, hist["train_mse"], "o-", label="训练 MSE")
    axes[0, 1].plot(eps, hist["val_mse"], "s-", label="验证 MSE")
    axes[0, 1].set_title("MSE 变化"); axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(eps, [a * 100 for a in hist["val_acc"]], "s-", color="green")
    axes[1, 0].set_title("验证集准确率"); axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("准确率 (%)"); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(eps, [f * 100 for f in hist["val_f1"]], "s-", color="purple")
    axes[1, 1].set_title("验证集 F1 值 (macro)"); axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1 (%)"); axes[1, 1].grid(alpha=0.3)
    fig.suptitle("最终模型 (MobileNetV2 迁移学习) 训练历史曲线")
    fig.tight_layout()
    # 在 结果图表 和 最终模型 各存一份（图表按系列编号 05，最终模型目录内不带编号）
    _curve_name = _du.MATERIAL_PREFIX + "05_最终模型训练历史曲线.png"
    fig.savefig(os.path.join(PLOT_DIR, _curve_name), dpi=150)
    fig.savefig(os.path.join(FINAL_DIR, _du.MATERIAL_PREFIX + "最终模型训练历史曲线.png"), dpi=150)
    print(f"训练曲线已保存:\n  {os.path.join(PLOT_DIR, _curve_name)}\n  {os.path.join(FINAL_DIR, _du.MATERIAL_PREFIX + '最终模型训练历史曲线.png')}", flush=True)


if __name__ == "__main__":
    main()
