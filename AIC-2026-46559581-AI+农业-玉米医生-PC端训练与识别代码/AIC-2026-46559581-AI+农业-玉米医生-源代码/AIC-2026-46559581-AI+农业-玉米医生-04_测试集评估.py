# -*- coding: utf-8 -*-
"""测试集评估（多进程/多线程并行推理）：混淆矩阵、各类别准确率、分类报告。

运行: python ...-源代码/04_测试集评估.py [模型路径]
默认评估 ...-输出结果/...-最终模型/...-最终模型最优权重.pth

输出：
  ...-结果图表/06_混淆矩阵_测试集.png
  ...-结果图表/07_归一化混淆矩阵_测试集.png
  ...-结果图表/08_各类别准确率_测试集.png
  ...-测试评估报告.json
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
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 文件名以数字开头，需通过 importlib 导入公共模块
_du = importlib.import_module("AIC-2026-46559581-AI+农业-玉米医生-01_数据集加载与公共工具")
CLASSES = _du.CLASSES
CLASS_CN = _du.CLASS_CN
OUT_DIR = _du.OUT_DIR
FINAL_DIR = _du.FINAL_DIR
PLOT_DIR = _du.PLOT_DIR
build_datasets = _du.build_datasets
make_loader = _du.make_loader

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BATCH = 64
WORKERS = 6  # 多进程并行读取与预处理图像


def safe_save(fig, path, attempts=3):
    """保存图片；若目标文件被占用（如正被图片查看器打开），重试，最后改存 _new.png。"""
    for i in range(attempts):
        try:
            fig.savefig(path, dpi=150)
            return
        except OSError:
            if i < attempts - 1:
                time.sleep(1.5)
    alt = path.replace(".png", "_new.png")
    fig.savefig(alt, dpi=150)
    print(f"警告: {path} 被其他程序占用，已改存为 {alt}", flush=True)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("arch") == "mobilenet_v2":
        m = models.mobilenet_v2()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    else:  # 实验用自建 CNN
        _te = importlib.import_module(_du.MATERIAL_PREFIX + "02_对比实验训练")
        cfg = ckpt["cfg"]
        m = _te.ConfigurableCNN(cfg["blocks"], cfg["dropout"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FINAL_DIR, _du.MATERIAL_PREFIX + "最终模型最优权重.pth")
    print(f"评估模型: {model_path}", flush=True)
    model = load_model(model_path)

    _, _, test_ds, names = build_datasets(img_size=224)
    # 多进程 DataLoader = 多线程并行读取/解码/增强图像，加速推理
    loader = make_loader(test_ds, BATCH, shuffle=False, workers=WORKERS)
    print(f"测试集: {len(test_ds)} 张图片, {WORKERS} 个并行工作进程推理中...", flush=True)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x), dim=1)
            all_probs.append(probs)
            all_preds.append(probs.argmax(1))
            all_labels.append(y)
    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc = float((preds == labels).mean())
    print(f"\n测试集总体准确率: {acc*100:.2f}%  ({(preds==labels).sum()}/{len(labels)})", flush=True)

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=CLASSES, digits=4)
    print("\n" + report, flush=True)

    # 06 混淆矩阵（数量）
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), [CLASS_CN[c] for c in CLASSES], fontsize=11)
    ax.set_yticks(range(len(CLASSES)), [CLASS_CN[c] for c in CLASSES], fontsize=11)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title(f"测试集混淆矩阵（张数）\n总体准确率 {acc*100:.2f}%")
    fig.colorbar(im); fig.tight_layout()
    safe_save(fig, os.path.join(PLOT_DIR, _du.MATERIAL_PREFIX + "06_混淆矩阵_测试集.png"))
    plt.close(fig)

    # 07 归一化混淆矩阵
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), [CLASS_CN[c] for c in CLASSES], fontsize=11)
    ax.set_yticks(range(len(CLASSES)), [CLASS_CN[c] for c in CLASSES], fontsize=11)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i,j]*100:.1f}%", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=11)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("测试集归一化混淆矩阵（每类召回率 %）")
    fig.colorbar(im); fig.tight_layout()
    safe_save(fig, os.path.join(PLOT_DIR, _du.MATERIAL_PREFIX + "07_归一化混淆矩阵_测试集.png"))
    plt.close(fig)

    # 08 各类别准确率
    per_acc = [cm[i, i] / cm[i].sum() * 100 for i in range(len(CLASSES))]
    support = cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(range(len(CLASSES)), per_acc,
                  color=plt.cm.viridis(np.linspace(0.2, 0.8, len(CLASSES))))
    ax.axhline(acc * 100, color="red", ls="--", label=f"总体 {acc*100:.2f}%")
    for b, a, s in zip(bars, per_acc, support):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.5, f"{a:.1f}%\n({s}张)", ha="center", fontsize=10)
    ax.set_xticks(range(len(CLASSES)), [CLASS_CN[c] for c in CLASSES], fontsize=10)
    ax.set_ylabel("准确率 (%)"); ax.set_ylim(0, 105)
    ax.set_title("测试集各类别识别准确率")
    ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    safe_save(fig, os.path.join(PLOT_DIR, _du.MATERIAL_PREFIX + "08_各类别准确率_测试集.png"))
    plt.close(fig)

    out = {"model_path": model_path, "test_accuracy": acc,
           "per_class_accuracy": dict(zip(CLASSES, per_acc)),
           "classification_report": report,
           "confusion_matrix": cm.tolist(), "test_size": len(labels)}
    with open(os.path.join(OUT_DIR, "测试评估报告.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n图表已保存到: {PLOT_DIR}", flush=True)


if __name__ == "__main__":
    main()
