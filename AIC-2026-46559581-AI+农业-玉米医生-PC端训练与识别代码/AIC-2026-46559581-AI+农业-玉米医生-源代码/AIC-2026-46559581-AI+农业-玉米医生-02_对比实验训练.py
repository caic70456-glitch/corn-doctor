# -*- coding: utf-8 -*-
"""实验脚本：用不同的网络层数 / Dropout / L2 正则化训练自建 CNN，记录 MSE 变化并绘图对比。

运行: python ...-源代码/02_对比实验训练.py

输出：
  ...-输出结果/...-实验过程模型/01~05_各实验最优模型.pth   每组实验的最优模型
  ...-输出结果/...-结果图表/01~04_各对比图.png
  ...-输出结果/...-实验结果数据.json        全部实验数据
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 文件名以数字开头，需通过 importlib 导入公共模块
_du = importlib.import_module("AIC-2026-46559581-AI+农业-玉米医生-01_数据集加载与公共工具")
MODEL_DIR = _du.MODEL_DIR
PLOT_DIR = _du.PLOT_DIR
RESULTS_JSON = _du.RESULTS_JSON
NUM_CLASSES = _du.NUM_CLASSES
build_datasets = _du.build_datasets
compute_mse = _du.compute_mse
make_loader = _du.make_loader

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE = torch.device("cpu")
IMG_SIZE = 128
BATCH = 32
EPOCHS = 10
LR = 1e-3
PATIENCE = 4

# ---------------- 实验配置：层数 / Dropout / L2 正则(权重衰减) ----------------
EXPERIMENTS = [
    {"name": "A_3层_基准",            "blocks": 3, "dropout": 0.2, "weight_decay": 0.0},
    {"name": "B_3层_高随机失活0.5",    "blocks": 3, "dropout": 0.5, "weight_decay": 0.0},
    {"name": "C_3层_L2正则1e-4",      "blocks": 3, "dropout": 0.2, "weight_decay": 1e-4},
    {"name": "D_5层_深网络",          "blocks": 5, "dropout": 0.2, "weight_decay": 0.0},
    {"name": "E_5层_强正则_失活0.5",  "blocks": 5, "dropout": 0.5, "weight_decay": 1e-4},
]

# 各实验最优模型文件名（统一命名格式：前缀 + 序号_材料名称，与交付文件一一对应）
EXP_MODEL_FILES = {
    "A_3层_基准":           "01_实验A_3层基准最优模型.pth",
    "B_3层_高随机失活0.5":   "02_实验B_3层高随机失活0.5最优模型.pth",
    "C_3层_L2正则1e-4":     "03_实验C_3层L2正则1e-4最优模型.pth",
    "D_5层_深网络":         "04_实验D_5层深网络最优模型.pth",
    "E_5层_强正则_失活0.5": "05_实验E_5层强正则失活0.5最优模型.pth",
}


class ConfigurableCNN(nn.Module):
    """可配置层数与 Dropout 的 CNN。每个块: Conv-BN-ReLU-MaxPool。"""

    def __init__(self, blocks=3, dropout=0.2, num_classes=NUM_CLASSES):
        super().__init__()
        chans = [32, 64, 128, 256, 256][:blocks]
        layers, in_c = [], 3
        for c in chans:
            layers += [nn.Conv2d(in_c, c, 3, padding=1), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            in_c = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chans[-1], 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def evaluate(model, loader):
    model.eval()
    ce_sum, mse_sum, correct, n = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            onehot = F.one_hot(y, NUM_CLASSES).float()
            ce_sum += F.cross_entropy(logits, y, reduction="sum").item()
            mse_sum += torch.sum((probs - onehot) ** 2).item() / NUM_CLASSES  # 归一化后的 MSE
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return ce_sum / n, mse_sum / n, correct / n


def run_experiment(cfg, train_ds, val_ds):
    torch.manual_seed(42)
    model = ConfigurableCNN(cfg["blocks"], cfg["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=cfg["weight_decay"])

    tr_loader = make_loader(train_ds, BATCH, shuffle=True, workers=4)
    va_loader = make_loader(val_ds, BATCH, shuffle=False, workers=2)

    hist = {"train_loss": [], "train_mse": [], "val_loss": [], "val_mse": [], "val_acc": []}
    best_acc, best_state, bad = 0.0, None, 0
    for ep in range(1, EPOCHS + 1):
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
        tr_loss, tr_mse = ce_sum / n, mse_sum / n
        va_loss, va_mse, va_acc = evaluate(model, va_loader)
        hist["train_loss"].append(tr_loss); hist["train_mse"].append(tr_mse)
        hist["val_loss"].append(va_loss);   hist["val_mse"].append(va_mse)
        hist["val_acc"].append(va_acc)
        mark = ""
        if va_acc > best_acc:
            best_acc, bad = va_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            mark = "  <-- 最优"
        else:
            bad += 1
        print(f"  [{cfg['name']}] epoch {ep}/{EPOCHS} "
              f"train_CE={tr_loss:.4f} MSE={tr_mse:.4f} | "
              f"val_CE={va_loss:.4f} MSE={va_mse:.4f} acc={va_acc:.4f} "
              f"({time.time()-t0:.0f}s){mark}", flush=True)
        if bad >= PATIENCE:
            print(f"  [{cfg['name']}] 早停（{PATIENCE} 轮无提升）", flush=True)
            break

    # 保存该组实验的最优模型
    path = os.path.join(MODEL_DIR, _du.MATERIAL_PREFIX + EXP_MODEL_FILES[cfg["name"]])
    torch.save({"state_dict": best_state, "cfg": cfg, "img_size": IMG_SIZE,
                "val_acc": best_acc}, path)
    print(f"  [{cfg['name']}] 最优模型已保存: {path} (val_acc={best_acc:.4f})", flush=True)
    return {"hist": hist, "best_val_acc": best_acc, "model_path": path, "cfg": cfg}


def main():
    train_ds, val_ds, test_ds, names = build_datasets(img_size=IMG_SIZE)
    print(f"数据集: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} 类别={names}", flush=True)

    results = {}
    for cfg in EXPERIMENTS:
        print(f"\n=== 实验 {cfg['name']} (层数={cfg['blocks']} dropout={cfg['dropout']} L2={cfg['weight_decay']}) ===", flush=True)
        results[cfg["name"]] = run_experiment(cfg, train_ds, val_ds)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---------------- 绘图（每张图标注清楚内容）----------------
    xs = [range(1, len(r["hist"]["val_mse"]) + 1) for r in results.values()]
    labels = list(results.keys())

    # 01 MSE 变化对比
    plt.figure(figsize=(9, 6))
    for (name, r), x in zip(results.items(), xs):
        plt.plot(x, r["hist"]["val_mse"], marker="o", label=name)
    plt.xlabel("Epoch（训练轮数）"); plt.ylabel("验证集 MSE（概率与onehot的均方误差）")
    plt.title("不同层数/正则化/随机失活下 验证集 MSE 变化对比")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, _du.MATERIAL_PREFIX + "01_均方误差变化对比_不同层数与正则化.png"), dpi=150)
    plt.close()

    # 02 交叉熵损失对比
    plt.figure(figsize=(9, 6))
    for (name, r), x in zip(results.items(), xs):
        plt.plot(x, r["hist"]["val_loss"], marker="o", label=name)
    plt.xlabel("Epoch（训练轮数）"); plt.ylabel("验证集交叉熵损失 (Cross-Entropy)")
    plt.title("各实验配置 验证集损失对比")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, _du.MATERIAL_PREFIX + "02_交叉熵损失对比_各实验.png"), dpi=150)
    plt.close()

    # 03 验证准确率对比
    plt.figure(figsize=(9, 6))
    for (name, r), x in zip(results.items(), xs):
        plt.plot(x, [a * 100 for a in r["hist"]["val_acc"]], marker="o", label=name)
    plt.xlabel("Epoch（训练轮数）"); plt.ylabel("验证集准确率 (%)")
    plt.title("各实验配置 验证集准确率对比")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, _du.MATERIAL_PREFIX + "03_验证准确率对比_各实验.png"), dpi=150)
    plt.close()

    # 04 最终准确率柱状图
    plt.figure(figsize=(10, 6))
    accs = [r["best_val_acc"] * 100 for r in results.values()]
    bars = plt.bar(range(len(labels)), accs,
                   color=plt.cm.viridis(np.linspace(0.15, 0.85, len(labels))))
    plt.xticks(range(len(labels)), [l.replace("_", "\n") for l in labels], fontsize=9)
    plt.ylabel("最优验证集准确率 (%)")
    plt.title("各实验配置最优验证准确率汇总\n(差异来自: 网络层数 / Dropout随机失活 / L2权重衰减)")
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.3, f"{a:.2f}%", ha="center", fontsize=10)
    plt.ylim(0, 100); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, _du.MATERIAL_PREFIX + "04_各实验最终验证准确率_柱状图.png"), dpi=150)
    plt.close()

    print("\n全部实验完成。图表与结果已保存到 ...-输出结果/", flush=True)
    for name, r in results.items():
        print(f"  {name}: best_val_acc={r['best_val_acc']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
