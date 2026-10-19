# -*- coding: utf-8 -*-
"""公共工具：数据集加载、类别定义、输出目录。

数据集（已按官方划分整理，扁平目录 + 文件名前缀标注类别）:
  数据集文件夹/train/images/   训练集 2931 张（train/labels 为 labelme JSON 标注）
  数据集文件夹/test/images/    测试集 1257 张（test/labels 为 labelme JSON 标注）
  文件名前缀 -> 类别: common_rust_/corn_blight_/gray_leaf_spot_/healthy_

目录约定（所有路径均基于本文件位置定位，从任意工作目录启动均有效）:
  <项目根>/
  ├── ...-PC端训练与识别代码/   ← 本文件夹
  │   ├── ...-源代码/           全部源码（按执行顺序 01-05 编号）
  │   ├── ...-输出结果/
  │   │   ├── ...-实验过程模型/   实验对比各组最优权重（01~05）
  │   │   ├── ...-结果图表/       实验对比图 01-04、测试集评估图 06-08
  │   │   ├── ...-最终模型/       ★最终交付：最终模型 + 训练历史曲线
  │   │   ├── ...-实验结果数据.json
  │   │   └── ...-测试评估报告.json
  │   └── ...-网页模板/          Flask 前端页面
  └── ...-玉米病害数据集/
      ├── train/  (images + labels)
      └── test/   (images + labels)

划分约定: 官方 test 文件夹保持封闭，仅做最终评估；val 从 train 内部按类别
分层抽取（默认 10%，seed=42 可复现）。
"""
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../PC端训练与识别代码/源代码
ROOT_DIR = os.path.dirname(BASE_DIR)                           # .../PC端训练与识别代码
DATA_ROOT = os.path.join(os.path.dirname(ROOT_DIR),
                         "AIC-2026-46559581-AI+农业-玉米医生-玉米病害数据集")  # .../玉米病害数据集
TRAIN_DIR = os.path.join(DATA_ROOT, "train", "images")
TEST_DIR = os.path.join(DATA_ROOT, "test", "images")
MATERIAL_PREFIX = "AIC-2026-46559581-AI+农业-玉米医生-"  # 统一命名前缀（参赛团队编号-赛题名称-作品名称-）
OUT_DIR = os.path.join(ROOT_DIR, MATERIAL_PREFIX + "输出结果")
MODEL_DIR = os.path.join(OUT_DIR, MATERIAL_PREFIX + "实验过程模型")  # 实验过程模型（02脚本保存的实验权重）
PLOT_DIR = os.path.join(OUT_DIR, MATERIAL_PREFIX + "结果图表")       # 实验对比图与测试集评估图
FINAL_DIR = os.path.join(OUT_DIR, MATERIAL_PREFIX + "最终模型")      # 最终交付：最终模型 + 训练历史曲线
RESULTS_JSON = os.path.join(OUT_DIR, MATERIAL_PREFIX + "实验结果数据.json")

for d in (MODEL_DIR, PLOT_DIR, FINAL_DIR):
    os.makedirs(d, exist_ok=True)

CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
CLASS_CN = {
    "Blight": "叶斑病",
    "Common_Rust": "普通锈病",
    "Gray_Leaf_Spot": "灰斑病",
    "Healthy": "健康",
}
# 文件名前缀 -> 代码内类别名（保持与历史实验模型/图表的类别顺序一致）
PREFIX_TO_CLASS = {
    "corn_blight": "Blight",
    "common_rust": "Common_Rust",
    "gray_leaf_spot": "Gray_Leaf_Spot",
    "healthy": "Healthy",
}
NUM_CLASSES = len(CLASSES)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

from torchvision import transforms  # noqa: E402


def make_transform(img_size=224, augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CornLeafDataset(Dataset):
    """从文件名前缀解析类别的扁平目录数据集（train/test 已官方划分）。"""

    def __init__(self, files, labels, transform):
        self.files, self.labels, self.transform = files, labels, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def _scan(dirpath):
    """扫描图片目录，按文件名前缀返回 (文件路径列表, 类别索引列表)。"""
    files, labels = [], []
    for fn in sorted(os.listdir(dirpath)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        prefix = fn.rsplit("_", 1)[0]          # healthy_0001.jpg -> healthy
        files.append(os.path.join(dirpath, fn))
        labels.append(CLASSES.index(PREFIX_TO_CLASS[prefix]))
    return files, labels


def build_datasets(img_size=224, val_split=0.1, seed=42, augment=True):
    """训练集内部分层抽取 val，官方 test 目录独立作最终测试集。

    返回 (train_ds, val_ds, test_ds, class_names)
    """
    tr_files, tr_labels = _scan(TRAIN_DIR)
    te_files, te_labels = _scan(TEST_DIR)
    assert tr_files and te_files, f"数据集为空: {TRAIN_DIR} / {TEST_DIR}"

    # 按类别分层划分 train -> train/val
    g = torch.Generator().manual_seed(seed)
    tr_idx, va_idx = [], []
    by_class = {}
    for i, y in enumerate(tr_labels):
        by_class.setdefault(y, []).append(i)
    for y, idxs in by_class.items():
        idxs = torch.tensor(idxs)[torch.randperm(len(idxs), generator=g)].tolist()
        n_val = max(1, int(len(idxs) * val_split))
        va_idx += idxs[:n_val]
        tr_idx += idxs[n_val:]

    train_ds = CornLeafDataset([tr_files[i] for i in tr_idx],
                               [tr_labels[i] for i in tr_idx],
                               make_transform(img_size, augment))
    val_ds = CornLeafDataset([tr_files[i] for i in va_idx],
                             [tr_labels[i] for i in va_idx],
                             make_transform(img_size, False))
    test_ds = CornLeafDataset(te_files, te_labels, make_transform(img_size, False))
    return train_ds, val_ds, test_ds, CLASSES


def make_loader(ds, batch_size=32, shuffle=False, workers=4):
    """DataLoader：workers>0 即多进程并行读取/预处理（Windows 需在 __main__ 中调用）。"""
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=False,
                      persistent_workers=workers > 0)


@torch.no_grad()
def compute_mse(probs, onehot):
    """MSE = mean((softmax概率 - onehot标签)^2)，用于观察不同超参下 MSE 的变化。"""
    return torch.mean((probs - onehot) ** 2).item()
