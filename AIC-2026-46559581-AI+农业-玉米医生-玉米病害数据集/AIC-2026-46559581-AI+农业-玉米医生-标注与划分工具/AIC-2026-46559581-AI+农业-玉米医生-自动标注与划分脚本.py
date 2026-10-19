# -*- coding: utf-8 -*-
"""
玉米病虫害数据集：自动生成 labelme JSON 标注 + 分层 7:3 拆分训练/测试集

原理：HSV 颜色空间机理规则分割病斑区域（非绿色组织 & 非背景），
     对每个病斑轮廓生成 labelme polygon 标注。
     标签按图像所属类别赋予（自拟标签名见 CLASS_LABELS）。

用法：
    python 自动标注与划分.py
可按需修改下方 CONFIG 常量。
"""
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ---------------- CONFIG ----------------
ROOT = Path(__file__).resolve().parents[1]   # 数据集根目录（本脚本的上一级）

SRC1 = ROOT / "原始数据集1_Kaggle四分类"      # 数据集1：Kaggle 4分类（PlantVillage+PlantDoc）
SRC2 = ROOT / "原始数据集2_田间感染"          # 数据集2：Kaggle 田间感染数据集（结构见 inspect）
OUT = ROOT / "合并输出"                       # 输出：train/ 与 test/（图像 + 同名 .json）
TRAIN_RATIO = 0.7
SEED = 42
MAX_SHAPES = 12          # 每张图最多标注病斑数
MIN_AREA_RATIO = 0.003   # 病斑最小面积（占全图比例）
POINTS_EPS = 0.004       # 轮廓多边形简化系数
MAX_POINTS = 40          # 单个多边形最多点数

# 自拟标签名：类别文件夹名 -> labelme 标签
CLASS_LABELS = {
    "Blight": "blight_lesion",            # 叶枯病病斑
    "Common_Rust": "rust_pustule",        # 锈病脓疱
    "Gray_Leaf_Spot": "gray_spot_lesion", # 灰斑病病斑
    "Healthy": "healthy",                 # 健康叶片（无病斑）
    "Infected": "field_lesion",           # 田间感染叶片（未定型病斑）
}
HEALTHY_KEYS = {"healthy", "health", "healthycorn"}


def collect_images(src: Path, prefix: str):
    """收集 src 下各类别子文件夹中的图像，返回 [(img_path, class_key), ...]"""
    items = []
    if not src.exists():
        return items
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        for f in sorted(class_dir.rglob("*")):
            if f.suffix.lower() in exts:
                items.append((f, f"{prefix}:{class_dir.name}"))
    return items


def lesion_mask(img_bgr: np.ndarray, class_key: str) -> np.ndarray:
    """HSV 机理规则：分割病斑（非绿色健康组织、非背景、非过暗区域）"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    green = (H >= 35) & (H <= 95) & (S >= 50)          # 健康叶绿区域
    background = ((S < 40) & (V > 170)) | (V < 35)     # 浅灰背景 / 过暗
    lesion = (~green) & (~background)

    if class_key == "Common_Rust":                     # 锈病：红褐色脓疱为主
        reddish = ((H <= 22) | (H >= 155)) & (S >= 60)
        tan = (H > 22) & (H <= 38) & (S >= 40)         # 允许偏黄褐部分
        lesion = lesion & (reddish | tan)
    elif class_key == "Gray_Leaf_Spot":                # 灰斑：允许低饱和度灰褐斑
        lesion = lesion & (V >= 50)
    else:                                              # 通用：褐色/黄色/灰色病斑
        lesion = lesion & (V >= 40)

    lesion = lesion.astype(np.uint8) * 255
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return lesion


def image_shapes(img_bgr: np.ndarray, label: str) -> list:
    """病斑掩膜 -> labelme polygon shapes"""
    if label == "healthy":
        return []
    h, w = img_bgr.shape[:2]
    mask = lesion_mask(img_bgr, label)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA_RATIO * h * w]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_SHAPES]
    shapes = []
    for c in contours:
        eps = POINTS_EPS * cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        if len(poly) > MAX_POINTS:                     # 点数过多时均匀抽样
            idx = np.linspace(0, len(poly) - 1, MAX_POINTS).astype(int)
            poly = poly[idx]
        shapes.append({
            "label": label,
            "points": [[int(x), int(y)] for x, y in poly],
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
        })
    return shapes


def labelme_json(shapes: list, img_name: str, h: int, w: int) -> dict:
    return {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }


def label_for(class_key: str) -> str:
    folder = class_key.split(":", 1)[1]
    for k, v in CLASS_LABELS.items():
        if folder.lower() == k.lower():
            return v
    if any(h in folder.lower() for h in ("health",)):
        return "healthy"
    return "field_lesion"


def main():
    random.seed(SEED)
    items = collect_images(SRC1, "ds1") + collect_images(SRC2, "ds2")
    if not items:
        print("未找到任何图像，请检查 SRC1 / SRC2 路径")
        return
    print(f"共收集图像 {len(items)} 张")

    # 按类别分层：各类内 shuffle 后前 70% 训练、其余测试
    by_class = {}
    for p, ck in items:
        by_class.setdefault(ck, []).append(p)

    (OUT / "train").mkdir(parents=True, exist_ok=True)
    (OUT / "test").mkdir(parents=True, exist_ok=True)

    stat = {}
    total_shapes = 0
    used_names = set()
    for ck in sorted(by_class):
        paths = sorted(by_class[ck])
        random.shuffle(paths)
        n_train = int(len(paths) * TRAIN_RATIO)
        label = label_for(ck)
        for i, img_path in enumerate(paths):
            split = "train" if i < n_train else "test"
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            shapes = image_shapes(img_bgr, label)
            total_shapes += len(shapes)

            stem = img_path.stem
            name = f"{ck.replace(':', '_').replace('/', '_')}__{stem}{img_path.suffix.lower()}"
            while name in used_names:                   # 防重名
                name = name.replace(img_path.suffix.lower(), f"_x{random.randint(0, 9999)}{img_path.suffix.lower()}")
            used_names.add(name)

            dst_img = OUT / split / name
            shutil.copy2(img_path, dst_img)
            dst_json = dst_img.with_suffix(".json")
            dst_json.write_text(
                json.dumps(labelme_json(shapes, name, h, w), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            key = (ck, split)
            stat[key] = stat.get(key, 0) + 1

    print(f"\n拆分完成（ratio={TRAIN_RATIO}, seed={SEED}） -> {OUT}")
    print(f"{'类别':<28}{'训练':>8}{'测试':>8}")
    classes = sorted({k[0] for k in stat})
    for ck in classes:
        tr = stat.get((ck, "train"), 0)
        te = stat.get((ck, "test"), 0)
        print(f"{ck:<28}{tr:>8}{te:>8}")
    tr_all = sum(v for k, v in stat.items() if k[1] == "train")
    te_all = sum(v for k, v in stat.items() if k[1] == "test")
    print(f"{'合计':<28}{tr_all:>8}{te_all:>8}")
    print(f"病斑多边形总数: {total_shapes}（均值 {total_shapes / max(tr_all + te_all, 1):.2f}/张）")


if __name__ == "__main__":
    main()
