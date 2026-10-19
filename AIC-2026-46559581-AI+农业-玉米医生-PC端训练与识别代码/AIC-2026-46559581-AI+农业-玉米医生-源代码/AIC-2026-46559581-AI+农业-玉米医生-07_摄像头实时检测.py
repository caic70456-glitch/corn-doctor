# -*- coding: utf-8 -*-
"""USB 摄像头实时病虫害检测（滑动窗口定位病灶坐标）。

原理（复用最终分类模型，无需重新训练）:
  1. 场景预检：统计画面 HSV 绿色(叶色)像素占比，光照不足或未对准叶片
     （手挡光/非植物场景）时直接提示，不做病害分析
  2. 每帧常驻多尺度滑动窗口扫描（限中央检测区，窗口边长取画面短边的
     1.0/0.7/0.5 倍，约 40% 重叠）。窗口是"局部视图"，窗口内病灶占比远高
     于整帧，小面积病灶也能检出；窗口级再做绿色预检，非叶片窗口直接跳过
  3. 叶片窗口批量分类，"非健康且置信度达标、且窗口内确有病斑色"的窗口
     经 NMS 去重合并后即为病虫害二维坐标框 (x1, y1, x2, y2)（像素坐标，
     相对摄像头原始画面）
  4. 时域确认：需连续 CONFIRM_FRAMES 帧检出病灶才显示/输出（过滤手伸入
     画面等瞬时扰动造成的闪现误判）

画面显示: 病灶框 + 中文类别 + 置信度；左上角显示整体状态与推理耗时
控制台  : 打印每个病灶框的坐标与置信度（结果变化时打印）
快捷键  : s 保存当前画面截图到 输出结果/...-摄像头截图/   q 退出

运行: python ...-源代码/07_摄像头实时检测.py [--camera 0] [--width 640] [--height 480]
模型: ...-输出结果/...-最终模型/...-最终模型最优.onnx（先运行 06_ONNX模型导出.py 生成）
依赖: onnxruntime opencv-python pillow（无需 PyTorch）
"""
import argparse
import importlib
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 文件名以数字开头，需通过 importlib 导入公共模块
_du = importlib.import_module("AIC-2026-46559581-AI+农业-玉米医生-01_数据集加载与公共工具")
FINAL_DIR = _du.FINAL_DIR
OUT_DIR = _du.OUT_DIR
MATERIAL_PREFIX = _du.MATERIAL_PREFIX
CLASS_CN = _du.CLASS_CN

MODEL_PATH = os.path.join(FINAL_DIR, MATERIAL_PREFIX + "最终模型最优.onnx")
SHOT_DIR = os.path.join(OUT_DIR, MATERIAL_PREFIX + "摄像头截图")

IMG_SIZE = 224
MEAN = np.array(_du.MEAN, dtype=np.float32)
STD = np.array(_du.STD, dtype=np.float32)

SCALES = (1.0, 0.7, 0.5, 0.35)   # 窗口边长 / 画面短边（多尺度覆盖近/中/远距离病灶）
STEP_RATIO = 0.6           # 滑窗步长 = 窗口边长 x 0.6（约 40% 重叠）
ROI_RATIO = 0.70           # 中央检测区占画面比例（滑窗只在该区域内，避免边缘/背景误框）
WIN_PROB = 0.80            # 窗口判定阈值：病害类置信度 >= 该值才记为病灶块
NMS_IOU = 0.35             # 病灶框合并的 IoU 阈值（帧内空间 NMS）
DARK_MEAN = 45.0           # 画面平均亮度低于该值 -> 光照不足，不做检测（防暗帧误判）
LEAF_MIN_RATIO = 0.02      # 画面绿色(叶色)占比低于该值 -> 视为未对准叶片（放宽兼容手机翻拍/远距拍摄）
LEAF_WIN_RATIO = 0.15      # 窗口绿色占比低于该值 -> 非叶片窗口，跳过不送模型（防非叶片误判）
LESION_MIN_RATIO = 0.03    # 病灶窗口内病斑色(橙红/黄褐)占比下限（过滤模型对杂乱背景的高置信误判）
CONFIRM_FRAMES = 5         # 时域确认：连续该帧数检出病灶才显示/输出坐标（防瞬时误判）

# 中文字体（Windows -> Linux 常见路径，找不到则退回英文标签）
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(size=20):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT = find_font()


def load_onnx(path=MODEL_PATH):
    """加载 ONNX 模型，并从元数据读类别顺序（读不到则用公共模块默认顺序）。"""
    if not os.path.exists(path):
        raise SystemExit(
            f"[错误] 未找到 ONNX 模型: {path}\n"
            f"请先运行: python ...-源代码/06_ONNX模型导出.py 生成。")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = list(_du.CLASSES)
    try:
        meta = {kv.key: kv.value for kv in sess.get_modelmeta().custom_metadata_map.items()}
        names = [meta[f"class_{i}"] for i in range(len(meta)) if f"class_{i}" in meta] or names
    except Exception:
        pass
    print(f"[初始化] ONNX 模型加载成功: {os.path.basename(path)}")
    print(f"[初始化] 类别顺序: {names}")
    return sess, sess.get_inputs()[0].name, names


def preprocess_patch(bgr):
    """BGR 图块 -> 归一化 CHW 输入（与训练预处理一致：resize/RGB/除255/标准化）。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))


def predict_batch(sess, input_name, patches):
    """批量推理，返回 (N, 类别数) 概率矩阵。"""
    batch = np.ascontiguousarray(np.stack(patches))
    return sess.run(None, {input_name: batch})[0]


def sliding_windows(h, w):
    """生成中央检测区内的多尺度滑动窗口 (x1, y1, x2, y2)。"""
    short = min(h, w)
    mx1, my1 = int(w * (1 - ROI_RATIO) / 2), int(h * (1 - ROI_RATIO) / 2)
    mx2, my2 = w - mx1, h - my1
    wins = []
    for s in SCALES:
        win = max(64, int(short * s))
        step = max(1, int(win * STEP_RATIO))
        for y in range(my1, max(my2 - win, my1) + 1, step):
            for x in range(mx1, max(mx2 - win, mx1) + 1, step):
                wins.append((x, y, x + win, y + win))
    return wins


def green_ratio(frame_bgr):
    """统计画面中绿色(植物叶色)像素占比，用于确认画面中确实存在玉米叶片。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))   # H绿色域/S不灰白/V不太暗
    return float(cv2.countNonZero(mask)) / mask.size


def lesion_ratio(frame_bgr):
    """统计画面中病斑色(橙红~黄褐)像素占比，用于确认病害窗口内确有病斑。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 40, 50), (40, 255, 255))    # 橙红~黄褐色域
    return float(cv2.countNonZero(mask)) / mask.size


def detect_frame(sess, input_name, class_names, frame):
    """单帧检测：常驻滑窗扫描 -> 窗口级叶片预检 -> 批量分类 -> NMS 合并。

    返回 (detections, infer_ms, kept, total)
      detections: [(x1,y1,x2,y2), conf, class_id]
      kept/total: 通过叶片预检的窗口数 / 滑窗总数（调试用）
    """
    t0 = time.perf_counter()
    h, w = frame.shape[:2]

    # 1) 滑窗 + 窗口级绿色预检：只把"叶片窗口"送模型，手/边框/背景等直接跳过
    detections = []
    patches, keep_wins = [], []
    wins = sliding_windows(h, w)
    total = len(wins)
    for (x1, y1, x2, y2) in wins:
        win = frame[y1:y2, x1:x2]
        if green_ratio(win) < LEAF_WIN_RATIO:
            continue
        patches.append(preprocess_patch(win))
        keep_wins.append((x1, y1, x2, y2))
    kept = len(patches)

    # 2) 叶片窗口批量分类，筛选病害窗口（模型判病害 + 窗口内确有病斑色）
    if patches:
        probs = predict_batch(sess, input_name, patches)
        boxes, confs, cls_ids = [], [], []
        for (x1, y1, x2, y2), p in zip(keep_wins, probs):
            t = int(p.argmax())
            if class_names[t] != "Healthy" and float(p[t]) >= WIN_PROB:
                if lesion_ratio(frame[y1:y2, x1:x2]) < LESION_MIN_RATIO:
                    continue    # 模型误判（窗口内无病斑色）-> 忽略，防背景/手指高置信误框
                boxes.append([x1, y1, x2, y2])
                confs.append(float(p[t]))
                cls_ids.append(t)
        if boxes:
            idx = np.array(cv2.dnn.NMSBoxes(boxes, confs, WIN_PROB, NMS_IOU)).reshape(-1)
            detections = [(boxes[i], confs[i], cls_ids[i]) for i in idx]

    infer_ms = (time.perf_counter() - t0) * 1000
    return detections, infer_ms, kept, total


def draw_overlay(frame, detections, class_names, status, color, infer_ms, fps,
                 leaf_pct=None, wins=None):
    """在画面上绘制病灶框（中文标签+坐标）、整体状态、预检数值与耗时。"""
    h, w = frame.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(img)

    # 中央检测区提示（白色角标）：病灶只在该区域内检出，请把叶片对准框内
    mx1, my1 = int(w * (1 - ROI_RATIO) / 2), int(h * (1 - ROI_RATIO) / 2)
    mx2, my2 = w - mx1, h - my1
    L = 26
    for (ax, ay, bx, by, cx, cy) in (
            (mx1, my1, mx1 + L, my1, mx1, my1 + L),
            (mx2 - L, my1, mx2, my1, mx2, my1 + L),
            (mx1, my2 - L, mx1 + L, my2, mx1, my2 - L),
            (mx2 - L, my2 - L, mx2, my2, mx2, my2 - L)):
        dr.line([(bx, ay), (ax, ay), (ax, cy)], fill=(255, 255, 255), width=2)
        dr.line([(ax, by), (ax, ay), (bx, ay)], fill=(255, 255, 255), width=2)

    for (x1, y1, x2, y2), c, ci in detections:
        cn = CLASS_CN.get(class_names[ci], class_names[ci])
        dr.rectangle([x1, y1, x2, y2], outline=(230, 30, 30), width=3)
        dr.text((x1 + 2, max(y1 - 24, 2)), f"{cn} {c:.2f}",
                fill=(255, 80, 80), font=FONT, stroke_width=2, stroke_fill=(0, 0, 0))
        dr.text((x1 + 2, min(y2 + 2, h - 24)), f"({x1},{y1})-({x2},{y2})",
                fill=(255, 200, 60), font=FONT, stroke_width=2, stroke_fill=(0, 0, 0))

    dr.text((10, 8), status, fill=color, font=FONT, stroke_width=2, stroke_fill=(0, 0, 0))
    extra = f"leaf {leaf_pct*100:.1f}%" if leaf_pct is not None else ""
    if wins is not None:
        extra += f" | 叶窗 {wins[0]}/{wins[1]}"
    dr.text((10, 34), f"{extra} | infer {infer_ms:.0f} ms | {fps:.1f} FPS",
            fill=(240, 240, 240), font=FONT, stroke_width=2, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="玉米病虫害 USB 摄像头实时检测")
    ap.add_argument("--camera", type=int, default=0, help="摄像头编号（默认 0）")
    ap.add_argument("--width", type=int, default=640, help="采集宽度（默认 640）")
    ap.add_argument("--height", type=int, default=480, help="采集高度（默认 480）")
    args = ap.parse_args()

    sess, input_name, class_names = load_onnx()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)   # Windows 用 DSHOW 启动更快
    if not cap.isOpened():                               # 非 Windows / 其他后端兜底
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"[错误] 无法打开摄像头 {args.camera}，请检查连接或换 --camera 编号。")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"[初始化] 摄像头已打开 ({args.width}x{args.height})，按 q 退出、s 保存截图\n")

    fps, last = 0.0, time.perf_counter()
    hit = 0                 # 连续判定为病害的帧数（时域确认计数器）
    prev_sig = None                                      # 上次打印结果（避免刷屏）
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[警告] 读取画面失败，重试中...")
            continue

        # 状态判定：暗帧/无叶片 -> 场景预检提示，不做病害分析；
        # 常驻滑窗扫描 -> 连续 CONFIRM_FRAMES 帧检出病灶才框选输出（瞬时误判不显示）
        gr = green_ratio(frame)                   # 画面绿色占比（预检 + 屏显调试）
        if float(frame.mean()) < DARK_MEAN:
            detections, infer_ms, hit = [], 0.0, 0
            kept = total = 0
            status, color, kind = "光照不足，请对准玉米叶片", (250, 210, 60), "dark"
        elif gr < LEAF_MIN_RATIO:
            detections, infer_ms, hit = [], 0.0, 0
            kept = total = 0
            status = f"未检测到玉米叶片，请对准叶片 (绿色占比 {gr*100:.1f}%)"
            color, kind = (250, 210, 60), "noleaf"
        else:
            detections, infer_ms, kept, total = detect_frame(sess, input_name, class_names, frame)
            if not detections:
                hit = 0
                status, color, kind = "正常", (60, 220, 60), "normal"
            elif hit < CONFIRM_FRAMES - 1:
                hit += 1
                detections = []               # 确认中：暂不显示/输出坐标
                status, color, kind = "正常", (60, 220, 60), "normal"
            else:
                hit += 1
                _, conf, ci = max(detections, key=lambda d: d[1])   # 最高置信度病灶作为主类别
                status = f"{CLASS_CN.get(class_names[ci], class_names[ci])}  {conf:.2f}"
                color, kind = (255, 70, 70), "disease"

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now

        # 控制台输出（状态类别或病灶结果变化时打印，避免刷屏）
        sig = (kind, tuple(sorted((d[2], round(d[1], 2)) for d in detections)))
        if sig != prev_sig:
            if kind == "noleaf":
                print(f"[{datetime.now():%H:%M:%S}] 未检测到玉米叶片 "
                      f"(画面绿色占比 {gr*100:.1f}% < 阈值 {LEAF_MIN_RATIO*100:.0f}%)")
            elif not detections:
                print(f"[{datetime.now():%H:%M:%S}] {status}")
            else:
                print(f"[{datetime.now():%H:%M:%S}] {status}  病灶数={len(detections)}")
                for (x1, y1, x2, y2), c, ci in detections:
                    print(f"    病灶 {CLASS_CN.get(class_names[ci], class_names[ci])}: "
                          f"坐标 (x1={x1}, y1={y1}, x2={x2}, y2={y2})  置信度={c:.2f}")
            prev_sig = sig

        show = draw_overlay(frame, detections, class_names, status, color, infer_ms, fps,
                            leaf_pct=gr, wins=(kept, total) if total else None)
        cv2.imshow("CornDoctor Camera (q=quit, s=save)", show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            os.makedirs(SHOT_DIR, exist_ok=True)
            path = os.path.join(SHOT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")
            cv2.imwrite(path, show)
            print(f"[截图] 已保存: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[退出] 摄像头检测已结束。")


if __name__ == "__main__":
    main()
