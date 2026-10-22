// utils/inference.js
// 纯 JS 前向推理（无后端、无插件、完全离线）
// 网络：Conv3x3 + BN(已融合进权重) + ReLU  x4  ->  GlobalAvgPool  ->  FC(64->4)  ->  softmax
// 权重文件：utils/model_weights.bin（由 tools/train_lite.py 导出，扁平 float32，按 meta 中的偏移切片）

const MODEL_BIN = "utils/model_weights.bin";
const MODEL_META = require("./model_meta.json");

// 绿叶检测阈值（与 PC 端 07_摄像头实时检测.py 一致）
const DARK_MEAN = 45;
const LEAF_MIN_RATIO = 0.02;
const LEAF_H_MIN = 70, LEAF_H_MAX = 180;   // 绿色色相（度）
const LEAF_S_MIN = 60 / 255, LEAF_V_MIN = 40 / 255;

let WEIGHTS = null;      // Float32Array
let loadPromise = null;

/* ---------- 读取权重（代码包内文件，一次加载常驻内存） ---------- */
function loadModel() {
  if (WEIGHTS) return Promise.resolve(WEIGHTS);
  if (loadPromise) return loadPromise;
  loadPromise = new Promise((resolve, reject) => {
    try {
      const buf = wx.getFileSystemManager().readFileSync(MODEL_BIN);
      const arr = new Float32Array(buf);
      if (arr.length !== MODEL_META.totalFloats) {
        return reject(new Error("模型权重长度不符：" + arr.length + " vs " + MODEL_META.totalFloats));
      }
      WEIGHTS = arr;
      resolve(arr);
    } catch (e) {
      reject(new Error("模型权重读取失败：" + (e && e.message ? e.message : e)));
    }
  });
  return loadPromise;
}

/* ---------- 图片 -> 96x96 RGB 张量 + 绿叶检测 ---------- */
function imageToTensor(filePath, size) {
  return new Promise((resolve, reject) => {
    let canvas, ctx;
    try {
      canvas = wx.createOffscreenCanvas({ type: "2d", width: size, height: size });
      ctx = canvas.getContext("2d");
    } catch (e) {
      return reject(new Error("当前微信版本不支持离屏画布，请升级微信"));
    }
    const img = canvas.createImage();
    img.onload = () => {
      try {
        ctx.drawImage(img, 0, 0, size, size);
        const rgba = ctx.getImageData(0, 0, size, size).data;
        resolve(buildTensor(rgba, size));
      } catch (e) {
        reject(e);
      }
    };
    img.onerror = () => reject(new Error("图片解码失败，请换一张试试"));
    img.src = filePath;
  });
}

function buildTensor(rgba, size) {
  const n = size * size;
  const chw = new Float32Array(3 * n);
  let green = 0, vsum = 0;

  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    // 归一化：(v/255 - 0.5) / 0.5 = v/127.5 - 1
    chw[i] = r / 127.5 - 1;
    chw[n + i] = g / 127.5 - 1;
    chw[2 * n + i] = b / 127.5 - 1;

    // 绿叶检测：RGB -> HSV（与后端 PIL HSV 口径一致）
    const rf = r / 255, gf = g / 255, bf = b / 255;
    const mx = Math.max(rf, gf, bf), mn = Math.min(rf, gf, bf), d = mx - mn;
    let h = 0;
    if (d > 1e-6) {
      if (mx === rf) h = 60 * (((gf - bf) / d) % 6);
      else if (mx === gf) h = 60 * ((bf - rf) / d + 2);
      else h = 60 * ((rf - gf) / d + 4);
    }
    if (h < 0) h += 360;
    const s = mx === 0 ? 0 : d / mx;
    vsum += mx;
    if (h >= LEAF_H_MIN && h <= LEAF_H_MAX && s >= LEAF_S_MIN && mx >= LEAF_V_MIN) green++;
  }

  const greenRatio = green / n;
  const brightness = (vsum / n) * 255;
  let isLeaf, message;
  if (brightness < DARK_MEAN) {
    isLeaf = false; message = "光照不足，请对准玉米叶片并保证光线充足";
  } else if (greenRatio < LEAF_MIN_RATIO) {
    isLeaf = false;
    message = "未检测到玉米叶片（绿色占比 " + (greenRatio * 100).toFixed(1) + "%），请对准叶片后重拍";
  } else {
    isLeaf = true; message = "已确认玉米叶片";
  }

  return {
    chw,
    leaf: {
      is_leaf: isLeaf,
      green_ratio: +(greenRatio * 100).toFixed(2),
      brightness: +brightness.toFixed(1),
      message
    }
  };
}

/* ---------- 单层 3x3 卷积 + ReLU ---------- */
function convRelu3x3(x, inC, inH, inW, w, b, outC, stride, pad) {
  const outH = Math.floor((inH + 2 * pad - 3) / stride) + 1;
  const outW = Math.floor((inW + 2 * pad - 3) / stride) + 1;
  const out = new Float32Array(outC * outH * outW);
  const planeIn = inH * inW;

  for (let oc = 0; oc < outC; oc++) {
    const wOff = oc * inC * 9;
    const bias = b[oc];
    const outBase = oc * outH * outW;
    for (let oy = 0; oy < outH; oy++) {
      const iy0 = oy * stride - pad;
      for (let ox = 0; ox < outW; ox++) {
        const ix0 = ox * stride - pad;
        let sum = bias;
        for (let ic = 0; ic < inC; ic++) {
          const wBase = wOff + ic * 9;
          const inBase = ic * planeIn;
          for (let ky = 0; ky < 3; ky++) {
            const iy = iy0 + ky;
            if (iy < 0 || iy >= inH) continue;
            const rowBase = inBase + iy * inW;
            const wRow = wBase + ky * 3;
            for (let kx = 0; kx < 3; kx++) {
              const ix = ix0 + kx;
              if (ix < 0 || ix >= inW) continue;
              sum += x[rowBase + ix] * w[wRow + kx];
            }
          }
        }
        out[outBase + oy * outW + ox] = sum > 0 ? sum : 0;   // 融合 ReLU
      }
    }
  }
  return { data: out, c: outC, h: outH, w: outW };
}

/* ---------- 前向推理 ---------- */
function forward(chw, size) {
  const meta = MODEL_META;
  let feat = { data: chw, c: 3, h: size, w: size };

  for (let i = 0; i < meta.layers.length; i++) {
    const L = meta.layers[i];
    feat = convRelu3x3(
      feat.data, feat.c, feat.h, feat.w,
      WEIGHTS.subarray(L.w_offset, L.w_offset + L.w_flat),
      WEIGHTS.subarray(L.b_offset, L.b_offset + L.b_flat),
      L.w[0], L.stride, L.pad
    );
  }

  // Global Average Pooling
  const plane = feat.h * feat.w;
  const gap = new Float32Array(feat.c);
  for (let c = 0; c < feat.c; c++) {
    let s = 0;
    const base = c * plane;
    for (let i = 0; i < plane; i++) s += feat.data[base + i];
    gap[c] = s / plane;
  }

  // 全连接
  const FC = meta.fc;
  const fcW = WEIGHTS.subarray(FC.w_offset, FC.w_offset + FC.w_flat);
  const fcB = WEIGHTS.subarray(FC.b_offset, FC.b_offset + FC.b_flat);
  const nCls = FC.b_flat;
  const logits = new Float32Array(nCls);
  for (let o = 0; o < nCls; o++) {
    let s = fcB[o];
    for (let i = 0; i < gap.length; i++) s += gap[i] * fcW[o * gap.length + i];
    logits[o] = s;
  }

  // softmax
  let mx = -Infinity;
  for (let i = 0; i < nCls; i++) if (logits[i] > mx) mx = logits[i];
  let sum = 0;
  const probs = new Float32Array(nCls);
  for (let i = 0; i < nCls; i++) { probs[i] = Math.exp(logits[i] - mx); sum += probs[i]; }
  for (let i = 0; i < nCls; i++) probs[i] /= sum;
  return probs;
}

/* ---------- 对外接口 ---------- */
async function predict(filePath) {
  const t0 = Date.now();
  await loadModel();
  const meta = MODEL_META;
  const { chw, leaf } = await imageToTensor(filePath, meta.input);
  if (!leaf.is_leaf) {
    return { ok: false, leaf, ms: Date.now() - t0 };
  }
  const probs = forward(chw, meta.input);

  const ranked = meta.classes.map((cls, i) => ({ cls, p: probs[i] }))
    .sort((a, b) => b.p - a.p);

  return {
    ok: true,
    leaf,
    probs: ranked,
    top: ranked[0],
    ms: Date.now() - t0
  };
}

module.exports = {
  predict,
  loadModel,
  CLASSES: MODEL_META.classes,
  INPUT_SIZE: MODEL_META.input,
  TEST_ACC: MODEL_META.testAcc
};
