// tools/test_inference.js
// 在 Node 里直接跑 utils/inference.js，与 Python 基准对比，验证小程序端的 JS 推理算法
// 用法: node tools/test_inference.js
const fs = require("fs");
const path = require("path");

// 仅 mock 小程序里用到的 base64 -> ArrayBuffer
global.wx = {
  base64ToArrayBuffer(b64) {
    const buf = Buffer.from(b64, "base64");
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
};

const inf = require("../utils/inference.js");

(async () => {
  const t0 = Date.now();
  await inf.loadModel();
  const loadMs = Date.now() - t0;

  const raw = fs.readFileSync(path.join(__dirname, "test_rgba.bin"));
  const rgba = new Uint8ClampedArray(raw.buffer, raw.byteOffset, raw.length);

  const t1 = Date.now();
  const built = inf.buildTensor(rgba, inf.INPUT_SIZE);
  const probs = inf.forward(built.chw, inf.INPUT_SIZE);
  const inferMs = Date.now() - t1;

  const base = JSON.parse(fs.readFileSync(path.join(__dirname, "test_baseline.json"), "utf8"));
  const jsPct = Array.from(probs).map((p) => +(p * 100).toFixed(4));
  const maxDiff = Math.max(...jsPct.map((p, i) => Math.abs(p - base.probs[i])));

  const report = {
    loadMs,
    inferMs,
    classes: inf.CLASSES,
    js: { probs: jsPct, leaf: built.leaf },
    python: { probs: base.probs, green_ratio: base.green_ratio, brightness: base.brightness },
    maxDiffPercent: +maxDiff.toFixed(4),
    pass: maxDiff < 0.5
  };
  console.log(JSON.stringify(report, null, 2));

  const top = jsPct.indexOf(Math.max(...jsPct));
  console.log("\nJS 判定:", inf.CLASSES[top], "置信度", jsPct[top].toFixed(2) + "%");
  console.log("与 Python 最大偏差:", maxDiff.toFixed(4), "个百分点 ->", report.pass ? "一致 ✓" : "不一致 ✗");
})().catch((e) => {
  console.error("失败:", e && e.stack ? e.stack : e);
  process.exit(1);
});
