# -*- coding: utf-8 -*-
"""把 model_meta.json / model_weights.bin 转成小程序可直接 require 的 js 模块。

原因：
  1. 微信小程序不支持 require(.json)，会直接报模块错误；
  2. readFileSync 读取代码包内二进制文件的兼容性不稳定。
两者都转成 js 模块后，只用最基础的 require 能力，最稳。
"""
import base64
import json
import os

D = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

meta = json.load(open(os.path.join(D, "model_meta.json"), encoding="utf-8"))
with open(os.path.join(D, "model_meta.js"), "w", encoding="utf-8") as f:
    f.write("// 模型元数据（由 tools/train_lite.py 生成；小程序不支持 require .json，故用 js 模块）\n")
    f.write("module.exports = " + json.dumps(meta, ensure_ascii=False, indent=1) + ";\n")

raw = open(os.path.join(D, "model_weights.bin"), "rb").read()
b64 = base64.b64encode(raw).decode()
with open(os.path.join(D, "model_weights_b64.js"), "w", encoding="utf-8") as f:
    f.write("// 模型权重：float32 平铺后 base64 编码，共 %d 字节 / %d 个 float32\n" % (len(raw), len(raw) // 4))
    f.write('module.exports = "' + b64 + '";\n')

print("model_meta.js         %d B" % os.path.getsize(os.path.join(D, "model_meta.js")))
print("model_weights_b64.js  %d B" % os.path.getsize(os.path.join(D, "model_weights_b64.js")))
