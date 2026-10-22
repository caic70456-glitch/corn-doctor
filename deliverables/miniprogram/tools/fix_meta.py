# -*- coding: utf-8 -*-
"""修复 model_meta.json 为小程序推理所需的扁平偏移结构（一次性工具）。

首次训练用的导出代码把 fc 也放进了 layers 且未写 w_flat/b_flat，
这里按权重形状重建偏移，不改动 model_weights.bin 本身。
"""
import json
import os

import numpy as np

P = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils", "model_meta.json"))
m = json.load(open(P, encoding="utf-8"))

allayers = m["layers"]
convs, fc = allayers[:4], allayers[-1]

off = 0
for L in convs:
    wc = int(np.prod(L["w"]))
    bc = int(np.prod(L["b"]))
    L["w_flat"], L["b_flat"] = wc, bc
    L["w_offset"], L["b_offset"] = off, off + wc
    off += wc + bc

fw, fb = int(np.prod(fc["w"])), int(np.prod(fc["b"]))
fc["w_flat"], fc["b_flat"] = fw, fb
fc["w_offset"], fc["b_offset"] = off, off + fw
off += fw + fb

m["layers"] = convs
m["fc"] = fc
m.pop("mean", None)
m.pop("std", None)

json.dump(m, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("累计", off, "/ total", m["totalFloats"], "OK" if off == m["totalFloats"] else "MISMATCH")
for i, L in enumerate(convs):
    print("conv%d w=%s stride=%d pad=%d off=%d/%d" % (i + 1, L["w"], L["stride"], L["pad"], L["w_offset"], L["b_offset"]))
print("fc w=%s b=%s off=%d/%d" % (fc["w"], fc["b"], fc["w_offset"], fc["b_offset"]))
