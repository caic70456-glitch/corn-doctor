// 模型元数据（由 tools/train_lite.py 生成；小程序不支持 require .json，故用 js 模块）
module.exports = {
 "input": 96,
 "classes": [
  "Blight",
  "Common_Rust",
  "Gray_Leaf_Spot",
  "Healthy"
 ],
 "classesSrc": [
  "corn_blight",
  "common_rust",
  "gray_leaf_spot",
  "healthy"
 ],
 "layers": [
  {
   "w": [
    16,
    3,
    3,
    3
   ],
   "b": [
    16
   ],
   "stride": 2,
   "pad": 1,
   "w_offset": 0,
   "b_offset": 432,
   "w_flat": 432,
   "b_flat": 16
  },
  {
   "w": [
    32,
    16,
    3,
    3
   ],
   "b": [
    32
   ],
   "stride": 2,
   "pad": 1,
   "w_offset": 448,
   "b_offset": 5056,
   "w_flat": 4608,
   "b_flat": 32
  },
  {
   "w": [
    64,
    32,
    3,
    3
   ],
   "b": [
    64
   ],
   "stride": 2,
   "pad": 1,
   "w_offset": 5088,
   "b_offset": 23520,
   "w_flat": 18432,
   "b_flat": 64
  },
  {
   "w": [
    64,
    64,
    3,
    3
   ],
   "b": [
    64
   ],
   "stride": 1,
   "pad": 1,
   "w_offset": 23584,
   "b_offset": 60448,
   "w_flat": 36864,
   "b_flat": 64
  }
 ],
 "fc": {
  "w": [
   4,
   64
  ],
  "b": [
   4
  ],
  "w_offset": 60512,
  "b_offset": 60768,
  "w_flat": 256,
  "b_flat": 4
 },
 "totalFloats": 60772,
 "testAcc": 92.12
};
