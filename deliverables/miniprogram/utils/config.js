// utils/config.js
// 【已停用】小程序改为本地离线推理（utils/inference.js + utils/model_weights.bin），
// 不再需要后端服务，本文件已不被任何页面引用，仅保留作历史参考。
//
// 如需切回「上传到服务器识别」的方式：
//   1) 启动后端（server/api.py 监听 5000，或 deliverables/web/server.py 监听 8000）
//   2) 下方 baseUrl 改为 http://<电脑局域网IP>:<端口>
//   3) pages/index/index.js 改回使用 wx.uploadFile 上传 file 字段
module.exports = {
  baseUrl: "",
  shareCode: ""
};
