// utils/config.js - 服务端地址配置
// 原 WorkBuddy 云端地址（https://corn-doctor-recognition.app.workbuddy.host）已失效，
// 改为指向本机局域网后端（与 server/api.py 或 deliverables/web/server.py 配合）。
// 注意：电脑 IP 变化后要同步修改 baseUrl；真机需与电脑连同一 WiFi 并开启调试模式。
module.exports = {
  baseUrl: "http://10.45.90.142:5000",
  shareCode: ""
  // 若改用 deliverables/web/server.py，端口为 8000，则改为 "http://10.45.90.142:8000"
};
