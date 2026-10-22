// utils/config.js - 识别服务地址配置
// 指向本机局域网后端（server/api.py，0.0.0.0:5000）
// 注意：电脑 IP 变化后要同步修改这里的 baseUrl（ipconfig 查看 IPv4 地址）
module.exports = {
  baseUrl: "http://10.45.90.142:5000",
  shareCode: ""
  // 开发者工具模拟器直接可用；真机需与电脑连同一 WiFi，
  // 并在手机端打开调试模式（跳过合法域名校验）
};
