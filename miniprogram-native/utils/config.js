// utils/config.js - 服务端地址配置
// 已指向 WorkBuddy 云端发布的线上后端（HTTPS），真机可直接用
// sharecode 是该发布链接的访问凭证，调用接口时会自动拼上
module.exports = {
  baseUrl: "https://corn-doctor-recognition.app.workbuddy.host",
  shareCode: "FES5LKCPYG"
  // 换成局域网调试时：baseUrl 改为 "http://192.168.x.x:8000"，shareCode 改为 ""
};
