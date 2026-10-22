// app.js
App({
  onLaunch() {},
  // 服务端地址：真机调试请改为局域网 IP，如 http://192.168.1.100:8000
  // 正式上线需 HTTPS 域名并在小程序后台配置 request/uploadFile 合法域名
  globalData: {
    baseUrl: "http://127.0.0.1:8000"
  }
});
