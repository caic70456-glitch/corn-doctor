// pages/index/index.js
const { baseUrl, shareCode } = require("../../utils/config.js");

// 拼接接口地址：线上链接需要 sharecode 访问凭证
const predictUrl = baseUrl + "/predict" + (shareCode ? "?sharecode=" + shareCode : "");

Page({
  data: {
    photoPath: "",       // 预览图路径
    cameraMode: false,   // 是否打开相机取景
    loading: false,
    result: null,
    error: ""
  },

  // ---------- 打开/关闭相机取景 ----------
  openCamera() {
    this.setData({ cameraMode: true, result: null, error: "" });
  },
  closeCamera() {
    this.setData({ cameraMode: false });
  },

  // ---------- 相机拍照 ----------
  takePhoto() {
    const ctx = wx.createCameraContext();
    ctx.takePhoto({
      quality: "high",
      success: (res) => {
        const path = res.tempImagePath;
        this.setData({ photoPath: path, cameraMode: false });
        this.predict(path);
      },
      fail: () => wx.showToast({ title: "拍照失败", icon: "none" })
    });
  },
  cameraError() {
    wx.showToast({ title: "相机权限被拒绝，请到设置中开启", icon: "none" });
  },

  // ---------- 相册选图 ----------
  chooseFromAlbum() {
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      sourceType: ["album", "camera"],
      sizeType: ["compressed"],
      success: (res) => {
        const path = res.tempFiles[0].tempFilePath;
        this.setData({ photoPath: path, result: null, error: "" });
        this.predict(path);
      }
    });
  },

  // ---------- 上传识别 ----------
  predict(filePath) {
    this.setData({ loading: true });
    wx.uploadFile({
      url: predictUrl,
      filePath,
      name: "file",
      success: (res) => {
        try {
          const data = JSON.parse(res.data);
          if (data.error) throw new Error(data.error);
          this.setData({ result: data, error: "" });
        } catch (e) {
          this.setData({ error: "识别失败: " + e.message });
        }
      },
      fail: () => {
        this.setData({
          error: "网络错误：请确认手机与服务器在同一局域网，且已修改 utils/config.js 中的 baseUrl"
        });
      },
      complete: () => this.setData({ loading: false })
    });
  },

  // 重拍/重新选择
  retake() {
    this.setData({ photoPath: "", result: null, error: "" });
  }
});
