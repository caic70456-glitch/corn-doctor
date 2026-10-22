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
    leaf: null,          // 绿叶预检结果（服务端返回）
    noLeaf: false,       // 画面不是玉米叶片
    error: ""
  },

  // ---------- 打开/关闭相机取景 ----------
  openCamera() {
    this.setData({ cameraMode: true, result: null, leaf: null, noLeaf: false, error: "" });
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
        this.setData({ photoPath: path, result: null, leaf: null, noLeaf: false, error: "" });
        this.predict(path);
      }
    });
  },

  // ---------- 上传识别 ----------
  predict(filePath) {
    this.setData({ loading: true, noLeaf: false, leaf: null });
    wx.uploadFile({
      url: predictUrl,
      filePath,
      name: "file",
      success: (res) => {
        try {
          const data = JSON.parse(res.data);
          // 绿叶预检未通过：画面里不是玉米叶片，服务端不会做病害判定
          if (data.leaf && !data.leaf.is_leaf) {
            this.setData({ leaf: data.leaf, noLeaf: true, result: null, error: "" });
            return;
          }
          if (data.error) throw new Error(data.error);
          this.setData({ result: data, leaf: data.leaf || null, noLeaf: false, error: "" });
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
    this.setData({ photoPath: "", result: null, leaf: null, noLeaf: false, error: "" });
  }
});
