// pages/index/index.js
// 本地离线识别：模型与推理全部在小程序内完成，不需要后端 / 服务器 / 域名配置
const { predict: localPredict, loadModel, TEST_ACC, INPUT_SIZE } = require("../../utils/inference.js");

const CLASS_CN = {
  Blight: "叶斑病",
  Common_Rust: "普通锈病",
  Gray_Leaf_Spot: "灰斑病",
  Healthy: "健康"
};
const ADVICE = {
  Blight: "玉米叶斑病：及时摘除下部重病叶并带出田外销毁；发病初期可用苯醚甲环唑、吡唑醚菌酯等药剂喷雾防治，间隔7~10天连喷2次；合理密植、增施钾肥提高抗性。",
  Common_Rust: "玉米普通锈病：选用抗病品种；发病初期喷施三唑酮、戊唑醇或苯醚甲环唑，间隔7~10天连喷2~3次；注意田间排水、降低湿度。",
  Gray_Leaf_Spot: "玉米灰斑病：收获后清除病残体减少越冬菌源；发病初期用嘧菌酯、吡唑醚菌酯或苯醚甲环唑喷雾，重点保护果穗上下叶片。",
  Healthy: "玉米长势健康，请继续保持良好水肥管理，定期巡查即可。"
};

Page({
  data: {
    photoPath: "",       // 预览图路径
    cameraMode: false,   // 是否打开相机取景
    loading: false,
    result: null,
    leaf: null,          // 绿叶预检结果
    noLeaf: false,       // 画面不是玉米叶片
    error: "",
    modelReady: false,
    modelInfo: ""
  },

  onLoad() {
    const info = "本地模型 · 输入 " + INPUT_SIZE + "×" + INPUT_SIZE + " · 测试准确率 " + TEST_ACC + "%";
    this.setData({ modelInfo: info });
    // 预热：提前把权重读入内存，首次识别更快
    loadModel()
      .then(() => this.setData({ modelReady: true }))
      .catch((e) => this.setData({ error: "模型加载失败：" + (e.message || e) }));
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

  // ---------- 本地识别（无网络请求） ----------
  async predict(filePath) {
    this.setData({ loading: true, noLeaf: false, leaf: null, error: "" });
    try {
      const res = await localPredict(filePath);

      // 绿叶预检未通过：不是玉米叶片，不做病害判定
      if (!res.ok) {
        this.setData({ leaf: res.leaf, noLeaf: true, result: null });
        return;
      }

      const top = res.top;
      this.setData({
        leaf: res.leaf,
        noLeaf: false,
        result: {
          class: top.cls,
          class_cn: CLASS_CN[top.cls] || top.cls,
          confidence: +(top.p * 100).toFixed(2),
          has_disease: top.cls !== "Healthy",
          advice: ADVICE[top.cls] || "",
          ms: res.ms,
          probs: res.probs.map((r) => ({
            class: r.cls,
            class_cn: CLASS_CN[r.cls] || r.cls,
            prob: +(r.p * 100).toFixed(2)
          }))
        }
      });
    } catch (e) {
      this.setData({ error: "识别失败：" + (e && e.message ? e.message : e) });
    } finally {
      this.setData({ loading: false });
    }
  },

  // 重拍/重新选择
  retake() {
    this.setData({ photoPath: "", result: null, leaf: null, noLeaf: false, error: "" });
  }
});
