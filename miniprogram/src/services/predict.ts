import Taro from '@tarojs/taro';

// 后端识别服务地址
// 真机调试时需改为运行 server/api.py 的电脑局域网 IP，例如 http://192.168.1.10:5000
// 微信开发者工具模拟器可直接用 http://localhost:5000
const API_BASE = 'http://localhost:5000';

export interface ProbItem {
  class: string;
  class_cn: string;
  prob: number;
}

export interface PredictResult {
  class: string;
  class_cn: string;
  confidence: number;
  has_disease: boolean;
  probs: ProbItem[];
}

/**
 * 上传玉米叶片图片并请求后端 ONNX 模型识别。
 * @param filePath 临时图片路径
 */
export const predictImage = (filePath: string): Promise<PredictResult> => {
  return new Promise((resolve, reject) => {
    console.log('[predict] 上传图片识别:', filePath);
    Taro.uploadFile({
      url: `${API_BASE}/predict`,
      filePath,
      name: 'file',
      success: (res) => {
        try {
          const data = JSON.parse(res.data);
          if (data.error) {
            console.error('[predict] 后端返回错误:', data.error);
            reject(new Error(data.error));
            return;
          }
          console.log('[predict] 识别结果:', data.class_cn, data.confidence + '%');
          resolve(data as PredictResult);
        } catch (e) {
          console.error('[predict] 响应解析失败:', e, res.data);
          reject(new Error('识别结果解析失败，请检查后端服务'));
        }
      },
      fail: (err) => {
        console.error('[predict] 上传失败:', err);
        reject(new Error('网络异常：无法连接识别服务，请检查后端是否启动及 API 地址'));
      }
    });
  });
};
