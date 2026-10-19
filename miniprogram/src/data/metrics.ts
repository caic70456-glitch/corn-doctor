// 指标与病害准确率数据（来源：玉米医生 RealSense 三维定位方案）
export interface MetricItem {
  value: string;
  label: string;
}

export const heroMetrics: MetricItem[] = [
  { value: '90.2%', label: '测试集整体准确率' },
  { value: '2 Hz', label: '推理/坐标发布频率' },
  { value: '115200', label: '串口波特率(8N1)' },
  { value: '<1%', label: '1m 内深度误差(D415)' }
];

export interface ClassAccuracy {
  name: string;
  en: string;
  acc: number;
}

export const classAccuracies: ClassAccuracy[] = [
  { name: '叶斑病', en: 'Blight', acc: 84.0 },
  { name: '普通锈病', en: 'Common_Rust', acc: 95.4 },
  { name: '灰斑病', en: 'Gray_Leaf_Spot', acc: 72.7 },
  { name: '健康', en: 'Healthy', acc: 99.1 }
];
