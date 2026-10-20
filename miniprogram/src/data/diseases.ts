// 玉米四类病害知识（症状 + 防治建议）
export interface DiseaseInfo {
  class: string;
  class_cn: string;
  isHealthy: boolean;
  symptom: string;
  treatment: string;
}

export const diseaseKnowledge: DiseaseInfo[] = [
  {
    class: 'Blight',
    class_cn: '叶斑病',
    isHealthy: false,
    symptom: '叶片出现椭圆形或梭形褐色病斑，病斑中央灰白色、边缘深褐色，严重时病斑汇合导致叶片枯死。',
    treatment: '及时清除病残体并销毁；发病初期喷施 75% 百菌清或 50% 多菌灵可湿性粉剂，间隔 7~10 天一次。'
  },
  {
    class: 'Common_Rust',
    class_cn: '普通锈病',
    isHealthy: false,
    symptom: '叶片正反面出现圆形或椭圆形红褐色疱状夏孢子堆，表皮破裂后散出铁锈色粉末，后期出现黑色冬孢子堆。',
    treatment: '选用抗病品种；发病初期喷施 25% 三唑酮可湿性粉剂或 12.5% 烯唑醇，均匀喷洒叶片正反面。'
  },
  {
    class: 'Gray_Leaf_Spot',
    class_cn: '灰斑病',
    isHealthy: false,
    symptom: '叶片出现与叶脉平行的矩形灰褐色病斑，初期为淡褐色小点，后扩展为长条形，严重时整片叶干枯。',
    treatment: '合理密植、增施磷钾肥；发病初期喷施 50% 退菌特或 70% 甲基托布津，注意喷匀叶背。'
  },
  {
    class: 'Healthy',
    class_cn: '健康',
    isHealthy: true,
    symptom: '叶片色泽正常、无病斑、无霉层，植株生长健壮。',
    treatment: '保持田间通风透光，合理施肥浇水，定期巡查即可。'
  }
];

export const getDiseaseInfo = (cls: string): DiseaseInfo | undefined =>
  diseaseKnowledge.find((d) => d.class === cls);
