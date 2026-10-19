import React from 'react';
import { View, Text } from '@tarojs/components';
import SectionCard from '@/components/SectionCard';
import CodeBlock from '@/components/CodeBlock';
import { deploySteps, paramTable, errorSources } from '@/data/plan';
import styles from './index.module.scss';

/** 部署指南页：环境 + 部署命令（可复制）+ 参数表 + 精度验证 + 误差对策 */
const DeployPage: React.FC = () => {
  return (
    <View className={styles.page}>
      {/* 环境说明 */}
      <SectionCard title="运行环境">
        <Text className={styles.theory}>
          硬件为树莓派 4B/5（≥4GB）+ Intel RealSense D415（兼容 D435/D435i/D455，其他深度相机重映射话题即可接入）+ STM32
          等任意单片机下位机。ONNX Runtime CPU 推理单帧约 0.2–0.5s。
        </Text>
        <View className={styles.envTags}>
          <Text className={styles.envTag}>Ubuntu 22.04</Text>
          <Text className={styles.envTag}>ROS 2 Humble</Text>
          <Text className={styles.envTag}>realsense2_camera</Text>
          <Text className={styles.envTag}>onnxruntime</Text>
          <Text className={styles.envTag}>pyserial</Text>
        </View>
      </SectionCard>

      {/* 部署步骤 */}
      <SectionCard title="部署与运行步骤" badge="08">
        {deploySteps.map((s) => (
          <View key={s.comment}>
            <CodeBlock caption={s.comment} lines={s.cmd} />
          </View>
        ))}
      </SectionCard>

      {/* 参数配置 */}
      <SectionCard title="参数配置表" badge="07">
        {paramTable.map((p) => (
          <View key={p.name} className={styles.paramRow}>
            <View className={styles.paramFirst}>
              <Text className={styles.paramName}>{p.name}</Text>
              <Text className={styles.paramValue}>{p.value}</Text>
            </View>
            <Text className={styles.paramMeta}>
              {p.node} · {p.note}
            </Text>
          </View>
        ))}
      </SectionCard>

      {/* 精度分析 */}
      <SectionCard title="精度分析与验证" badge="09">
        <Text className={styles.theory}>
          理论精度：D415 在 1m 内深度误差 &lt;1%，2m 约 2%；横向误差由内参决定（0.5m 处 1 像素约 0.36mm），实际主要误差来自手抖引起的瞄准偏移而非算法。
        </Text>
        <View className={styles.callout}>
          <Text className={styles.calloutIcon}>📏</Text>
          <Text className={styles.calloutText}>
            建议实测流程：在 0.5 / 1.0 / 1.5m 三个距离用卷尺对照 /target_3d 的 Z 值，各测 10
            次记录均值与标准差，验证精度满足云台瞄准需求。
          </Text>
        </View>
      </SectionCard>

      {/* 误差对策 */}
      <SectionCard title="误差来源与对策">
        {errorSources.map((e) => (
          <View key={e.source} className={styles.errRow}>
            <View className={styles.errFirst}>
              <Text className={styles.errSource}>{e.source}</Text>
              <Text className={styles.errImpact}>{e.impact}</Text>
            </View>
            <Text className={styles.errFix}>{e.fix}</Text>
          </View>
        ))}
      </SectionCard>
    </View>
  );
};

export default DeployPage;
