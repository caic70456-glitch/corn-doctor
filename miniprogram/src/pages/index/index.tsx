import React from 'react';
import { View, Text } from '@tarojs/components';
import SectionCard from '@/components/SectionCard';
import MetricCard from '@/components/MetricCard';
import { heroMetrics, classAccuracies } from '@/data/metrics';
import styles from './index.module.scss';

/** 概览首页：项目信息 + 核心指标 + 病害准确率 + 方案要点 */
const HomePage: React.FC = () => {
  return (
    <View className={styles.page}>
      {/* Hero */}
      <View className={styles.hero}>
        <Text className={styles.tagline}>AIC-2026-46559581 · AI+农业赛道</Text>
        <View>
          <Text className={styles.heroTitle}>玉米医生 · RealSense 三维定位方案</Text>
        </View>
        <Text className={styles.heroSub}>
          树莓派 4B/5 · Ubuntu 22.04 · ROS 2 Humble · Intel RealSense D415 · MobileNetV2 (ONNX)
        </Text>
      </View>

      {/* 核心指标 */}
      <View className={styles.metricGrid}>
        {heroMetrics.map((m) => (
          <View key={m.label} className={styles.metricCell}>
            <MetricCard value={m.value} label={m.label} />
          </View>
        ))}
      </View>

      {/* 方案概述 */}
      <SectionCard title="方案概述">
        <Text className={styles.introText}>
          在树莓派边缘端接入 RealSense 深度相机，对玉米叶片完成病害识别的同时，输出目标叶片在相机坐标系下的三维坐标
          (X, Y, Z)，经串口实时下发给下位单片机，引导云台 / 末端执行器对准病叶。
        </Text>
        <View className={styles.keyPoint}>
          整图分类（生病/健康 + 病害种类） + 画面中心点深度反投影（三维坐标）
        </View>
        <Text className={styles.introText}>
          操作者将病叶对准画面中心，系统即输出该点三维坐标；方案与分类模型完全配套，计算量小，可实时运行。
        </Text>
      </SectionCard>

      {/* 测试集准确率 */}
      <SectionCard title="测试集准确率（1257 张）">
        {classAccuracies.map((c) => (
          <View key={c.en} className={styles.barRow}>
            <View className={styles.barTop}>
              <View>
                <Text className={styles.barName}>{c.name}</Text>
                <Text className={styles.barEn}>{c.en}</Text>
              </View>
              <Text className={styles.barVal}>{c.acc.toFixed(1)}%</Text>
            </View>
            <View className={styles.barBg}>
              <View className={styles.barFill} style={{ width: `${c.acc}%` }} />
            </View>
          </View>
        ))}
      </SectionCard>

      <Text className={styles.footTip}>技术架构 → 「技术」Tab · 上手部署 → 「部署」Tab</Text>
    </View>
  );
};

export default HomePage;
