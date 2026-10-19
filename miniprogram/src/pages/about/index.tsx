import React from 'react';
import { View, Text } from '@tarojs/components';
import Taro from '@tarojs/taro';
import SectionCard from '@/components/SectionCard';
import { systemModules, risks, LINK_REPO, LINK_PAGE } from '@/data/plan';
import styles from './index.module.scss';

const MODULE_ICONS = ['🌾', '💻', '🤖'];

/** 关于页：项目信息 + 系统组成 + 风险扩展 + 相关链接 */
const AboutPage: React.FC = () => {
  const copyLink = (url: string, label: string) => {
    Taro.setClipboardData({
      data: url,
      success: () => {
        console.log('[About] 已复制链接:', label);
        Taro.showToast({ title: '链接已复制', icon: 'success' });
      },
      fail: (err) => {
        console.error('[About] 复制链接失败:', err);
        Taro.showToast({ title: '复制失败', icon: 'none' });
      }
    });
  };

  return (
    <View className={styles.page}>
      {/* 项目信息 */}
      <View className={styles.infoCard}>
        <Text className={styles.infoName}>玉米医生</Text>
        <Text className={styles.infoId}>AIC-2026-46559581 · AI+农业赛道</Text>
        <Text className={styles.infoDesc}>
          对玉米四大类病害（叶斑病 / 普通锈病 / 灰斑病 / 健康）进行识别的完整落地系统：PC 端训练与网页应用 + 树莓派
          ROS 2 边缘部署（深度相机三维定位 + USB 相机手持机）。测试集整体准确率 90.2%。
        </Text>
      </View>

      {/* 系统组成 */}
      <SectionCard title="系统组成">
        {systemModules.map((m, i) => (
          <View key={m.name} className={styles.moduleItem}>
            <View className={styles.moduleIcon}>
              <Text>{MODULE_ICONS[i]}</Text>
            </View>
            <View style={{ flex: 1, minWidth: 0 }}>
              <Text className={styles.moduleName}>{m.name}</Text>
              <Text className={styles.moduleDesc}>{m.desc}</Text>
            </View>
          </View>
        ))}
      </SectionCard>

      {/* 风险与扩展 */}
      <SectionCard title="风险与后续扩展">
        {risks.map((r) => (
          <View key={r.no} className={styles.riskItem}>
            <View className={styles.riskHead}>
              <Text className={styles.riskNo}>{r.no}</Text>
              <Text className={styles.riskTitle}>{r.title}</Text>
            </View>
            <Text className={styles.riskDesc}>{r.desc}</Text>
          </View>
        ))}
      </SectionCard>

      {/* 相关链接 */}
      <SectionCard title="相关链接">
        <View className={styles.linkBtn} onClick={() => copyLink(LINK_REPO, 'GitHub 仓库')}>
          <View style={{ flex: 1, minWidth: 0 }}>
            <Text className={styles.linkName}>GitHub 仓库（点击复制）</Text>
            <Text className={styles.linkUrl}>{LINK_REPO}</Text>
          </View>
          <Text className={styles.linkArrow}>›</Text>
        </View>
        <View className={styles.linkBtn} onClick={() => copyLink(LINK_PAGE, '网页版方案')}>
          <View style={{ flex: 1, minWidth: 0 }}>
            <Text className={styles.linkName}>网页版方案（点击复制）</Text>
            <Text className={styles.linkUrl}>{LINK_PAGE}</Text>
          </View>
          <Text className={styles.linkArrow}>›</Text>
        </View>
      </SectionCard>
    </View>
  );
};

export default AboutPage;
