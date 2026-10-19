import React from 'react';
import { View, Text } from '@tarojs/components';
import styles from './index.module.scss';

interface SectionCardProps {
  /** 卡片标题（可选） */
  title?: string;
  /** 左侧编号徽章文本（可选，如 "01"） */
  badge?: string;
  children: React.ReactNode;
}

/** 通用章节卡片：白底 + 圆角 + 阴影 */
const SectionCard: React.FC<SectionCardProps> = ({ title, badge, children }) => {
  return (
    <View className={styles.card}>
      {(title || badge) && (
        <View className={styles.header}>
          {badge && <Text className={styles.badge}>{badge}</Text>}
          {title && <Text className={styles.title}>{title}</Text>}
        </View>
      )}
      {children}
    </View>
  );
};

export default SectionCard;
