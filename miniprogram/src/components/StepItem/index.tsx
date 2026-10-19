import React from 'react';
import { View, Text } from '@tarojs/components';
import styles from './index.module.scss';

interface StepItemProps {
  /** 步骤序号（从 1 开始） */
  index: number;
  title: string;
  desc: string;
  /** 右侧标签（可选，如 "硬件"） */
  tag?: string;
  /** 是否为最后一步（不显示连接线） */
  last?: boolean;
}

/** 步骤条目：左侧圆点编号 + 竖向连接线 */
const StepItem: React.FC<StepItemProps> = ({ index, title, desc, tag, last }) => {
  return (
    <View className={styles.step}>
      <View className={styles.leftCol}>
        <View className={styles.dot}>
          <Text className={styles.dotText}>{index}</Text>
        </View>
        {!last && <View className={styles.line} />}
      </View>
      <View className={styles.body}>
        <View className={styles.titleRow}>
          <Text className={styles.title}>{title}</Text>
          {tag && <Text className={styles.tag}>{tag}</Text>}
        </View>
        <Text className={styles.desc}>{desc}</Text>
      </View>
    </View>
  );
};

export default StepItem;
