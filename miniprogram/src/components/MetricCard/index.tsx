import React from 'react';
import { View, Text } from '@tarojs/components';
import styles from './index.module.scss';

interface MetricCardProps {
  value: string;
  label: string;
}

/** 指标卡：绿色渐变底 + 玉米金数值 */
const MetricCard: React.FC<MetricCardProps> = ({ value, label }) => {
  return (
    <View className={styles.metric}>
      <Text className={styles.value}>{value}</Text>
      <Text className={styles.label}>{label}</Text>
    </View>
  );
};

export default MetricCard;
