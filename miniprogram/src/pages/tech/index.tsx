import React from 'react';
import { View, Text } from '@tarojs/components';
import { diseaseKnowledge } from '@/data/diseases';
import styles from './index.module.scss';

const KnowledgePage: React.FC = () => {
  return (
    <View className={styles.page}>
      <View className={styles.intro}>
        <Text className={styles.introText}>
          本系统可识别玉米 4 类状态：叶斑病、普通锈病、灰斑病及健康。以下为各类病害的症状特征与防治建议，供田间巡查参考。
        </Text>
      </View>

      {diseaseKnowledge.map((d) => (
        <View
          key={d.class}
          className={`${styles.card} ${d.isHealthy ? styles.cardHealthy : styles.cardSick}`}
        >
          <View className={styles.cardHead}>
            <Text className={styles.cardName}>{d.class_cn}</Text>
            <Text className={styles.cardEn}>{d.class}</Text>
            <View className={`${styles.tag} ${d.isHealthy ? styles.tagHealthy : styles.tagSick}`}>
              <Text>{d.isHealthy ? '健康' : '病害'}</Text>
            </View>
          </View>

          <View className={styles.section}>
            <Text className={styles.sectionTitle}>症状特征</Text>
            <Text className={styles.sectionText}>{d.symptom}</Text>
          </View>
          <View className={styles.section}>
            <Text className={styles.sectionTitle}>防治建议</Text>
            <Text className={styles.sectionText}>{d.treatment}</Text>
          </View>
        </View>
      ))}
    </View>
  );
};

export default KnowledgePage;
