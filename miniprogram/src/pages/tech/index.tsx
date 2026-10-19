import React from 'react';
import { View, Text } from '@tarojs/components';
import SectionCard from '@/components/SectionCard';
import StepItem from '@/components/StepItem';
import CodeBlock from '@/components/CodeBlock';
import { architectureFlow, axes, keyTechs, projectFormula, rosTopics } from '@/data/plan';
import styles from './index.module.scss';

/** 技术方案页：架构数据流 + 坐标系 + 关键技术 + 输出接口 */
const TechPage: React.FC = () => {
  return (
    <View className={styles.page}>
      {/* 系统架构数据流 */}
      <SectionCard title="系统架构与数据流" badge="03">
        <Text className={styles.intro}>
          三个 ROS 节点职责单一、通过话题解耦：驱动节点由官方包提供，detect_node 负责「识别 + 定位」，serial_bridge_node
          负责「下发」，任意一级都可独立替换调试。
        </Text>
        <View style={{ height: '24rpx' }} />
        {architectureFlow.map((s, i) => (
          <StepItem key={s.title} index={i + 1} title={s.title} desc={s.desc} tag={s.tag} last={i === architectureFlow.length - 1} />
        ))}
      </SectionCard>

      {/* 坐标系定义 */}
      <SectionCard title="坐标系定义" badge="04">
        <Text className={styles.intro}>/target_3d 采用相机光学坐标系（RealSense color 光学帧约定，与 OpenCV 一致），原点为彩色相机光心，单位米。</Text>
        <View style={{ height: '24rpx' }} />
        {axes.map((a) => (
          <View key={a.axis} className={styles.axisRow}>
            <View className={styles.axisBadge} style={{ background: a.color }}>
              <Text>{a.axis}</Text>
            </View>
            <Text className={styles.axisDesc}>{a.dir}</Text>
          </View>
        ))}
        <View className={styles.axisNote}>
          Z 即目标直线距离，可直接用于云台俯仰/偏航解算；需转换到基座坐标系时，仅在下位机或增加 tf2
          静态变换节点做外参旋转平移即可。
        </View>
      </SectionCard>

      {/* 关键技术 */}
      <SectionCard title="关键技术环节" badge="05">
        {keyTechs.map((t) => (
          <View key={t.no} className={styles.techItem}>
            <View style={{ display: 'flex', alignItems: 'center' }}>
              <Text className={styles.techNo}>{t.no}</Text>
              <Text className={styles.techTitle}>{t.title}</Text>
            </View>
            <Text className={styles.techDesc}>{t.desc}</Text>
          </View>
        ))}
        <CodeBlock caption="针孔模型反投影公式" lines={projectFormula} />
      </SectionCard>

      {/* ROS 话题 */}
      <SectionCard title="ROS 话题接口" badge="6.1">
        {rosTopics.map((t) => (
          <View key={t.topic} className={styles.topicRow}>
            <View className={styles.topicFirst}>
              <Text className={styles.topicName}>{t.topic}</Text>
              <Text className={`${styles.dirTag} ${t.dir === '发布' ? styles.dirPub : styles.dirSub}`}>{t.dir}</Text>
            </View>
            <View className={styles.topicMeta}>
              <Text className={styles.topicType}>{t.type}</Text>
              <Text className={styles.topicNote}>{t.note}</Text>
            </View>
          </View>
        ))}
      </SectionCard>

      {/* 串口协议 */}
      <SectionCard title="串口协议（对下位机）" badge="6.2">
        <Text className={styles.intro}>
          UART 默认 /dev/ttyUSB0，波特率 115200，8N1。每收到一帧坐标输出一行 ASCII 文本，以 \r\n 结尾，便于单片机逐行解析；单位为米，无效深度时不发送该帧。
        </Text>
        <View className={styles.serialLine}>X:0.123,Y:-0.045,Z:0.876</View>
        <Text className={styles.intro}>
          {'\n'}单片机侧解析：以 \n 分帧 → 查找 X: Y: Z: 关键字 → strtof 提取三个浮点数；超时未收到即视为目标丢失。
        </Text>
      </SectionCard>
    </View>
  );
};

export default TechPage;
