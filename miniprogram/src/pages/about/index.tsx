import React from 'react';
import { View, Text } from '@tarojs/components';
import Taro from '@tarojs/taro';
import styles from './index.module.scss';

const REPO = 'https://github.com/caic70456-glitch/corn-doctor';
const PAGE = 'https://caic70456-glitch.github.io/corn-doctor/';

const AboutPage: React.FC = () => {
  const copy = (text: string) => {
    Taro.setClipboardData({ data: text });
  };

  return (
    <View className={styles.page}>
      {/* 项目信息 */}
      <View className={styles.infoCard}>
        <Text className={styles.infoName}>玉米医生</Text>
        <Text className={styles.infoId}>AIC-2026-46559581 · AI+农业赛道</Text>
        <Text className={styles.infoDesc}>
          玉米病害识别小程序。前端拍照上传，后端 MobileNetV2 ONNX 模型推理，输出 4 类结果（叶斑病 / 普通锈病 / 灰斑病 / 健康）。
        </Text>
      </View>

      {/* 模型信息 */}
      <View className={styles.card}>
        <Text className={styles.cardTitle}>识别模型</Text>
        <View className={styles.kvRow}><Text className={styles.k}>模型</Text><Text className={styles.v}>MobileNetV2（迁移学习）</Text></View>
        <View className={styles.kvRow}><Text className={styles.k}>格式</Text><Text className={styles.v}>ONNX（已含 softmax）</Text></View>
        <View className={styles.kvRow}><Text className={styles.k}>输入</Text><Text className={styles.v}>224×224×3 RGB</Text></View>
        <View className={styles.kvRow}><Text className={styles.k}>测试准确率</Text><Text className={styles.v}>90.2%</Text></View>
        <View className={styles.kvRow}><Text className={styles.k}>预处理</Text><Text className={styles.v}>Resize → /255 → ImageNet 归一化</Text></View>
      </View>

      {/* 后端配置 */}
      <View className={styles.card}>
        <Text className={styles.cardTitle}>后端识别服务</Text>
        <Text className={styles.cardText}>
          本小程序需配合后端服务使用。在项目根目录运行：
        </Text>
        <View className={styles.codeBox}>
          <Text className={styles.code}>pip install flask pillow numpy onnxruntime</Text>
          <Text className={styles.code}>python server/api.py</Text>
        </View>
        <Text className={styles.cardText}>
          服务默认监听 0.0.0.0:5000。{'\n'}
          修改 API 地址：<Text className={styles.highlight}>src/services/predict.ts</Text> 中的{' '}
          <Text className={styles.highlight}>API_BASE</Text>，真机调试时改为电脑局域网 IP。
        </Text>
      </View>

      {/* 相关链接 */}
      <View className={styles.card}>
        <Text className={styles.cardTitle}>相关链接</Text>
        <View className={styles.linkBtn} onClick={() => copy(REPO)}>
          <View style={{ flex: 1, minWidth: 0 }}>
            <Text className={styles.linkName}>GitHub 仓库（点击复制）</Text>
            <Text className={styles.linkUrl}>{REPO}</Text>
          </View>
          <Text className={styles.linkArrow}>›</Text>
        </View>
        <View className={styles.linkBtn} onClick={() => copy(PAGE)}>
          <View style={{ flex: 1, minWidth: 0 }}>
            <Text className={styles.linkName}>方案网页版（点击复制）</Text>
            <Text className={styles.linkUrl}>{PAGE}</Text>
          </View>
          <Text className={styles.linkArrow}>›</Text>
        </View>
      </View>
    </View>
  );
};

export default AboutPage;
