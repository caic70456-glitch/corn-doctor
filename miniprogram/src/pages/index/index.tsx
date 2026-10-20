import React, { useState } from 'react';
import { View, Text, Image, Button } from '@tarojs/components';
import Taro from '@tarojs/taro';
import { predictImage, PredictResult } from '@/services/predict';
import { getDiseaseInfo } from '@/data/diseases';
import styles from './index.module.scss';

type Status = 'idle' | 'loading' | 'success' | 'error';

const RecognizePage: React.FC = () => {
  const [imgPath, setImgPath] = useState<string>('');
  const [status, setStatus] = useState<Status>('idle');
  const [result, setResult] = useState<PredictResult | null>(null);
  const [errMsg, setErrMsg] = useState<string>('');

  const handleChoose = (source: 'camera' | 'album') => {
    setResult(null);
    setErrMsg('');
    Taro.chooseImage({
      count: 1,
      sourceType: [source],
      sizeType: ['compressed'],
      success: (res) => {
        const path = res.tempFilePaths[0];
        setImgPath(path);
        setStatus('loading');
        predictImage(path)
          .then((r) => {
            setResult(r);
            setStatus('success');
          })
          .catch((e) => {
            setErrMsg(e.message);
            setStatus('error');
          });
      },
      fail: () => {
        setStatus('idle');
      }
    });
  };

  const info = result ? getDiseaseInfo(result.class) : null;

  return (
    <View className={styles.page}>
      {/* 图片预览区 */}
      <View className={styles.previewBox}>
        {imgPath ? (
          <Image className={styles.previewImg} src={imgPath} mode="aspectFill" />
        ) : (
          <View className={styles.placeholder}>
            <Text className={styles.placeholderIcon}>🌽</Text>
            <Text className={styles.placeholderText}>拍照或选择玉米叶片图片</Text>
            <Text className={styles.placeholderHint}>支持叶斑病 / 普通锈病 / 灰斑病 / 健康 4 类识别</Text>
          </View>
        )}

        {status === 'loading' && (
          <View className={styles.loadingMask}>
            <View className={styles.spinner} />
            <Text className={styles.loadingText}>AI 识别中…</Text>
          </View>
        )}
      </View>

      {/* 操作按钮 */}
      <View className={styles.btnRow}>
        <Button className={styles.btnPrimary} onClick={() => handleChoose('camera')}>拍照识别</Button>
        <Button className={styles.btnGhost} onClick={() => handleChoose('album')}>相册选择</Button>
      </View>

      {/* 错误提示 */}
      {status === 'error' && (
        <View className={styles.errorBox}>
          <Text className={styles.errorText}>识别失败：{errMsg}</Text>
        </View>
      )}

      {/* 空状态下的拍摄建议 */}
      {status === 'idle' && (
        <View className={styles.tipCard}>
          <Text className={styles.tipTitle}>拍摄建议</Text>
          <View className={styles.tipRow}>
            <Text className={styles.tipDot}>1</Text>
            <Text className={styles.tipText}>让叶片占满画面，只拍一片叶子效果最好</Text>
          </View>
          <View className={styles.tipRow}>
            <Text className={styles.tipDot}>2</Text>
            <Text className={styles.tipText}>在自然光下拍摄，避免逆光和阴影遮挡</Text>
          </View>
          <View className={styles.tipRow}>
            <Text className={styles.tipDot}>3</Text>
            <Text className={styles.tipText}>尽量拍清病斑、锈斑等细节，避免模糊</Text>
          </View>
        </View>
      )}

      {/* 识别结果 */}
      {status === 'success' && result && info && (
        <View className={styles.resultCard}>
          {/* 结论头 */}
          <View className={`${styles.resultHead} ${info.isHealthy ? styles.headHealthy : styles.headSick}`}>
            <Text className={styles.resultLabel}>{info.isHealthy ? '✅ 叶片健康' : '⚠️ 检测到病害'}</Text>
            <Text className={styles.resultName}>{result.class_cn}</Text>
            <Text className={styles.resultConf}>置信度 {result.confidence}%</Text>
          </View>

          {/* 各类别概率 */}
          <View className={styles.probs}>
            <Text className={styles.probsTitle}>各类别概率</Text>
            {result.probs.map((p) => (
              <View key={p.class} className={styles.probRow}>
                <View className={styles.probName}>
                  <Text className={styles.probCn}>{p.class_cn}</Text>
                  <Text className={styles.probEn}>{p.class}</Text>
                </View>
                <View className={styles.probBarBg}>
                  <View
                    className={p.class === result.class ? styles.probBarTop : styles.probBar}
                    style={{ width: `${p.prob}%` }}
                  />
                </View>
                <Text className={styles.probVal}>{p.prob.toFixed(1)}%</Text>
              </View>
            ))}
          </View>

          {/* 症状与防治 */}
          <View className={styles.detailBlock}>
            <Text className={styles.detailTitle}>症状特征</Text>
            <Text className={styles.detailText}>{info.symptom}</Text>
          </View>
          <View className={styles.detailBlock}>
            <Text className={styles.detailTitle}>防治建议</Text>
            <Text className={styles.detailText}>{info.treatment}</Text>
          </View>
        </View>
      )}
    </View>
  );
};

export default RecognizePage;
