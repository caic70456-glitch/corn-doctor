import React from 'react';
import { View, Text } from '@tarojs/components';
import Taro from '@tarojs/taro';
import styles from './index.module.scss';

interface CodeBlockProps {
  /** 代码行数组（按行渲染，保留空行） */
  lines: string[];
  /** 可选标题（如步骤注释） */
  caption?: string;
}

/** 代码块：深色终端风格，点击整体复制 */
const CodeBlock: React.FC<CodeBlockProps> = ({ lines, caption }) => {
  const handleCopy = () => {
    const text = lines.join('\n');
    Taro.setClipboardData({
      data: text,
      success: () => {
        console.log('[CodeBlock] 已复制代码, 行数:', lines.length);
      },
      fail: (err) => {
        console.error('[CodeBlock] 复制失败:', err);
        Taro.showToast({ title: '复制失败', icon: 'none' });
      }
    });
  };

  return (
    <View className={styles.block} onClick={handleCopy}>
      <View className={styles.head}>
        <Text className={styles.caption}>{caption || '命令 · 点击复制'}</Text>
        <Text className={styles.copyBtn}>复制</Text>
      </View>
      <View className={styles.codeArea}>
        {lines.map((line, i) => (
          <Text key={i} className={styles.line}>
            {line || ' '}
          </Text>
        ))}
      </View>
    </View>
  );
};

export default CodeBlock;
