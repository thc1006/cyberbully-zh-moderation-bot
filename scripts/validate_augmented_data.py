#!/usr/bin/env python3
"""
驗證增強資料品質的腳本
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_text_quality(df: pd.DataFrame) -> dict:
    """分析文本品質"""

    # 文本長度統計
    text_lengths = df['TEXT'].str.len()
    length_stats = {
        'mean': float(text_lengths.mean()),
        'std': float(text_lengths.std()),
        'min': int(text_lengths.min()),
        'max': int(text_lengths.max()),
        'median': float(text_lengths.median()),
        'q25': float(text_lengths.quantile(0.25)),
        'q75': float(text_lengths.quantile(0.75))
    }

    # 檢查重複文本
    duplicates = df['TEXT'].duplicated().sum()
    duplicate_ratio = duplicates / len(df)

    # 檢查空文本或過短文本
    empty_texts = df['TEXT'].isna().sum()
    short_texts = (df['TEXT'].str.len() < 5).sum()

    # 檢查特殊字符比例
    special_char_pattern = r'[^\u4e00-\u9fff\u3400-\u4dbf\w\s\.\!\?\,\;\:\'\"\-\(\)]'
    special_char_texts = df['TEXT'].str.contains(special_char_pattern, regex=True, na=False).sum()

    return {
        'text_length_stats': length_stats,
        'total_samples': len(df),
        'duplicates': int(duplicates),
        'duplicate_ratio': float(duplicate_ratio),
        'empty_texts': int(empty_texts),
        'short_texts': int(short_texts),
        'special_char_texts': int(special_char_texts),
        'quality_issues': {
            'duplicate_ratio_high': bool(duplicate_ratio > 0.1),
            'too_many_empty': bool(empty_texts > 10),
            'too_many_short': bool(short_texts > 100),
            'too_many_special_chars': bool(special_char_texts > len(df) * 0.05)
        }
    }

def compare_distributions(original_df: pd.DataFrame, augmented_df: pd.DataFrame) -> dict:
    """比較原始和增強資料的分佈"""

    # 標籤分佈比較
    orig_label_dist = original_df['label'].value_counts(normalize=True).to_dict()
    aug_label_dist = augmented_df['label'].value_counts(normalize=True).to_dict()

    # 主題分佈比較
    orig_topic_dist = original_df['topic'].value_counts(normalize=True).to_dict()
    aug_topic_dist = augmented_df['topic'].value_counts(normalize=True).to_dict()

    # 文本長度分佈比較
    orig_length_stats = {
        'mean': float(original_df['TEXT'].str.len().mean()),
        'std': float(original_df['TEXT'].str.len().std())
    }
    aug_length_stats = {
        'mean': float(augmented_df['TEXT'].str.len().mean()),
        'std': float(augmented_df['TEXT'].str.len().std())
    }

    return {
        'label_distribution': {
            'original': orig_label_dist,
            'augmented': aug_label_dist,
            'difference': {k: aug_label_dist.get(k, 0) - orig_label_dist.get(k, 0)
                          for k in set(list(orig_label_dist.keys()) + list(aug_label_dist.keys()))}
        },
        'topic_distribution': {
            'original': orig_topic_dist,
            'augmented': aug_topic_dist,
            'difference': {k: aug_topic_dist.get(k, 0) - orig_topic_dist.get(k, 0)
                          for k in set(list(orig_topic_dist.keys()) + list(aug_topic_dist.keys()))}
        },
        'length_statistics': {
            'original': orig_length_stats,
            'augmented': aug_length_stats,
            'mean_difference': aug_length_stats['mean'] - orig_length_stats['mean']
        }
    }

def sample_augmented_examples(df: pd.DataFrame, n_samples: int = 10) -> list:
    """採樣增強樣本進行人工檢查"""

    # 假設原始資料在前面，增強資料在後面
    total_original = len(df) // 3  # 假設增強了3倍
    augmented_samples = df.iloc[total_original:].sample(n=min(n_samples, len(df) - total_original))

    examples = []
    for _, row in augmented_samples.iterrows():
        examples.append({
            'text': row['TEXT'],
            'label': row['label'],
            'topic': row['topic'],
            'length': len(row['TEXT'])
        })

    return examples

def detect_augmentation_patterns(df: pd.DataFrame) -> dict:
    """檢測增強模式"""

    # 尋找可能的增強模式
    patterns = {
        'exclamation_marks': int(df['TEXT'].str.count('!').sum()),
        'question_marks': int(df['TEXT'].str.count(r'\?').sum()),
        'multiple_periods': int(df['TEXT'].str.count(r'\.\.').sum()),
        'tone_words': 0,
        'template_phrases': 0
    }

    # 檢查語氣詞
    tone_words = ['真的', '超級', '非常', '特別', '極其', '難道', '莫非', '絕對', '一定']
    for word in tone_words:
        patterns['tone_words'] += int(df['TEXT'].str.count(word).sum())

    # 檢查模板短語
    template_phrases = ['我覺得', '就是', '明明', '根本', '簡直', '怎麼']
    for phrase in template_phrases:
        patterns['template_phrases'] += int(df['TEXT'].str.count(phrase).sum())

    return patterns

def generate_validation_report(original_file: str, augmented_file: str, output_file: str):
    """生成驗證報告"""

    print("讀取資料...")
    original_df = pd.read_csv(original_file)
    augmented_df = pd.read_csv(augmented_file)

    print("分析文本品質...")
    quality_analysis = analyze_text_quality(augmented_df)

    print("比較分佈...")
    distribution_comparison = compare_distributions(original_df, augmented_df)

    print("採樣增強例子...")
    sample_examples = sample_augmented_examples(augmented_df, n_samples=20)

    print("檢測增強模式...")
    augmentation_patterns = detect_augmentation_patterns(augmented_df)

    # 生成綜合報告
    report = {
        'summary': {
            'original_size': len(original_df),
            'augmented_size': len(augmented_df),
            'expansion_ratio': len(augmented_df) / len(original_df),
            'validation_timestamp': pd.Timestamp.now().isoformat()
        },
        'quality_analysis': quality_analysis,
        'distribution_comparison': distribution_comparison,
        'augmentation_patterns': augmentation_patterns,
        'sample_examples': sample_examples,
        'recommendations': []
    }

    # 生成建議
    recommendations = []

    if quality_analysis['quality_issues']['duplicate_ratio_high']:
        recommendations.append("檢測到高重複率，建議檢查增強算法避免產生過多重複文本")

    if quality_analysis['quality_issues']['too_many_short']:
        recommendations.append("檢測到過多短文本，建議增加最小長度限制")

    if abs(distribution_comparison['label_distribution']['difference'].get('0', 0)) > 0.05:
        recommendations.append("標籤分佈發生較大變化，建議檢查標籤平衡")

    if distribution_comparison['length_statistics']['mean_difference'] > 10:
        recommendations.append("平均文本長度明顯增加，可能過度增強")
    elif distribution_comparison['length_statistics']['mean_difference'] < -5:
        recommendations.append("平均文本長度明顯減少，可能增強品質不佳")

    if not recommendations:
        recommendations.append("資料增強品質良好，可以用於訓練")

    report['recommendations'] = recommendations

    # 保存報告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n=== 資料增強驗證報告 ===")
    print(f"原始大小: {len(original_df):,}")
    print(f"增強後大小: {len(augmented_df):,}")
    print(f"擴充比例: {len(augmented_df) / len(original_df):.2f}x")
    print(f"重複率: {quality_analysis['duplicate_ratio']:.2%}")
    print(f"平均文本長度變化: {distribution_comparison['length_statistics']['mean_difference']:.1f}")

    print("\n=== 標籤分佈比較 ===")
    for label, diff in distribution_comparison['label_distribution']['difference'].items():
        print(f"標籤 {label}: {diff:+.3f}")

    print("\n=== 建議 ===")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    print(f"\n詳細報告已保存至: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="驗證增強資料品質")
    parser.add_argument("--original", required=True, help="原始資料檔案")
    parser.add_argument("--augmented", required=True, help="增強資料檔案")
    parser.add_argument("--output", required=True, help="驗證報告輸出檔案")

    args = parser.parse_args()

    generate_validation_report(args.original, args.augmented, args.output)

if __name__ == "__main__":
    main()