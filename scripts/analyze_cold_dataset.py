#!/usr/bin/env python3
"""
分析 COLD 資料集的詳細統計資訊
"""

import pandas as pd
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_cold_dataset():
    """分析 COLD 資料集"""

    # 設定路徑
    data_dir = Path("C:/Users/thc1006/Desktop/dev/cyberbully-zh-moderation-bot/data/raw/cold/COLDataset")
    output_dir = Path("C:/Users/thc1006/Desktop/dev/cyberbully-zh-moderation-bot/data/processed")
    output_dir.mkdir(exist_ok=True)

    # 讀取資料
    train_df = pd.read_csv(data_dir / "train.csv")
    dev_df = pd.read_csv(data_dir / "dev.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    print("=== COLD 資料集統計分析 ===\n")

    # 基本統計
    print("## 基本統計")
    print(f"訓練集: {len(train_df):,} 筆")
    print(f"驗證集: {len(dev_df):,} 筆")
    print(f"測試集: {len(test_df):,} 筆")
    print(f"總計: {len(train_df) + len(dev_df) + len(test_df):,} 筆\n")

    # 分析每個檔案的欄位結構
    print("## 欄位結構")
    print("訓練集欄位:", list(train_df.columns))
    print("驗證集欄位:", list(dev_df.columns))
    print("測試集欄位:", list(test_df.columns))
    print()

    # 檢查是否有缺失值
    print("## 缺失值檢查")
    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"{name} 缺失值:")
            print(missing[missing > 0])
        else:
            print(f"{name}: 無缺失值")
    print()

    # 標籤分佈分析
    print("## 標籤分佈 (label)")
    stats = {}

    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        label_counts = df['label'].value_counts().sort_index()
        stats[name] = {
            'total': len(df),
            'toxic': int(label_counts.get(1, 0)),
            'non_toxic': int(label_counts.get(0, 0)),
            'toxic_ratio': label_counts.get(1, 0) / len(df)
        }

        print(f"{name}:")
        print(f"  非毒性 (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df):.2%})")
        print(f"  毒性 (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df):.2%})")
        print()

    # 主題分佈分析
    print("## 主題分佈 (topic)")
    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        topic_counts = df['topic'].value_counts()
        print(f"{name}:")
        for topic, count in topic_counts.items():
            print(f"  {topic}: {count:,} ({count/len(df):.2%})")
        print()

    # 分析測試集的細粒度標籤 (如果存在)
    if 'fine-grained-label' in test_df.columns:
        print("## 測試集細粒度標籤分佈")
        fine_counts = test_df['fine-grained-label'].value_counts().sort_index()
        for label, count in fine_counts.items():
            print(f"  級別 {label}: {count:,} ({count/len(test_df):.2%})")
        print()

    # 文本長度分析
    print("## 文本長度統計")
    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        df['text_length'] = df['TEXT'].str.len()
        length_stats = df['text_length'].describe()
        print(f"{name}:")
        print(f"  平均長度: {length_stats['mean']:.1f}")
        print(f"  中位數: {length_stats['50%']:.1f}")
        print(f"  最短: {length_stats['min']:.0f}")
        print(f"  最長: {length_stats['max']:.0f}")
        print()

    # 保存統計結果
    summary_stats = {
        'dataset_info': {
            'train_size': len(train_df),
            'dev_size': len(dev_df),
            'test_size': len(test_df),
            'total_size': len(train_df) + len(dev_df) + len(test_df),
            'columns': {
                'train': list(train_df.columns),
                'dev': list(dev_df.columns),
                'test': list(test_df.columns)
            }
        },
        'label_distribution': stats,
        'topic_distribution': {
            name: df['topic'].value_counts().to_dict()
            for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]
        },
        'text_length_stats': {
            name: {
                'mean': float(df['TEXT'].str.len().mean()),
                'std': float(df['TEXT'].str.len().std()),
                'min': int(df['TEXT'].str.len().min()),
                'max': int(df['TEXT'].str.len().max()),
                'median': float(df['TEXT'].str.len().median())
            }
            for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]
        }
    }

    # 如果有細粒度標籤，也加入統計
    if 'fine-grained-label' in test_df.columns:
        summary_stats['fine_grained_distribution'] = test_df['fine-grained-label'].value_counts().to_dict()

    # 保存統計結果到 JSON
    with open(output_dir / "cold_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    print(f"統計結果已保存至: {output_dir / 'cold_analysis.json'}")

    return summary_stats, train_df, dev_df, test_df

if __name__ == "__main__":
    stats, train_df, dev_df, test_df = analyze_cold_dataset()