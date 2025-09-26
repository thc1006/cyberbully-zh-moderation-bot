#!/usr/bin/env python3
"""
生成資料增強的最終統計報告
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

def generate_comprehensive_report(
    original_file: str,
    augmented_file: str,
    augmentation_stats_file: str,
    validation_report_file: str,
    output_file: str
):
    """生成綜合的資料增強報告"""

    # 讀取所有相關文件
    original_df = pd.read_csv(original_file)
    augmented_df = pd.read_csv(augmented_file)

    with open(augmentation_stats_file, 'r', encoding='utf-8') as f:
        aug_stats = json.load(f)

    with open(validation_report_file, 'r', encoding='utf-8') as f:
        validation_report = json.load(f)

    # 生成綜合報告
    report = {
        "metadata": {
            "report_date": datetime.now().isoformat(),
            "original_dataset": original_file,
            "augmented_dataset": augmented_file,
            "augmentation_method": "中文霸凌偵測資料增強 (同義詞替換 + 句式變換)",
            "version": "1.0"
        },

        "dataset_overview": {
            "original": {
                "total_samples": len(original_df),
                "label_distribution": original_df['label'].value_counts().to_dict(),
                "topic_distribution": original_df['topic'].value_counts().to_dict(),
                "avg_text_length": float(original_df['TEXT'].str.len().mean()),
                "text_length_std": float(original_df['TEXT'].str.len().std())
            },
            "augmented": {
                "total_samples": len(augmented_df),
                "label_distribution": augmented_df['label'].value_counts().to_dict(),
                "topic_distribution": augmented_df['topic'].value_counts().to_dict(),
                "avg_text_length": float(augmented_df['TEXT'].str.len().mean()),
                "text_length_std": float(augmented_df['TEXT'].str.len().std())
            }
        },

        "augmentation_results": {
            "expansion_ratio": aug_stats["expansion_ratio"],
            "samples_generated": aug_stats["augmented_size"] - aug_stats["original_size"],
            "augmentation_parameters": aug_stats["parameters"],
            "label_balance_maintained": abs(
                (aug_stats["augmented_distribution"]["0"] / aug_stats["augmented_size"]) -
                (aug_stats["original_distribution"]["0"] / aug_stats["original_size"])
            ) < 0.01
        },

        "quality_assessment": {
            "duplicate_rate": validation_report["quality_analysis"]["duplicate_ratio"],
            "quality_issues": validation_report["quality_analysis"]["quality_issues"],
            "text_length_change": validation_report["distribution_comparison"]["length_statistics"]["mean_difference"],
            "recommendations": validation_report["recommendations"]
        },

        "technical_details": {
            "augmentation_techniques": [
                "同義詞替換 (詞彙級別多樣化)",
                "語氣詞添加 (語調變化)",
                "句式變換 (語法結構變化)",
                "輕微文本變化 (標點變化等)"
            ],
            "intensity_level": aug_stats["parameters"]["intensity"],
            "target_categories": ["race", "region", "gender"],
            "preservation_features": [
                "標籤分佈平衡",
                "主題分佈一致性",
                "語義內容完整性"
            ]
        },

        "usage_recommendations": {
            "training_readiness": True,
            "suggested_splits": {
                "train": 0.8,
                "validation": 0.1,
                "test": 0.1
            },
            "model_recommendations": [
                "chinese-roberta-wwm-ext (推薦用於語義理解)",
                "chinese-macbert-base (適合多任務學習)",
                "bert-base-chinese (基準模型)"
            ],
            "evaluation_metrics": [
                "macro F1-score",
                "precision per class",
                "recall per class",
                "confusion matrix analysis"
            ]
        },

        "file_paths": {
            "original_data": original_file,
            "augmented_data": augmented_file,
            "statistics": augmentation_stats_file,
            "validation_report": validation_report_file
        }
    }

    # 生成樣本展示
    print("生成樣本展示...")
    sample_augmented = augmented_df.iloc[len(original_df):].sample(n=5)
    sample_examples = []

    for _, row in sample_augmented.iterrows():
        sample_examples.append({
            "text": row['TEXT'],
            "label": int(row['label']),
            "topic": row['topic'],
            "length": len(row['TEXT'])
        })

    report["sample_examples"] = sample_examples

    # 保存報告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print("\n" + "="*60)
    print("              資料增強最終報告              ")
    print("="*60)

    print(f"\n[資料集概覽]:")
    print(f"   原始樣本數: {len(original_df):,}")
    print(f"   增強後樣本數: {len(augmented_df):,}")
    print(f"   擴充比例: {aug_stats['expansion_ratio']:.1f}x")
    print(f"   新增樣本: {aug_stats['augmented_size'] - aug_stats['original_size']:,}")

    print(f"\n[標籤分佈]:")
    for label in [0, 1]:
        orig_count = aug_stats["original_distribution"][str(label)]
        aug_count = aug_stats["augmented_distribution"][str(label)]
        orig_pct = orig_count / aug_stats["original_size"] * 100
        aug_pct = aug_count / aug_stats["augmented_size"] * 100
        print(f"   標籤 {label}: {orig_count:,} -> {aug_count:,} ({orig_pct:.1f}% -> {aug_pct:.1f}%)")

    print(f"\n[文本品質]:")
    print(f"   重複率: {validation_report['quality_analysis']['duplicate_ratio']:.2%}")
    print(f"   平均長度變化: {validation_report['distribution_comparison']['length_statistics']['mean_difference']:+.1f} 字符")

    print(f"\n[增強技術]:")
    for tech in report["technical_details"]["augmentation_techniques"]:
        print(f"   - {tech}")

    print(f"\n[品質評估]:")
    quality_issues = validation_report["quality_analysis"]["quality_issues"]
    issues_found = sum(quality_issues.values())
    if issues_found == 0:
        print("   [OK] 未發現品質問題")
    else:
        print(f"   [WARNING] 發現 {issues_found} 個潛在問題")

    print(f"\n[輸出檔案]:")
    print(f"   增強資料: {augmented_file}")
    print(f"   統計報告: {augmentation_stats_file}")
    print(f"   驗證報告: {validation_report_file}")
    print(f"   最終報告: {output_file}")

    print(f"\n[使用建議]:")
    for rec in validation_report["recommendations"]:
        print(f"   - {rec}")

    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="生成資料增強最終報告")
    parser.add_argument("--original", required=True, help="原始資料檔案")
    parser.add_argument("--augmented", required=True, help="增強資料檔案")
    parser.add_argument("--augmentation-stats", required=True, help="增強統計檔案")
    parser.add_argument("--validation-report", required=True, help="驗證報告檔案")
    parser.add_argument("--output", required=True, help="最終報告輸出檔案")

    args = parser.parse_args()

    generate_comprehensive_report(
        args.original,
        args.augmented,
        args.augmentation_stats,
        args.validation_report,
        args.output
    )

if __name__ == "__main__":
    main()