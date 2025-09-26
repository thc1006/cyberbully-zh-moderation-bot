#!/usr/bin/env python3
"""
重新標註資料集腳本
使用改進的標籤映射邏輯重新標註所有資料集
"""

import os
import sys
import json
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cyberpuppy.labeling.improved_label_map import ImprovedLabelMapper
from src.cyberpuppy.labeling.label_map import LabelMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetRelabeler:
    """資料集重新標註器"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.improved_mapper = ImprovedLabelMapper()
        self.original_mapper = LabelMapper()

        # 確保輸出目錄存在
        self.output_dir = self.data_dir / "processed" / "relabeled"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reports_dir = self.data_dir / "processed" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def relabel_cold_dataset(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        重新標註 COLD 資料集

        Args:
            input_path: 輸入檔案路徑
            output_path: 輸出檔案路徑

        Returns:
            重新標註結果統計
        """
        logger.info(f"開始重新標註 COLD 資料集: {input_path}")

        # 讀取原始資料
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.json'):
            df = pd.read_json(input_path)
        else:
            raise ValueError(f"不支援的檔案格式: {input_path}")

        # 檢查必要欄位
        required_columns = ['text', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要欄位: {missing_columns}")

        # 原始標籤統計
        original_labels = df['label'].tolist()
        texts = df['text'].tolist()

        logger.info(f"原始資料集大小: {len(df)} 筆")
        logger.info(f"原始標籤分佈: {pd.Series(original_labels).value_counts().to_dict()}")

        # 使用原始映射
        original_unified = [self.original_mapper.from_cold_to_unified(label) for label in original_labels]

        # 使用改進映射
        improved_unified = self.improved_mapper.batch_improve_cold_labels(original_labels, texts)

        # 創建新的資料框
        new_df = df.copy()

        # 添加改進後的標籤欄位
        new_df['original_toxicity'] = [label.toxicity.value for label in original_unified]
        new_df['original_bullying'] = [label.bullying.value for label in original_unified]

        new_df['improved_toxicity'] = [label.toxicity.value for label in improved_unified]
        new_df['improved_bullying'] = [label.bullying.value for label in improved_unified]
        new_df['improved_toxicity_score'] = [label.toxicity_score for label in improved_unified]
        new_df['improved_bullying_score'] = [label.bullying_score for label in improved_unified]
        new_df['improved_role'] = [label.role.value for label in improved_unified]
        new_df['improved_emotion'] = [label.emotion.value for label in improved_unified]
        new_df['improved_emotion_intensity'] = [label.emotion_intensity for label in improved_unified]

        # 保存結果
        if output_path is None:
            output_path = self.output_dir / f"cold_relabeled_{Path(input_path).stem}.csv"

        new_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"重新標註後的資料已保存到: {output_path}")

        # 生成比較報告
        comparison_report = self.improved_mapper.compare_with_original(original_labels, texts)

        return {
            'input_path': input_path,
            'output_path': str(output_path),
            'original_size': len(df),
            'relabeled_size': len(new_df),
            'comparison_report': comparison_report
        }

    def create_training_datasets(self, relabeled_data_path: str) -> Dict[str, str]:
        """
        從重新標註的資料創建訓練資料集

        Args:
            relabeled_data_path: 重新標註後的資料路徑

        Returns:
            創建的資料集路徑
        """
        logger.info("創建訓練資料集...")

        df = pd.read_csv(relabeled_data_path)

        datasets = {}

        # 毒性檢測資料集
        toxicity_df = df[['text', 'improved_toxicity', 'improved_toxicity_score']].copy()
        toxicity_df.columns = ['text', 'label', 'score']
        toxicity_path = self.output_dir / "toxicity_dataset.csv"
        toxicity_df.to_csv(toxicity_path, index=False, encoding='utf-8')
        datasets['toxicity'] = str(toxicity_path)

        # 霸凌檢測資料集
        bullying_df = df[['text', 'improved_bullying', 'improved_bullying_score']].copy()
        bullying_df.columns = ['text', 'label', 'score']
        bullying_path = self.output_dir / "bullying_dataset.csv"
        bullying_df.to_csv(bullying_path, index=False, encoding='utf-8')
        datasets['bullying'] = str(bullying_path)

        # 角色檢測資料集
        role_df = df[['text', 'improved_role']].copy()
        role_df.columns = ['text', 'label']
        role_path = self.output_dir / "role_dataset.csv"
        role_df.to_csv(role_path, index=False, encoding='utf-8')
        datasets['role'] = str(role_path)

        # 情緒檢測資料集
        emotion_df = df[['text', 'improved_emotion', 'improved_emotion_intensity']].copy()
        emotion_df.columns = ['text', 'emotion', 'intensity']
        emotion_path = self.output_dir / "emotion_dataset.csv"
        emotion_df.to_csv(emotion_path, index=False, encoding='utf-8')
        datasets['emotion'] = str(emotion_path)

        # 多任務資料集
        multitask_df = df[[
            'text', 'improved_toxicity', 'improved_bullying',
            'improved_role', 'improved_emotion', 'improved_emotion_intensity'
        ]].copy()
        multitask_df.columns = [
            'text', 'toxicity', 'bullying', 'role', 'emotion', 'emotion_intensity'
        ]
        multitask_path = self.output_dir / "multitask_dataset.csv"
        multitask_df.to_csv(multitask_path, index=False, encoding='utf-8')
        datasets['multitask'] = str(multitask_path)

        logger.info(f"創建了 {len(datasets)} 個訓練資料集")
        return datasets

    def generate_label_statistics(self, relabeled_data_path: str) -> Dict[str, Any]:
        """
        生成標籤統計報告

        Args:
            relabeled_data_path: 重新標註後的資料路徑

        Returns:
            統計報告
        """
        logger.info("生成標籤統計報告...")

        df = pd.read_csv(relabeled_data_path)

        stats = {
            'dataset_info': {
                'total_samples': len(df),
                'file_path': relabeled_data_path
            },
            'original_distribution': {
                'toxicity': df['original_toxicity'].value_counts().to_dict(),
                'bullying': df['original_bullying'].value_counts().to_dict()
            },
            'improved_distribution': {
                'toxicity': df['improved_toxicity'].value_counts().to_dict(),
                'bullying': df['improved_bullying'].value_counts().to_dict(),
                'role': df['improved_role'].value_counts().to_dict(),
                'emotion': df['improved_emotion'].value_counts().to_dict()
            },
            'score_statistics': {
                'toxicity_score': {
                    'mean': df['improved_toxicity_score'].mean(),
                    'std': df['improved_toxicity_score'].std(),
                    'min': df['improved_toxicity_score'].min(),
                    'max': df['improved_toxicity_score'].max()
                },
                'bullying_score': {
                    'mean': df['improved_bullying_score'].mean(),
                    'std': df['improved_bullying_score'].std(),
                    'min': df['improved_bullying_score'].min(),
                    'max': df['improved_bullying_score'].max()
                }
            }
        }

        # 分析標籤變化
        toxicity_changes = sum(1 for i, row in df.iterrows()
                             if row['original_toxicity'] != row['improved_toxicity'])

        bullying_changes = sum(1 for i, row in df.iterrows()
                             if row['original_bullying'] != row['improved_bullying'])

        stats['label_changes'] = {
            'toxicity_changed': toxicity_changes,
            'bullying_changed': bullying_changes,
            'toxicity_change_rate': toxicity_changes / len(df),
            'bullying_change_rate': bullying_changes / len(df)
        }

        # 計算標籤分離度
        toxic_not_bullying = sum(1 for i, row in df.iterrows()
                               if row['improved_toxicity'] != 'none'
                               and row['improved_bullying'] == 'none')

        bullying_not_toxic = sum(1 for i, row in df.iterrows()
                               if row['improved_bullying'] != 'none'
                               and row['improved_toxicity'] == 'none')

        both_toxic_and_bullying = sum(1 for i, row in df.iterrows()
                                    if row['improved_toxicity'] != 'none'
                                    and row['improved_bullying'] != 'none')

        stats['separation_analysis'] = {
            'toxic_not_bullying': toxic_not_bullying,
            'bullying_not_toxic': bullying_not_toxic,
            'both_toxic_and_bullying': both_toxic_and_bullying,
            'separation_score': (toxic_not_bullying + bullying_not_toxic) / len(df)
        }

        return stats

    def estimate_f1_improvement(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """
        估算 F1 分數改進

        Args:
            stats: 統計報告

        Returns:
            F1 改進估算
        """
        separation_score = stats.get('separation_analysis', {}).get('separation_score', 0)
        change_rate = stats.get('label_changes', {})

        # 基於分離度和標籤變化率估算改進
        toxicity_change_rate = change_rate.get('toxicity_change_rate', 0)
        bullying_change_rate = change_rate.get('bullying_change_rate', 0)

        # 經驗估算公式
        estimated_improvements = {
            'toxicity_f1_improvement': min(0.15, separation_score * 0.2 + toxicity_change_rate * 0.1),
            'bullying_f1_improvement': min(0.20, separation_score * 0.3 + bullying_change_rate * 0.15),
            'overall_f1_improvement': min(0.18, separation_score * 0.25 +
                                        (toxicity_change_rate + bullying_change_rate) * 0.05)
        }

        return estimated_improvements

    def run_complete_relabeling(self, cold_data_path: str) -> Dict[str, Any]:
        """
        執行完整的重新標註流程

        Args:
            cold_data_path: COLD 資料集路徑

        Returns:
            完整的處理結果
        """
        logger.info("開始完整的重新標註流程...")

        # 步驟1: 重新標註 COLD 資料集
        relabel_result = self.relabel_cold_dataset(cold_data_path)

        # 步驟2: 創建訓練資料集
        training_datasets = self.create_training_datasets(relabel_result['output_path'])

        # 步驟3: 生成統計報告
        stats = self.generate_label_statistics(relabel_result['output_path'])

        # 步驟4: 估算 F1 改進
        f1_estimates = self.estimate_f1_improvement(stats)

        # 合併結果
        complete_result = {
            'relabel_result': relabel_result,
            'training_datasets': training_datasets,
            'statistics': stats,
            'f1_improvement_estimates': f1_estimates,
            'summary': {
                'total_samples_processed': stats['dataset_info']['total_samples'],
                'separation_achieved': stats['separation_analysis']['separation_score'] > 0.1,
                'expected_toxicity_f1_gain': f1_estimates['toxicity_f1_improvement'],
                'expected_bullying_f1_gain': f1_estimates['bullying_f1_improvement']
            }
        }

        # 保存完整報告
        report_path = self.reports_dir / "relabeling_complete_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(complete_result, f, ensure_ascii=False, indent=2)

        logger.info(f"完整報告已保存到: {report_path}")
        return complete_result


def main():
    parser = argparse.ArgumentParser(description="重新標註資料集")
    parser.add_argument("--cold-data", required=True, help="COLD 資料集路徑")
    parser.add_argument("--data-dir", default="data", help="資料目錄")
    parser.add_argument("--output-summary", action="store_true", help="輸出摘要報告")

    args = parser.parse_args()

    relabeler = DatasetRelabeler(args.data_dir)

    try:
        result = relabeler.run_complete_relabeling(args.cold_data)

        if args.output_summary:
            print("\n=== 重新標註摘要報告 ===")
            print(f"處理樣本數: {result['summary']['total_samples_processed']}")
            print(f"標籤分離達成: {result['summary']['separation_achieved']}")
            print(f"預期毒性 F1 提升: {result['summary']['expected_toxicity_f1_gain']:.3f}")
            print(f"預期霸凌 F1 提升: {result['summary']['expected_bullying_f1_gain']:.3f}")

            print("\n=== 標籤分佈變化 ===")
            stats = result['statistics']
            print("毒性標籤:")
            for label, count in stats['improved_distribution']['toxicity'].items():
                print(f"  {label}: {count}")
            print("霸凌標籤:")
            for label, count in stats['improved_distribution']['bullying'].items():
                print(f"  {label}: {count}")

            print(f"\n分離分數: {stats['separation_analysis']['separation_score']:.3f}")

        logger.info("重新標註流程完成!")

    except Exception as e:
        logger.error(f"重新標註過程中發生錯誤: {e}")
        raise


if __name__ == "__main__":
    main()