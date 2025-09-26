#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
標註品質控制系統
包含多人標註一致性檢查、困難樣本管理、標註結果驗證等功能
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnnotationQualityController:
    """標註品質控制器"""

    def __init__(self, output_dir: str = "data/annotation/quality_control"):
        """初始化品質控制器

        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 標籤維度
        self.label_dimensions = {
            'toxicity': ['none', 'toxic', 'severe'],
            'bullying': ['none', 'harassment', 'threat'],
            'role': ['none', 'perpetrator', 'victim', 'bystander'],
            'emotion': ['pos', 'neu', 'neg'],
            'emotion_strength': ['0', '1', '2', '3', '4']
        }

        # 一致性閾值
        self.consistency_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'acceptable': 0.4,
            'poor': 0.2
        }

        logger.info(f"標註品質控制器初始化完成，輸出目錄: {self.output_dir}")

    def load_annotations(self, annotation_files: List[str]) -> Dict[str, List[Dict]]:
        """載入多個標註者的標註結果

        Args:
            annotation_files: 標註檔案路徑列表

        Returns:
            {annotator_id: [annotations]}
        """
        annotations_by_annotator = {}

        for file_path in annotation_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取標註者ID
                if isinstance(data, dict) and 'annotator_id' in data:
                    annotator_id = data['annotator_id']
                    annotations = data.get('annotations', {})
                elif isinstance(data, list) and len(data) > 0:
                    # 假設第一個記錄包含標註者資訊
                    annotator_id = data[0].get('annotator_id', Path(file_path).stem)
                    annotations = {}
                    for item in data:
                        if 'annotation' in item:
                            idx = str(item.get('id', item.get('original_index', len(annotations))))
                            annotations[idx] = item['annotation']
                else:
                    annotator_id = Path(file_path).stem
                    annotations = data

                annotations_by_annotator[annotator_id] = annotations
                logger.info(f"載入標註者 {annotator_id} 的 {len(annotations)} 個標註")

            except Exception as e:
                logger.error(f"載入標註檔案失敗 {file_path}: {str(e)}")

        return annotations_by_annotator

    def calculate_inter_annotator_agreement(self, annotations_by_annotator: Dict[str, List[Dict]]) -> Dict[str, float]:
        """計算標註者間一致性 (Inter-Annotator Agreement)

        Args:
            annotations_by_annotator: 標註者標註結果

        Returns:
            各維度的 Kappa 係數
        """
        logger.info("計算標註者間一致性...")

        results = {}

        # 找到所有標註者共同標註的樣本
        annotator_ids = list(annotations_by_annotator.keys())
        if len(annotator_ids) < 2:
            logger.warning("需要至少兩個標註者的資料才能計算一致性")
            return results

        # 找到共同樣本
        common_samples = set(annotations_by_annotator[annotator_ids[0]].keys())
        for annotator_id in annotator_ids[1:]:
            common_samples &= set(annotations_by_annotator[annotator_id].keys())

        logger.info(f"找到 {len(common_samples)} 個共同標註樣本")

        if len(common_samples) < 10:
            logger.warning("共同樣本數量過少，一致性計算可能不準確")

        # 對每個維度計算一致性
        for dimension in self.label_dimensions:
            try:
                # 準備標註資料
                annotator_labels = []
                for annotator_id in annotator_ids:
                    labels = []
                    for sample_id in sorted(common_samples):
                        annotation = annotations_by_annotator[annotator_id][sample_id]
                        label = annotation.get(dimension, 'none')
                        labels.append(label)
                    annotator_labels.append(labels)

                # 計算每對標註者的 Kappa 係數
                kappa_scores = []
                for i in range(len(annotator_ids)):
                    for j in range(i + 1, len(annotator_ids)):
                        try:
                            kappa = cohen_kappa_score(annotator_labels[i], annotator_labels[j])
                            kappa_scores.append(kappa)
                        except Exception as e:
                            logger.warning(f"計算 {dimension} 維度 Kappa 失敗: {str(e)}")

                if kappa_scores:
                    avg_kappa = np.mean(kappa_scores)
                    results[dimension] = avg_kappa
                    logger.info(f"{dimension} 維度平均 Kappa: {avg_kappa:.3f}")

            except Exception as e:
                logger.error(f"計算 {dimension} 維度一致性失敗: {str(e)}")

        return results

    def identify_disagreement_cases(self, annotations_by_annotator: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
        """識別不一致標註案例

        Args:
            annotations_by_annotator: 標註者標註結果

        Returns:
            {dimension: [disagreement_sample_ids]}
        """
        logger.info("識別不一致標註案例...")

        disagreement_cases = defaultdict(list)
        annotator_ids = list(annotations_by_annotator.keys())

        # 找到共同樣本
        common_samples = set(annotations_by_annotator[annotator_ids[0]].keys())
        for annotator_id in annotator_ids[1:]:
            common_samples &= set(annotations_by_annotator[annotator_id].keys())

        # 檢查每個樣本的一致性
        for sample_id in common_samples:
            for dimension in self.label_dimensions:
                labels = []
                for annotator_id in annotator_ids:
                    annotation = annotations_by_annotator[annotator_id][sample_id]
                    label = annotation.get(dimension, 'none')
                    labels.append(label)

                # 如果標籤不一致，記錄為不一致案例
                if len(set(labels)) > 1:
                    disagreement_cases[dimension].append(sample_id)

        # 記錄統計資訊
        for dimension, cases in disagreement_cases.items():
            disagreement_rate = len(cases) / len(common_samples) * 100
            logger.info(f"{dimension} 維度不一致案例: {len(cases)} ({disagreement_rate:.1f}%)")

        return dict(disagreement_cases)

    def analyze_annotator_performance(self, annotations_by_annotator: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """分析標註者表現

        Args:
            annotations_by_annotator: 標註者標註結果

        Returns:
            標註者表現統計
        """
        logger.info("分析標註者表現...")

        performance_stats = {}

        for annotator_id, annotations in annotations_by_annotator.items():
            stats = {
                'total_annotations': len(annotations),
                'dimensions_distribution': {},
                'difficult_cases': 0,
                'annotation_speed': None
            }

            # 計算各維度分布
            for dimension in self.label_dimensions:
                dimension_labels = []
                for annotation in annotations.values():
                    label = annotation.get(dimension, 'none')
                    dimension_labels.append(label)

                distribution = Counter(dimension_labels)
                stats['dimensions_distribution'][dimension] = dict(distribution)

            # 計算困難案例數量
            difficult_count = sum(1 for ann in annotations.values() if ann.get('difficult', False))
            stats['difficult_cases'] = difficult_count

            # 計算標註速度（如果有時間戳）
            timestamps = []
            for annotation in annotations.values():
                if 'timestamp' in annotation:
                    try:
                        timestamp = datetime.fromisoformat(annotation['timestamp'].replace('Z', '+00:00'))
                        timestamps.append(timestamp)
                    except:
                        pass

            if len(timestamps) > 1:
                timestamps.sort()
                duration = (timestamps[-1] - timestamps[0]).total_seconds()
                speed = len(annotations) / (duration / 3600)  # 樣本/小時
                stats['annotation_speed'] = speed

            performance_stats[annotator_id] = stats

        return performance_stats

    def generate_quality_report(self, annotations_by_annotator: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """生成品質報告

        Args:
            annotations_by_annotator: 標註者標註結果

        Returns:
            品質報告
        """
        logger.info("生成品質報告...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'inter_annotator_agreement': {},
            'disagreement_cases': {},
            'annotator_performance': {},
            'recommendations': []
        }

        # 基本統計
        total_annotators = len(annotations_by_annotator)
        total_annotations = sum(len(ann) for ann in annotations_by_annotator.values())
        report['summary'] = {
            'total_annotators': total_annotators,
            'total_annotations': total_annotations,
            'average_annotations_per_annotator': total_annotations / total_annotators if total_annotators > 0 else 0
        }

        # 標註者間一致性
        if total_annotators >= 2:
            iaa_results = self.calculate_inter_annotator_agreement(annotations_by_annotator)
            report['inter_annotator_agreement'] = iaa_results

            # 不一致案例
            disagreement_cases = self.identify_disagreement_cases(annotations_by_annotator)
            report['disagreement_cases'] = disagreement_cases

        # 標註者表現
        performance_stats = self.analyze_annotator_performance(annotations_by_annotator)
        report['annotator_performance'] = performance_stats

        # 產生建議
        recommendations = self.generate_recommendations(report)
        report['recommendations'] = recommendations

        return report

    def generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """基於報告生成改進建議

        Args:
            report: 品質報告

        Returns:
            建議列表
        """
        recommendations = []

        # 檢查一致性
        iaa_results = report.get('inter_annotator_agreement', {})
        for dimension, kappa in iaa_results.items():
            if kappa < self.consistency_thresholds['poor']:
                recommendations.append(f"{dimension} 維度一致性極低 (κ={kappa:.3f})，建議重新培訓標註者並澄清標註標準")
            elif kappa < self.consistency_thresholds['acceptable']:
                recommendations.append(f"{dimension} 維度一致性較低 (κ={kappa:.3f})，建議討論困難案例並優化標註指引")
            elif kappa < self.consistency_thresholds['good']:
                recommendations.append(f"{dimension} 維度一致性中等 (κ={kappa:.3f})，可考慮增加標註範例")

        # 檢查不一致案例
        disagreement_cases = report.get('disagreement_cases', {})
        for dimension, cases in disagreement_cases.items():
            if len(cases) > 10:
                recommendations.append(f"{dimension} 維度有 {len(cases)} 個不一致案例，建議組織討論會解決分歧")

        # 檢查標註者表現差異
        performance_stats = report.get('annotator_performance', {})
        if len(performance_stats) > 1:
            speeds = [stats.get('annotation_speed', 0) for stats in performance_stats.values() if stats.get('annotation_speed')]
            if speeds and max(speeds) / min(speeds) > 3:
                recommendations.append("標註者間速度差異較大，建議檢查是否存在品質問題")

        # 困難案例建議
        total_difficult = sum(stats.get('difficult_cases', 0) for stats in performance_stats.values())
        if total_difficult > 20:
            recommendations.append(f"共有 {total_difficult} 個困難案例，建議安排專家審核")

        if not recommendations:
            recommendations.append("標註品質整體良好，建議繼續保持")

        return recommendations

    def visualize_agreement_matrix(self, annotations_by_annotator: Dict[str, List[Dict]], dimension: str = 'toxicity'):
        """視覺化標註一致性矩陣

        Args:
            annotations_by_annotator: 標註者標註結果
            dimension: 要視覺化的維度
        """
        logger.info(f"生成 {dimension} 維度一致性視覺化...")

        annotator_ids = list(annotations_by_annotator.keys())
        if len(annotator_ids) < 2:
            logger.warning("需要至少兩個標註者才能生成一致性矩陣")
            return

        # 找到共同樣本
        common_samples = set(annotations_by_annotator[annotator_ids[0]].keys())
        for annotator_id in annotator_ids[1:]:
            common_samples &= set(annotations_by_annotator[annotator_id].keys())

        # 建立一致性矩陣
        n_annotators = len(annotator_ids)
        agreement_matrix = np.zeros((n_annotators, n_annotators))

        for i in range(n_annotators):
            for j in range(n_annotators):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    # 計算兩個標註者的一致性
                    labels_i = []
                    labels_j = []
                    for sample_id in common_samples:
                        label_i = annotations_by_annotator[annotator_ids[i]][sample_id].get(dimension, 'none')
                        label_j = annotations_by_annotator[annotator_ids[j]][sample_id].get(dimension, 'none')
                        labels_i.append(label_i)
                        labels_j.append(label_j)

                    try:
                        kappa = cohen_kappa_score(labels_i, labels_j)
                        agreement_matrix[i, j] = max(0, kappa)  # 避免負值
                    except:
                        agreement_matrix[i, j] = 0

        # 繪製熱圖
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            agreement_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=annotator_ids,
            yticklabels=annotator_ids,
            cbar_kws={'label': 'Cohen\'s Kappa'}
        )
        plt.title(f'{dimension.capitalize()} 維度標註者間一致性 (Cohen\'s Kappa)')
        plt.tight_layout()

        # 儲存圖片
        output_path = self.output_dir / f"agreement_matrix_{dimension}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"一致性矩陣已儲存到: {output_path}")

    def generate_confusion_matrices(self, annotations_by_annotator: Dict[str, List[Dict]], reference_annotator: str = None):
        """生成混淆矩陣比較不同標註者

        Args:
            annotations_by_annotator: 標註者標註結果
            reference_annotator: 參考標註者（如專家標註者）
        """
        logger.info("生成混淆矩陣...")

        annotator_ids = list(annotations_by_annotator.keys())
        if not reference_annotator:
            reference_annotator = annotator_ids[0]

        if reference_annotator not in annotator_ids:
            logger.error(f"參考標註者 {reference_annotator} 不存在")
            return

        # 找到共同樣本
        common_samples = set(annotations_by_annotator[reference_annotator].keys())
        for annotator_id in annotator_ids:
            common_samples &= set(annotations_by_annotator[annotator_id].keys())

        for dimension in self.label_dimensions:
            labels = self.label_dimensions[dimension]

            # 參考標註者的標籤
            ref_labels = []
            for sample_id in sorted(common_samples):
                label = annotations_by_annotator[reference_annotator][sample_id].get(dimension, 'none')
                ref_labels.append(label)

            # 與其他標註者比較
            for annotator_id in annotator_ids:
                if annotator_id == reference_annotator:
                    continue

                # 當前標註者的標籤
                ann_labels = []
                for sample_id in sorted(common_samples):
                    label = annotations_by_annotator[annotator_id][sample_id].get(dimension, 'none')
                    ann_labels.append(label)

                # 生成混淆矩陣
                cm = confusion_matrix(ref_labels, ann_labels, labels=labels)

                # 繪製混淆矩陣
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels
                )
                plt.title(f'{dimension.capitalize()} 維度混淆矩陣\n{reference_annotator} vs {annotator_id}')
                plt.xlabel(f'{annotator_id} 預測')
                plt.ylabel(f'{reference_annotator} 實際')
                plt.tight_layout()

                # 儲存圖片
                output_path = self.output_dir / f"confusion_matrix_{dimension}_{reference_annotator}_vs_{annotator_id}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

    def export_disagreement_cases(self, annotations_by_annotator: Dict[str, List[Dict]], original_samples: List[Dict]):
        """匯出不一致案例供討論

        Args:
            annotations_by_annotator: 標註者標註結果
            original_samples: 原始樣本資料
        """
        logger.info("匯出不一致案例...")

        # 建立樣本ID到內容的映射
        sample_content_map = {}
        for i, sample in enumerate(original_samples):
            sample_id = str(sample.get('id', sample.get('annotation_metadata', {}).get('original_index', i)))

            # 提取文字內容
            text_fields = ['text', 'content', 'message', 'sentence', 'comment']
            content = ""
            for field in text_fields:
                if field in sample and sample[field]:
                    content = str(sample[field])
                    break

            sample_content_map[sample_id] = content

        # 識別不一致案例
        disagreement_cases = self.identify_disagreement_cases(annotations_by_annotator)

        # 為每個維度匯出不一致案例
        for dimension, case_ids in disagreement_cases.items():
            if not case_ids:
                continue

            disagreement_data = []
            annotator_ids = list(annotations_by_annotator.keys())

            for case_id in case_ids:
                case_info = {
                    'sample_id': case_id,
                    'content': sample_content_map.get(case_id, "未找到內容"),
                    'annotations': {}
                }

                # 收集所有標註者對此案例的標註
                for annotator_id in annotator_ids:
                    if case_id in annotations_by_annotator[annotator_id]:
                        annotation = annotations_by_annotator[annotator_id][case_id]
                        case_info['annotations'][annotator_id] = {
                            dimension: annotation.get(dimension, 'none'),
                            'note': annotation.get('note', ''),
                            'timestamp': annotation.get('timestamp', '')
                        }

                disagreement_data.append(case_info)

            # 儲存不一致案例
            output_path = self.output_dir / f"disagreement_cases_{dimension}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(disagreement_data, f, ensure_ascii=False, indent=2)

            logger.info(f"{dimension} 維度不一致案例已儲存到: {output_path}")

    def save_quality_report(self, report: Dict[str, Any], filename: str = None):
        """儲存品質報告

        Args:
            report: 品質報告
            filename: 檔案名稱
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"品質報告已儲存到: {output_path}")

        # 同時生成簡化的文字報告
        text_report_path = self.output_dir / filename.replace('.json', '_summary.txt')
        self.generate_text_summary(report, text_report_path)

    def generate_text_summary(self, report: Dict[str, Any], output_path: Path):
        """生成文字摘要報告

        Args:
            report: 品質報告
            output_path: 輸出路徑
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== 標註品質控制報告 ===\n\n")
            f.write(f"生成時間: {report['timestamp']}\n\n")

            # 基本統計
            summary = report['summary']
            f.write("=== 基本統計 ===\n")
            f.write(f"標註者數量: {summary['total_annotators']}\n")
            f.write(f"總標註數量: {summary['total_annotations']}\n")
            f.write(f"平均每人標註: {summary['average_annotations_per_annotator']:.1f}\n\n")

            # 一致性結果
            if 'inter_annotator_agreement' in report and report['inter_annotator_agreement']:
                f.write("=== 標註者間一致性 (Cohen's Kappa) ===\n")
                for dimension, kappa in report['inter_annotator_agreement'].items():
                    consistency_level = "優秀" if kappa >= 0.8 else "良好" if kappa >= 0.6 else "可接受" if kappa >= 0.4 else "差"
                    f.write(f"{dimension}: {kappa:.3f} ({consistency_level})\n")
                f.write("\n")

            # 不一致案例
            if 'disagreement_cases' in report and report['disagreement_cases']:
                f.write("=== 不一致案例統計 ===\n")
                for dimension, cases in report['disagreement_cases'].items():
                    f.write(f"{dimension}: {len(cases)} 個案例\n")
                f.write("\n")

            # 建議
            if 'recommendations' in report and report['recommendations']:
                f.write("=== 改進建議 ===\n")
                for i, recommendation in enumerate(report['recommendations'], 1):
                    f.write(f"{i}. {recommendation}\n")

        logger.info(f"文字摘要報告已儲存到: {output_path}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="標註品質控制系統")

    parser.add_argument("--annotation_files", nargs='+', required=True, help="標註檔案路徑列表")
    parser.add_argument("--original_samples", help="原始樣本檔案路徑")
    parser.add_argument("--output_dir", default="data/annotation/quality_control", help="輸出目錄")
    parser.add_argument("--reference_annotator", help="參考標註者ID（用於混淆矩陣）")
    parser.add_argument("--visualize", action="store_true", help="生成視覺化圖表")

    args = parser.parse_args()

    # 初始化品質控制器
    controller = AnnotationQualityController(args.output_dir)

    # 載入標註資料
    annotations_by_annotator = controller.load_annotations(args.annotation_files)

    if not annotations_by_annotator:
        logger.error("未能載入任何標註資料")
        return

    # 生成品質報告
    report = controller.generate_quality_report(annotations_by_annotator)

    # 儲存報告
    controller.save_quality_report(report)

    # 載入原始樣本（如果提供）
    original_samples = []
    if args.original_samples:
        try:
            with open(args.original_samples, 'r', encoding='utf-8') as f:
                if args.original_samples.endswith('.json'):
                    original_samples = json.load(f)
                elif args.original_samples.endswith('.jsonl'):
                    original_samples = []
                    for line in f:
                        original_samples.append(json.loads(line.strip()))
                elif args.original_samples.endswith('.csv'):
                    df = pd.read_csv(args.original_samples)
                    original_samples = df.to_dict('records')

            # 匯出不一致案例
            controller.export_disagreement_cases(annotations_by_annotator, original_samples)

        except Exception as e:
            logger.error(f"載入原始樣本失敗: {str(e)}")

    # 生成視覺化（如果要求）
    if args.visualize:
        for dimension in controller.label_dimensions:
            controller.visualize_agreement_matrix(annotations_by_annotator, dimension)

        if args.reference_annotator:
            controller.generate_confusion_matrices(annotations_by_annotator, args.reference_annotator)

    # 輸出摘要
    print("\n=== 品質控制摘要 ===")
    print(f"標註者數量: {report['summary']['total_annotators']}")
    print(f"總標註數量: {report['summary']['total_annotations']}")

    if 'inter_annotator_agreement' in report:
        print("\n=== 一致性結果 ===")
        for dimension, kappa in report['inter_annotator_agreement'].items():
            print(f"{dimension}: κ = {kappa:.3f}")

    print("\n=== 建議 ===")
    for recommendation in report['recommendations']:
        print(f"• {recommendation}")

    print(f"\n詳細報告已儲存到: {args.output_dir}")


if __name__ == "__main__":
    main()