#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
標註結果驗證和一致性檢查系統
提供標註結果自動驗證、一致性檢查、異常檢測等功能
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnnotationValidator:
    """標註結果驗證器"""

    def __init__(self, output_dir: str = "data/annotation/validation"):
        """初始化驗證器

        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 標籤維度和有效值
        self.valid_labels = {
            'toxicity': {'none', 'toxic', 'severe'},
            'bullying': {'none', 'harassment', 'threat'},
            'role': {'none', 'perpetrator', 'victim', 'bystander'},
            'emotion': {'pos', 'neu', 'neg'},
            'emotion_strength': {'0', '1', '2', '3', '4'}
        }

        # 邏輯規則
        self.logic_rules = {
            'toxicity_bullying': {
                # 如果有霸凌行為，毒性程度不應該是none
                'rule': 'if bullying != "none" then toxicity != "none"',
                'description': '有霸凌行為時，毒性程度不應為無'
            },
            'severe_toxicity_bullying': {
                # 嚴重毒性通常伴隨霸凌行為
                'rule': 'if toxicity == "severe" then bullying != "none"',
                'description': '嚴重毒性通常伴隨霸凌行為'
            },
            'perpetrator_toxicity': {
                # 施害者角色通常伴隨毒性內容
                'rule': 'if role == "perpetrator" then toxicity != "none"',
                'description': '施害者角色通常伴隨毒性內容'
            },
            'emotion_strength_consistency': {
                # 負面情緒強度與毒性程度的一致性
                'rule': 'if emotion == "neg" and emotion_strength in ["3", "4"] then toxicity != "none"',
                'description': '強烈負面情緒通常伴隨毒性內容'
            }
        }

        # 異常檢測閾值
        self.anomaly_thresholds = {
            'min_annotation_time': 5,  # 最少標註時間（秒）
            'max_annotation_time': 600,  # 最多標註時間（秒）
            'max_note_length': 500,  # 最大備註長度
            'min_difficult_ratio': 0.05,  # 最少困難案例比例
            'max_difficult_ratio': 0.30,  # 最多困難案例比例
        }

        logger.info(f"標註驗證器初始化完成，輸出目錄: {self.output_dir}")

    def validate_single_annotation(self, annotation: Dict[str, Any], sample_id: str = None) -> Dict[str, Any]:
        """驗證單個標註

        Args:
            annotation: 標註資料
            sample_id: 樣本ID

        Returns:
            驗證結果
        """
        validation_result = {
            'sample_id': sample_id,
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'logic_violations': []
        }

        # 1. 格式驗證
        format_errors = self.validate_format(annotation)
        validation_result['errors'].extend(format_errors)

        # 2. 邏輯一致性檢查
        logic_violations = self.check_logic_consistency(annotation)
        validation_result['logic_violations'].extend(logic_violations)

        # 3. 時間合理性檢查
        time_warnings = self.check_annotation_time(annotation)
        validation_result['warnings'].extend(time_warnings)

        # 4. 內容合理性檢查
        content_warnings = self.check_content_reasonableness(annotation)
        validation_result['warnings'].extend(content_warnings)

        # 判斷整體有效性
        if validation_result['errors'] or validation_result['logic_violations']:
            validation_result['is_valid'] = False

        return validation_result

    def validate_format(self, annotation: Dict[str, Any]) -> List[str]:
        """驗證標註格式

        Args:
            annotation: 標註資料

        Returns:
            錯誤列表
        """
        errors = []

        # 檢查必要欄位
        required_fields = ['toxicity', 'bullying', 'role', 'emotion', 'emotion_strength']
        for field in required_fields:
            if field not in annotation:
                errors.append(f"缺少必要欄位: {field}")
            elif annotation[field] not in self.valid_labels[field]:
                errors.append(f"欄位 {field} 的值 '{annotation[field]}' 不在有效值範圍內")

        # 檢查標註者ID
        if 'annotator_id' not in annotation or not annotation['annotator_id']:
            errors.append("缺少標註者ID")

        # 檢查時間戳
        if 'timestamp' in annotation:
            try:
                datetime.fromisoformat(annotation['timestamp'].replace('Z', '+00:00'))
            except:
                errors.append("時間戳格式錯誤")

        return errors

    def check_logic_consistency(self, annotation: Dict[str, Any]) -> List[str]:
        """檢查邏輯一致性

        Args:
            annotation: 標註資料

        Returns:
            邏輯違規列表
        """
        violations = []

        # 檢查每個邏輯規則
        for rule_name, rule_info in self.logic_rules.items():
            violation = self.check_single_logic_rule(annotation, rule_name, rule_info)
            if violation:
                violations.append(violation)

        return violations

    def check_single_logic_rule(self, annotation: Dict[str, Any], rule_name: str, rule_info: Dict) -> Optional[str]:
        """檢查單個邏輯規則

        Args:
            annotation: 標註資料
            rule_name: 規則名稱
            rule_info: 規則資訊

        Returns:
            違規描述或None
        """
        try:
            toxicity = annotation.get('toxicity', 'none')
            bullying = annotation.get('bullying', 'none')
            role = annotation.get('role', 'none')
            emotion = annotation.get('emotion', 'neu')
            emotion_strength = annotation.get('emotion_strength', '0')

            if rule_name == 'toxicity_bullying':
                if bullying != 'none' and toxicity == 'none':
                    return f"邏輯違規: {rule_info['description']}"

            elif rule_name == 'severe_toxicity_bullying':
                if toxicity == 'severe' and bullying == 'none':
                    return f"邏輯違規: {rule_info['description']}"

            elif rule_name == 'perpetrator_toxicity':
                if role == 'perpetrator' and toxicity == 'none':
                    return f"邏輯違規: {rule_info['description']}"

            elif rule_name == 'emotion_strength_consistency':
                if emotion == 'neg' and emotion_strength in ['3', '4'] and toxicity == 'none':
                    return f"邏輯違規: {rule_info['description']}"

        except Exception as e:
            return f"邏輯規則檢查錯誤: {str(e)}"

        return None

    def check_annotation_time(self, annotation: Dict[str, Any]) -> List[str]:
        """檢查標註時間合理性

        Args:
            annotation: 標註資料

        Returns:
            警告列表
        """
        warnings = []

        # 檢查標註時間（需要外部提供參考時間）
        if 'annotation_duration' in annotation:
            duration = annotation['annotation_duration']
            if duration < self.anomaly_thresholds['min_annotation_time']:
                warnings.append(f"標註時間過短: {duration}秒")
            elif duration > self.anomaly_thresholds['max_annotation_time']:
                warnings.append(f"標註時間過長: {duration}秒")

        return warnings

    def check_content_reasonableness(self, annotation: Dict[str, Any]) -> List[str]:
        """檢查內容合理性

        Args:
            annotation: 標註資料

        Returns:
            警告列表
        """
        warnings = []

        # 檢查備註長度
        if 'note' in annotation and annotation['note']:
            note_length = len(annotation['note'])
            if note_length > self.anomaly_thresholds['max_note_length']:
                warnings.append(f"備註過長: {note_length}字符")

        # 檢查困難標記的合理性
        if annotation.get('difficult', False):
            # 困難案例應該有備註說明
            if not annotation.get('note', '').strip():
                warnings.append("困難案例缺少說明備註")

        return warnings

    def validate_batch_annotations(self, annotations_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """批次驗證標註結果

        Args:
            annotations_data: {sample_id: annotation}

        Returns:
            批次驗證結果
        """
        logger.info(f"開始批次驗證 {len(annotations_data)} 個標註")

        batch_result = {
            'total_annotations': len(annotations_data),
            'valid_annotations': 0,
            'invalid_annotations': 0,
            'validation_details': {},
            'summary': {
                'format_errors': 0,
                'logic_violations': 0,
                'warnings': 0
            },
            'anomaly_detection': {},
            'recommendations': []
        }

        # 逐個驗證標註
        for sample_id, annotation in annotations_data.items():
            validation_result = self.validate_single_annotation(annotation, sample_id)
            batch_result['validation_details'][sample_id] = validation_result

            if validation_result['is_valid']:
                batch_result['valid_annotations'] += 1
            else:
                batch_result['invalid_annotations'] += 1

            # 統計錯誤類型
            batch_result['summary']['format_errors'] += len(validation_result['errors'])
            batch_result['summary']['logic_violations'] += len(validation_result['logic_violations'])
            batch_result['summary']['warnings'] += len(validation_result['warnings'])

        # 異常檢測
        batch_result['anomaly_detection'] = self.detect_batch_anomalies(annotations_data)

        # 生成建議
        batch_result['recommendations'] = self.generate_validation_recommendations(batch_result)

        logger.info(f"批次驗證完成: {batch_result['valid_annotations']}/{batch_result['total_annotations']} 有效")

        return batch_result

    def detect_batch_anomalies(self, annotations_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """檢測批次異常

        Args:
            annotations_data: 標註資料

        Returns:
            異常檢測結果
        """
        anomalies = {
            'statistical_anomalies': [],
            'pattern_anomalies': [],
            'temporal_anomalies': []
        }

        if not annotations_data:
            return anomalies

        # 統計分析
        label_distributions = defaultdict(Counter)
        difficult_count = 0
        timestamps = []

        for annotation in annotations_data.values():
            # 統計標籤分布
            for dimension in self.valid_labels:
                if dimension in annotation:
                    label_distributions[dimension][annotation[dimension]] += 1

            # 統計困難案例
            if annotation.get('difficult', False):
                difficult_count += 1

            # 收集時間戳
            if 'timestamp' in annotation:
                try:
                    timestamp = datetime.fromisoformat(annotation['timestamp'].replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except:
                    pass

        # 檢測統計異常
        total_annotations = len(annotations_data)

        # 困難案例比例異常
        difficult_ratio = difficult_count / total_annotations
        if difficult_ratio < self.anomaly_thresholds['min_difficult_ratio']:
            anomalies['statistical_anomalies'].append(f"困難案例比例過低: {difficult_ratio:.1%}")
        elif difficult_ratio > self.anomaly_thresholds['max_difficult_ratio']:
            anomalies['statistical_anomalies'].append(f"困難案例比例過高: {difficult_ratio:.1%}")

        # 標籤分布異常
        for dimension, distribution in label_distributions.items():
            if distribution:
                # 檢查是否過度偏向某個標籤
                max_count = max(distribution.values())
                max_ratio = max_count / total_annotations
                if max_ratio > 0.95:  # 95%以上都是同一個標籤
                    most_common = max(distribution.items(), key=lambda x: x[1])
                    anomalies['pattern_anomalies'].append(
                        f"{dimension} 維度過度偏向 '{most_common[0]}': {max_ratio:.1%}"
                    )

        # 檢測時間模式異常
        if len(timestamps) > 1:
            timestamps.sort()

            # 檢查是否在極短時間內完成大量標註
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            if time_span > 0:
                annotation_rate = len(timestamps) / (time_span / 60)  # 每分鐘標註數
                if annotation_rate > 10:  # 每分鐘超過10個標註
                    anomalies['temporal_anomalies'].append(f"標註速度異常: {annotation_rate:.1f} 個/分鐘")

            # 檢查時間分布
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds()
                        for i in range(len(timestamps)-1)]
            if intervals:
                avg_interval = np.mean(intervals)
                if avg_interval < 10:  # 平均間隔少於10秒
                    anomalies['temporal_anomalies'].append(f"平均標註間隔過短: {avg_interval:.1f} 秒")

        return anomalies

    def generate_validation_recommendations(self, batch_result: Dict[str, Any]) -> List[str]:
        """生成驗證建議

        Args:
            batch_result: 批次驗證結果

        Returns:
            建議列表
        """
        recommendations = []

        total = batch_result['total_annotations']
        invalid = batch_result['invalid_annotations']

        # 基於驗證結果的建議
        if invalid > 0:
            invalid_ratio = invalid / total
            if invalid_ratio > 0.1:
                recommendations.append(f"無效標註比例過高 ({invalid_ratio:.1%})，建議重新培訓標註者")
            elif invalid_ratio > 0.05:
                recommendations.append(f"無效標註比例較高 ({invalid_ratio:.1%})，建議澄清標註指引")

        # 基於錯誤類型的建議
        format_errors = batch_result['summary']['format_errors']
        logic_violations = batch_result['summary']['logic_violations']

        if format_errors > total * 0.05:
            recommendations.append("格式錯誤較多，建議檢查標註介面和輸入驗證")

        if logic_violations > total * 0.1:
            recommendations.append("邏輯一致性問題較多，建議加強邏輯規則培訓")

        # 基於異常檢測的建議
        anomalies = batch_result['anomaly_detection']

        if anomalies['statistical_anomalies']:
            recommendations.append("發現統計異常，建議檢查標註者表現和資料品質")

        if anomalies['pattern_anomalies']:
            recommendations.append("發現標籤分布異常，建議檢查樣本多樣性和標註者偏見")

        if anomalies['temporal_anomalies']:
            recommendations.append("發現時間模式異常，建議檢查標註速度和品質關聯")

        if not recommendations:
            recommendations.append("標註品質整體良好")

        return recommendations

    def check_inter_annotator_consistency(self, multi_annotations: Dict[str, Dict[str, Dict]]) -> Dict[str, Any]:
        """檢查多標註者一致性

        Args:
            multi_annotations: {sample_id: {annotator_id: annotation}}

        Returns:
            一致性檢查結果
        """
        logger.info("檢查多標註者一致性")

        consistency_result = {
            'total_samples': len(multi_annotations),
            'overlap_samples': 0,
            'consistency_by_dimension': {},
            'disagreement_cases': defaultdict(list),
            'annotator_pairs_agreement': {},
            'overall_consistency': 0.0
        }

        # 找出有多個標註者的樣本
        overlap_samples = {
            sample_id: annotations
            for sample_id, annotations in multi_annotations.items()
            if len(annotations) > 1
        }

        consistency_result['overlap_samples'] = len(overlap_samples)

        if not overlap_samples:
            logger.warning("沒有重疊標註的樣本，無法計算一致性")
            return consistency_result

        # 計算各維度一致性
        for dimension in self.valid_labels:
            dimension_agreements = []
            disagreements = []

            for sample_id, annotations in overlap_samples.items():
                # 提取該維度的所有標註
                labels = []
                annotators = []
                for annotator_id, annotation in annotations.items():
                    if dimension in annotation:
                        labels.append(annotation[dimension])
                        annotators.append(annotator_id)

                if len(labels) > 1:
                    # 檢查是否一致
                    unique_labels = set(labels)
                    if len(unique_labels) == 1:
                        dimension_agreements.append(1.0)  # 完全一致
                    else:
                        dimension_agreements.append(0.0)  # 不一致
                        disagreements.append({
                            'sample_id': sample_id,
                            'annotators': annotators,
                            'labels': labels
                        })

            # 計算該維度的一致性比例
            if dimension_agreements:
                consistency_ratio = np.mean(dimension_agreements)
                consistency_result['consistency_by_dimension'][dimension] = {
                    'consistency_ratio': consistency_ratio,
                    'total_comparisons': len(dimension_agreements),
                    'agreements': sum(dimension_agreements),
                    'disagreements': len(disagreements)
                }

                if disagreements:
                    consistency_result['disagreement_cases'][dimension] = disagreements

        # 計算整體一致性
        if consistency_result['consistency_by_dimension']:
            overall_consistency = np.mean([
                result['consistency_ratio']
                for result in consistency_result['consistency_by_dimension'].values()
            ])
            consistency_result['overall_consistency'] = overall_consistency

        return consistency_result

    def export_validation_report(self, validation_results: Dict[str, Any],
                               filename: str = None) -> str:
        """匯出驗證報告

        Args:
            validation_results: 驗證結果
            filename: 檔案名稱

        Returns:
            報告檔案路徑
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.json"

        # JSON 報告
        json_file = self.output_dir / filename
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, ensure_ascii=False, indent=2)

        # 文字摘要報告
        text_file = self.output_dir / filename.replace('.json', '_summary.txt')
        self.generate_text_validation_report(validation_results, text_file)

        logger.info(f"驗證報告已匯出: {json_file}")
        return str(json_file)

    def generate_text_validation_report(self, validation_results: Dict[str, Any],
                                      output_path: Path):
        """生成文字驗證報告

        Args:
            validation_results: 驗證結果
            output_path: 輸出路徑
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== 標註驗證報告 ===\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 基本統計
            f.write("=== 基本統計 ===\n")
            f.write(f"總標註數量: {validation_results['total_annotations']}\n")
            f.write(f"有效標註: {validation_results['valid_annotations']}\n")
            f.write(f"無效標註: {validation_results['invalid_annotations']}\n")
            valid_ratio = validation_results['valid_annotations'] / validation_results['total_annotations']
            f.write(f"有效比例: {valid_ratio:.1%}\n\n")

            # 錯誤統計
            f.write("=== 錯誤統計 ===\n")
            summary = validation_results['summary']
            f.write(f"格式錯誤: {summary['format_errors']}\n")
            f.write(f"邏輯違規: {summary['logic_violations']}\n")
            f.write(f"警告數量: {summary['warnings']}\n\n")

            # 異常檢測
            if 'anomaly_detection' in validation_results:
                anomalies = validation_results['anomaly_detection']
                f.write("=== 異常檢測 ===\n")

                for anomaly_type, anomaly_list in anomalies.items():
                    if anomaly_list:
                        f.write(f"{anomaly_type}:\n")
                        for anomaly in anomaly_list:
                            f.write(f"  - {anomaly}\n")
                        f.write("\n")

            # 一致性檢查（如果有）
            if 'consistency_by_dimension' in validation_results:
                f.write("=== 一致性檢查 ===\n")
                f.write(f"重疊樣本數: {validation_results.get('overlap_samples', 0)}\n")
                f.write(f"整體一致性: {validation_results.get('overall_consistency', 0):.3f}\n\n")

                for dimension, consistency in validation_results['consistency_by_dimension'].items():
                    f.write(f"{dimension} 維度:\n")
                    f.write(f"  一致性比例: {consistency['consistency_ratio']:.3f}\n")
                    f.write(f"  比較次數: {consistency['total_comparisons']}\n")
                    f.write(f"  一致案例: {consistency['agreements']}\n")
                    f.write(f"  不一致案例: {consistency['disagreements']}\n\n")

            # 建議
            if 'recommendations' in validation_results:
                f.write("=== 改進建議 ===\n")
                for i, recommendation in enumerate(validation_results['recommendations'], 1):
                    f.write(f"{i}. {recommendation}\n")

        logger.info(f"文字驗證報告已生成: {output_path}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="標註結果驗證和一致性檢查系統")

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 單一標註驗證
    single_parser = subparsers.add_parser('validate_single', help='驗證單個標註檔案')
    single_parser.add_argument('--annotation_file', required=True, help='標註檔案路徑')
    single_parser.add_argument('--output_dir', help='輸出目錄')

    # 批次驗證
    batch_parser = subparsers.add_parser('validate_batch', help='批次驗證標註檔案')
    batch_parser.add_argument('--annotation_files', nargs='+', required=True, help='標註檔案路徑列表')
    batch_parser.add_argument('--output_dir', help='輸出目錄')

    # 一致性檢查
    consistency_parser = subparsers.add_parser('check_consistency', help='檢查多標註者一致性')
    consistency_parser.add_argument('--annotation_files', nargs='+', required=True, help='標註檔案路徑列表')
    consistency_parser.add_argument('--output_dir', help='輸出目錄')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 初始化驗證器
    output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else "data/annotation/validation"
    validator = AnnotationValidator(output_dir)

    if args.command == 'validate_single':
        # 載入單個標註檔案
        try:
            with open(args.annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取標註資料
            if isinstance(data, dict) and 'annotations' in data:
                annotations_data = data['annotations']
            elif isinstance(data, list):
                annotations_data = {}
                for item in data:
                    if 'annotation' in item:
                        sample_id = str(item.get('id', len(annotations_data)))
                        annotations_data[sample_id] = item['annotation']
            else:
                annotations_data = data

            # 執行驗證
            validation_results = validator.validate_batch_annotations(annotations_data)

            # 匯出報告
            report_file = validator.export_validation_report(validation_results)
            print(f"驗證報告已生成: {report_file}")

        except Exception as e:
            logger.error(f"驗證失敗: {str(e)}")

    elif args.command == 'validate_batch':
        # 載入多個標註檔案
        all_annotations = {}

        for annotation_file in args.annotation_files:
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取標註資料
                if isinstance(data, dict) and 'annotations' in data:
                    file_annotations = data['annotations']
                elif isinstance(data, list):
                    file_annotations = {}
                    for item in data:
                        if 'annotation' in item:
                            sample_id = str(item.get('id', len(file_annotations)))
                            file_annotations[sample_id] = item['annotation']
                else:
                    file_annotations = data

                # 添加到總體資料中
                for sample_id, annotation in file_annotations.items():
                    # 添加檔案來源資訊
                    annotation['source_file'] = annotation_file
                    all_annotations[f"{Path(annotation_file).stem}_{sample_id}"] = annotation

            except Exception as e:
                logger.error(f"載入檔案失敗 {annotation_file}: {str(e)}")

        if all_annotations:
            # 執行批次驗證
            validation_results = validator.validate_batch_annotations(all_annotations)

            # 匯出報告
            report_file = validator.export_validation_report(validation_results)
            print(f"批次驗證報告已生成: {report_file}")
        else:
            print("沒有有效的標註資料")

    elif args.command == 'check_consistency':
        # 載入多標註者資料
        multi_annotations = defaultdict(dict)

        for annotation_file in args.annotation_files:
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取標註者ID
                if isinstance(data, dict) and 'annotator_id' in data:
                    annotator_id = data['annotator_id']
                    annotations = data.get('annotations', {})
                elif isinstance(data, list) and len(data) > 0:
                    annotator_id = data[0].get('annotator_id', Path(annotation_file).stem)
                    annotations = {}
                    for item in data:
                        if 'annotation' in item:
                            sample_id = str(item.get('id', len(annotations)))
                            annotations[sample_id] = item['annotation']
                else:
                    annotator_id = Path(annotation_file).stem
                    annotations = data

                # 組織多標註者資料
                for sample_id, annotation in annotations.items():
                    multi_annotations[sample_id][annotator_id] = annotation

            except Exception as e:
                logger.error(f"載入檔案失敗 {annotation_file}: {str(e)}")

        if multi_annotations:
            # 執行一致性檢查
            consistency_results = validator.check_inter_annotator_consistency(multi_annotations)

            # 匯出報告
            report_file = validator.export_validation_report(consistency_results,
                                                           f"consistency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            print(f"一致性檢查報告已生成: {report_file}")

            # 輸出摘要
            print("\n=== 一致性檢查摘要 ===")
            print(f"總樣本數: {consistency_results['total_samples']}")
            print(f"重疊樣本數: {consistency_results['overlap_samples']}")
            print(f"整體一致性: {consistency_results['overall_consistency']:.3f}")

            if consistency_results['consistency_by_dimension']:
                print("\n各維度一致性:")
                for dimension, result in consistency_results['consistency_by_dimension'].items():
                    print(f"  {dimension}: {result['consistency_ratio']:.3f}")
        else:
            print("沒有有效的多標註者資料")


if __name__ == "__main__":
    main()