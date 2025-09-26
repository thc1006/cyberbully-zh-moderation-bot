"""
資料品質驗證器
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """資料品質驗證器"""

    def __init__(self):
        """初始化驗證器"""
        # 定義有效標籤
        self.valid_labels = {
            'toxicity': {'none', 'toxic', 'severe'},
            'bullying': {'none', 'harassment', 'threat'},
            'role': {'none', 'perpetrator', 'victim', 'bystander'},
            'emotion': {'pos', 'neu', 'neg'},
            'emotion_strength': set(range(5))  # 0-4
        }

        # 文字品質檢查規則
        self.min_text_length = 3
        self.max_text_length = 1000
        self.min_chinese_ratio = 0.1

    def validate_text_quality(self, text: str) -> Dict[str, Any]:
        """
        驗證文字品質

        Args:
            text: 輸入文字

        Returns:
            驗證結果
        """
        if not text or pd.isna(text):
            return {
                'valid': False,
                'reason': 'empty_text',
                'severity': 'high'
            }

        text = str(text).strip()

        # 檢查長度
        if len(text) < self.min_text_length:
            return {
                'valid': False,
                'reason': 'text_too_short',
                'severity': 'medium',
                'length': len(text)
            }

        if len(text) > self.max_text_length:
            return {
                'valid': False,
                'reason': 'text_too_long',
                'severity': 'low',
                'length': len(text)
            }

        # 檢查中文字符比例
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        total_chars = len(text)
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0

        if chinese_ratio < self.min_chinese_ratio:
            return {
                'valid': False,
                'reason': 'insufficient_chinese',
                'severity': 'medium',
                'chinese_ratio': chinese_ratio
            }

        # 檢查是否為亂碼或重複字符
        if self._is_gibberish(text):
            return {
                'valid': False,
                'reason': 'gibberish_text',
                'severity': 'high'
            }

        # 檢查是否為廣告或垃圾訊息
        if self._is_spam(text):
            return {
                'valid': False,
                'reason': 'spam_content',
                'severity': 'medium'
            }

        return {
            'valid': True,
            'reason': 'valid',
            'severity': 'none',
            'metrics': {
                'length': len(text),
                'chinese_ratio': chinese_ratio,
                'char_diversity': self._calculate_char_diversity(text)
            }
        }

    def validate_label_format(self, label: Dict[str, Any]) -> Dict[str, Any]:
        """
        驗證標籤格式

        Args:
            label: 標籤字典

        Returns:
            驗證結果
        """
        errors = []
        warnings = []

        # 檢查必要欄位
        required_fields = set(self.valid_labels.keys())
        missing_fields = required_fields - set(label.keys())

        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")

        # 檢查標籤值
        for field, value in label.items():
            if field in self.valid_labels:
                valid_values = self.valid_labels[field]
                if value not in valid_values:
                    errors.append(f"Invalid {field} value: {value}, expected one of {valid_values}")

        # 檢查標籤一致性
        consistency_issues = self._check_label_consistency(label)
        warnings.extend(consistency_issues)

        is_valid = len(errors) == 0

        return {
            'valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'severity': 'high' if errors else 'low' if warnings else 'none'
        }

    def validate_sample(self, text: str, label: Dict[str, Any]) -> Dict[str, Any]:
        """
        驗證單個樣本

        Args:
            text: 文字內容
            label: 標籤

        Returns:
            驗證結果
        """
        text_validation = self.validate_text_quality(text)
        label_validation = self.validate_label_format(label)

        return {
            'text': text_validation,
            'label': label_validation,
            'overall_valid': text_validation['valid'] and label_validation['valid']
        }

    def validate_dataset(self,
                        data: List[Tuple[str, Dict]],
                        sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        驗證整個資料集

        Args:
            data: 資料列表
            sample_size: 抽樣大小（None表示全部驗證）

        Returns:
            驗證報告
        """
        # 抽樣驗證（如果資料太大）
        if sample_size and len(data) > sample_size:
            import random
            data = random.sample(data, sample_size)
            logger.info(f"Sampling {sample_size} samples for validation")

        validation_results = {
            'total_samples': len(data),
            'valid_samples': 0,
            'invalid_samples': 0,
            'text_issues': defaultdict(int),
            'label_issues': defaultdict(int),
            'severity_counts': defaultdict(int),
            'detailed_issues': []
        }

        for i, (text, label) in enumerate(data):
            sample_validation = self.validate_sample(text, label)

            if sample_validation['overall_valid']:
                validation_results['valid_samples'] += 1
            else:
                validation_results['invalid_samples'] += 1

                # 記錄問題詳情
                issue_detail = {
                    'sample_index': i,
                    'text_length': len(text) if text else 0,
                    'issues': []
                }

                # 文字問題
                if not sample_validation['text']['valid']:
                    reason = sample_validation['text']['reason']
                    severity = sample_validation['text']['severity']
                    validation_results['text_issues'][reason] += 1
                    validation_results['severity_counts'][severity] += 1
                    issue_detail['issues'].append({
                        'type': 'text',
                        'reason': reason,
                        'severity': severity
                    })

                # 標籤問題
                if not sample_validation['label']['valid']:
                    for error in sample_validation['label']['errors']:
                        validation_results['label_issues'][error] += 1
                        validation_results['severity_counts']['high'] += 1
                        issue_detail['issues'].append({
                            'type': 'label',
                            'reason': error,
                            'severity': 'high'
                        })

                if len(validation_results['detailed_issues']) < 100:  # 限制詳細問題數量
                    validation_results['detailed_issues'].append(issue_detail)

        # 計算品質分數
        validation_results['quality_score'] = (
            validation_results['valid_samples'] / validation_results['total_samples']
            if validation_results['total_samples'] > 0 else 0
        )

        return validation_results

    def validate_data_file(self, file_path: str) -> Dict[str, Any]:
        """
        驗證資料檔案

        Args:
            file_path: 檔案路徑

        Returns:
            驗證報告
        """
        path = Path(file_path)

        if not path.exists():
            return {
                'valid': False,
                'error': 'file_not_found',
                'file_path': str(path)
            }

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 轉換為標準格式
            processed_data = []
            for item in data:
                if isinstance(item, dict) and 'text' in item and 'label' in item:
                    processed_data.append((item['text'], item['label']))

            validation_result = self.validate_dataset(processed_data)
            validation_result['file_path'] = str(path)
            validation_result['file_size'] = path.stat().st_size
            validation_result['valid'] = True

            return validation_result

        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'file_path': str(path)
            }

    def generate_validation_report(self,
                                  data_dir: str,
                                  output_path: str) -> Dict[str, Any]:
        """
        生成完整的驗證報告

        Args:
            data_dir: 資料目錄
            output_path: 輸出路徑

        Returns:
            驗證報告
        """
        data_dir = Path(data_dir)
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'data_directory': str(data_dir),
            'files': {},
            'overall_summary': {
                'total_files': 0,
                'valid_files': 0,
                'total_samples': 0,
                'valid_samples': 0,
                'overall_quality_score': 0.0
            }
        }

        # 驗證所有JSON檔案
        json_files = list(data_dir.glob('*.json'))

        for file_path in json_files:
            if file_path.name.startswith('.'):
                continue

            file_validation = self.validate_data_file(str(file_path))
            report['files'][file_path.name] = file_validation

            report['overall_summary']['total_files'] += 1

            if file_validation.get('valid', False):
                report['overall_summary']['valid_files'] += 1
                report['overall_summary']['total_samples'] += file_validation['total_samples']
                report['overall_summary']['valid_samples'] += file_validation['valid_samples']

        # 計算整體品質分數
        if report['overall_summary']['total_samples'] > 0:
            report['overall_summary']['overall_quality_score'] = (
                report['overall_summary']['valid_samples'] /
                report['overall_summary']['total_samples']
            )

        # 儲存報告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Validation report saved to {output_path}")
        return report

    def _is_gibberish(self, text: str) -> bool:
        """檢查是否為亂碼"""
        # 檢查重複字符比例
        char_counts = Counter(text)
        max_char_ratio = max(count / len(text) for count in char_counts.values())

        if max_char_ratio > 0.5:  # 超過50%為同一字符
            return True

        # 檢查字符多樣性
        unique_chars = len(set(text))
        if len(text) > 20 and unique_chars < 5:
            return True

        return False

    def _is_spam(self, text: str) -> bool:
        """檢查是否為垃圾訊息"""
        spam_patterns = [
            r'http[s]?://\S+',  # URL
            r'\b[\w.-]+@[\w.-]+\.\w+\b',  # Email
            r'\d{3}-?\d{4}-?\d{4}',  # 電話號碼
            r'[優惠|促銷|特價|免費|贈送]{2,}',  # 廣告用詞
        ]

        for pattern in spam_patterns:
            if re.search(pattern, text):
                return True

        return False

    def _calculate_char_diversity(self, text: str) -> float:
        """計算字符多樣性"""
        if not text:
            return 0.0

        unique_chars = len(set(text))
        total_chars = len(text)

        return unique_chars / total_chars

    def _check_label_consistency(self, label: Dict[str, Any]) -> List[str]:
        """檢查標籤一致性"""
        warnings = []

        # 檢查毒性和霸凌標籤的一致性
        toxicity = label.get('toxicity', 'none')
        bullying = label.get('bullying', 'none')

        if toxicity in ['toxic', 'severe'] and bullying == 'none':
            warnings.append("Toxic content but no bullying behavior marked")

        if bullying in ['harassment', 'threat'] and toxicity == 'none':
            warnings.append("Bullying behavior but no toxicity marked")

        # 檢查情緒和情緒強度的一致性
        emotion = label.get('emotion', 'neu')
        emotion_strength = label.get('emotion_strength', 0)

        if emotion == 'neu' and emotion_strength > 1:
            warnings.append("Neutral emotion with high strength")

        if emotion in ['pos', 'neg'] and emotion_strength == 0:
            warnings.append("Non-neutral emotion with zero strength")

        return warnings


# 便利函數
def validate_training_data(data_dir: str,
                          output_dir: str = None) -> Dict[str, Any]:
    """
    驗證訓練資料的便利函數

    Args:
        data_dir: 資料目錄
        output_dir: 輸出目錄

    Returns:
        驗證報告
    """
    validator = DataQualityValidator()

    if output_dir is None:
        output_dir = data_dir

    output_path = Path(output_dir) / 'validation_report.json'

    return validator.generate_validation_report(data_dir, str(output_path))


if __name__ == "__main__":
    # 測試驗證器
    validator = DataQualityValidator()

    # 測試文字驗證
    test_texts = [
        "這是一個正常的中文句子。",
        "abc",  # 太短
        "aaaaaaaaaaaaaaaaaaaaa",  # 重複字符
        "",  # 空字串
        "這是一個很長的句子" + "a" * 1000,  # 太長
    ]

    for text in test_texts:
        result = validator.validate_text_quality(text)
        print(f"Text: '{text[:20]}...' -> {result}")

    # 測試標籤驗證
    test_labels = [
        {'toxicity': 'toxic', 'bullying': 'harassment', 'role': 'none', 'emotion': 'neg', 'emotion_strength': 3},
        {'toxicity': 'invalid'},  # 無效標籤
        {'toxicity': 'none'},  # 缺少欄位
    ]

    for label in test_labels:
        result = validator.validate_label_format(label)
        print(f"Label: {label} -> {result}")