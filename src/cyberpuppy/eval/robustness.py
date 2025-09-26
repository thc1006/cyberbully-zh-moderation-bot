"""
穩健性測試模組
評估霸凌偵測模型在各種攻擊和變化下的穩健性
"""

import logging
import numpy as np
import torch
import random
import string
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import pandas as pd
import json
import os
from datetime import datetime
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class RobustnessTestResult:
    """穩健性測試結果"""
    test_name: str
    original_text: str
    modified_text: str
    original_prediction: str
    modified_prediction: str
    original_confidence: float
    modified_confidence: float
    confidence_drop: float
    prediction_changed: bool
    attack_success: bool
    modification_type: str

@dataclass
class AdversarialAttackResult:
    """對抗攻擊結果"""
    attack_method: str
    success_rate: float
    average_confidence_drop: float
    successful_attacks: List[RobustnessTestResult]
    failed_attacks: List[RobustnessTestResult]
    attack_statistics: Dict[str, Any]

class RobustnessTestSuite:
    """穩健性測試套件"""

    def __init__(self, model, tokenizer, output_dir: str = "robustness_results"):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 設定設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # 常見的中文字符替換
        self.char_substitutions = {
            '的': ['得', '地'],
            '你': ['妳', '您'],
            '他': ['她', '它'],
            '了': ['嘞', '啦'],
            '是': ['係', '事'],
            '在': ['再'],
            '有': ['友', '又'],
            '不': ['布', '步'],
            '一': ['壹', '1'],
            '二': ['貳', '2'],
            '三': ['叁', '3'],
            '四': ['肆', '4'],
            '五': ['伍', '5']
        }

        # 同義詞替換
        self.synonym_substitutions = {
            '笨蛋': ['白痴', '蠢貨', '傻瓜'],
            '垃圾': ['廢物', '渣滓', '糞便'],
            '死': ['掛', '嗝屁', '歸西'],
            '滾': ['走開', '消失', '離開'],
            '討厭': ['煩人', '令人厭惡', '不喜歡'],
            '噁心': ['惡劣', '糟糕', '令人作嘔']
        }

    def run_comprehensive_test(self,
                              texts: List[str],
                              labels: List[str],
                              test_types: List[str] = None) -> Dict[str, Any]:
        """
        運行全面的穩健性測試

        Args:
            texts: 測試文本列表
            labels: 真實標籤列表
            test_types: 測試類型列表

        Returns:
            測試結果彙總
        """

        if test_types is None:
            test_types = [
                'character_substitution',
                'typo_injection',
                'synonym_replacement',
                'case_variation',
                'punctuation_variation',
                'space_insertion',
                'repetition_attack',
                'length_variation'
            ]

        logger.info(f"開始運行穩健性測試，測試類型: {test_types}")

        all_results = {}
        summary_stats = {}

        for test_type in test_types:
            logger.info(f"執行 {test_type} 測試...")

            if test_type == 'character_substitution':
                results = self._test_character_substitution(texts, labels)
            elif test_type == 'typo_injection':
                results = self._test_typo_injection(texts, labels)
            elif test_type == 'synonym_replacement':
                results = self._test_synonym_replacement(texts, labels)
            elif test_type == 'case_variation':
                results = self._test_case_variation(texts, labels)
            elif test_type == 'punctuation_variation':
                results = self._test_punctuation_variation(texts, labels)
            elif test_type == 'space_insertion':
                results = self._test_space_insertion(texts, labels)
            elif test_type == 'repetition_attack':
                results = self._test_repetition_attack(texts, labels)
            elif test_type == 'length_variation':
                results = self._test_length_variation(texts, labels)
            else:
                logger.warning(f"未知的測試類型: {test_type}")
                continue

            all_results[test_type] = results
            summary_stats[test_type] = self._calculate_test_statistics(results)

        # 生成總體統計
        overall_stats = self._generate_overall_statistics(all_results, summary_stats)

        # 保存結果
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'test_types': test_types,
            'detailed_results': all_results,
            'summary_statistics': summary_stats,
            'overall_statistics': overall_stats,
            'recommendations': self._generate_robustness_recommendations(summary_stats)
        }

        self._save_results(comprehensive_results, 'comprehensive_robustness_test')

        logger.info("穩健性測試完成")

        return comprehensive_results

    def _get_prediction(self, text: str) -> Tuple[str, float]:
        """獲取模型預測結果"""

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions.max().item()

        label_mapping = {0: 'none', 1: 'toxic', 2: 'severe'}
        prediction = label_mapping.get(predicted_class, 'unknown')

        return prediction, confidence

    def _test_character_substitution(self, texts: List[str], labels: List[str]) -> List[RobustnessTestResult]:
        """測試字符替換攻擊"""

        results = []

        for text, label in zip(texts[:50], labels[:50]):  # 限制測試數量
            original_pred, original_conf = self._get_prediction(text)

            # 隨機選擇要替換的字符
            chars_to_replace = [char for char in text if char in self.char_substitutions]

            if not chars_to_replace:
                continue

            # 替換隨機字符
            modified_text = text
            for char in chars_to_replace[:2]:  # 最多替換2個字符
                substitutes = self.char_substitutions[char]
                substitute = random.choice(substitutes)
                modified_text = modified_text.replace(char, substitute, 1)

            modified_pred, modified_conf = self._get_prediction(modified_text)

            result = RobustnessTestResult(
                test_name='character_substitution',
                original_text=text,
                modified_text=modified_text,
                original_prediction=original_pred,
                modified_prediction=modified_pred,
                original_confidence=original_conf,
                modified_confidence=modified_conf,
                confidence_drop=original_conf - modified_conf,
                prediction_changed=original_pred != modified_pred,
                attack_success=(original_pred != modified_pred and original_pred != 'none'),
                modification_type='字符替換'
            )

            results.append(result)

        return results

    def _test_typo_injection(self, texts: List[str], labels: List[str]) -> List[RobustnessTestResult]:
        """測試拼寫錯誤注入攻擊"""

        results = []

        for text, label in zip(texts[:50], labels[:50]):
            original_pred, original_conf = self._get_prediction(text)

            # 隨機插入拼寫錯誤
            modified_text = self._inject_typos(text)

            modified_pred, modified_conf = self._get_prediction(modified_text)

            result = RobustnessTestResult(
                test_name='typo_injection',
                original_text=text,
                modified_text=modified_text,
                original_prediction=original_pred,
                modified_prediction=modified_pred,
                original_confidence=original_conf,
                modified_confidence=modified_conf,
                confidence_drop=original_conf - modified_conf,
                prediction_changed=original_pred != modified_pred,
                attack_success=(original_pred != modified_pred and original_pred != 'none'),
                modification_type='拼寫錯誤注入'
            )

            results.append(result)

        return results

    def _inject_typos(self, text: str, typo_rate: float = 0.1) -> str:
        """向文本中注入拼寫錯誤"""

        chars = list(text)
        num_typos = max(1, int(len(chars) * typo_rate))

        for _ in range(num_typos):
            if len(chars) < 2:
                break

            # 隨機選擇位置
            pos = random.randint(1, len(chars) - 2)

            # 隨機選擇錯誤類型
            typo_type = random.choice(['swap', 'delete', 'insert', 'replace'])

            if typo_type == 'swap' and pos < len(chars) - 1:
                # 交換相鄰字符
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            elif typo_type == 'delete':
                # 刪除字符
                chars.pop(pos)
            elif typo_type == 'insert':
                # 插入隨機字符
                random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                chars.insert(pos, random_char)
            elif typo_type == 'replace':
                # 替換為隨機字符
                chars[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')

        return ''.join(chars)

    def _test_synonym_replacement(self, texts: List[str], labels: List[str]) -> List[RobustnessTestResult]:
        """測試同義詞替換攻擊"""

        results = []

        for text, label in zip(texts[:50], labels[:50]):
            original_pred, original_conf = self._get_prediction(text)

            # 查找可替換的詞
            modified_text = text
            for word, synonyms in self.synonym_substitutions.items():
                if word in text:
                    synonym = random.choice(synonyms)
                    modified_text = modified_text.replace(word, synonym, 1)
                    break  # 只替換一個詞

            # 如果沒有找到可替換的詞，跳過
            if modified_text == text:
                continue

            modified_pred, modified_conf = self._get_prediction(modified_text)

            result = RobustnessTestResult(
                test_name='synonym_replacement',
                original_text=text,
                modified_text=modified_text,
                original_prediction=original_pred,
                modified_prediction=modified_pred,
                original_confidence=original_conf,
                modified_confidence=modified_conf,
                confidence_drop=original_conf - modified_conf,
                prediction_changed=original_pred != modified_pred,
                attack_success=(original_pred != modified_pred and original_pred != 'none'),
                modification_type='同義詞替換'
            )

            results.append(result)

        return results

    def _test_case_variation(self, texts: List[str], labels: List[str]) -> List[RobustnessTestResult]:
        """測試大小寫變化攻擊"""

        results = []

        for text, label in zip(texts[:50], labels[:50]):
            original_pred, original_conf = self._get_prediction(text)

            # 隨機改變英文字母大小寫
            modified_text = self._vary_case(text)

            # 如果沒有英文字母，跳過
            if modified_text == text:
                continue

            modified_pred, modified_conf = self._get_prediction(modified_text)

            result = RobustnessTestResult(
                test_name='case_variation',
                original_text=text,
                modified_text=modified_text,
                original_prediction=original_pred,
                modified_prediction=modified_pred,
                original_confidence=original_conf,
                modified_confidence=modified_conf,
                confidence_drop=original_conf - modified_conf,
                prediction_changed=original_pred != modified_pred,
                attack_success=(original_pred != modified_pred and original_pred != 'none'),
                modification_type='大小寫變化'
            )

            results.append(result)

        return results

    def _vary_case(self, text: str) -> str:
        """隨機變化文本中的大小寫"""

        chars = list(text)
        for i, char in enumerate(chars):
            if char.isalpha() and random.random() < 0.3:  # 30% 機率改變大小寫
                if char.islower():
                    chars[i] = char.upper()
                else:
                    chars[i] = char.lower()

        return ''.join(chars)

    def _test_punctuation_variation(self, texts: List[str], labels: List[str]) -> List[RobustnessTestResult]:
        """測試標點符號變化攻擊"""

        results = []

        for text, label in zip(texts[:50], labels[:50]):
            original_pred, original_conf = self._get_prediction(text)

            # 修改標點符號
            modified_text = self._vary_punctuation(text)

            modified_pred, modified_conf = self._get_prediction(modified_text)

            result = RobustnessTestResult(
                test_name='punctuation_variation',
                original_text=text,
                modified_text=modified_text,
                original_prediction=original_pred,
                modified_prediction=modified_pred,
                original_confidence=original_conf,
                modified_confidence=modified_conf,
                confidence_drop=original_conf - modified_conf,
                prediction_changed=original_pred != modified_pred,
                attack_success=(original_pred != modified_pred and original_pred != 'none'),
                modification_type='標點符號變化'
            )

            results.append(result)

        return results

    def _vary_punctuation(self, text: str) -> str:
        """變化文本中的標點符號"""

        # 添加額外的標點符號
        punctuation_chars = ['!', '?', '.', ',', ';', ':', '...']
        modified_text = text

        # 隨機添加標點符號
        if random.random() < 0.5:
            punct = random.choice(punctuation_chars)
            insert_pos = random.randint(0, len(modified_text))
            modified_text = modified_text[:insert_pos] + punct + modified_text[insert_pos:]

        # 隨機刪除標點符號
        if random.random() < 0.3:
            for punct in punctuation_chars:
                if punct in modified_text:
                    modified_text = modified_text.replace(punct, '', 1)
                    break

        return modified_text

    def _test_space_insertion(self, texts: List[str], labels: List[str]) -> List[RobustnessTestResult]:
        """測試空格插入攻擊"""

        results = []

        for text, label in zip(texts[:50], labels[:50]):
            original_pred, original_conf = self._get_prediction(text)

            # 在隨機位置插入空格
            modified_text = self._insert_spaces(text)

            modified_pred, modified_conf = self._get_prediction(modified_text)

            result = RobustnessTestResult(
                test_name='space_insertion',
                original_text=text,
                modified_text=modified_text,
                original_prediction=original_pred,
                modified_prediction=modified_pred,
                original_confidence=original_conf,
                modified_confidence=modified_conf,
                confidence_drop=original_conf - modified_conf,
                prediction_changed=original_pred != modified_pred,
                attack_success=(original_pred != modified_pred and original_pred != 'none'),
                modification_type='空格插入'
            )

            results.append(result)

        return results

    def _insert_spaces(self, text: str, insertion_rate: float = 0.15) -> str:
        """在文本中隨機插入空格"""

        chars = list(text)
        num_insertions = max(1, int(len(chars) * insertion_rate))

        for _ in range(num_insertions):
            if len(chars) < 2:
                break

            pos = random.randint(1, len(chars) - 1)
            chars.insert(pos, ' ')

        return ''.join(chars)

    def _test_repetition_attack(self, texts: List[str], labels: List[str]) -> List[RobustnessTestResult]:
        """測試字符重複攻擊"""

        results = []

        for text, label in zip(texts[:50], labels[:50]):
            original_pred, original_conf = self._get_prediction(text)

            # 重複某些字符
            modified_text = self._repeat_characters(text)

            modified_pred, modified_conf = self._get_prediction(modified_text)

            result = RobustnessTestResult(
                test_name='repetition_attack',
                original_text=text,
                modified_text=modified_text,
                original_prediction=original_pred,
                modified_prediction=modified_pred,
                original_confidence=original_conf,
                modified_confidence=modified_conf,
                confidence_drop=original_conf - modified_conf,
                prediction_changed=original_pred != modified_pred,
                attack_success=(original_pred != modified_pred and original_pred != 'none'),
                modification_type='字符重複'
            )

            results.append(result)

        return results

    def _repeat_characters(self, text: str, repetition_rate: float = 0.1) -> str:
        """重複文本中的某些字符"""

        chars = list(text)
        num_repetitions = max(1, int(len(chars) * repetition_rate))

        for _ in range(num_repetitions):
            if len(chars) < 1:
                break

            pos = random.randint(0, len(chars) - 1)
            char_to_repeat = chars[pos]

            # 重複字符2-4次
            repeat_count = random.randint(2, 4)
            chars[pos] = char_to_repeat * repeat_count

        return ''.join(chars)

    def _test_length_variation(self, texts: List[str], labels: List[str]) -> List[RobustnessTestResult]:
        """測試長度變化攻擊"""

        results = []

        for text, label in zip(texts[:50], labels[:50]):
            original_pred, original_conf = self._get_prediction(text)

            # 隨機截斷或延長文本
            if random.random() < 0.5:
                # 截斷文本
                if len(text) > 10:
                    cut_length = random.randint(5, len(text) - 5)
                    modified_text = text[:cut_length]
                else:
                    modified_text = text
            else:
                # 延長文本（添加無意義內容）
                padding = "。這是額外的內容。" * random.randint(1, 3)
                modified_text = text + padding

            modified_pred, modified_conf = self._get_prediction(modified_text)

            result = RobustnessTestResult(
                test_name='length_variation',
                original_text=text,
                modified_text=modified_text,
                original_prediction=original_pred,
                modified_prediction=modified_pred,
                original_confidence=original_conf,
                modified_confidence=modified_conf,
                confidence_drop=original_conf - modified_conf,
                prediction_changed=original_pred != modified_pred,
                attack_success=(original_pred != modified_pred and original_pred != 'none'),
                modification_type='長度變化'
            )

            results.append(result)

        return results

    def _calculate_test_statistics(self, results: List[RobustnessTestResult]) -> Dict[str, Any]:
        """計算測試統計信息"""

        if not results:
            return {}

        total_tests = len(results)
        prediction_changes = sum(1 for r in results if r.prediction_changed)
        attack_successes = sum(1 for r in results if r.attack_success)

        confidence_drops = [r.confidence_drop for r in results]

        stats = {
            'total_tests': total_tests,
            'prediction_changes': prediction_changes,
            'attack_successes': attack_successes,
            'prediction_change_rate': prediction_changes / total_tests if total_tests > 0 else 0,
            'attack_success_rate': attack_successes / total_tests if total_tests > 0 else 0,
            'average_confidence_drop': np.mean(confidence_drops) if confidence_drops else 0,
            'max_confidence_drop': max(confidence_drops) if confidence_drops else 0,
            'min_confidence_drop': min(confidence_drops) if confidence_drops else 0,
            'std_confidence_drop': np.std(confidence_drops) if confidence_drops else 0
        }

        return stats

    def _generate_overall_statistics(self,
                                   all_results: Dict[str, List[RobustnessTestResult]],
                                   summary_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成總體統計信息"""

        overall_stats = {
            'total_attack_types': len(all_results),
            'overall_robustness_score': 0.0,
            'vulnerability_ranking': [],
            'most_vulnerable_attack': '',
            'least_vulnerable_attack': '',
            'robustness_level': 'unknown'
        }

        if not summary_stats:
            return overall_stats

        # 計算各攻擊類型的脆弱性分數
        vulnerability_scores = {}
        for attack_type, stats in summary_stats.items():
            success_rate = stats.get('attack_success_rate', 0)
            avg_conf_drop = stats.get('average_confidence_drop', 0)

            # 綜合評分（成功率 * 0.7 + 置信度下降 * 0.3）
            vulnerability_score = success_rate * 0.7 + min(avg_conf_drop, 1.0) * 0.3
            vulnerability_scores[attack_type] = vulnerability_score

        if vulnerability_scores:
            # 排序漏洞
            sorted_vulnerabilities = sorted(
                vulnerability_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            overall_stats['vulnerability_ranking'] = sorted_vulnerabilities
            overall_stats['most_vulnerable_attack'] = sorted_vulnerabilities[0][0]
            overall_stats['least_vulnerable_attack'] = sorted_vulnerabilities[-1][0]

            # 計算總體穩健性分數（1 - 平均脆弱性分數）
            avg_vulnerability = np.mean(list(vulnerability_scores.values()))
            overall_stats['overall_robustness_score'] = 1.0 - avg_vulnerability

            # 確定穩健性等級
            robustness_score = overall_stats['overall_robustness_score']
            if robustness_score >= 0.8:
                overall_stats['robustness_level'] = 'high'
            elif robustness_score >= 0.6:
                overall_stats['robustness_level'] = 'medium'
            else:
                overall_stats['robustness_level'] = 'low'

        return overall_stats

    def _generate_robustness_recommendations(self, summary_stats: Dict[str, Dict[str, Any]]) -> List[str]:
        """生成穩健性改進建議"""

        recommendations = []

        for attack_type, stats in summary_stats.items():
            success_rate = stats.get('attack_success_rate', 0)

            if success_rate > 0.3:  # 如果攻擊成功率超過30%
                if attack_type == 'character_substitution':
                    recommendations.append("增強對字符變體的識別能力，考慮使用字符正規化預處理")
                elif attack_type == 'typo_injection':
                    recommendations.append("提高對拼寫錯誤的容忍度，可考慮添加拼寫檢查模組")
                elif attack_type == 'synonym_replacement':
                    recommendations.append("改進語義理解能力，增加同義詞訓練數據")
                elif attack_type == 'case_variation':
                    recommendations.append("在預處理階段統一大小寫格式")
                elif attack_type == 'punctuation_variation':
                    recommendations.append("增強對標點符號變化的穩健性，考慮標點符號正規化")
                elif attack_type == 'space_insertion':
                    recommendations.append("改進詞彙切分能力，添加空格處理邏輯")
                elif attack_type == 'repetition_attack':
                    recommendations.append("添加字符重複檢測和正規化功能")
                elif attack_type == 'length_variation':
                    recommendations.append("提高對不同文本長度的適應性，考慮分段處理")

        # 通用建議
        if len([s for s in summary_stats.values() if s.get('attack_success_rate', 0) > 0.2]) > 3:
            recommendations.extend([
                "考慮使用對抗訓練提高模型穩健性",
                "增加數據增強技術",
                "實施多模型集成策略",
                "加強預處理和後處理規則"
            ])

        return recommendations

    def _save_results(self, results: Dict[str, Any], filename_prefix: str):
        """保存測試結果"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # 轉換為可序列化格式
        serializable_results = self._make_serializable(results)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        logger.info(f"穩健性測試結果已保存至: {filepath}")

    def _make_serializable(self, obj):
        """轉換對象為可序列化格式"""

        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

class AdversarialTester:
    """對抗攻擊測試器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_gradient_based_attack(self,
                                 texts: List[str],
                                 labels: List[str],
                                 epsilon: float = 0.01,
                                 num_steps: int = 10) -> AdversarialAttackResult:
        """
        運行基於梯度的對抗攻擊

        Args:
            texts: 輸入文本
            labels: 真實標籤
            epsilon: 攻擊強度
            num_steps: 攻擊步數

        Returns:
            攻擊結果
        """

        logger.info("開始執行基於梯度的對抗攻擊...")

        successful_attacks = []
        failed_attacks = []

        for text, label in zip(texts[:20], labels[:20]):  # 限制測試數量
            try:
                attack_result = self._single_gradient_attack(text, label, epsilon, num_steps)
                if attack_result.attack_success:
                    successful_attacks.append(attack_result)
                else:
                    failed_attacks.append(attack_result)
            except Exception as e:
                logger.error(f"攻擊 '{text[:20]}...' 時發生錯誤: {str(e)}")

        success_rate = len(successful_attacks) / len(texts[:20]) if texts else 0
        avg_confidence_drop = np.mean([
            attack.confidence_drop for attack in successful_attacks + failed_attacks
        ]) if successful_attacks + failed_attacks else 0

        attack_statistics = {
            'total_attempts': len(texts[:20]),
            'successful_attacks': len(successful_attacks),
            'failed_attacks': len(failed_attacks),
            'epsilon': epsilon,
            'num_steps': num_steps
        }

        return AdversarialAttackResult(
            attack_method='gradient_based',
            success_rate=success_rate,
            average_confidence_drop=avg_confidence_drop,
            successful_attacks=successful_attacks,
            failed_attacks=failed_attacks,
            attack_statistics=attack_statistics
        )

    def _single_gradient_attack(self,
                               text: str,
                               label: str,
                               epsilon: float,
                               num_steps: int) -> RobustnessTestResult:
        """對單個樣本執行基於梯度的攻擊"""

        # 獲取原始預測
        original_pred, original_conf = self._get_prediction(text)

        # 編碼文本
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # 設定目標標籤（與原始預測不同）
        target_labels = {'none': 1, 'toxic': 2, 'severe': 0}  # 切換標籤
        target_label = target_labels.get(original_pred, 0)

        # 將 input_ids 轉換為可求導的 embeddings
        embeddings = self.model.get_input_embeddings()
        input_embeddings = embeddings(input_ids)
        input_embeddings.requires_grad_(True)

        # 進行多步攻擊
        perturbed_embeddings = input_embeddings.clone()

        for step in range(num_steps):
            perturbed_embeddings.requires_grad_(True)

            # 前向傳播
            outputs = self.model(
                inputs_embeds=perturbed_embeddings,
                attention_mask=attention_mask
            )

            # 計算損失（目標是讓模型預測錯誤的標籤）
            loss = torch.nn.functional.cross_entropy(
                outputs.logits,
                torch.tensor([target_label]).to(self.device)
            )

            # 反向傳播獲取梯度
            loss.backward()

            # 更新 embeddings
            with torch.no_grad():
                grad = perturbed_embeddings.grad
                perturbation = epsilon * grad.sign()
                perturbed_embeddings = perturbed_embeddings + perturbation

                # 清零梯度
                perturbed_embeddings.grad = None

        # 將擾動後的 embeddings 轉換回文本（這是一個近似過程）
        # 實際上，從 embeddings 精確還原文本是困難的，這裡我們使用近似方法
        with torch.no_grad():
            # 找到最接近的 token
            vocab_embeddings = embeddings.weight  # (vocab_size, embedding_dim)

            # 計算每個位置最接近的 token
            perturbed_ids = []
            for pos_embedding in perturbed_embeddings.squeeze(0):
                # 計算與詞彙表中所有 embedding 的距離
                distances = torch.norm(vocab_embeddings - pos_embedding, dim=1)
                closest_token_id = torch.argmin(distances).item()
                perturbed_ids.append(closest_token_id)

            # 轉換為文本
            modified_text = self.tokenizer.decode(
                perturbed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

        # 獲取修改後的預測
        modified_pred, modified_conf = self._get_prediction(modified_text)

        return RobustnessTestResult(
            test_name='gradient_based_attack',
            original_text=text,
            modified_text=modified_text,
            original_prediction=original_pred,
            modified_prediction=modified_pred,
            original_confidence=original_conf,
            modified_confidence=modified_conf,
            confidence_drop=original_conf - modified_conf,
            prediction_changed=original_pred != modified_pred,
            attack_success=(original_pred != modified_pred),
            modification_type='基於梯度的對抗攻擊'
        )

    def _get_prediction(self, text: str) -> Tuple[str, float]:
        """獲取模型預測結果"""

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions.max().item()

        label_mapping = {0: 'none', 1: 'toxic', 2: 'severe'}
        prediction = label_mapping.get(predicted_class, 'unknown')

        return prediction, confidence