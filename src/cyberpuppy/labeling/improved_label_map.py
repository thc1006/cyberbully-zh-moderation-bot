#!/usr/bin/env python3
"""
改進的標籤映射模組
解決霸凌與毒性標籤完美相關的問題
"""

import re
import logging
from typing import Dict, List, Set, Optional, Any, Union
from dataclasses import dataclass

from .label_map import (
    ToxicityLevel, BullyingLevel, RoleType, EmotionType, UnifiedLabel
)

logger = logging.getLogger(__name__)


@dataclass
class TextFeatures:
    """文本特徵分析結果"""
    has_profanity: bool = False
    has_personal_attack: bool = False
    has_threat: bool = False
    has_discrimination: bool = False
    has_exclusion: bool = False
    has_repetitive_pattern: bool = False
    target_specific: bool = False
    emotional_intensity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'has_profanity': self.has_profanity,
            'has_personal_attack': self.has_personal_attack,
            'has_threat': self.has_threat,
            'has_discrimination': self.has_discrimination,
            'has_exclusion': self.has_exclusion,
            'has_repetitive_pattern': self.has_repetitive_pattern,
            'target_specific': self.target_specific,
            'emotional_intensity': self.emotional_intensity
        }


class ImprovedLabelMapper:
    """改進的標籤映射器，解決合成標籤問題"""

    def __init__(self):
        self.profanity_keywords = self._load_profanity_keywords()
        self.bullying_patterns = self._load_bullying_patterns()
        self.threat_keywords = self._load_threat_keywords()
        self.discrimination_keywords = self._load_discrimination_keywords()
        self.exclusion_patterns = self._load_exclusion_patterns()

    def _load_profanity_keywords(self) -> Set[str]:
        """載入粗俗語言關鍵詞"""
        return {
            # 基本粗俗語言
            '幹', '靠', '媽的', '操', '屎', '屁', '笨蛋', '白痴', '智障',
            '垃圾', '廢物', '低能', '腦殘', '神經病', '有病', '瘋子',
            # 性相關
            '色情', '淫穢', '猥褻', 'H', '18禁',
            # 英文粗話
            'fuck', 'shit', 'damn', 'stupid', 'idiot', 'moron'
        }

    def _load_bullying_patterns(self) -> List[Dict[str, Any]]:
        """載入霸凌模式"""
        return [
            # 人身攻擊模式
            {
                'pattern': r'你\s*(就是|真的是|根本是|明明是)\s*(笨蛋|白痴|智障|廢物|垃圾)',
                'type': 'personal_attack',
                'weight': 0.8
            },
            {
                'pattern': r'(你|妳)\s*(長得|樣子|外表)\s*(很|真|好)\s*(醜|噁心|討厭|可怕)',
                'type': 'appearance_attack',
                'weight': 0.9
            },
            # 排擠模式
            {
                'pattern': r'(不要|別|沒人)\s*(跟|和|理|鳥)\s*(你|妳|他|她)',
                'type': 'exclusion',
                'weight': 0.7
            },
            {
                'pattern': r'(大家|我們)\s*(都|全部)\s*(討厭|不喜歡|害怕)\s*(你|妳|他|她)',
                'type': 'group_exclusion',
                'weight': 0.8
            },
            # 威脅模式
            {
                'pattern': r'(要|會|讓你|讓妳)\s*(死|完蛋|好看|後悔|倒霉)',
                'type': 'threat',
                'weight': 1.0
            },
            {
                'pattern': r'(小心|等著|準備)\s*(點|吧|著|受死)',
                'type': 'warning_threat',
                'weight': 0.9
            }
        ]

    def _load_threat_keywords(self) -> Set[str]:
        """載入威脅關鍵詞"""
        return {
            '殺', '死', '打', '揍', '扁', '修理', '整', '教訓',
            '報復', '復仇', '後果', '完蛋', '倒霉', '好看',
            '小心', '等著', '準備', '受死', '活該', '該死'
        }

    def _load_discrimination_keywords(self) -> Set[str]:
        """載入歧視關鍵詞"""
        return {
            # 外表歧視
            '醜', '胖', '瘦', '矮', '高', '黑', '白', '噁心',
            # 能力歧視
            '笨', '蠢', '智障', '低能', '腦殘', '弱智', '殘廢',
            # 身份歧視
            '窮', '富', '窮鬼', '暴發戶', '土', '俗', '沒水準',
            # 性別歧視
            '娘', '娘砲', '男人婆', '公主病', '媽寶'
        }

    def _load_exclusion_patterns(self) -> List[str]:
        """載入排擠模式"""
        return [
            r'不要.*理.*',
            r'沒人.*喜歡.*',
            r'大家.*討厭.*',
            r'離.*遠一點',
            r'不准.*參加',
            r'你.*不配',
            r'滾.*',
            r'消失.*'
        ]

    def analyze_text_features(self, text: str) -> TextFeatures:
        """分析文本特徵"""
        if not text:
            return TextFeatures()

        text_lower = text.lower()
        features = TextFeatures()

        # 檢查粗俗語言
        features.has_profanity = any(word in text for word in self.profanity_keywords)

        # 檢查威脅
        features.has_threat = any(keyword in text for keyword in self.threat_keywords)

        # 檢查歧視
        features.has_discrimination = any(word in text for word in self.discrimination_keywords)

        # 檢查人身攻擊和排擠模式
        for pattern_info in self.bullying_patterns:
            if re.search(pattern_info['pattern'], text):
                if pattern_info['type'] in ['personal_attack', 'appearance_attack']:
                    features.has_personal_attack = True
                elif pattern_info['type'] in ['exclusion', 'group_exclusion']:
                    features.has_exclusion = True
                elif pattern_info['type'] in ['threat', 'warning_threat']:
                    features.has_threat = True

        # 額外的人身攻擊檢查（針對直接侮辱）
        personal_attack_patterns = [
            r'你\s*(就是|真的是|根本是|明明是)\s*(白痴|笨蛋|智障|廢物|垃圾)',
            r'你\s*(這個|真是個)\s*(白痴|笨蛋|智障|廢物|垃圾)',
            r'(白痴|笨蛋|智障|廢物|垃圾)\s*(你|妳)'
        ]

        for pattern in personal_attack_patterns:
            if re.search(pattern, text):
                features.has_personal_attack = True
                break

        # 檢查排擠模式
        for pattern in self.exclusion_patterns:
            if re.search(pattern, text):
                features.has_exclusion = True
                break

        # 檢查是否針對特定目標
        personal_pronouns = ['你', '妳', '他', '她', '你們', '妳們', '他們', '她們']
        features.target_specific = any(pronoun in text for pronoun in personal_pronouns)

        # 檢查重複模式（霸凌常見特徵）
        words = text.split()
        if len(words) > 2:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_count = max(word_counts.values()) if word_counts else 0
            features.has_repetitive_pattern = max_count >= 3

        # 計算情緒強度（基於標點符號和大寫）
        exclamation_count = text.count('！') + text.count('!')
        question_count = text.count('？') + text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        features.emotional_intensity = min(1.0, (exclamation_count * 0.3 +
                                                question_count * 0.2 +
                                                caps_ratio * 0.5))

        return features

    def improved_cold_mapping(self, cold_label: int, text: str = "") -> UnifiedLabel:
        """
        改進的 COLD 標籤映射，解決完美相關問題

        Args:
            cold_label: COLD 原始標籤 (0 或 1)
            text: 文本內容，用於特徵分析

        Returns:
            UnifiedLabel 物件
        """
        features = self.analyze_text_features(text) if text else TextFeatures()

        if cold_label == 0:
            # 非冒犯性內容
            return UnifiedLabel(
                toxicity=ToxicityLevel.NONE,
                toxicity_score=0.0,
                bullying=BullyingLevel.NONE,
                bullying_score=0.0,
                role=RoleType.NONE,
                emotion=EmotionType.NEUTRAL,
                emotion_intensity=2,
                confidence=0.95,
                source_dataset="cold_improved"
            )

        elif cold_label == 1:
            # 冒犯性內容 - 需要進一步分析

            # 計算毒性分數
            toxicity_score = 0.3  # 基礎分數
            if features.has_profanity:
                toxicity_score += 0.3
            if features.has_discrimination:
                toxicity_score += 0.2
            if features.emotional_intensity > 0.5:
                toxicity_score += 0.2

            toxicity_score = min(1.0, toxicity_score)

            # 確定毒性等級
            if features.has_threat:
                toxicity = ToxicityLevel.SEVERE
                toxicity_score = max(0.8, toxicity_score)
            elif toxicity_score >= 0.8:
                toxicity = ToxicityLevel.SEVERE
            elif toxicity_score >= 0.4:
                toxicity = ToxicityLevel.TOXIC
            else:
                toxicity = ToxicityLevel.NONE

            # 計算霸凌分數 - 關鍵改進點
            bullying_score = 0.0
            bullying = BullyingLevel.NONE

            # 只有具備霸凌特徵才認定為霸凌
            bullying_indicators = 0

            if features.has_personal_attack:
                bullying_score += 0.4
                bullying_indicators += 1

            if features.has_threat:
                bullying_score += 0.5
                bullying_indicators += 1

            if features.has_exclusion:
                bullying_score += 0.3
                bullying_indicators += 1

            if features.target_specific and bullying_indicators > 0:
                bullying_score += 0.2

            if features.has_repetitive_pattern and bullying_indicators > 0:
                bullying_score += 0.1

            bullying_score = min(1.0, bullying_score)

            # 確定霸凌等級
            if bullying_score >= 0.7:
                if features.has_threat:
                    bullying = BullyingLevel.THREAT
                else:
                    bullying = BullyingLevel.HARASSMENT
            elif bullying_score >= 0.3:
                bullying = BullyingLevel.HARASSMENT
            else:
                bullying = BullyingLevel.NONE

            # 確定角色（如果有霸凌行為）
            role = RoleType.NONE
            if bullying != BullyingLevel.NONE:
                role = RoleType.PERPETRATOR

            # 確定情緒
            emotion = EmotionType.NEGATIVE
            emotion_intensity = min(4, max(1, int(features.emotional_intensity * 4) + 1))

            return UnifiedLabel(
                toxicity=toxicity,
                toxicity_score=toxicity_score,
                bullying=bullying,
                bullying_score=bullying_score,
                role=role,
                emotion=emotion,
                emotion_intensity=emotion_intensity,
                confidence=0.85,
                source_dataset="cold_improved"
            )

        else:
            # 未知標籤，預設為安全
            logger.warning(f"Unknown COLD label: {cold_label}")
            return UnifiedLabel(source_dataset="cold_improved")

    def batch_improve_cold_labels(self, labels: List[int], texts: List[str]) -> List[UnifiedLabel]:
        """
        批次改進 COLD 標籤

        Args:
            labels: COLD 標籤列表
            texts: 對應的文本列表

        Returns:
            改進後的 UnifiedLabel 列表
        """
        if len(labels) != len(texts):
            raise ValueError("Labels and texts must have the same length")

        return [
            self.improved_cold_mapping(label, text)
            for label, text in zip(labels, texts)
        ]

    def analyze_label_distribution(self, labels: List[UnifiedLabel]) -> Dict[str, Any]:
        """
        分析標籤分佈，檢查改進效果

        Args:
            labels: UnifiedLabel 列表

        Returns:
            分佈統計資訊
        """
        total = len(labels)
        if total == 0:
            return {}

        # 統計各類標籤
        toxicity_counts = {level.value: 0 for level in ToxicityLevel}
        bullying_counts = {level.value: 0 for level in BullyingLevel}
        role_counts = {role.value: 0 for role in RoleType}

        for label in labels:
            toxicity_counts[label.toxicity.value] += 1
            bullying_counts[label.bullying.value] += 1
            role_counts[label.role.value] += 1

        # 計算相關性（檢查是否仍然完美相關）
        toxic_and_bullying = sum(1 for label in labels
                               if label.toxicity != ToxicityLevel.NONE
                               and label.bullying != BullyingLevel.NONE)

        toxic_not_bullying = sum(1 for label in labels
                               if label.toxicity != ToxicityLevel.NONE
                               and label.bullying == BullyingLevel.NONE)

        bullying_not_toxic = sum(1 for label in labels
                               if label.bullying != BullyingLevel.NONE
                               and label.toxicity == ToxicityLevel.NONE)

        # 計算分離度（越高表示分離得越好）
        separation_score = (toxic_not_bullying + bullying_not_toxic) / total if total > 0 else 0

        return {
            'total_samples': total,
            'toxicity_distribution': {k: v/total for k, v in toxicity_counts.items()},
            'bullying_distribution': {k: v/total for k, v in bullying_counts.items()},
            'role_distribution': {k: v/total for k, v in role_counts.items()},
            'correlation_analysis': {
                'toxic_and_bullying': toxic_and_bullying,
                'toxic_not_bullying': toxic_not_bullying,
                'bullying_not_toxic': bullying_not_toxic,
                'separation_score': separation_score
            },
            'improvement_metrics': {
                'label_separation_achieved': separation_score > 0.1,
                'bullying_precision_improved': toxic_not_bullying > 0,
                'toxicity_recall_maintained': toxic_and_bullying > 0
            }
        }

    def compare_with_original(self, original_labels: List[int], texts: List[str]) -> Dict[str, Any]:
        """
        與原始映射比較，展示改進效果

        Args:
            original_labels: 原始 COLD 標籤
            texts: 文本內容

        Returns:
            比較結果
        """
        from .label_map import LabelMapper

        # 原始映射
        original_mapper = LabelMapper()
        original_unified = [original_mapper.from_cold_to_unified(label) for label in original_labels]

        # 改進映射
        improved_unified = self.batch_improve_cold_labels(original_labels, texts)

        # 分析分佈
        original_dist = self.analyze_label_distribution(original_unified)
        improved_dist = self.analyze_label_distribution(improved_unified)

        # 計算改進指標
        original_separation = original_dist.get('correlation_analysis', {}).get('separation_score', 0)
        improved_separation = improved_dist.get('correlation_analysis', {}).get('separation_score', 0)

        return {
            'original_distribution': original_dist,
            'improved_distribution': improved_dist,
            'improvement_summary': {
                'separation_improvement': improved_separation - original_separation,
                'toxicity_diversity_increased': (
                    len([x for x in improved_unified if x.toxicity != ToxicityLevel.NONE]) !=
                    len([x for x in improved_unified if x.bullying != BullyingLevel.NONE])
                ),
                'expected_f1_improvement': self._estimate_f1_improvement(
                    original_separation, improved_separation
                )
            }
        }

    def _estimate_f1_improvement(self, original_sep: float, improved_sep: float) -> float:
        """
        估算 F1 分數改進

        基於標籤分離度的改進來估算 F1 提升
        """
        if improved_sep <= original_sep:
            return 0.0

        # 經驗公式：分離度改進轉換為 F1 改進
        # 假設完美分離（separation_score=1）能帶來最多 0.15 的 F1 提升
        max_f1_gain = 0.15
        separation_improvement = improved_sep - original_sep
        estimated_f1_gain = separation_improvement * max_f1_gain

        return min(max_f1_gain, estimated_f1_gain)


# 便利函式
def improved_cold_to_unified(label: int, text: str = "") -> UnifiedLabel:
    """改進的 COLD 到統一格式轉換"""
    mapper = ImprovedLabelMapper()
    return mapper.improved_cold_mapping(label, text)


def analyze_text_bullying_features(text: str) -> Dict[str, Any]:
    """分析文本的霸凌特徵"""
    mapper = ImprovedLabelMapper()
    features = mapper.analyze_text_features(text)
    return features.to_dict()