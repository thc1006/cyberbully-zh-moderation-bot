"""
特徵提取器模組
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class NTUSDFeatureExtractor:
    """NTUSD詞典特徵提取器"""

    def __init__(self, ntusd_path: Optional[str] = None):
        """
        初始化NTUSD特徵提取器

        Args:
            ntusd_path: NTUSD詞典檔案路徑
        """
        self.positive_words: Set[str] = set()
        self.negative_words: Set[str] = set()
        self.loaded = False

        if ntusd_path and Path(ntusd_path).exists():
            self.load_ntusd(ntusd_path)

    def load_ntusd(self, ntusd_path: str) -> bool:
        """
        載入NTUSD詞典

        Args:
            ntusd_path: 詞典檔案路徑

        Returns:
            是否載入成功
        """
        try:
            path = Path(ntusd_path)

            if path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    ntusd_data = json.load(f)

                self.positive_words = set(ntusd_data.get('positive', []))
                self.negative_words = set(ntusd_data.get('negative', []))

            elif path.suffix == '.txt':
                # 假設格式為：每行一個詞，正面詞在前，負面詞在後，用空行分隔
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                current_set = self.positive_words
                for line in lines:
                    line = line.strip()
                    if not line:
                        current_set = self.negative_words
                        continue
                    current_set.add(line)

            else:
                logger.warning(f"Unsupported NTUSD file format: {path.suffix}")
                return False

            self.loaded = True
            logger.info(f"Loaded NTUSD: {len(self.positive_words)} positive, "
                       f"{len(self.negative_words)} negative words")
            return True

        except Exception as e:
            logger.warning(f"Failed to load NTUSD from {ntusd_path}: {e}")
            return False

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        提取NTUSD特徵

        Args:
            text: 輸入文字

        Returns:
            特徵字典
        """
        if not text or not self.loaded:
            return {
                'ntusd_pos_ratio': 0.0,
                'ntusd_neg_ratio': 0.0,
                'ntusd_sentiment': 0.0,
                'ntusd_pos_count': 0.0,
                'ntusd_neg_count': 0.0
            }

        # 中文按字符分割，也可以考慮詞級別分割
        chars = list(text)

        # 也嘗試詞級別匹配
        words = self._segment_words(text)
        all_units = chars + words

        pos_count = sum(1 for unit in all_units if unit in self.positive_words)
        neg_count = sum(1 for unit in all_units if unit in self.negative_words)
        total_units = len(all_units)

        pos_ratio = pos_count / total_units if total_units > 0 else 0.0
        neg_ratio = neg_count / total_units if total_units > 0 else 0.0
        sentiment = pos_ratio - neg_ratio

        return {
            'ntusd_pos_ratio': pos_ratio,
            'ntusd_neg_ratio': neg_ratio,
            'ntusd_sentiment': sentiment,
            'ntusd_pos_count': float(pos_count),
            'ntusd_neg_count': float(neg_count)
        }

    def _segment_words(self, text: str) -> List[str]:
        """簡單的詞彙切分"""
        # 這裡可以整合更sophisticated的分詞工具如jieba
        # 暫時使用簡單的規則
        words = []

        # 提取2-4字詞
        for length in [2, 3, 4]:
            for i in range(len(text) - length + 1):
                word = text[i:i+length]
                if self._is_valid_word(word):
                    words.append(word)

        return words

    def _is_valid_word(self, word: str) -> bool:
        """檢查是否為有效詞彙"""
        # 基本檢查：至少包含一個中文字符
        return bool(re.search(r'[\u4e00-\u9fff]', word))


class TextFeatureExtractor:
    """文字統計特徵提取器"""

    def __init__(self):
        pass

    def extract_basic_features(self, text: str) -> Dict[str, float]:
        """
        提取基本文字特徵

        Args:
            text: 輸入文字

        Returns:
            基本特徵字典
        """
        if not text:
            return self._get_empty_basic_features()

        features = {}

        # 長度特徵
        features['text_length'] = float(len(text))
        features['char_count'] = float(len(text))
        features['word_count'] = float(len(text.split()))

        # 字符類型特徵
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_chars = len([c for c in text if c.isalpha() and c.isascii()])
        digit_chars = len([c for c in text if c.isdigit()])
        punct_chars = len([c for c in text if not c.isalnum() and not c.isspace()])

        total_chars = len(text)
        features['chinese_ratio'] = chinese_chars / total_chars if total_chars > 0 else 0.0
        features['english_ratio'] = english_chars / total_chars if total_chars > 0 else 0.0
        features['digit_ratio'] = digit_chars / total_chars if total_chars > 0 else 0.0
        features['punct_ratio'] = punct_chars / total_chars if total_chars > 0 else 0.0

        # 重複字符特徵
        features['repeated_chars'] = self._count_repeated_chars(text)
        features['repeated_char_ratio'] = features['repeated_chars'] / total_chars if total_chars > 0 else 0.0

        # 大寫字母特徵
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / total_chars if total_chars > 0 else 0.0

        return features

    def extract_punctuation_features(self, text: str) -> Dict[str, float]:
        """
        提取標點符號特徵

        Args:
            text: 輸入文字

        Returns:
            標點符號特徵字典
        """
        if not text:
            return self._get_empty_punct_features()

        features = {}

        # 常見標點符號計數
        exclamation_count = text.count('!') + text.count('！')
        question_count = text.count('?') + text.count('？')
        period_count = text.count('.') + text.count('。')
        comma_count = text.count(',') + text.count('，')

        total_chars = len(text)
        features['exclamation_ratio'] = exclamation_count / total_chars if total_chars > 0 else 0.0
        features['question_ratio'] = question_count / total_chars if total_chars > 0 else 0.0
        features['period_ratio'] = period_count / total_chars if total_chars > 0 else 0.0
        features['comma_ratio'] = comma_count / total_chars if total_chars > 0 else 0.0

        # 連續標點符號
        features['consecutive_punct'] = len(re.findall(r'[!！?？。.]{2,}', text))

        return features

    def extract_emotion_indicators(self, text: str) -> Dict[str, float]:
        """
        提取情緒指示特徵

        Args:
            text: 輸入文字

        Returns:
            情緒指示特徵字典
        """
        if not text:
            return self._get_empty_emotion_features()

        features = {}

        # 情緒詞列表（簡化版）
        positive_words = {'好', '棒', '讚', '開心', '高興', '喜歡', '愛', '滿意', '不錯', '優秀'}
        negative_words = {'壞', '糟', '爛', '討厭', '生氣', '憤怒', '痛苦', '失望', '難過', '垃圾'}
        intense_words = {'超', '非常', '很', '特別', '極', '太', '超級', '最', '絕對', '完全'}

        # 計算情緒詞比例
        total_chars = len(text)
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        intense_count = sum(1 for word in intense_words if word in text)

        features['positive_words'] = float(pos_count)
        features['negative_words'] = float(neg_count)
        features['intense_words'] = float(intense_count)
        features['emotion_word_ratio'] = (pos_count + neg_count) / total_chars if total_chars > 0 else 0.0

        # 全大寫詞（表示強調）
        features['caps_words'] = len(re.findall(r'\b[A-Z]{2,}\b', text))

        return features

    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        提取所有文字特徵

        Args:
            text: 輸入文字

        Returns:
            完整特徵字典
        """
        features = {}

        features.update(self.extract_basic_features(text))
        features.update(self.extract_punctuation_features(text))
        features.update(self.extract_emotion_indicators(text))

        return features

    def _count_repeated_chars(self, text: str) -> float:
        """計算重複字符數量"""
        repeated_count = 0
        for match in re.finditer(r'(.)\1+', text):
            repeated_count += len(match.group()) - 1
        return float(repeated_count)

    def _get_empty_basic_features(self) -> Dict[str, float]:
        """獲取空的基本特徵"""
        return {
            'text_length': 0.0,
            'char_count': 0.0,
            'word_count': 0.0,
            'chinese_ratio': 0.0,
            'english_ratio': 0.0,
            'digit_ratio': 0.0,
            'punct_ratio': 0.0,
            'repeated_chars': 0.0,
            'repeated_char_ratio': 0.0,
            'uppercase_ratio': 0.0
        }

    def _get_empty_punct_features(self) -> Dict[str, float]:
        """獲取空的標點特徵"""
        return {
            'exclamation_ratio': 0.0,
            'question_ratio': 0.0,
            'period_ratio': 0.0,
            'comma_ratio': 0.0,
            'consecutive_punct': 0.0
        }

    def _get_empty_emotion_features(self) -> Dict[str, float]:
        """獲取空的情緒特徵"""
        return {
            'positive_words': 0.0,
            'negative_words': 0.0,
            'intense_words': 0.0,
            'emotion_word_ratio': 0.0,
            'caps_words': 0.0
        }


class CombinedFeatureExtractor:
    """組合特徵提取器"""

    def __init__(self, ntusd_path: Optional[str] = None):
        """
        初始化組合特徵提取器

        Args:
            ntusd_path: NTUSD詞典路徑
        """
        self.ntusd_extractor = NTUSDFeatureExtractor(ntusd_path)
        self.text_extractor = TextFeatureExtractor()

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        提取所有特徵

        Args:
            text: 輸入文字

        Returns:
            完整特徵字典
        """
        features = {}

        # NTUSD詞典特徵
        features.update(self.ntusd_extractor.extract_features(text))

        # 文字統計特徵
        features.update(self.text_extractor.extract_all_features(text))

        return features

    def get_feature_names(self) -> List[str]:
        """獲取所有特徵名稱"""
        # 提取一個空文字的特徵來獲取所有特徵名稱
        empty_features = self.extract_features("")
        return list(empty_features.keys())

    def extract_batch_features(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        批量提取特徵

        Args:
            texts: 文字列表

        Returns:
            特徵列表
        """
        return [self.extract_features(text) for text in texts]