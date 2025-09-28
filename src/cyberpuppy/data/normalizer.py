"""
資料正規化與標籤統一化處理器
"""

import hashlib
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import opencc

    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False
    logging.warning("OpenCC not available. Traditional/Simplified conversion disabled.")

logger = logging.getLogger(__name__)


class DataNormalizer:
    """資料正規化處理器"""

    def __init__(self, target_format: str = "traditional"):
        """
        初始化資料正規化器

        Args:
            target_format: 目標格式 ('traditional' 或 'simplified')
        """
        self.target_format = target_format

        # 繁簡轉換器
        if OPENCC_AVAILABLE:
            try:
                if target_format == "traditional":
                    self.converter = opencc.OpenCC("s2t")  # 簡體轉繁體
                else:
                    self.converter = opencc.OpenCC("t2s")  # 繁體轉簡體
            except Exception as e:
                logger.warning(f"Failed to initialize OpenCC: {e}")
                self.converter = None
        else:
            self.converter = None

    def normalize_text(self, text: str) -> str:
        """
        文字正規化

        Args:
            text: 輸入文字

        Returns:
            正規化後的文字
        """
        if not text or pd.isna(text):
            return ""

        text = str(text).strip()

        # 移除控制字符
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # 統一空白字符
        text = re.sub(r"\s+", " ", text)

        # 移除多餘的標點符號
        text = re.sub(r"[。！？]{2,}", "。", text)
        text = re.sub(r"[，、]{2,}", "，", text)

        # 繁簡轉換
        if self.converter:
            try:
                text = self.converter.convert(text)
            except Exception as e:
                logger.warning(f"Conversion failed: {e}")

        return text.strip()

    def clean_text(self, text: str) -> str:
        """
        深度清理文字

        Args:
            text: 輸入文字

        Returns:
            清理後的文字
        """
        if not text:
            return ""

        # 移除URL
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # 移除電子郵件
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

        # 移除電話號碼
        text = re.sub(r"\b\d{4}-\d{4}-\d{4}\b|\b\d{3}-\d{4}-\d{4}\b", "", text)

        # 移除過多的重複字符
        text = re.sub(r"(.)\1{3,}", r"\1\1", text)

        # 移除HTML標籤
        text = re.sub(r"<[^>]+>", "", text)

        return text.strip()

    def remove_duplicates(
        self, data: List[Tuple[str, Dict]], similarity_threshold: float = 0.95
    ) -> List[Tuple[str, Dict]]:
        """
        移除重複樣本

        Args:
            data: 資料列表
            similarity_threshold: 相似度閾值

        Returns:
            去重後的資料列表
        """
        unique_data = []
        seen_hashes = set()

        for text, label in data:
            # 計算文字雜湊
            normalized_text = self.normalize_text(text.lower())
            text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_data.append((text, label))

        removed_count = len(data) - len(unique_data)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate samples")

        return unique_data

    def balance_samples(
        self, data: List[Tuple[str, Dict]], balance_key: str = "toxicity", max_ratio: float = 3.0
    ) -> List[Tuple[str, Dict]]:
        """
        平衡樣本分佈

        Args:
            data: 資料列表
            balance_key: 平衡的標籤鍵
            max_ratio: 最大類別比例

        Returns:
            平衡後的資料列表
        """
        label_groups = defaultdict(list)

        # 按標籤分組
        for text, label in data:
            key_value = label[balance_key]
            label_groups[key_value].append((text, label))

        # 計算目標樣本數
        min_count = min(len(group) for group in label_groups.values())
        max_count = int(min_count * max_ratio)

        balanced_data = []
        for key_value, group in label_groups.items():
            if len(group) > max_count:
                # 隨機採樣
                np.random.shuffle(group)
                group = group[:max_count]
            balanced_data.extend(group)

        logger.info(f"Balanced dataset: {len(data)} -> {len(balanced_data)} samples")
        return balanced_data


class LabelUnifier:
    """標籤統一化處理器"""

    # 統一標籤定義
    TOXICITY_LABELS = {"none", "toxic", "severe"}
    BULLYING_LABELS = {"none", "harassment", "threat"}
    ROLE_LABELS = {"none", "perpetrator", "victim", "bystander"}
    EMOTION_LABELS = {"pos", "neu", "neg"}
    EMOTION_STRENGTH_RANGE = list(range(5))  # 0-4

    def __init__(self):
        pass

    def create_default_label(self) -> Dict[str, Any]:
        """創建預設標籤"""
        return {
            "toxicity": "none",
            "bullying": "none",
            "role": "none",
            "emotion": "neu",
            "emotion_strength": 0,
        }

    def unify_cold_labels(self, row: pd.Series) -> Dict[str, Any]:
        """
        統一COLD資料標籤

        Args:
            row: 資料行

        Returns:
            統一後的標籤
        """
        label = self.create_default_label()

        # 檢查不同的標籤欄位
        cold_label = None
        if "label" in row:
            cold_label = int(row["label"])
        elif "offensive" in row:
            cold_label = int(row["offensive"])
        elif "hate" in row:
            cold_label = int(row["hate"])

        if cold_label == 1:
            label["toxicity"] = "toxic"
            label["bullying"] = "harassment"

            # 檢查是否為嚴重等級
            if "hate" in row and int(row.get("hate", 0)) == 1:
                label["toxicity"] = "severe"
                label["bullying"] = "threat"

        return label

    def unify_sentiment_labels(
        self, sentiment_label: int, dataset_name: str = ""
    ) -> Dict[str, Any]:
        """
        統一情緒分析標籤

        Args:
            sentiment_label: 情緒標籤
            dataset_name: 資料集名稱

        Returns:
            統一後的標籤
        """
        label = self.create_default_label()

        if dataset_name.lower() == "chnsenticorp":
            # ChnSentiCorp: 0=負面, 1=正面
            if sentiment_label == 1:
                label["emotion"] = "pos"
                label["emotion_strength"] = 3
            elif sentiment_label == 0:
                label["emotion"] = "neg"
                label["emotion_strength"] = 3

        elif dataset_name.lower() == "dmsc":
            # DMSC: 1-5分等級
            if sentiment_label >= 4:
                label["emotion"] = "pos"
                label["emotion_strength"] = min(4, sentiment_label - 1)
            elif sentiment_label <= 2:
                label["emotion"] = "neg"
                label["emotion_strength"] = min(4, 3 - sentiment_label)
            else:
                label["emotion"] = "neu"
                label["emotion_strength"] = 1

        else:
            # 其他資料集的預設處理
            if sentiment_label > 0:
                label["emotion"] = "pos"
                label["emotion_strength"] = min(4, sentiment_label)
            elif sentiment_label < 0:
                label["emotion"] = "neg"
                label["emotion_strength"] = min(4, abs(sentiment_label))

        return label

    def unify_manual_labels(self, manual_label: Dict[str, Any]) -> Dict[str, Any]:
        """
        統一人工標註標籤

        Args:
            manual_label: 人工標註標籤

        Returns:
            統一後的標籤
        """
        label = self.create_default_label()

        # 複製有效的標籤
        for key, value in manual_label.items():
            if key in label and self._validate_label_value(key, value):
                label[key] = value

        return label

    def validate_labels(self, label: Dict[str, Any]) -> bool:
        """
        驗證標籤格式

        Args:
            label: 標籤字典

        Returns:
            是否有效
        """
        required_keys = {"toxicity", "bullying", "role", "emotion", "emotion_strength"}

        # 檢查必要鍵
        if not all(key in label for key in required_keys):
            return False

        # 檢查值的有效性
        return all(self._validate_label_value(key, value) for key, value in label.items())

    def _validate_label_value(self, key: str, value: Any) -> bool:
        """驗證單個標籤值"""
        if key == "toxicity":
            return value in self.TOXICITY_LABELS
        elif key == "bullying":
            return value in self.BULLYING_LABELS
        elif key == "role":
            return value in self.ROLE_LABELS
        elif key == "emotion":
            return value in self.EMOTION_LABELS
        elif key == "emotion_strength":
            return value in self.EMOTION_STRENGTH_RANGE

        return True

    def get_label_statistics(self, data: List[Tuple[str, Dict]]) -> Dict[str, Dict[str, int]]:
        """
        獲取標籤統計資訊

        Args:
            data: 資料列表

        Returns:
            標籤統計
        """
        stats = {
            "toxicity": defaultdict(int),
            "bullying": defaultdict(int),
            "role": defaultdict(int),
            "emotion": defaultdict(int),
            "emotion_strength": defaultdict(int),
        }

        for _, label in data:
            for key in stats:
                if key in label:
                    stats[key][label[key]] += 1

        # 轉換為普通字典
        return {key: dict(value) for key, value in stats.items()}
