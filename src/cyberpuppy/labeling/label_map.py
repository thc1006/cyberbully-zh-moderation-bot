#!/usr/bin/env python3
"""
標籤映射模組
將不同資料集的標籤映射到統一的 schema
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ToxicityLevel(Enum):
    """毒性等級"""

    NONE = "none"
    TOXIC = "toxic"
    SEVERE = "severe"


class BullyingLevel(Enum):
    """霸凌等級"""

    NONE = "none"
    HARASSMENT = "harassment"
    THREAT = "threat"


class RoleType(Enum):
    """角色類型"""

    NONE = "none"
    PERPETRATOR = "perpetrator"  # 加害者
    VICTIM = "victim"  # 受害者
    BYSTANDER = "bystander"  # 旁觀者


class EmotionType(Enum):
    """情緒類型"""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


@dataclass
class UnifiedLabel:
    """統一標籤格式"""

    # 毒性相關
    toxicity: ToxicityLevel = ToxicityLevel.NONE
    toxicity_score: float = 0.0  # 0.0 - 1.0

    # 霸凌相關
    bullying: BullyingLevel = BullyingLevel.NONE
    bullying_score: float = 0.0  # 0.0 - 1.0

    # 角色相關
    role: RoleType = RoleType.NONE

    # 情緒相關
    emotion: EmotionType = EmotionType.NEUTRAL
    emotion_intensity: int = 2  # 0-4 (0=very negative, 2=neutral, 4=very positive)

    # 元資料
    confidence: float = 1.0  # 標籤置信度
    source_dataset: str = ""  # 來源資料集
    original_label: Any = None  # 原始標籤

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "toxicity": self.toxicity.value,
            "toxicity_score": self.toxicity_score,
            "bullying": self.bullying.value,
            "bullying_score": self.bullying_score,
            "role": self.role.value,
            "emotion": self.emotion.value,
            "emotion_intensity": self.emotion_intensity,
            "confidence": self.confidence,
            "source_dataset": self.source_dataset,
            "original_label": self.original_label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedLabel":
        """從字典建立"""
        return cls(
            toxicity=ToxicityLevel(data.get("toxicity", "none")),
            toxicity_score=data.get("toxicity_score", 0.0),
            bullying=BullyingLevel(data.get("bullying", "none")),
            bullying_score=data.get("bullying_score", 0.0),
            role=RoleType(data.get("role", "none")),
            emotion=EmotionType(data.get("emotion", "neutral")),
            emotion_intensity=data.get("emotion_intensity", 2),
            confidence=data.get("confidence", 1.0),
            source_dataset=data.get("source_dataset", ""),
            original_label=data.get("original_label"),
        )


class LabelMapper:
    """標籤映射器"""

    # COLD 資料集映射規則
    COLD_MAPPING = {
        0: {  # 非冒犯
            "toxicity": ToxicityLevel.NONE,
            "toxicity_score": 0.0,
            "bullying": BullyingLevel.NONE,
            "bullying_score": 0.0,
        },
        1: {  # 冒犯
            "toxicity": ToxicityLevel.TOXIC,
            "toxicity_score": 0.7,
            "bullying": BullyingLevel.HARASSMENT,
            "bullying_score": 0.6,
        },
    }

    # SCCD 會話級標籤映射（假設的標籤結構）
    SCCD_MAPPING = {
        "non_bullying": {
            "toxicity": ToxicityLevel.NONE,
            "bullying": BullyingLevel.NONE,
            "bullying_score": 0.0,
        },
        "mild_bullying": {
            "toxicity": ToxicityLevel.TOXIC,
            "bullying": BullyingLevel.HARASSMENT,
            "bullying_score": 0.5,
        },
        "severe_bullying": {
            "toxicity": ToxicityLevel.SEVERE,
            "bullying": BullyingLevel.THREAT,
            "bullying_score": 0.9,
        },
        "harassment": {
            "toxicity": ToxicityLevel.TOXIC,
            "bullying": BullyingLevel.HARASSMENT,
            "bullying_score": 0.7,
        },
        "threat": {
            "toxicity": ToxicityLevel.SEVERE,
            "bullying": BullyingLevel.THREAT,
            "bullying_score": 1.0,
        },
    }

    # CHNCI 事件級標籤映射（假設的標籤結構）
    CHNCI_MAPPING = {
        "event_type": {
            "none": ToxicityLevel.NONE,
            "verbal_abuse": ToxicityLevel.TOXIC,
            "threat": ToxicityLevel.SEVERE,
            "discrimination": ToxicityLevel.TOXIC,
        },
        "role_mapping": {
            "aggressor": RoleType.PERPETRATOR,
            "target": RoleType.VICTIM,
            "witness": RoleType.BYSTANDER,
            "neutral": RoleType.NONE,
        },
    }

    # 情感標籤映射
    SENTIMENT_MAPPING = {
        # 二元情感
        0: {"emotion": EmotionType.NEGATIVE, "emotion_intensity": 1},  # 負面
        1: {"emotion": EmotionType.POSITIVE, "emotion_intensity": 3},  # 正面
        # 五星評分映射
        "1.0": {"emotion": EmotionType.NEGATIVE, "emotion_intensity": 0},  # 1星
        "2.0": {"emotion": EmotionType.NEGATIVE, "emotion_intensity": 1},  # 2星
        "3.0": {"emotion": EmotionType.NEUTRAL, "emotion_intensity": 2},  # 3星
        "4.0": {"emotion": EmotionType.POSITIVE, "emotion_intensity": 3},  # 4星
        "5.0": {"emotion": EmotionType.POSITIVE, "emotion_intensity": 4},  # 5星
    }

    @classmethod
    def from_cold_to_unified(cls, label: int, **kwargs) -> UnifiedLabel:
        """
        將 COLD 標籤轉換為統一格式

        Args:
            label: COLD 標籤 (0 或 1)
            **kwargs: 額外參數

        Returns:
            UnifiedLabel 物件
        """
        if label not in cls.COLD_MAPPING:
            logger.warning(f"Unknown COLD label: {label}, treating as non-toxic")
            label = 0

        mapping = cls.COLD_MAPPING[label]

        return UnifiedLabel(
            toxicity=mapping["toxicity"],
            toxicity_score=mapping["toxicity_score"],
            bullying=mapping["bullying"],
            bullying_score=mapping["bullying_score"],
            role=RoleType.NONE,
            emotion=EmotionType.NEUTRAL,
            emotion_intensity=2,
            confidence=kwargs.get("confidence", 1.0),
            source_dataset="COLD",
            original_label=label,
        )

    @classmethod
    def from_sccd_to_unified(
        cls,
        label: str,
        role: Optional[str] = None,
        **kwargs
    ) -> UnifiedLabel:
        """
        將 SCCD 標籤轉換為統一格式

        Args:
            label: SCCD 霸凌標籤
            role: 角色標籤（可選）
            **kwargs: 額外參數

        Returns:
            UnifiedLabel 物件
        """
        if label not in cls.SCCD_MAPPING:
            logger.warning(f"Unknown SCCD label: {label}, treating as non-bullying")
            label = "non_bullying"

        mapping = cls.SCCD_MAPPING[label]

        # 處理角色
        role_type = RoleType.NONE
        if role and role in ["perpetrator", "victim", "bystander"]:
            role_type = RoleType(role)

        return UnifiedLabel(
            toxicity=mapping["toxicity"],
            toxicity_score=mapping.get("toxicity_score", 0.5),
            bullying=mapping["bullying"],
            bullying_score=mapping["bullying_score"],
            role=role_type,
            emotion=EmotionType.NEUTRAL,
            emotion_intensity=2,
            confidence=kwargs.get("confidence", 1.0),
            source_dataset="SCCD",
            original_label={"label": label, "role": role},
        )

    @classmethod
    def from_chnci_to_unified(
        cls, event_type: str, role: str, severity: Optional[float] = None,
            **kwargs
    ) -> UnifiedLabel:
        """
        將 CHNCI 標籤轉換為統一格式

        Args:
            event_type: 事件類型
            role: 角色
            severity: 嚴重程度 (0-1)
            **kwargs: 額外參數

        Returns:
            UnifiedLabel 物件
        """
        # 處理事件類型
        if event_type not in cls.CHNCI_MAPPING["event_type"]:
            logger.warning(f"Unknown CHNCI event type: {event_type}")
            event_type = "none"

        event_mapping = cls.CHNCI_MAPPING["event_type"][event_type]

        # 處理角色
        role_type = cls.CHNCI_MAPPING["role_mapping"].get(role, RoleType.NONE)

        # 計算分數
        if severity is not None:
            toxicity_score = severity
            bullying_score = severity
        else:
            toxicity_score = 0.5 if event_type != "none" else 0.0
            bullying_score = 0.5 if event_type != "none" else 0.0

        return UnifiedLabel(
            toxicity=event_mapping["toxicity"],
            toxicity_score=toxicity_score,
            bullying=event_mapping["bullying"],
            bullying_score=bullying_score,
            role=role_type,
            emotion=EmotionType.NEUTRAL,
            emotion_intensity=2,
            confidence=kwargs.get("confidence", 1.0),
            source_dataset="CHNCI",
            original_label={"event_type": event_type}
        )

    @classmethod
    def from_sentiment_to_unified(
        cls, label: Union[int, float], text_emotion: Optional[str] = None,
            **kwargs
    ) -> UnifiedLabel:
        """
        將情感標籤轉換為統一格式

        Args:
            label: 情感標籤 (0/1 或 1-5 星級)
            text_emotion: 文字情緒標籤（可選）
            **kwargs: 額外參數

        Returns:
            UnifiedLabel 物件
        """
        # 處理不同類型的情感標籤
        if label in cls.SENTIMENT_MAPPING:
            mapping = cls.SENTIMENT_MAPPING[label]
        else:
            # 嘗試轉換為最接近的標籤
            if isinstance(label, (int, float)):
                if label <= 0:
                    mapping = cls.SENTIMENT_MAPPING[0]
                elif label >= 1:
                    mapping = cls.SENTIMENT_MAPPING[1]
                else:
                    mapping = {"emotion": EmotionType.NEUTRAL, "emotion_intensity": 2}
            else:
                logger.warning(f"Unknown sentiment label: {label}")
                mapping = {"emotion": EmotionType.NEUTRAL, "emotion_intensity": 2} 

        # 覆蓋文字情緒（如果提供）
        if text_emotion:
            if text_emotion.lower() in ["positive", "pos"]:
                mapping["emotion"] = EmotionType.POSITIVE
            elif text_emotion.lower() in ["negative", "neg"]:
                mapping["emotion"] = EmotionType.NEGATIVE
            else:
                mapping["emotion"] = EmotionType.NEUTRAL

        return UnifiedLabel(
            toxicity=ToxicityLevel.NONE,
            toxicity_score=0.0,
            bullying=BullyingLevel.NONE,
            bullying_score=0.0,
            role=RoleType.NONE,
            emotion=mapping["emotion"],
            emotion_intensity=mapping["emotion_intensity"],
            confidence=kwargs.get("confidence", 1.0),
            source_dataset="SENTIMENT",
            original_label=label,
        )

    @classmethod
    def to_cold_label(cls, unified: UnifiedLabel) -> int:
        """
        將統一標籤轉換回 COLD 格式

        Args:
            unified: UnifiedLabel 物件

        Returns:
            COLD 標籤 (0 或 1)
        """
        if unified.toxicity == ToxicityLevel.NONE:
            return 0
        else:
            return 1

    @classmethod
    def to_sccd_label(cls, unified: UnifiedLabel) -> Dict[str, str]:
        """
        將統一標籤轉換回 SCCD 格式

        Args:
            unified: UnifiedLabel 物件

        Returns:
            SCCD 標籤字典
        """
        # 根據霸凌等級決定標籤
        if unified.bullying == BullyingLevel.NONE:
            label = "non_bullying"
        elif unified.bullying == BullyingLevel.HARASSMENT:
            if unified.bullying_score >= 0.7:
                label = "harassment"
            else:
                label = "mild_bullying"
        elif unified.bullying == BullyingLevel.THREAT:
            label = "severe_bullying"
        else:
            label = "non_bullying"

        # 角色映射
        role = unified.role.value if unified.role != RoleType.NONE else None

        return {"label": label, "role": role}

    @classmethod
    def to_chnci_label(cls, unified: UnifiedLabel) -> Dict[str, Any]:
        """
        將統一標籤轉換回 CHNCI 格式

        Args:
            unified: UnifiedLabel 物件

        Returns:
            CHNCI 標籤字典
        """
        # 事件類型映射
        if unified.toxicity == ToxicityLevel.NONE:
            event_type = "none"
        elif unified.toxicity == ToxicityLevel.SEVERE:
            event_type = "threat"
        elif unified.bullying == BullyingLevel.HARASSMENT:
            event_type = "verbal_abuse"
        else:
            event_type = "discrimination"

        # 角色反向映射
        role_mapping_reverse = {
            RoleType.PERPETRATOR: "aggressor",
            RoleType.VICTIM: "target",
            RoleType.BYSTANDER: "witness",
            RoleType.NONE: "neutral",
        }

        role = role_mapping_reverse.get(unified.role, "neutral")

        return {
            "event_type": event_type,
            "role": role,
            "severity": max(unified.toxicity_score, unified.bullying_score),
        }

    @classmethod
    def to_sentiment_label(
        cls, unified: UnifiedLabel, format: str = "binary"
    ) -> Union[int, float, str]:
        """
        將統一標籤轉換回情感標籤

        Args:
            unified: UnifiedLabel 物件
            format: 輸出格式 ('binary', 'star', 'text')

        Returns:
            情感標籤
        """
        if format == "binary":
            # 二元分類
            if unified.emotion == EmotionType.POSITIVE:
                return 1
            elif unified.emotion == EmotionType.NEGATIVE:
                return 0
            else:
                # 中性視為正面（可調整）
                return 1 if unified.emotion_intensity > 2 else 0

        elif format == "star":
            # 五星評分
            intensity_to_star = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}
            return intensity_to_star.get(unified.emotion_intensity, 3.0)

        elif format == "text":
            # 文字標籤
            return unified.emotion.value

        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def merge_labels(cls, labels: List[UnifiedLabel]) -> UnifiedLabel:
        """
        合併多個統一標籤（用於多標註者或多模型的情況）

        Args:
            labels: UnifiedLabel 物件列表

        Returns:
            合併後的 UnifiedLabel
        """
        if not labels:
            return UnifiedLabel()

        if len(labels) == 1:
            return labels[0]

        # 計算平均分數
        avg_toxicity_score = sum(label.toxicity_score for label in labels) / len(labels)
        avg_bullying_score = sum(label.bullying_score for label in labels) / len(labels)
        avg_emotion_intensity = sum(label.emotion_intensity for label in labels) / len(labels)
        avg_confidence = sum(label.confidence for label in labels) / len(labels)

        # 投票決定類別
        toxicity_votes = {}
        bullying_votes = {}
        role_votes = {}
        emotion_votes = {}

        for label in labels:
            toxicity_votes[label.toxicity] = toxicity_votes.get(
                label.toxicity,
                0
            ) + 1
            bullying_votes[label.bullying] = bullying_votes.get(
                label.bullying,
                0
            ) + 1
            role_votes[label.role] = role_votes.get(label.role, 0) + 1
            emotion_votes[label.emotion] = emotion_votes.get(
                label.emotion,
                0
            ) + 1

        # 選擇最多票的類別
        toxicity = max(toxicity_votes, key=toxicity_votes.get)
        bullying = max(bullying_votes, key=bullying_votes.get)
        role = max(role_votes, key=role_votes.get)
        emotion = max(emotion_votes, key=emotion_votes.get)

        return UnifiedLabel(
            toxicity=toxicity,
            toxicity_score=avg_toxicity_score,
            bullying=bullying,
            bullying_score=avg_bullying_score,
            role=role,
            emotion=emotion,
            emotion_intensity=round(avg_emotion_intensity),
            confidence=avg_confidence,
            source_dataset="MERGED",
            original_label=[label.to_dict() for label in labels],
        )


# 便利函式
def from_cold_to_unified(label: int, **kwargs) -> UnifiedLabel:
    """COLD 到統一格式"""
    return LabelMapper.from_cold_to_unified(label, **kwargs)


def from_sccd_to_unified(
    label: str,
    role: Optional[str] = None,
    **kwargs
) -> UnifiedLabel:
    """SCCD 到統一格式"""
    return LabelMapper.from_sccd_to_unified(label, role, **kwargs)


def from_chnci_to_unified(
    event_type: str, role: str, severity: Optional[float] = None, **kwargs
) -> UnifiedLabel:
    """CHNCI 到統一格式"""
    return LabelMapper.from_chnci_to_unified(
        event_type,
        role,
        severity,
        **kwargs
    )


def from_sentiment_to_unified(
    label: Union[int, float], text_emotion: Optional[str] = None, **kwargs
) -> UnifiedLabel:
    """情感標籤到統一格式"""
    return LabelMapper.from_sentiment_to_unified(label, text_emotion, **kwargs)


def to_cold_label(unified: UnifiedLabel) -> int:
    """統一格式到 COLD"""
    return LabelMapper.to_cold_label(unified)


def to_sccd_label(unified: UnifiedLabel) -> Dict[str, str]:
    """統一格式到 SCCD"""
    return LabelMapper.to_sccd_label(unified)


def to_chnci_label(unified: UnifiedLabel) -> Dict[str, Any]:
    """統一格式到 CHNCI"""
    return LabelMapper.to_chnci_label(unified)


def to_sentiment_label(
    unified: UnifiedLabel,
    format: str = "binary"
) -> Union[int, float, str]:
    """統一格式到情感標籤"""
    return LabelMapper.to_sentiment_label(unified, format)
