#!/usr/bin/env python3
"""
測試標籤映射系統
"""
from unittest.mock import patch

import pytest

from src.cyberpuppy.labeling import (BullyingLevel, EmotionType, LabelMapper,
                                     RoleType, ToxicityLevel, UnifiedLabel,
                                     from_chnci_to_unified,
                                     from_cold_to_unified,
                                     from_sccd_to_unified,
                                     from_sentiment_to_unified, to_chnci_label,
                                     to_cold_label, to_sccd_label,
                                     to_sentiment_label)


class TestEnums:
    """測試枚舉類型"""

    def test_toxicity_level_values(self):
        """測試毒性等級枚舉值"""
        assert ToxicityLevel.NONE.value == "none"
        assert ToxicityLevel.TOXIC.value == "toxic"
        assert ToxicityLevel.SEVERE.value == "severe"

    def test_bullying_level_values(self):
        """測試霸凌等級枚舉值"""
        assert BullyingLevel.NONE.value == "none"
        assert BullyingLevel.HARASSMENT.value == "harassment"
        assert BullyingLevel.THREAT.value == "threat"

    def test_role_type_values(self):
        """測試角色類型枚舉值"""
        assert RoleType.NONE.value == "none"
        assert RoleType.PERPETRATOR.value == "perpetrator"
        assert RoleType.VICTIM.value == "victim"
        assert RoleType.BYSTANDER.value == "bystander"

    def test_emotion_type_values(self):
        """測試情緒類型枚舉值"""
        assert EmotionType.POSITIVE.value == "pos"
        assert EmotionType.NEUTRAL.value == "neu"
        assert EmotionType.NEGATIVE.value == "neg"


class TestUnifiedLabel:
    """測試統一標籤結構"""

    def test_unified_label_creation(self):
        """測試統一標籤建立"""
        label = UnifiedLabel(
            toxicity=ToxicityLevel.TOXIC,
            bullying=BullyingLevel.HARASSMENT,
            role=RoleType.PERPETRATOR,
            emotion=EmotionType.NEGATIVE,
            emotion_intensity=3,
            source_dataset="test",
        )

        assert label.toxicity == ToxicityLevel.TOXIC
        assert label.bullying == BullyingLevel.HARASSMENT
        assert label.role == RoleType.PERPETRATOR
        assert label.emotion == EmotionType.NEGATIVE
        assert label.emotion_intensity == 3
        assert label.source_dataset == "test"

    def test_unified_label_defaults(self):
        """測試統一標籤預設值"""
        label = UnifiedLabel()

        assert label.toxicity == ToxicityLevel.NONE
        assert label.bullying == BullyingLevel.NONE
        assert label.role == RoleType.NONE
        assert label.emotion == EmotionType.NEUTRAL
        assert label.emotion_intensity == 2
        assert label.source_dataset == ""

    def test_unified_label_to_dict(self):
        """測試統一標籤轉字典"""
        label = UnifiedLabel(
            toxicity=ToxicityLevel.SEVERE, emotion=EmotionType.NEGATIVE,
                emotion_intensity=4
        )

        result = label.to_dict()
        expected = {
            "toxicity": "severe",
            "bullying": "none",
            "role": "none",
            "emotion": "neg",
            "emotion_intensity": 4,
            "source_dataset": "",
        }

        assert result == expected


class TestColdConversion:
    """測試 COLD 資料集標籤轉換"""

    def test_cold_non_offensive_to_unified(self):
        """測試非冒犯語言轉換"""
        result = from_cold_to_unified(0)

        assert result.toxicity == ToxicityLevel.NONE
        assert result.bullying == BullyingLevel.NONE
        assert result.emotion == EmotionType.NEUTRAL
        assert result.source_dataset == "cold"

    def test_cold_offensive_to_unified(self):
        """測試冒犯語言轉換"""
        result = from_cold_to_unified(1)

        assert result.toxicity == ToxicityLevel.TOXIC
        assert result.bullying == BullyingLevel.HARASSMENT
        assert result.emotion == EmotionType.NEGATIVE
        assert result.emotion_intensity == 2
        assert result.source_dataset == "cold"

    def test_cold_invalid_label(self):
        """測試無效 COLD 標籤"""
        result = from_cold_to_unified(99)

        assert result.toxicity == ToxicityLevel.NONE
        assert result.source_dataset == "cold"

    def test_unified_to_cold(self):
        """測試統一標籤轉 COLD"""
        # 非毒性
        label1 = UnifiedLabel(toxicity=ToxicityLevel.NONE)
        assert to_cold_label(label1) == 0

        # 毒性
        label2 = UnifiedLabel(toxicity=ToxicityLevel.TOXIC)
        assert to_cold_label(label2) == 1

        # 嚴重毒性
        label3 = UnifiedLabel(toxicity=ToxicityLevel.SEVERE)
        assert to_cold_label(label3) == 1


class TestSentimentConversion:
    """測試情感分析標籤轉換"""

    def test_positive_sentiment_to_unified(self):
        """測試正面情感轉換"""
        result = from_sentiment_to_unified(1)

        assert result.emotion == EmotionType.POSITIVE
        assert result.emotion_intensity == 2
        assert result.toxicity == ToxicityLevel.NONE
        assert result.source_dataset == "sentiment"

    def test_negative_sentiment_to_unified(self):
        """測試負面情感轉換"""
        result = from_sentiment_to_unified(0)

        assert result.emotion == EmotionType.NEGATIVE
        assert result.emotion_intensity == 2
        assert result.toxicity == ToxicityLevel.NONE
        assert result.source_dataset == "sentiment"

    def test_invalid_sentiment_label(self):
        """測試無效情感標籤"""
        result = from_sentiment_to_unified(99)

        assert result.emotion == EmotionType.NEUTRAL
        assert result.source_dataset == "sentiment"

    def test_unified_to_sentiment(self):
        """測試統一標籤轉情感"""
        # 正面
        label1 = UnifiedLabel(emotion=EmotionType.POSITIVE)
        assert to_sentiment_label(label1) == 1

        # 負面
        label2 = UnifiedLabel(emotion=EmotionType.NEGATIVE)
        assert to_sentiment_label(label2) == 0

        # 中性（預設為正面）
        label3 = UnifiedLabel(emotion=EmotionType.NEUTRAL)
        assert to_sentiment_label(label3) == 1


class TestSccdConversion:
    """測試 SCCD 會話級霸凌標籤轉換"""

    def test_sccd_normal_to_unified(self):
        """測試正常會話轉換"""
        result = from_sccd_to_unified("normal")

        assert result.toxicity == ToxicityLevel.NONE
        assert result.bullying == BullyingLevel.NONE
        assert result.role == RoleType.NONE
        assert result.source_dataset == "sccd"

    def test_sccd_harassment_to_unified(self):
        """測試騷擾會話轉換"""
        result = from_sccd_to_unified("harassment")

        assert result.toxicity == ToxicityLevel.TOXIC
        assert result.bullying == BullyingLevel.HARASSMENT
        assert result.emotion == EmotionType.NEGATIVE
        assert result.emotion_intensity == 2
        assert result.source_dataset == "sccd"

    def test_sccd_threat_to_unified(self):
        """測試威脅會話轉換"""
        result = from_sccd_to_unified("threat")

        assert result.toxicity == ToxicityLevel.SEVERE
        assert result.bullying == BullyingLevel.THREAT
        assert result.emotion == EmotionType.NEGATIVE
        assert result.emotion_intensity == 4
        assert result.source_dataset == "sccd"

    def test_sccd_invalid_label(self):
        """測試無效 SCCD 標籤"""
        result = from_sccd_to_unified("invalid")

        assert result.toxicity == ToxicityLevel.NONE
        assert result.bullying == BullyingLevel.NONE
        assert result.source_dataset == "sccd"

    def test_unified_to_sccd(self):
        """測試統一標籤轉 SCCD"""
        # 正常
        label1 = UnifiedLabel()
        assert to_sccd_label(label1) == "normal"

        # 騷擾
        label2 = UnifiedLabel(bullying=BullyingLevel.HARASSMENT)
        assert to_sccd_label(label2) == "harassment"

        # 威脅
        label3 = UnifiedLabel(bullying=BullyingLevel.THREAT)
        assert to_sccd_label(label3) == "threat"


class TestChnciConversion:
    """測試 CHNCI 事件級霸凌標籤轉換"""

    def test_chnci_data_conversion(self):
        """測試 CHNCI 資料轉換"""
        chnci_data = {
            "bullying_type": "cyberbullying",
            "severity": "moderate",
            "role": "perpetrator",
        }

        result = from_chnci_to_unified(chnci_data)

        assert result.toxicity == ToxicityLevel.TOXIC
        assert result.bullying == BullyingLevel.HARASSMENT
        assert result.role == RoleType.PERPETRATOR
        assert result.emotion == EmotionType.NEGATIVE
        assert result.source_dataset == "chnci"

    def test_chnci_severe_bullying(self):
        """測試嚴重霸凌事件"""
        chnci_data = {"bullying_type": "threat", "severity": "high"}

        result = from_chnci_to_unified(chnci_data)

        assert result.toxicity == ToxicityLevel.SEVERE
        assert result.bullying == BullyingLevel.THREAT
        assert result.role == RoleType.VICTIM
        assert result.emotion_intensity == 4

    def test_chnci_no_bullying(self):
        """測試無霸凌事件"""
        chnci_data = {"bullying_type": "none", "severity": "low"}

        result = from_chnci_to_unified(chnci_data)

        assert result.toxicity == ToxicityLevel.NONE
        assert result.bullying == BullyingLevel.NONE
        assert result.role == RoleType.BYSTANDER

    def test_chnci_invalid_data(self):
        """測試無效 CHNCI 資料"""
        # 空字典
        result = from_chnci_to_unified({})
        assert result.source_dataset == "chnci"

        # 無效類型
        result2 = from_chnci_to_unified("invalid")
        assert result2.source_dataset == "chnci"

    def test_unified_to_chnci(self):
        """測試統一標籤轉 CHNCI"""
        # 正常
        label1 = UnifiedLabel()
        result1 = to_chnci_label(label1)
        expected1 = {"bullying_type": "none", "severity": "low", "role": "none"}
        assert result1 == expected1

        # 霸凌事件
        label2 = UnifiedLabel(
            bullying=BullyingLevel.HARASSMENT,
            role=RoleType.PERPETRATOR,
            toxicity=ToxicityLevel.TOXIC,
        )
        result2 = to_chnci_label(label2)
        expected2 = {
            "bullying_type": "cyberbullying",
            "severity": "moderate",
            "role": "perpetrator",
        }
        assert result2 == expected2

        # 嚴重威脅
        label3 = UnifiedLabel(
            bullying=BullyingLevel.THREAT, role=RoleType.VICTIM,
                toxicity=ToxicityLevel.SEVERE
        )
        result3 = to_chnci_label(label3)
        expected3 = {"bullying_type": "threat", "severity": "high", "role": "victim"}
        assert result3 == expected3


class TestLabelMapper:
    """測試標籤映射器"""

    def setUp(self):
        """設置測試環境"""
        self.mapper = LabelMapper()

    def test_label_mapper_creation(self):
        """測試標籤映射器建立"""
        mapper = LabelMapper()
        assert mapper is not None

    def test_batch_convert_cold_labels(self):
        """測試批次轉換 COLD 標籤"""
        mapper = LabelMapper()
        cold_labels = [0, 1, 0, 1]

        results = mapper.batch_convert_from_cold(cold_labels)

        assert len(results) == 4
        assert results[0].toxicity == ToxicityLevel.NONE
        assert results[1].toxicity == ToxicityLevel.TOXIC
        assert results[2].toxicity == ToxicityLevel.NONE
        assert results[3].toxicity == ToxicityLevel.TOXIC

    def test_batch_convert_sentiment_labels(self):
        """測試批次轉換情感標籤"""
        mapper = LabelMapper()
        sentiment_labels = [1, 0, 1]

        results = mapper.batch_convert_from_sentiment(sentiment_labels)

        assert len(results) == 3
        assert results[0].emotion == EmotionType.POSITIVE
        assert results[1].emotion == EmotionType.NEGATIVE
        assert results[2].emotion == EmotionType.POSITIVE

    def test_get_label_statistics(self):
        """測試標籤統計"""
        mapper = LabelMapper()
        labels = [
            UnifiedLabel(toxicity=ToxicityLevel.TOXIC),
            UnifiedLabel(toxicity=ToxicityLevel.NONE),
            UnifiedLabel(emotion=EmotionType.POSITIVE),
            UnifiedLabel(bullying=BullyingLevel.HARASSMENT),
        ]

        stats = mapper.get_label_statistics(labels)

        assert "toxicity" in stats
        assert "emotion" in stats
        assert "bullying" in stats
        assert stats["toxicity"]["toxic"] == 1
        assert stats["toxicity"]["none"] == 3  # 包含其他未設定毒性的標籤
        assert stats["emotion"]["pos"] == 1


class TestEdgeCases:
    """測試邊界情況"""

    def test_none_input(self):
        """測試 None 輸入"""
        result = from_cold_to_unified(None)
        assert result.source_dataset == "cold"

    def test_empty_string_input(self):
        """測試空字串輸入"""
        result = from_sccd_to_unified("")
        assert result.source_dataset == "sccd"

    def test_extreme_emotion_intensity(self):
        """測試極端情緒強度值"""
        # 超出範圍的情緒強度應該被限制
        label = UnifiedLabel(emotion_intensity=10)
        assert label.emotion_intensity == 10  # 允許超出範圍但記錄原值

        label2 = UnifiedLabel(emotion_intensity=-1)
        assert label2.emotion_intensity == -1  # 允許負值但記錄原值

    def test_case_insensitive_sccd(self):
        """測試 SCCD 標籤不區分大小寫"""
        result1 = from_sccd_to_unified("HARASSMENT")
        result2 = from_sccd_to_unified("harassment")
        result3 = from_sccd_to_unified("Harassment")

        assert result1.bullying == result2.bullying == result3.bullying
        assert result1.bullying == BullyingLevel.HARASSMENT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
