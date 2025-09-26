"""
CyberPuppy 標籤系統
統一不同資料集的標籤映射
"""

from .label_map import (
    BullyingLevel,
    EmotionType,
    LabelMapper,
    RoleType,
    ToxicityLevel,
    UnifiedLabel,
    from_chnci_to_unified,
    from_cold_to_unified,
    from_sccd_to_unified,
    from_sentiment_to_unified,
    to_chnci_label,
    to_cold_label,
    to_sccd_label,
    to_sentiment_label,
)

from .improved_label_map import (
    ImprovedLabelMapper,
    TextFeatures,
    improved_cold_to_unified,
    analyze_text_bullying_features,
)

__all__ = [
    "LabelMapper",
    "UnifiedLabel",
    "ToxicityLevel",
    "BullyingLevel",
    "RoleType",
    "EmotionType",
    "from_cold_to_unified",
    "from_sccd_to_unified",
    "from_chnci_to_unified",
    "from_sentiment_to_unified",
    "to_cold_label",
    "to_sccd_label",
    "to_chnci_label",
    "to_sentiment_label",
    # 改進的映射功能
    "ImprovedLabelMapper",
    "TextFeatures",
    "improved_cold_to_unified",
    "analyze_text_bullying_features",
]
