#!/usr/bin/env python3
"""
簡化測試改進的標籤映射
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cyberpuppy.labeling.improved_label_map import ImprovedLabelMapper
from src.cyberpuppy.labeling.label_map import LabelMapper, ToxicityLevel, BullyingLevel


def test_basic_functionality():
    """測試基本功能"""
    print("測試改進的標籤映射功能")
    print("="*50)

    # 創建映射器
    improved_mapper = ImprovedLabelMapper()
    original_mapper = LabelMapper()

    # 測試案例
    test_cases = [
        (0, "今天天氣很好"),
        (1, "這個遊戲真爛"),  # 毒性但非霸凌
        (1, "你就是個白痴"),  # 霸凌
        (1, "我要殺了你"),   # 威脅
    ]

    print("\n測試結果:")
    print("-" * 80)
    print(f"{'文本':<20} {'原毒性':<10} {'原霸凌':<10} {'新毒性':<10} {'新霸凌':<10}")
    print("-" * 80)

    separation_count = 0
    total_toxic_samples = 0

    for label, text in test_cases:
        # 原始映射
        orig_result = original_mapper.from_cold_to_unified(label)

        # 改進映射
        impr_result = improved_mapper.improved_cold_mapping(label, text)

        print(f"{text:<20} {orig_result.toxicity.value:<10} {orig_result.bullying.value:<10} "
              f"{impr_result.toxicity.value:<10} {impr_result.bullying.value:<10}")

        # 統計分離效果
        if label == 1:  # 冒犯性內容
            total_toxic_samples += 1
            if (impr_result.toxicity != ToxicityLevel.NONE and
                impr_result.bullying == BullyingLevel.NONE):
                separation_count += 1

    print("-" * 80)
    print(f"\n分離效果統計:")
    print(f"毒性但非霸凌樣本: {separation_count}/{total_toxic_samples}")
    print(f"分離成功率: {separation_count/total_toxic_samples*100:.1f}%" if total_toxic_samples > 0 else "N/A")

    if separation_count > 0:
        print("✓ 成功實現標籤分離")
    else:
        print("✗ 未實現標籤分離")


def test_text_features():
    """測試文本特徵分析"""
    print("\n\n測試文本特徵分析")
    print("="*50)

    mapper = ImprovedLabelMapper()

    test_texts = [
        "你就是個白痴",
        "這個遊戲很爛",
        "我要殺了你",
        "大家都討厭你",
    ]

    for text in test_texts:
        features = mapper.analyze_text_features(text)
        print(f"\n文本: {text}")
        print(f"  粗俗語言: {features.has_profanity}")
        print(f"  人身攻擊: {features.has_personal_attack}")
        print(f"  威脅: {features.has_threat}")
        print(f"  排擠: {features.has_exclusion}")
        print(f"  針對性: {features.target_specific}")


if __name__ == "__main__":
    test_basic_functionality()
    test_text_features()