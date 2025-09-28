#!/usr/bin/env python3
"""
測試改進的標籤映射邏輯
驗證霸凌與毒性標籤分離效果
"""

import sys
from pathlib import Path

import pytest

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cyberpuppy.labeling.improved_label_map import (
    ImprovedLabelMapper, analyze_text_bullying_features,
    improved_cold_to_unified)
from src.cyberpuppy.labeling.label_map import (BullyingLevel, LabelMapper,
                                               RoleType, ToxicityLevel)


class TestTextFeatures:
    """測試文本特徵分析"""

    def test_profanity_detection(self):
        """測試粗俗語言檢測"""
        mapper = ImprovedLabelMapper()

        # 包含粗俗語言的文本
        profane_texts = ["這個人真是個白痴", "你就是個廢物", "操你媽的垃圾"]

        for text in profane_texts:
            features = mapper.analyze_text_features(text)
            assert features.has_profanity, f"應該檢測到粗俗語言: {text}"

        # 不包含粗俗語言的文本
        clean_texts = ["今天天氣很好", "我不同意你的觀點", "這個想法需要改進"]

        for text in clean_texts:
            features = mapper.analyze_text_features(text)
            assert not features.has_profanity, f"不應該檢測到粗俗語言: {text}"

    def test_personal_attack_detection(self):
        """測試人身攻擊檢測"""
        mapper = ImprovedLabelMapper()

        # 人身攻擊
        attack_texts = ["你就是個笨蛋", "你長得真醜", "你根本是個智障"]

        for text in attack_texts:
            features = mapper.analyze_text_features(text)
            assert features.has_personal_attack, f"應該檢測到人身攻擊: {text}"

        # 非人身攻擊
        non_attack_texts = ["這個想法很笨", "這個設計很醜", "這個方案有問題"]

        for text in non_attack_texts:
            features = mapper.analyze_text_features(text)
            assert not features.has_personal_attack, f"不應該檢測到人身攻擊: {text}"

    def test_threat_detection(self):
        """測試威脅檢測"""
        mapper = ImprovedLabelMapper()

        # 威脅文本
        threat_texts = ["要你好看", "等著受死吧", "我會殺了你"]

        for text in threat_texts:
            features = mapper.analyze_text_features(text)
            assert features.has_threat, f"應該檢測到威脅: {text}"

    def test_exclusion_detection(self):
        """測試排擠檢測"""
        mapper = ImprovedLabelMapper()

        # 排擠文本
        exclusion_texts = ["不要理他", "大家都討厭你", "你不配參加"]

        for text in exclusion_texts:
            features = mapper.analyze_text_features(text)
            assert features.has_exclusion, f"應該檢測到排擠: {text}"

    def test_target_specific_detection(self):
        """測試針對性檢測"""
        mapper = ImprovedLabelMapper()

        # 針對特定目標
        targeted_texts = ["你真的很討厭", "他就是個笨蛋", "你們都是白痴"]

        for text in targeted_texts:
            features = mapper.analyze_text_features(text)
            assert features.target_specific, f"應該檢測到針對性: {text}"

        # 非針對性
        general_texts = ["這件事很討厭", "這個想法很笨", "這種行為很愚蠢"]

        for text in general_texts:
            features = mapper.analyze_text_features(text)
            assert not features.target_specific, f"不應該檢測到針對性: {text}"


class TestImprovedLabelMapping:
    """測試改進的標籤映射"""

    def test_non_offensive_content(self):
        """測試非冒犯性內容"""
        mapper = ImprovedLabelMapper()

        # 非冒犯性文本
        non_offensive_texts = ["今天天氣真好", "我喜歡這個電影", "謝謝你的幫助"]

        for text in non_offensive_texts:
            result = mapper.improved_cold_mapping(0, text)
            assert result.toxicity == ToxicityLevel.NONE
            assert result.bullying == BullyingLevel.NONE
            assert result.role == RoleType.NONE

    def test_toxic_but_not_bullying(self):
        """測試毒性但非霸凌的內容"""
        mapper = ImprovedLabelMapper()

        # 毒性但非霸凌（粗俗語言，但不針對特定個人）
        toxic_not_bullying = [
            "這個遊戲真他媽的難",  # 粗俗語言但非針對性
            "靠北這個系統有夠爛",  # 抱怨但非霸凌
            "幹這個作業好煩",  # 表達不滿但非霸凌
        ]

        for text in toxic_not_bullying:
            result = mapper.improved_cold_mapping(1, text)
            # 應該有毒性但沒有霸凌
            assert result.toxicity != ToxicityLevel.NONE, f"應該檢測到毒性: {text}"
            assert result.bullying == BullyingLevel.NONE, f"不應該檢測到霸凌: {text}"

    def test_bullying_behavior(self):
        """測試霸凌行為"""
        mapper = ImprovedLabelMapper()

        # 明確的霸凌行為
        bullying_texts = [
            "你就是個笨蛋廢物",  # 人身攻擊
            "你長得真醜，沒人會喜歡你",  # 外表攻擊
            "大家都討厭你，滾遠點",  # 排擠
        ]

        for text in bullying_texts:
            result = mapper.improved_cold_mapping(1, text)
            # 應該同時有毒性和霸凌
            assert result.toxicity != ToxicityLevel.NONE, f"應該檢測到毒性: {text}"
            assert result.bullying != BullyingLevel.NONE, f"應該檢測到霸凌: {text}"
            assert result.role == RoleType.PERPETRATOR, f"應該識別為加害者角色: {text}"

    def test_threat_behavior(self):
        """測試威脅行為"""
        mapper = ImprovedLabelMapper()

        # 威脅文本
        threat_texts = ["我要殺了你", "等著受死吧", "小心點，要你好看"]

        for text in threat_texts:
            result = mapper.improved_cold_mapping(1, text)
            # 應該是嚴重毒性和威脅級霸凌
            assert result.toxicity == ToxicityLevel.SEVERE, f"應該是嚴重毒性: {text}"
            assert result.bullying == BullyingLevel.THREAT, f"應該是威脅級霸凌: {text}"

    def test_label_separation(self):
        """測試標籤分離效果"""
        mapper = ImprovedLabelMapper()

        # 測試案例：毒性但非霸凌 vs 霸凌行為
        test_cases = [
            # (文本, 預期毒性, 預期霸凌)
            ("這個遊戲真他媽的爛", "toxic", "none"),  # 毒性但非霸凌
            ("你這個白痴給我滾", "toxic", "harassment"),  # 霸凌
            ("靠北這個系統", "toxic", "none"),  # 毒性但非霸凌
            ("你就是個廢物沒人愛", "toxic", "harassment"),  # 霸凌
            ("今天心情不錯", "none", "none"),  # 都不是
        ]

        results = []
        for text, expected_toxicity, expected_bullying in test_cases:
            result = mapper.improved_cold_mapping(1 if expected_toxicity != "none" else 0, text)
            results.append(result)

            if expected_toxicity == "none":
                assert result.toxicity == ToxicityLevel.NONE, f"毒性判斷錯誤: {text}"
            else:
                assert result.toxicity != ToxicityLevel.NONE, f"毒性判斷錯誤: {text}"

            if expected_bullying == "none":
                assert result.bullying == BullyingLevel.NONE, f"霸凌判斷錯誤: {text}"
            else:
                assert result.bullying != BullyingLevel.NONE, f"霸凌判斷錯誤: {text}"

        # 檢查分離效果
        separation_stats = mapper.analyze_label_distribution(results)
        separation_score = separation_stats["correlation_analysis"]["separation_score"]
        assert separation_score > 0, "應該實現標籤分離"

    def test_batch_processing(self):
        """測試批次處理"""
        mapper = ImprovedLabelMapper()

        labels = [0, 1, 1, 0, 1]
        texts = ["今天天氣很好", "這個遊戲很爛", "你就是個白痴", "謝謝你的幫助", "我要殺了你"]

        results = mapper.batch_improve_cold_labels(labels, texts)
        assert len(results) == len(labels)

        # 檢查每個結果
        assert results[0].toxicity == ToxicityLevel.NONE  # 正常文本
        assert results[1].toxicity != ToxicityLevel.NONE  # 毒性文本
        assert results[1].bullying == BullyingLevel.NONE  # 但非霸凌
        assert results[2].bullying != BullyingLevel.NONE  # 霸凌文本
        assert results[4].bullying == BullyingLevel.THREAT  # 威脅


class TestComparisonWithOriginal:
    """測試與原始映射的比較"""

    def test_improvement_metrics(self):
        """測試改進指標"""
        improved_mapper = ImprovedLabelMapper()
        LabelMapper()

        # 測試資料
        labels = [0, 1, 1, 1, 0, 1, 1]
        texts = [
            "今天天氣很好",  # 非冒犯
            "這個遊戲真爛",  # 毒性但非霸凌
            "你就是個白痴",  # 霸凌
            "靠北這個系統",  # 毒性但非霸凌
            "謝謝你的幫助",  # 非冒犯
            "你長得真醜",  # 霸凌
            "我要殺了你",  # 威脅
        ]

        # 比較結果
        comparison = improved_mapper.compare_with_original(labels, texts)

        # 檢查改進指標
        assert "improvement_summary" in comparison
        improvement = comparison["improvement_summary"]

        # 應該有分離度改進
        assert improvement["separation_improvement"] > 0, "應該有分離度改進"

        # 應該實現毒性多樣性
        assert improvement["toxicity_diversity_increased"], "應該實現毒性標籤多樣性"

        # 應該有 F1 改進預期
        assert improvement["expected_f1_improvement"] > 0, "應該有 F1 改進預期"

    def test_label_distribution_change(self):
        """測試標籤分佈變化"""
        improved_mapper = ImprovedLabelMapper()

        # 構造測試案例，確保有分離效果
        labels = [1] * 10  # 全部是冒犯性標籤
        texts = [
            "這個遊戲真爛",  # 毒性但非霸凌
            "靠北這個問題",  # 毒性但非霸凌
            "你就是個白痴",  # 霸凌
            "你長得真醜",  # 霸凌
            "幹這個作業",  # 毒性但非霸凌
            "大家都討厭你",  # 霸凌
            "這個系統有夠爛",  # 毒性但非霸凌
            "你不配參加",  # 霸凌
            "操這個軟體",  # 毒性但非霸凌
            "我要殺了你",  # 威脅
        ]

        results = improved_mapper.batch_improve_cold_labels(labels, texts)
        stats = improved_mapper.analyze_label_distribution(results)

        # 檢查是否實現了標籤分離
        toxic_not_bullying = stats["correlation_analysis"]["toxic_not_bullying"]
        stats["correlation_analysis"]["bullying_not_toxic"]
        separation_score = stats["correlation_analysis"]["separation_score"]

        assert toxic_not_bullying > 0, "應該有毒性但非霸凌的案例"
        assert separation_score > 0, "應該實現標籤分離"


class TestUtilityFunctions:
    """測試工具函數"""

    def test_improved_cold_to_unified(self):
        """測試便利函數"""
        result = improved_cold_to_unified(1, "你就是個白痴")
        assert result.toxicity != ToxicityLevel.NONE
        assert result.bullying != BullyingLevel.NONE

    def test_analyze_text_bullying_features(self):
        """測試文本霸凌特徵分析"""
        features = analyze_text_bullying_features("你就是個白痴廢物")
        assert features["has_profanity"]
        assert features["has_personal_attack"]
        assert features["target_specific"]


if __name__ == "__main__":
    pytest.main([__file__])
