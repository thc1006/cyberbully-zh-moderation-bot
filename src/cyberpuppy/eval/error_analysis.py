"""
錯誤分析模組
提供霸凌偵測系統的深度錯誤分析功能
"""

import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ErrorCase:
    """錯誤案例數據結構"""

    text: str
    true_label: str
    predicted_label: str
    confidence: float
    error_type: str  # 'false_positive', 'false_negative'
    toxicity_score: Optional[float] = None
    bullying_score: Optional[float] = None
    emotion_score: Optional[float] = None
    text_length: Optional[int] = None
    contains_keywords: Optional[List[str]] = None
    difficulty_level: Optional[str] = None  # 'easy', 'medium', 'hard'


@dataclass
class ErrorPattern:
    """錯誤模式分析結果"""

    pattern_type: str
    description: str
    frequency: int
    examples: List[ErrorCase]
    confidence_range: Tuple[float, float]
    suggested_improvements: List[str]


class ErrorAnalyzer:
    """綜合錯誤分析器"""

    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 關鍵詞庫用於分析
        self.bullying_keywords = [
            "笨蛋",
            "蠢",
            "死",
            "滾",
            "閉嘴",
            "噁心",
            "垃圾",
            "廢物",
            "豬",
            "狗",
            "畜生",
            "白痴",
            "智障",
            "神經病",
            "瘋子",
        ]

        self.positive_keywords = [
            "好",
            "棒",
            "讚",
            "厲害",
            "優秀",
            "開心",
            "快樂",
            "愛",
            "謝謝",
            "感謝",
            "幫助",
            "支持",
            "鼓勵",
            "加油",
        ]

    def analyze_errors(
        self,
        true_labels: List[str],
        predicted_labels: List[str],
        texts: List[str],
        confidences: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        全面錯誤分析

        Args:
            true_labels: 真實標籤
            predicted_labels: 預測標籤
            texts: 原始文本
            confidences: 預測信心分數
            metadata: 額外元數據

        Returns:
            錯誤分析結果
        """

        logger.info("開始執行錯誤分析...")

        # 建立錯誤案例
        error_cases = self._create_error_cases(
            true_labels, predicted_labels, texts, confidences, metadata
        )

        # 分析錯誤模式
        error_patterns = self._analyze_error_patterns(error_cases)

        # 分析困難樣本
        difficult_cases = self._identify_difficult_cases(error_cases)

        # 分析信心分數分布
        confidence_analysis = self._analyze_confidence_distribution(error_cases)

        # 文本特徵分析
        text_analysis = self._analyze_text_features(error_cases)

        # 生成改進建議
        improvements = self._generate_improvement_suggestions(
            error_patterns, difficult_cases, confidence_analysis
        )

        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "total_errors": len(error_cases),
            "error_cases": [asdict(case) for case in error_cases],
            "error_patterns": [asdict(pattern) for pattern in error_patterns],
            "difficult_cases": [asdict(case) for case in difficult_cases],
            "confidence_analysis": confidence_analysis,
            "text_analysis": text_analysis,
            "improvement_suggestions": improvements,
            "statistics": self._generate_error_statistics(error_cases),
        }

        # 保存結果
        self._save_analysis_results(analysis_results)

        logger.info(f"錯誤分析完成，共發現 {len(error_cases)} 個錯誤案例")

        return analysis_results

    def _create_error_cases(
        self,
        true_labels: List[str],
        predicted_labels: List[str],
        texts: List[str],
        confidences: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ErrorCase]:
        """建立錯誤案例列表"""

        error_cases = []

        for i, (true_label, pred_label, text, confidence) in enumerate(
            zip(true_labels, predicted_labels, texts, confidences)
        ):
            if true_label != pred_label:
                # 判斷錯誤類型
                if true_label == "none" and pred_label in ["toxic", "severe"]:
                    error_type = "false_positive"
                elif true_label in ["toxic", "severe"] and pred_label == "none":
                    error_type = "false_negative"
                else:
                    error_type = "misclassification"

                # 分析文本特徵
                contains_keywords = self._find_keywords_in_text(text)
                difficulty_level = self._assess_difficulty_level(
                    text, true_label, pred_label, confidence
                )

                error_case = ErrorCase(
                    text=text,
                    true_label=true_label,
                    predicted_label=pred_label,
                    confidence=confidence,
                    error_type=error_type,
                    text_length=len(text),
                    contains_keywords=contains_keywords,
                    difficulty_level=difficulty_level,
                )

                # 添加元數據
                if metadata and i < len(metadata.get("toxicity_scores", [])):
                    error_case.toxicity_score = metadata["toxicity_scores"][i]
                if metadata and i < len(metadata.get("bullying_scores", [])):
                    error_case.bullying_score = metadata["bullying_scores"][i]
                if metadata and i < len(metadata.get("emotion_scores", [])):
                    error_case.emotion_score = metadata["emotion_scores"][i]

                error_cases.append(error_case)

        return error_cases

    def _find_keywords_in_text(self, text: str) -> List[str]:
        """找出文本中包含的關鍵詞"""
        found_keywords = []

        for keyword in self.bullying_keywords + self.positive_keywords:
            if keyword in text:
                found_keywords.append(keyword)

        return found_keywords

    def _assess_difficulty_level(
        self, text: str, true_label: str, pred_label: str, confidence: float
    ) -> str:
        """評估樣本困難程度"""

        # 基於信心分數
        if confidence > 0.8:
            base_difficulty = "hard"  # 高信心但錯誤 = 困難
        elif confidence > 0.6:
            base_difficulty = "medium"
        else:
            base_difficulty = "easy"

        # 基於文本特徵調整
        text_length = len(text)
        contains_bullying_keywords = any(keyword in text for keyword in self.bullying_keywords)
        contains_positive_keywords = any(keyword in text for keyword in self.positive_keywords)

        # 調整困難度
        if text_length < 10 and not contains_bullying_keywords:
            return "hard"  # 短文本且無明顯關鍵詞
        elif contains_bullying_keywords and contains_positive_keywords:
            return "hard"  # 包含混合情感

        return base_difficulty

    def _analyze_error_patterns(self, error_cases: List[ErrorCase]) -> List[ErrorPattern]:
        """分析錯誤模式"""

        patterns = []

        # 按錯誤類型分組
        error_type_groups = defaultdict(list)
        for case in error_cases:
            error_type_groups[case.error_type].append(case)

        for error_type, cases in error_type_groups.items():
            if len(cases) < 2:
                continue

            # 分析信心分數分布
            confidences = [case.confidence for case in cases]
            confidence_range = (min(confidences), max(confidences))

            # 生成改進建議
            improvements = []
            if error_type == "false_positive":
                improvements.extend(
                    [
                        "提高模型對正常語言的辨識能力",
                        "增加正面樣本的訓練數據",
                        "調整決策閾值以降低誤報率",
                    ]
                )
            elif error_type == "false_negative":
                improvements.extend(
                    ["加強對隱含霸凌語言的識別", "增加更多霸凌樣本進行訓練", "改進特徵提取方法"]
                )

            pattern = ErrorPattern(
                pattern_type=error_type,
                description=f"{error_type} 錯誤模式分析",
                frequency=len(cases),
                examples=cases[:5],  # 取前5個例子
                confidence_range=confidence_range,
                suggested_improvements=improvements,
            )

            patterns.append(pattern)

        # 分析基於關鍵詞的模式
        keyword_patterns = self._analyze_keyword_patterns(error_cases)
        patterns.extend(keyword_patterns)

        # 分析基於文本長度的模式
        length_patterns = self._analyze_length_patterns(error_cases)
        patterns.extend(length_patterns)

        return patterns

    def _analyze_keyword_patterns(self, error_cases: List[ErrorCase]) -> List[ErrorPattern]:
        """分析關鍵詞相關的錯誤模式"""

        patterns = []

        # 分析包含霸凌關鍵詞但被誤判為正常的案例
        fp_with_keywords = [
            case
            for case in error_cases
            if case.error_type == "false_positive"
            and any(keyword in self.bullying_keywords for keyword in (case.contains_keywords or []))
        ]

        if fp_with_keywords:
            pattern = ErrorPattern(
                pattern_type="false_positive_with_bullying_keywords",
                description="包含霸凌關鍵詞但被誤判為正常的案例",
                frequency=len(fp_with_keywords),
                examples=fp_with_keywords[:3],
                confidence_range=(
                    min(case.confidence for case in fp_with_keywords),
                    max(case.confidence for case in fp_with_keywords),
                ),
                suggested_improvements=[
                    "改進上下文理解能力",
                    "增加諷刺和反語的識別",
                    "考慮詞彙的語境依賴性",
                ],
            )
            patterns.append(pattern)

        # 分析不包含明顯關鍵詞但被誤判為霸凌的案例
        fn_without_keywords = [
            case
            for case in error_cases
            if case.error_type == "false_negative"
            and not any(
                keyword in self.bullying_keywords for keyword in (case.contains_keywords or [])
            )
        ]

        if fn_without_keywords:
            pattern = ErrorPattern(
                pattern_type="false_negative_without_keywords",
                description="不包含明顯關鍵詞但實際為霸凌的案例",
                frequency=len(fn_without_keywords),
                examples=fn_without_keywords[:3],
                confidence_range=(
                    min(case.confidence for case in fn_without_keywords),
                    max(case.confidence for case in fn_without_keywords),
                ),
                suggested_improvements=[
                    "提升對隱含霸凌語言的理解",
                    "加強語義層面的分析",
                    "增加更多隱蔽霸凌的訓練樣本",
                ],
            )
            patterns.append(pattern)

        return patterns

    def _analyze_length_patterns(self, error_cases: List[ErrorCase]) -> List[ErrorPattern]:
        """分析文本長度相關的錯誤模式"""

        patterns = []

        # 分析短文本錯誤
        short_errors = [case for case in error_cases if (case.text_length or 0) < 20]
        if short_errors:
            pattern = ErrorPattern(
                pattern_type="short_text_errors",
                description="短文本（<20字符）錯誤模式",
                frequency=len(short_errors),
                examples=short_errors[:3],
                confidence_range=(
                    min(case.confidence for case in short_errors),
                    max(case.confidence for case in short_errors),
                ),
                suggested_improvements=[
                    "改進短文本的特徵提取",
                    "增加短文本訓練樣本",
                    "調整模型架構以適應短序列",
                ],
            )
            patterns.append(pattern)

        # 分析長文本錯誤
        long_errors = [case for case in error_cases if (case.text_length or 0) > 100]
        if long_errors:
            pattern = ErrorPattern(
                pattern_type="long_text_errors",
                description="長文本（>100字符）錯誤模式",
                frequency=len(long_errors),
                examples=long_errors[:3],
                confidence_range=(
                    min(case.confidence for case in long_errors),
                    max(case.confidence for case in long_errors),
                ),
                suggested_improvements=[
                    "改進長序列建模能力",
                    "使用注意力機制突出重要片段",
                    "考慮分段處理策略",
                ],
            )
            patterns.append(pattern)

        return patterns

    def _identify_difficult_cases(self, error_cases: List[ErrorCase]) -> List[ErrorCase]:
        """識別困難樣本"""

        # 按困難程度和信心分數排序
        difficult_cases = [
            case for case in error_cases if case.difficulty_level == "hard" or case.confidence > 0.7
        ]

        # 按信心分數降序排列（高信心的錯誤更值得關注）
        difficult_cases.sort(key=lambda x: x.confidence, reverse=True)

        return difficult_cases[:20]  # 返回前20個最困難的案例

    def _analyze_confidence_distribution(self, error_cases: List[ErrorCase]) -> Dict[str, Any]:
        """分析信心分數分布"""

        confidences = [case.confidence for case in error_cases]

        if not confidences:
            return {}

        analysis = {
            "mean_confidence": np.mean(confidences),
            "median_confidence": np.median(confidences),
            "std_confidence": np.std(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "high_confidence_errors": len([c for c in confidences if c > 0.8]),
            "medium_confidence_errors": len([c for c in confidences if 0.5 < c <= 0.8]),
            "low_confidence_errors": len([c for c in confidences if c <= 0.5]),
            "confidence_quartiles": {
                "Q1": np.percentile(confidences, 25),
                "Q2": np.percentile(confidences, 50),
                "Q3": np.percentile(confidences, 75),
            },
        }

        return analysis

    def _analyze_text_features(self, error_cases: List[ErrorCase]) -> Dict[str, Any]:
        """分析文本特徵"""

        if not error_cases:
            return {}

        text_lengths = [case.text_length or 0 for case in error_cases]
        keyword_counts = [len(case.contains_keywords or []) for case in error_cases]

        analysis = {
            "length_statistics": {
                "mean_length": np.mean(text_lengths),
                "median_length": np.median(text_lengths),
                "min_length": min(text_lengths),
                "max_length": max(text_lengths),
                "std_length": np.std(text_lengths),
            },
            "keyword_statistics": {
                "mean_keywords": np.mean(keyword_counts),
                "median_keywords": np.median(keyword_counts),
                "max_keywords": max(keyword_counts),
                "cases_with_keywords": len([c for c in keyword_counts if c > 0]),
            },
            "error_type_distribution": Counter([case.error_type for case in error_cases]),
            "difficulty_distribution": Counter([case.difficulty_level for case in error_cases]),
        }

        return analysis

    def _generate_improvement_suggestions(
        self,
        error_patterns: List[ErrorPattern],
        difficult_cases: List[ErrorCase],
        confidence_analysis: Dict[str, Any],
    ) -> List[str]:
        """生成改進建議"""

        suggestions = []

        # 基於錯誤模式的建議
        for pattern in error_patterns:
            suggestions.extend(pattern.suggested_improvements)

        # 基於信心分數分析的建議
        if confidence_analysis:
            high_conf_errors = confidence_analysis.get("high_confidence_errors", 0)
            total_errors = len(difficult_cases) if difficult_cases else 1

            if high_conf_errors / total_errors > 0.3:
                suggestions.append("模型過度自信，建議添加不確定性估計")
                suggestions.append("考慮使用校準技術改善信心分數")

        # 基於困難案例的建議
        if difficult_cases:
            hard_cases = [case for case in difficult_cases if case.difficulty_level == "hard"]
            if len(hard_cases) > 5:
                suggestions.append("增加更多困難樣本進行訓練")
                suggestions.append("考慮使用主動學習選擇困難樣本")

        # 去除重複建議
        suggestions = list(set(suggestions))

        return suggestions

    def _generate_error_statistics(self, error_cases: List[ErrorCase]) -> Dict[str, Any]:
        """生成錯誤統計信息"""

        if not error_cases:
            return {}

        stats = {
            "total_errors": len(error_cases),
            "error_types": {
                "false_positive": len([c for c in error_cases if c.error_type == "false_positive"]),
                "false_negative": len([c for c in error_cases if c.error_type == "false_negative"]),
                "misclassification": len(
                    [c for c in error_cases if c.error_type == "misclassification"]
                ),
            },
            "difficulty_levels": {
                "easy": len([c for c in error_cases if c.difficulty_level == "easy"]),
                "medium": len([c for c in error_cases if c.difficulty_level == "medium"]),
                "hard": len([c for c in error_cases if c.difficulty_level == "hard"]),
            },
            "confidence_distribution": {
                "high": len([c for c in error_cases if c.confidence > 0.8]),
                "medium": len([c for c in error_cases if 0.5 < c.confidence <= 0.8]),
                "low": len([c for c in error_cases if c.confidence <= 0.5]),
            },
        }

        return stats

    def _save_analysis_results(self, results: Dict[str, Any]):
        """保存分析結果"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_analysis_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"錯誤分析結果已保存至: {filepath}")


class FalsePositiveAnalyzer:
    """False Positive 專門分析器"""

    def __init__(self):
        self.common_fp_patterns = [
            "正面詞彙被誤判",
            "中性表達被誤判",
            "疑問句被誤判",
            "引用他人言論被誤判",
        ]

    def analyze_false_positives(self, error_cases: List[ErrorCase]) -> Dict[str, Any]:
        """分析 False Positive 案例"""

        fp_cases = [case for case in error_cases if case.error_type == "false_positive"]

        if not fp_cases:
            return {"message": "未發現 False Positive 案例"}

        # 按信心分數分組
        high_conf_fp = [case for case in fp_cases if case.confidence > 0.8]
        medium_conf_fp = [case for case in fp_cases if 0.5 < case.confidence <= 0.8]
        low_conf_fp = [case for case in fp_cases if case.confidence <= 0.5]

        analysis = {
            "total_false_positives": len(fp_cases),
            "confidence_groups": {
                "high_confidence": {
                    "count": len(high_conf_fp),
                    "examples": [case.text for case in high_conf_fp[:3]],
                    "avg_confidence": (
                        np.mean([case.confidence for case in high_conf_fp]) if high_conf_fp else 0
                    ),
                },
                "medium_confidence": {
                    "count": len(medium_conf_fp),
                    "examples": [case.text for case in medium_conf_fp[:3]],
                    "avg_confidence": (
                        np.mean([case.confidence for case in medium_conf_fp])
                        if medium_conf_fp
                        else 0
                    ),
                },
                "low_confidence": {
                    "count": len(low_conf_fp),
                    "examples": [case.text for case in low_conf_fp[:3]],
                    "avg_confidence": (
                        np.mean([case.confidence for case in low_conf_fp]) if low_conf_fp else 0
                    ),
                },
            },
            "text_length_analysis": {
                "short_texts": len([case for case in fp_cases if (case.text_length or 0) < 20]),
                "medium_texts": len(
                    [case for case in fp_cases if 20 <= (case.text_length or 0) < 50]
                ),
                "long_texts": len([case for case in fp_cases if (case.text_length or 0) >= 50]),
            },
            "recommendations": [
                "增加正面語言樣本的訓練",
                "改進上下文理解能力",
                "調整決策閾值以降低誤報率",
                "加強對中性語言的識別",
            ],
        }

        return analysis


class FalseNegativeAnalyzer:
    """False Negative 專門分析器"""

    def __init__(self):
        self.subtle_bullying_indicators = ["隱含威脅", "間接羞辱", "社交排斥", "身份攻擊"]

    def analyze_false_negatives(self, error_cases: List[ErrorCase]) -> Dict[str, Any]:
        """分析 False Negative 案例"""

        fn_cases = [case for case in error_cases if case.error_type == "false_negative"]

        if not fn_cases:
            return {"message": "未發現 False Negative 案例"}

        # 分析沒有明顯關鍵詞的案例
        subtle_cases = [
            case
            for case in fn_cases
            if not any(
                keyword in (case.contains_keywords or [])
                for keyword in ["笨蛋", "蠢", "死", "滾", "閉嘴", "垃圾"]
            )
        ]

        analysis = {
            "total_false_negatives": len(fn_cases),
            "subtle_bullying_cases": {
                "count": len(subtle_cases),
                "percentage": len(subtle_cases) / len(fn_cases) * 100 if fn_cases else 0,
                "examples": [case.text for case in subtle_cases[:5]],
            },
            "confidence_analysis": {
                "mean_confidence": np.mean([case.confidence for case in fn_cases]),
                "low_confidence_cases": len([case for case in fn_cases if case.confidence < 0.3]),
            },
            "severity_analysis": {
                "mild_cases": len([case for case in fn_cases if case.true_label == "toxic"]),
                "severe_cases": len([case for case in fn_cases if case.true_label == "severe"]),
            },
            "recommendations": [
                "增加隱含霸凌語言的訓練樣本",
                "改進語義理解能力",
                "加強對間接攻擊的識別",
                "使用更複雜的特徵提取方法",
                "考慮多輪對話的上下文信息",
            ],
        }

        return analysis
