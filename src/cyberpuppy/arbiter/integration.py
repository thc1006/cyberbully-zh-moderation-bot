"""
仲裁服務整合模組
將外部 API 整合到主要分析流程中
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from .config import arbiter_config
from .perspective import PerspectiveAPI, PerspectiveResult, UncertaintyDetector

logger = logging.getLogger(__name__)


class ArbiterService:
    """
    仲裁服務統一介面

    整合多個外部驗證服務，提供統一的不確定性檢測與外部驗證功能
    """

    def __init__(self):
        """初始化仲裁服務"""
        self.uncertainty_detector = UncertaintyDetector(
            uncertainty_threshold=arbiter_config.uncertainty.uncertainty_threshold,
            confidence_threshold=arbiter_config.uncertainty.confidence_threshold,
            min_confidence_gap=arbiter_config.uncertainty.min_confidence_gap,
        )

        self._perspective_api: Optional[PerspectiveAPI] = None
        self._is_perspective_available = arbiter_config.is_perspective_enabled()

        logger.info(
            f"仲裁服務已初始化 - Perspective API 可用: {self._is_perspective_available}"
        )

    async def __aenter__(self):
        """異步上下文管理器進入"""
        if self._is_perspective_available:
            self._perspective_api = PerspectiveAPI(
                api_key=arbiter_config.perspective.api_key,
                rate_limit=arbiter_config.get_perspective_rate_limit(),
            )
            await self._perspective_api.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器退出"""
        if self._perspective_api:
            await self._perspective_api.__aexit__(exc_type, exc_val, exc_tb)

    async def validate_prediction(
        self, text: str, local_prediction: Dict[str, Any], context: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        驗證本地模型預測

        Args:
            text: 原始文本
            local_prediction: 本地模型預測結果
            context: 可選的上下文資訊

        Returns:
            Tuple[enhanced_prediction, validation_metadata]: 增強預測結果與驗證元資料
        """
        validation_metadata = {
            "used_external_validation": False,
            "uncertainty_analysis": None,
            "perspective_result": None,
            "recommendation": "local_prediction_sufficient",
        }

        try:
            # 檢查是否需要外部驗證
            should_validate, uncertainty_analysis = (
                self.uncertainty_detector.should_use_perspective(local_prediction)
            )

            validation_metadata["uncertainty_analysis"] = {
                "is_uncertain": uncertainty_analysis.is_uncertain,
                "confidence_score": uncertainty_analysis.confidence_score,
                "rea" "sons": [reason.value for reason in uncertainty_analysis.reasons],
                "recommendation": uncertainty_analysis.recommendation,
                "threshold_details": uncertainty_analysis.threshold_details,
            }

            enhanced_prediction = local_prediction.copy()

            if (
                should_validate
                and self._is_perspective_available
                and self._perspective_api
            ):
                logger.info("執行 Perspective API 外部驗證")

                try:
                    # 呼叫 Perspective API
                    perspective_result = await self._perspective_api.analyze_comment(
                        text=text, lang="zh"
                    )

                    validation_metadata["used_external_validation"] = True
                    validation_metadata["perspective_result"] = {
                        "toxicity_score": perspective_result.toxicity_score,
                        "severe_tox"
                        "icity_score": perspective_result.severe_toxicity_score,
                        "threat_score": perspective_result.threat_score,
                        "processin" "g_time_ms": perspective_result.processing_time_ms,
                        "confidence"
                        "_assessment": self._assess_perspective_confidence(
                            perspective_result
                        ),
                    }

                    # 整合外部驗證結果
                    enhanced_prediction = self._integrate_external_validation(
                        local_prediction, perspective_result, uncertainty_analysis
                    )

                    validation_metadata["recommendation"] = (
                        "External validation successful"
                    )

                except Exception as e:
                    logger.error(f"Perspective API 驗證失敗: {e}")
                    validation_metadata["perspective_error"] = str(e)
                    validation_metadata["recommendation"] = (
                        "External validation failed, use with caution"
                    )

            elif should_validate and not self._is_perspective_available:
                logger.warning("需要外部驗證但 Perspective API 不可用")
                validation_metadata["recommendation"] = (
                    "External validation unavailable, use local prediction with caution"
                )

            return enhanced_prediction, validation_metadata

        except Exception as e:
            logger.error(f"驗證過程發生錯誤: {e}")
            validation_metadata["validation_error"] = str(e)
            validation_metadata["recommendation"] = (
                "Validation error occurred, use prediction with extreme caution"
            )
            return local_prediction, validation_metadata

    def _assess_perspective_confidence(
        self, result: PerspectiveResult
    ) -> Dict[str, Any]:
        """評估 Perspective API 結果的信心度"""
        # 基於多個屬性評估整體信心度
        scores = [
            result.toxicity_score,
            result.severe_toxicity_score,
            result.threat_score,
            result.insult_score,
        ]

        max_score = max(scores)
        score_variance = sum(
            (s - sum(scores) / len(scores)) ** 2 for s in scores
        ) / len(scores)

        # 信心度評估
        if max_score > 0.8 or max_score < 0.2:
            confidence_level = "high"
        elif max_score > 0.6 or max_score < 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        return {
            "confidence_level": confidence_level,
            "max_score": max_score,
            "score_variance": score_variance,
            "language_detected": result.language_detected,
            "processing_time_ms": result.processing_time_ms,
        }

    def _integrate_external_validation(
        self,
        local_prediction: Dict[str, Any],
        perspective_result: PerspectiveResult,
        uncertainty_analysis,
    ) -> Dict[str, Any]:
        """整合外部驗證結果到本地預測"""
        enhanced = local_prediction.copy()

        # 添加外部驗證資訊
        enhanced["external_validation"] = {
            "perspective_toxicity": perspective_result.toxicity_score,
            "perspective_severe": perspective_result.severe_toxicity_score,
            "perspective_threat": perspective_result.threat_score,
        }

        # 計算調整後的信心度
        local_scores = local_prediction.get("scores", {}).get("toxicity", [0.0])
        local_max_score = max(local_scores) if local_scores else 0.0
        perspective_score = perspective_result.toxicity_score

        # 如果外部驗證與本地預測差異很大，降低整體信心度
        score_difference = abs(local_max_score - perspective_score)
        if score_difference > 0.3:
            enhanced["confidence_adjustment"] = "low_agreement"
            enhanced["validation_note"] = (
                "External validation shows significant disagreement"
            )
        elif score_difference < 0.1:
            enhanced["confidence_adjustment"] = "high_agreement"
            enhanced["validation_note"] = (
                "External validation strongly agrees with local prediction"
            )
        else:
            enhanced["confidence_adjustment"] = "moderate_agreement"
            enhanced["validation_note"] = (
                "External validation moderately agrees with local prediction"
            )

        return enhanced

    async def get_service_status(self) -> Dict[str, Any]:
        """取得仲裁服務狀態"""
        status = {
            "arbiter_service": "active",
            "uncertainty_detector": "active",
            "perspective_api": "unavailable",
        }

        if self._perspective_api:
            try:
                quota = await self._perspective_api.get_quota_status()
                status["perspective_api"] = "active"
                status["perspective_quota"] = quota
            except Exception as e:
                status["perspective_api"] = "error"
                status["perspective_error"] = str(e)

        status["configuration"] = arbiter_config.to_dict()
        return status


# 便利函式
async def validate_with_arbiter(
    text: str, local_prediction: Dict[str, Any], context: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    使用仲裁服務驗證預測的便利函式

    Args:
        text: 原始文本
        local_prediction: 本地模型預測
        context: 可選上下文

    Returns:
        Tuple[enhanced_prediction, validation_metadata]
    """
    async with ArbiterService() as arbiter:
        return await arbiter.validate_prediction(text, local_prediction, context)


# 使用範例
async def example_integration():
    """整合使用範例"""

    # 模擬本地模型預測結果
    local_prediction = {
        "toxicity": "none",
        "bullying": "none",
        "emotion": "neg",
        "emotion_strength": 3,
        "scores": {"toxicity": {"none": 0.45, "toxic": 0.4, "severe": 0.15}},
    }

    text = "這個產品真的很糟糕，完全不推薦"

    try:
        enhanced_prediction, metadata = await validate_with_arbiter(
            text=text, local_prediction=local_prediction
        )

        logger.info(f"原始預測毒性: {local_prediction.get('toxicity')}")
        logger.info(f"是否使用外部驗證: {metadata['used_external_validation']}")

        if metadata["used_external_validation"]:
            perspective_result = metadata["perspective_result"]
            perspective_score = perspective_result.toxicity_score
            logger.info(f"Perspective 毒性分數: {perspective_score:.3f}")
            logger.info(
                f"信心度調整: {enhanced_prediction.get('confidence_adjustment')}"
            )

        logger.info(f"建議: {metadata['recommendation']}")

    except Exception as e:
        logger.error(f"仲裁驗證失敗: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_integration())
