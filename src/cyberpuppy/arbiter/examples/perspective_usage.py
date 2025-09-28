"""
Perspective API 使用範例
展示如何整合 Perspective API 到 CyberPuppy 系統中
"""

import asyncio
import logging
import os

from ..integration import validate_with_arbiter
# 導入 CyberPuppy 模組
from ..perspective import PerspectiveAPI, UncertaintyDetector

# 設定日誌
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def basic_perspective_example():
    """基本 Perspective API 使用範例"""

    logger.info("=== 基本 Perspective API 使用範例 ===")

    # 檢查 API Key
    api_key = os.getenv("PERSPECTIVE_API_KEY")
    if not api_key:
        logger.warning("未設定 PERSPECTIVE_API_KEY，跳過此範例")
        return

    # 測試文本
    test_texts = [
        "你好，今天天氣很好！",  # 正面文本
        "這個產品真的很糟糕",  # 輕微負面
        "你這個白痴，給我滾開！",  # 明顯毒性
        "I hate you and want to hurt you badly",  # 英文威脅性文本
    ]

    async with PerspectiveAPI(api_key=api_key) as perspective:

        for i, text in enumerate(test_texts, 1):
            logger.info(f"\n--- 測試文本 {i}: {text} ---")

            try:
                # 分析文本
                result = await perspective.analyze_comment(
                    text=text,
                    lang=("zh" if any(ord(c) > 127 for c in text) else "en"),
                )

                # 顯示結果
                logger.info(f"毒性分數: {result.toxicity_score:.3f}")
                logger.info(f"嚴重毒性: {result.severe_toxicity_score:.3f}")
                logger.info(f"威脅分數: {result.threat_score:.3f}")
                logger.info(f"侮辱分數: {result.insult_score:.3f}")
                logger.info(f"處理時間: {result.processing_time_ms:.1f}ms")

                if result.language_detected:
                    logger.info(f"檢測語言: {result.language_detected}")

                # 分析信心度
                max_score = max(
                    result.toxicity_score,
                    result.severe_toxicity_score,
                    result.threat_score,
                    result.insult_score,
                )

                if max_score > 0.7:
                    logger.info("⚠️  高毒性風險")
                elif max_score > 0.4:
                    logger.info("⚡ 中等毒性風險")
                else:
                    logger.info("✅ 低毒性風險")

            except Exception as e:
                logger.error(f"分析失敗: {e}")

        # 取得配額狀態
        quota = await perspective.get_quota_status()
        logger.info("\n=== API 配額狀態 ===")
        logger.info(f"今日已使用: {quota['daily_requests_used']}")
        logger.info(f"每日限制: {quota['daily_requests_limit']}")


async def uncertainty_detection_example():
    """不確定性檢測範例"""

    logger.info("\n=== 不確定性檢測範例 ===")

    detector = UncertaintyDetector(
        uncertainty_threshold=0.4, confidence_threshold=0.6, min_confidence_gap=0.1
    )

    # 模擬不同的本地模型預測情況
    test_cases = [
        {
            "name": "高信心度預測（無需外部驗證）",
            "prediction": {
                "toxicity": "none",
                "scores": {"toxicity": {"none": 0.85, "toxic": 0.12, "severe": 0.03}},
                "emotion": "neu",
            },
        },
        {
            "name": "邊界分數（需要外部驗證）",
            "prediction": {
                "toxicity": "none",
                "scores": {"toxicity": {"none": 0.45, "toxic": 0.4, "severe": 0.15}},
                "emotion": "neu",
            },
        },
        {
            "name": "情緒衝突信號（需要驗證）",
            "prediction": {
                "toxicity": "none",
                "scores": {"toxicity": {"none": 0.8, "toxic": 0.15, "severe": 0.05}},
                "emotion": "neg",
                "emotion_strength": 4,
            },
        },
        {
            "name": "低信心度差距",
            "prediction": {
                "toxicity": "toxic",
                "scores": {"toxicity": {"none": 0.48, "toxic": 0.47, "severe": 0.05}},
                "emotion": "neu",
            },
        },
    ]

    for case in test_cases:
        logger.info(f"\n--- {case['name']} ---")

        should_use, analysis = detector.should_use_perspective(case["prediction"])

        logger.info(f"是否需要外部驗證: {should_use}")
        logger.info(f"信心度分數: {analysis.confidence_score:.3f}")
        logger.info(f"不確定性原因: {[reason.value for reason in analysis.reasons]}")
        logger.info(f"建議: {analysis.recommendation}")

        if analysis.threshold_details:
            logger.info(f"閾值詳情: {analysis.threshold_details}")


async def integrated_validation_example():
    """整合驗證範例"""

    logger.info("\n=== 整合驗證範例 ===")

    # 模擬需要外部驗證的情況
    uncertain_prediction = {
        "toxicity": "none",
        "bullying": "none",
        "role": "none",
        "emotion": "neg",
        "emotion_strength": 3,
        "scores": {"toxicity": {"none": 0.45, "toxic": 0.4, "severe": 0.15}},
    }

    test_texts = [
        "這個產品真的很爛，完全不推薦",
        "你為什麼要這樣對我？好難過...",
        "去死吧，我恨死你了！",
    ]

    for text in test_texts:
        logger.info(f"\n--- 分析文本: {text} ---")

        try:
            enhanced_prediction, metadata = await validate_with_arbiter(
                text=text, local_prediction=uncertain_prediction.copy()
            )

            logger.info(f"原始毒性預測: {uncertain_prediction['toxicity']}")
            logger.info(f"是否使用外部驗證: {metadata['used_external_validation']}")

            if metadata["uncertainty_analysis"]:
                ua = metadata["uncertainty_analysis"]
                logger.info(
                    f"不確定性分析: {ua['is_uncertain']} " f"({ua['confidence_score']:.3f})"
                )
                logger.info(f"不確定原因: {ua['reasons']}")

            if metadata["used_external_validation"]:
                pr = metadata["perspective_result"]
                logger.info(f"Perspective 毒性分數: {pr['toxicity_score']:.3f}")
                logger.info(f"Perspective 威脅分數: {pr['threat_score']:.3f}")
                logger.info(f"信心度評估: {pr['confidence_assessment']['confidence_level']}")

                if "confidence_adjustment" in enhanced_prediction:
                    logger.info(f"信心度調整: {enhanced_prediction['confidence_adjustment']}")
                    logger.info(f"驗證備註: {enhanced_prediction['validation_note']}")

            logger.info(f"最終建議: {metadata['recommendation']}")

        except Exception as e:
            logger.error(f"整合驗證失敗: {e}")


async def performance_analysis_example():
    """效能分析範例"""

    logger.info("\n=== 效能分析範例 ===")

    # 檢查 API Key
    if not os.getenv("PERSPECTIVE_API_KEY"):
        logger.warning("未設定 PERSPECTIVE_API_KEY，跳過效能測試")
        return

    import time

    async with PerspectiveAPI() as perspective:

        # 測試多個請求的效能
        test_texts = [
            "測試文本 1",
            "Testing text 2",
            "テストテキスト 3",
            "Texto de prueba 4",
            "Тестовый текст 5",
        ]

        logger.info("執行效能測試...")
        start_time = time.time()

        results = []
        for i, text in enumerate(test_texts, 1):
            logger.info(f"處理文本 {i}/{len(test_texts)}")

            try:
                result = await perspective.analyze_comment(text)
                results.append(result)
                logger.info(f"  處理時間: {result.processing_time_ms:.1f}ms")

            except Exception as e:
                logger.error(f"  處理失敗: {e}")

        total_time = time.time() - start_time
        logger.info("\n效能統計:")
        logger.info("總處理時間: %.2fs" % total_time)
        logger.info("平均每個請求: %.2fs" % (total_time / len(test_texts)))

        if results:
            avg_api_time = sum(r.processing_time_ms for r in results) / len(results)
            logger.info(f"平均 API 處理時間: {avg_api_time:.1f}ms")

        # 配額使用情況
        quota = await perspective.get_quota_status()
        logger.info(
            f"API 配額使用: {quota['daily_requests_used']}/" f"{quota['daily_requests_limit']}"
        )


async def main():
    """主要範例執行函式"""

    logger.info("🐕 CyberPuppy Perspective API 整合範例")
    logger.info("=" * 50)

    try:
        # 執行所有範例
        await basic_perspective_example()
        await uncertainty_detection_example()
        await integrated_validation_example()
        await performance_analysis_example()

        logger.info("\n✅ 所有範例執行完成")

    except KeyboardInterrupt:
        logger.info("\n❌ 使用者中斷執行")
    except Exception as e:
        logger.error(f"\n💥 執行過程中發生錯誤: {e}")


if __name__ == "__main__":
    asyncio.run(main())
