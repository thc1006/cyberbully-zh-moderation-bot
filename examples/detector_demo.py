"""
Demonstration of CyberPuppy Detection System.

This script shows how to use the result classes and demonstrates
the complete workflow without requiring actual ML models.
"""

from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cyberpuppy.models.result import (  # noqa: E402
    DetectionResult, ToxicityResult, EmotionResult, BullyingResult, RoleResult,
    ExplanationResult, ModelPrediction, ToxicityLevel, EmotionType,
    BullyingType, RoleType, ResultAggregator, ConfidenceThresholds
)
import torch  # noqa: E402


def create_sample_result(text: str, risk_level: str = "med"
    "ium") -> DetectionResult:
    """Create a sample detection result for demonstration."""

    if risk_level == "high":
        toxicity = ToxicityResult(
            prediction=ToxicityLevel.SEVERE,
            confidence=0.95,
            raw_scores={'none': 0.02, 'toxic': 0.03, 'severe': 0.95},
            threshold_met=True
        )
        emotion = EmotionResult(
            prediction=EmotionType.NEGATIVE,
            confidence=0.92,
            strength=4,
            raw_scores={'pos': 0.03, 'neu': 0.05, 'neg': 0.92},
            threshold_met=True
        )
        bullying = BullyingResult(
            prediction=BullyingType.THREAT,
            confidence=0.88,
            raw_scores={'none': 0.07, 'harassment': 0.05, 'threat': 0.88},
            threshold_met=True
        )
        role = RoleResult(
            prediction=RoleType.PERPETRATOR,
            confidence=0.85,
            raw_scores={'none': 0.05, 'perpetrator': 0.85, 'victim': 0.05,
                'bystander': 0.05},
            threshold_met=True
        )
    elif risk_level == "low":
        toxicity = ToxicityResult(
            prediction=ToxicityLevel.NONE,
            confidence=0.92,
            raw_scores={'none': 0.92, 'toxic': 0.05, 'severe': 0.03},
            threshold_met=True
        )
        emotion = EmotionResult(
            prediction=EmotionType.POSITIVE,
            confidence=0.85,
            strength=2,
            raw_scores={'pos': 0.85, 'neu': 0.10, 'neg': 0.05},
            threshold_met=True
        )
        bullying = BullyingResult(
            prediction=BullyingType.NONE,
            confidence=0.95,
            raw_scores={'none': 0.95, 'harassment': 0.03, 'threat': 0.02},
            threshold_met=True
        )
        role = RoleResult(
            prediction=RoleType.NONE,
            confidence=0.88,
            raw_scores={'none': 0.88, 'perpetrator': 0.04, 'victim': 0.04,
                'bystander': 0.04},
            threshold_met=True
        )
    else:  # medium risk
        toxicity = ToxicityResult(
            prediction=ToxicityLevel.TOXIC,
            confidence=0.78,
            raw_scores={'none': 0.15, 'toxic': 0.78, 'severe': 0.07},
            threshold_met=True
        )
        emotion = EmotionResult(
            prediction=EmotionType.NEGATIVE,
            confidence=0.72,
            strength=2,
            raw_scores={'pos': 0.10, 'neu': 0.18, 'neg': 0.72},
            threshold_met=True
        )
        bullying = BullyingResult(
            prediction=BullyingType.HARASSMENT,
            confidence=0.65,
            raw_scores={'none': 0.25, 'harassment': 0.65, 'threat': 0.10},
            threshold_met=True
        )
        role = RoleResult(
            prediction=RoleType.VICTIM,
            confidence=0.62,
            raw_scores={'none': 0.28, 'perpetrator': 0.05, 'victim': 0.62,
                'bystander': 0.05},
            threshold_met=True
        )

    # Create explanation
    explanation = ExplanationResult(
        attributions=[0.1, 0.3, -0.1, 0.8, -0.2, 0.6],
        tokens=text.split() if len(text.split()) <= 6 else text.split()[:6],
        explanation_text=f"Key words contribute to {r"
            "isk_level} risk assessment",
        top_contributing_words=[
            (text.split()[min(3, len(text.split())-1)], 0.8),
            (text.split()[min(1, len(text.split())-1)], 0.6)
        ] if text.split() else [("word", 0.8)],
        method='integrated_gradients'
    )

    # Create model prediction
    model_pred = ModelPrediction(
        model_name='demo_model',
        predictions={
            'toxicity': torch.tensor(list(toxicity.raw_scores.values())),
            'emotion': torch.tensor(list(emotion.raw_scores.values())),
            'bullying': torch.tensor(list(bullying.raw_scores.values())),
            'role': torch.tensor(list(role.raw_scores.values()))
        },
        confidence_scores={
            'toxicity': toxicity.confidence,
            'emotion': emotion.confidence,
            'bullying': bullying.confidence,
            'role': role.confidence
        },
        processing_time=0.12
    )

    return DetectionResult(
        text=text,
        timestamp=datetime.now(),
        processing_time=0.15,
        toxicity=toxicity,
        emotion=emotion,
        bullying=bullying,
        role=role,
        explanations={'toxicity': explanation, 'emotion': explanation},
        model_predictions={'demo_model': model_pred},
        ensemble_weights={'demo_model': 1.0}
    )


def main():
    """Demonstrate the CyberPuppy detection system."""

    # Set UTF-8 encoding for Windows
    import codecs
    if sys.platform.startswith('win'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

    print("CyberPuppy Detection System Demo")
    print("=" * 50)

    # Create sample texts with different risk levels
    texts = [
        ("我今天心情很好，謝謝大家的支持！", "low"),
        ("這個人很討厭，我不喜歡他的態度", "medium"),
        ("我要殺了你，你這個白痴去死吧", "high")
    ]

    results = []

    print("\nIndividual Results:")
    print("-" * 30)

    for text, risk_level in texts:
        result = create_sample_result(text, risk_level)
        results.append(result)

        print(f"\n文本: {text}")
        print(f"風險等級: {risk_level.upper()}")
        print(f"毒性: {result.toxicity.prediction.value}"
            " (信心: {result.toxicity.confidence:.2f})")
        print(f"情緒: {result.emotion.prediction.val"
            "ue} (強度: {result.emotion.strength})")
        print(f"霸凌: {result.bullying.prediction.value}"
            " (信心: {result.bullying.confidence:.2f})")
        print(f"角色: {result.role.prediction.value}"
            " (信心: {result.role.confidence:.2f})")
        print(f"高風險: {'是' if result.is_high_risk() else '否'}")

        # Show summary
        summary = result.get_summary()
        print(f"處理時間: {summary['processing_time']:.3f}秒")

    print("\nBatch Analysis:")
    print("-" * 30)

    # Aggregate results
    stats = ResultAggregator.aggregate_batch_results(results)
    print(f"總文本數: {stats['total_results']}")
    print(f"高風險文本: {stats['high_risk_count']} ({"
        "stats['high_risk_percentage']:.1f}%)")

    print("\n毒性分佈:")
    for level, count in stats['prediction_counts']['toxicity'].items():
        print(f"  {level}: {count}")

    print("\n情緒分佈:")
    for emotion, count in stats['prediction_counts']['emotion'].items():
        print(f"  {emotion}: {count}")

    print("\n處理時間統計:")
    time_stats = stats['processing_time_statistics']
    print(f"  平均: {time_stats['mean']:.3f}秒")
    print(f"  最小: {time_stats['min']:.3f}秒")
    print(f"  最大: {time_stats['max']:.3f}秒")

    # Filter high confidence results
    print("\nHigh Confidence Toxicity Results:")
    print("-" * 40)
    high_conf_results = ResultAggregator.filter_results_by_confidence(
        results, 'toxicity', 0.8
    )
    for result in high_conf_results:
        print(f"文本: {result.text[:30]}...")
        print(f"毒性信心: {result.toxicity.confidence:.2f}")

    # Get top risk results
    print("\nTop Risk Results:")
    print("-" * 25)
    top_risks = ResultAggregator.get_top_risk_results(results, top_k=2)
    for i, result in enumerate(top_risks, 1):
        print(f"{i}. {result.text[:40]}...")
        print(f"   毒性: {result.toxicity.prediction.value}")
        print(f"   霸凌: {result.bullying.prediction.value}")

    # Demonstrate JSON serialization
    print("\nJSON Serialization Demo:")
    print("-" * 35)
    sample_result = results[0]
    json_str = sample_result.to_json(indent=2)
    print(f"JSON長度: {len(json_str)} 字符")

    # Test round-trip
    restored = DetectionResult.from_json(json_str)
    print(f"序列化測試: {'通過' if restored.text == sample_result.text else '失敗'}")

    # Confidence threshold management
    print("\nConfidence Threshold Management:")
    print("-" * 45)
    print("Default thresholds:")
    for task in ['toxicity', 'emotion', 'bullying', 'role']:
        if task in ConfidenceThresholds.DEFAULT_THRESHOLDS:
            thresholds = ConfidenceThresholds.DEFAULT_THRESHOLDS[task]
            print(f"  {task}: {thresholds}")

    print("\nDemo completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
