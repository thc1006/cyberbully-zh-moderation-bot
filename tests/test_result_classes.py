"""
Test module for CyberPuppy result classes.

This module tests the core result classes that are independent
of ML model dependencies, following TDD London School principles.
"""

import pytest
from datetime import datetime
import torch

# Import result classes that don't depend on ML models
from cyberpuppy.models.result import (
    DetectionResult, ToxicityResult, EmotionResult, BullyingResult, RoleResult,
    ExplanationResult, ModelPrediction, ToxicityLevel, EmotionType,
    BullyingType, RoleType, ResultAggregator, ConfidenceThresholds
)


class TestResultClasses:
    """Test all result classes comprehensively."""

    def test_toxicity_result_complete_workflow(self):
        """Test complete ToxicityResult workflow."""
        # Create result
        result = ToxicityResult(
            prediction=ToxicityLevel.SEVERE,
            confidence=0.92,
            raw_scores={'none': 0.03, 'toxic': 0.05, 'severe': 0.92},
            threshold_met=True
        )

        # Test basic properties
        assert result.prediction == ToxicityLevel.SEVERE
        assert result.confidence == 0.92
        assert result.threshold_met is True

        # Test serialization
        data = result.to_dict()
        assert data['prediction'] == 'severe'
        assert data['confidence'] == 0.92

        # Test deserialization
        restored = ToxicityResult.from_dict(data)
        assert restored.prediction == ToxicityLevel.SEVERE
        assert restored.confidence == 0.92

    def test_emotion_result_strength_validation(self):
        """Test EmotionResult strength validation and workflow."""
        # Valid strength
        result = EmotionResult(
            prediction=EmotionType.NEGATIVE,
            confidence=0.85,
            strength=4,
            raw_scores={'pos': 0.05, 'neu': 0.1, 'neg': 0.85},
            threshold_met=True
        )
        assert result.strength == 4

        # Invalid strength should raise error
        with pytest.raises(ValueError):
            EmotionResult(
                prediction=EmotionType.NEGATIVE,
                confidence=0.85,
                strength=5,  # Invalid
                raw_scores={'pos': 0.05, 'neu': 0.1, 'neg': 0.85},
                threshold_met=True
            )

    def test_complete_detection_result_workflow(self):
        """Test complete DetectionResult creation and manipulation."""
        # Create individual results
        toxicity = ToxicityResult(
            ToxicityLevel.TOXIC, 0.8, {'none': 0.1, 'toxic': 0.8, 'severe':
                0.1}, True
        )
        emotion = EmotionResult(
            EmotionType.NEGATIVE, 0.75, 3, {'pos': 0.1, 'neu': 0.15, 'neg':
                0.75}, True
        )
        bullying = BullyingResult(
            BullyingType.HARASSMENT, 0.7, {'none': 0.2, 'harassment': 0.7,
                'threat': 0.1}, True
        )
        role = RoleResult(
            RoleType.VICTIM, 0.65, {'none': 0.2, 'perpetrator': 0.1, 'victim':
                0.65, 'bystander': 0.05}, True
        )

        explanation = ExplanationResult(
            attributions=[0.1, 0.3, -0.2, 0.8],
            tokens=['這', '個', '人', '討厭'],
            explanation_text="'討厭' contributes most to negative prediction",
            top_contributing_words=[('討厭', 0.8), ('個', 0.3)],
            method='integrated_gradients'
        )

        model_pred = ModelPrediction(
            model_name='baseline',
            predictions={'toxicity': torch.tensor([0.1, 0.8, 0.1])},
            confidence_scores={'toxicity': 0.8},
            processing_time=0.15
        )

        # Create complete detection result
        result = DetectionResult(
            text="這個人很討厭",
            timestamp=datetime.now(),
            processing_time=0.25,
            toxicity=toxicity,
            emotion=emotion,
            bullying=bullying,
            role=role,
            explanations={'toxicity': explanation},
            model_predictions={'baseline': model_pred},
            ensemble_weights={'baseline': 1.0}
        )

        # Test high risk assessment
        assert result.is_high_risk() is True

        # Test summary
        summary = result.get_summary()
        assert summary['high_risk'] is True
        assert summary['toxicity'] == 'toxic'
        assert summary['emotion_strength'] == 3

        # Test JSON round trip
        json_str = result.to_json()
        assert isinstance(json_str, str)

        restored = DetectionResult.from_json(json_str)
        assert restored.text == result.text
        assert restored.toxicity.prediction == ToxicityLevel.TOXIC
        assert restored.emotion.strength == 3

    def test_result_aggregator_comprehensive(self):
        """Test ResultAggregator with multiple scenarios."""
        # Create diverse results
        results = []

        # High risk result
        high_risk = DetectionResult(
            text="威脅性文字",
            timestamp=datetime.now(),
            processing_time=0.3,
            toxicity=ToxicityResult(ToxicityLevel.SEVERE, 0.95, {}, True),
            emotion=EmotionResult(EmotionType.NEGATIVE, 0.9, 4, {}, True),
            bullying=BullyingResult(BullyingType.THREAT, 0.9, {}, True),
            role=RoleResult(RoleType.PERPETRATOR, 0.8, {}, True),
            explanations={},
            model_predictions={},
            ensemble_weights={}
        )

        # Medium risk result
        medium_risk = DetectionResult(
            text="一般負面文字",
            timestamp=datetime.now(),
            processing_time=0.2,
            toxicity=ToxicityResult(ToxicityLevel.TOXIC, 0.7, {}, True),
            emotion=EmotionResult(EmotionType.NEGATIVE, 0.6, 2, {}, True),
            bullying=BullyingResult(BullyingType.NONE, 0.8, {}, True),
            role=RoleResult(RoleType.NONE, 0.7, {}, True),
            explanations={},
            model_predictions={},
            ensemble_weights={}
        )

        # Low risk result
        low_risk = DetectionResult(
            text="正常文字",
            timestamp=datetime.now(),
            processing_time=0.1,
            toxicity=ToxicityResult(ToxicityLevel.NONE, 0.9, {}, True),
            emotion=EmotionResult(EmotionType.NEUTRAL, 0.8, 0, {}, True),
            bullying=BullyingResult(BullyingType.NONE, 0.9, {}, True),
            role=RoleResult(RoleType.NONE, 0.8, {}, True),
            explanations={},
            model_predictions={},
            ensemble_weights={}
        )

        results = [high_risk, medium_risk, low_risk]

        # Test aggregation
        stats = ResultAggregator.aggregate_batch_results(results)
        assert stats['total_results'] == 3
        assert stats['high_risk_count'] == 2  # high_risk and medium_risk should be flagged
        assert stats['high_risk_percentage'] == (2 / 3) * 100

        # Test confidence filtering
        high_conf_results = ResultAggregator.filter_results_by_confidence(
            results, 'toxicity', 0.8
        )
        assert len(high_conf_results) == 2  # high_risk and low_risk

        # Test top risk results
        top_risks = ResultAggregator.get_top_risk_results(results, top_k=2)
        assert len(top_risks) == 2
        assert top_risks[0] == high_risk  # Should be first due to highest risk

    def test_confidence_thresholds_management(self):
        """Test ConfidenceThresholds utility."""
        # Test default thresholds
        assert ConfidenceThresholds.get_threshold('toxicity', 'severe') == 0.8
        assert ConfidenceThresholds.get_threshold('emotion', 'pos') == 0.6

        # Test threshold updates
        original = ConfidenceThresholds.get_threshold('toxicity', 'toxic')

        ConfidenceThresholds.update_thresholds({
            'toxicity': {'toxic': 0.9}
        })

        assert ConfidenceThresholds.get_threshold('toxicity', 'toxic') == 0.9

        # Restore original
        ConfidenceThresholds.update_thresholds({
            'toxicity': {'toxic': original}
        })

        # Test validation
        valid_thresholds = {
            'toxicity': {'none': 0.5, 'toxic': 0.7}
        }
        assert ConfidenceThresholds.validate_thresholds(valid_thresholds) is True

        invalid_thresholds = {
            'toxicity': {'none': 1.5}  # Invalid: > 1.0
        }
        assert ConfidenceThresholds.validate_thresholds(invalid_thresholds) is False

    def test_detection_result_validation(self):
        """Test DetectionResult input validation."""
        # Test invalid confidence scores
        with pytest.raises(ValueError):
            DetectionResult(
                text="test",
                timestamp=datetime.now(),
                processing_time=0.1,
                toxicity=ToxicityResult(
                    ToxicityLevel.NONE,
                    1.5,
                    {},
                    True
                ),  # Invalid confidence
                emotion=EmotionResult(EmotionType.NEUTRAL, 0.8, 0, {}, True),
                bullying=BullyingResult(BullyingType.NONE, 0.8, {}, True),
                role=RoleResult(RoleType.NONE, 0.8, {}, True),
                explanations={},
                model_predictions={},
                ensemble_weights={}
            )

        # Test invalid processing time
        with pytest.raises(ValueError):
            DetectionResult(
                text="test",
                timestamp=datetime.now(),
                processing_time=-0.1,  # Invalid: negative
                toxicity=ToxicityResult(ToxicityLevel.NONE, 0.8, {}, True),
                emotion=EmotionResult(EmotionType.NEUTRAL, 0.8, 0, {}, True),
                bullying=BullyingResult(BullyingType.NONE, 0.8, {}, True),
                role=RoleResult(RoleType.NONE, 0.8, {}, True),
                explanations={},
                model_predictions={},
                ensemble_weights={}
            )

    def test_model_prediction_serialization(self):
        """Test ModelPrediction serialization with torch tensors."""
        pred = ModelPrediction(
            model_name='test_model',
            predictions={
                'task1': torch.tensor([0.2, 0.8]),
                'task2': torch.tensor([0.1, 0.4, 0.5])
            },
            confidence_scores={'task1': 0.8, 'task2': 0.5},
            processing_time=0.1
        )

        # Test serialization
        data = pred.to_dict()
        assert data['model_name'] == 'test_model'
        assert isinstance(data['predictions']['task1'], list)
        # Use approximate comparison for floating point
        assert abs(data['predictions']['task1'][0] - 0.2) < 1e-6
        assert abs(data['predictions']['task1'][1] - 0.8) < 1e-6

        # Test deserialization
        restored = ModelPrediction.from_dict(data)
        assert restored.model_name == 'test_model'
        assert isinstance(restored.predictions['task1'], torch.Tensor)
        assert torch.allclose(
            restored.predictions['task1'],
            torch.tensor([0.2, 0.8])
        )

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        # Test empty batch aggregation
        empty_stats = ResultAggregator.aggregate_batch_results([])
        assert empty_stats == {}

        # Test unknown task filtering
        result = DetectionResult(
            text="test",
            timestamp=datetime.now(),
            processing_time=0.1,
            toxicity=ToxicityResult(ToxicityLevel.NONE, 0.8, {}, True),
            emotion=EmotionResult(EmotionType.NEUTRAL, 0.8, 0, {}, True),
            bullying=BullyingResult(BullyingType.NONE, 0.8, {}, True),
            role=RoleResult(RoleType.NONE, 0.8, {}, True),
            explanations={},
            model_predictions={},
            ensemble_weights={}
        )

        # Should handle unknown task gracefully
        filtered = ResultAggregator.filter_results_by_confidence(
            [result],
            'unknown_task',
            0.5
        )
        assert filtered == []

    def test_comprehensive_risk_assessment(self):
        """Test comprehensive risk assessment with custom thresholds."""
        result = DetectionResult(
            text="測試文字",
            timestamp=datetime.now(),
            processing_time=0.1,
            toxicity=ToxicityResult(ToxicityLevel.TOXIC, 0.75, {}, True),
            emotion=EmotionResult(EmotionType.NEGATIVE, 0.85, 3, {}, True),
            bullying=BullyingResult(BullyingType.HARASSMENT, 0.72, {}, True),
            role=RoleResult(RoleType.VICTIM, 0.6, {}, True),
            explanations={},
            model_predictions={},
            ensemble_weights={}
        )

        # Test with default thresholds
        assert result.is_high_risk() is True

        # Test with custom thresholds (more strict)
        strict_thresholds = {
            'toxicity_toxic': 0.9,  # Higher than our 0.75
            'bullying_harassment': 0.8,  # Higher than our 0.72
            'emotion_negative_strong': 0.9  # Higher than our 0.85
        }
        assert result.is_high_risk(strict_thresholds) is False

        # Test with lenient thresholds
        lenient_thresholds = {
            'toxicity_toxic': 0.5,
            'bullying_harassment': 0.5,
            'emotion_negative_strong': 0.5
        }
        assert result.is_high_risk(lenient_thresholds) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
