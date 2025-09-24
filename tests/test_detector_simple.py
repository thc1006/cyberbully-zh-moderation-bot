"""
Simplified tests for CyberPuppyDetector and DetectionResult classes.

This focuses on the core TDD principles and behavior verification
without complex mocking of dependencies that don't exist yet.
"""

import pytest
from unittest.mock import patch
from datetime import datetime
import torch


# Import our actual classes
from src.cyberpuppy.models.result import (
    DetectionResult, ToxicityResult, EmotionResult, BullyingResult, RoleResult,
    ToxicityLevel, EmotionType, BullyingType, RoleType,
    ExplanationResult, ModelPrediction, ResultAggregator, ConfidenceThresholds
)


@pytest.fixture
def sample_toxicity_result():
    """Sample toxicity result for testing."""
    return ToxicityResult(
        prediction=ToxicityLevel.TOXIC,
        confidence=0.85,
        raw_scores={'none': 0.1, 'toxic': 0.8, 'severe': 0.1},
        threshold_met=True
    )


@pytest.fixture
def sample_emotion_result():
    """Sample emotion result for testing."""
    return EmotionResult(
        prediction=EmotionType.NEGATIVE,
        confidence=0.78,
        strength=3,
        raw_scores={'pos': 0.1, 'neu': 0.12, 'neg': 0.78},
        threshold_met=True
    )


@pytest.fixture
def sample_bullying_result():
    """Sample bullying result for testing."""
    return BullyingResult(
        prediction=BullyingType.HARASSMENT,
        confidence=0.72,
        raw_scores={'none': 0.18, 'harassment': 0.72, 'threat': 0.1},
        threshold_met=True
    )


@pytest.fixture
def sample_role_result():
    """Sample role result for testing."""
    return RoleResult(
        prediction=RoleType.VICTIM,
        confidence=0.65,
        raw_scores={'none': 0.2, 'perpetrator': 0.1, 'victim': 0.65,
            'bystander': 0.05},
        threshold_met=True
    )


@pytest.fixture
def sample_explanation_result():
    """Sample explanation result for testing."""
    return ExplanationResult(
        attributions=[0.1, 0.3, -0.2, 0.8, -0.1],
        tokens=['這', '個', '人', '很', '討厭'],
        explanation_text="Word '很' and '討厭' contribut"
            "e most to toxicity detection",
        top_contributing_words=[('很', 0.8), ('討厭', 0.3)],
        method='integrated_gradients'
    )


@pytest.fixture
def sample_model_prediction():
    """Sample model prediction for testing."""
    return ModelPrediction(
        model_name='baseline',
        predictions={
            'toxicity': torch.tensor([0.1, 0.8, 0.1]),
            'emotion': torch.tensor([0.1, 0.2, 0.7])
        },
        confidence_scores={'toxicity': 0.8, 'emotion': 0.7},
        processing_time=0.15
    )


@pytest.fixture
def sample_detection_result(
    sample_toxicity_result,
    sample_emotion_result,
    sample_bullying_result,
    sample_role_result,
    sample_explanation_result,
    sample_model_prediction
):
    """Complete sample detection result for testing."""
    return DetectionResult(
        text="這個人真的很討厭，我很生氣",
        timestamp=datetime.now(),
        processing_time=0.25,
        toxicity=sample_toxicity_result,
        emotion=sample_emotion_result,
        bullying=sample_bullying_result,
        role=sample_role_result,
        explanations={'toxicity': sample_explanation_result},
        model_predictions={'baseline': sample_model_prediction},
        ensemble_weights={'baseline': 0.4, 'contextual': 0.35,
            'weak_supervision': 0.25}
    )


class TestToxicityResult:
    """Test ToxicityResult class."""

    def test_toxicity_result_creation(self, sample_toxicity_result):
        """Test ToxicityResult can be created with valid data."""
        assert sample_toxicity_result.prediction == ToxicityLevel.TOXIC
        assert sample_toxicity_result.confidence == 0.85
        assert sample_toxicity_result.threshold_met is True

    def test_toxicity_result_to_dict(self, sample_toxicity_result):
        """Test ToxicityResult serialization to dict."""
        result_dict = sample_toxicity_result.to_dict()

        assert result_dict['prediction'] == 'toxic'
        assert result_dict['confidence'] == 0.85
        assert 'raw_scores' in result_dict
        assert result_dict['threshold_met'] is True

    def test_toxicity_result_from_dict(self):
        """Test ToxicityResult deserialization from dict."""
        data = {
            'prediction': 'severe',
            'confidence': 0.95,
            'raw_scores': {'none': 0.02, 'toxic': 0.03, 'severe': 0.95},
            'threshold_met': True
        }

        result = ToxicityResult.from_dict(data)
        assert result.prediction == ToxicityLevel.SEVERE
        assert result.confidence == 0.95


class TestEmotionResult:
    """Test EmotionResult class."""

    def test_emotion_result_creation(self, sample_emotion_result):
        """Test EmotionResult can be created with valid data."""
        assert sample_emotion_result.prediction == EmotionType.NEGATIVE
        assert sample_emotion_result.confidence == 0.78
        assert sample_emotion_result.strength == 3
        assert sample_emotion_result.threshold_met is True

    def test_emotion_strength_validation(self):
        """Test emotion strength validation."""
        with pytest.raises(ValueError):
            EmotionResult(
                prediction=EmotionType.POSITIVE,
                confidence=0.8,
                strength=5,  # Invalid: should be 0-4
                raw_scores={'pos': 0.8, 'neu': 0.1, 'neg': 0.1},
                threshold_met=True
            )


class TestDetectionResult:
    """Test DetectionResult class."""

    def test_detection_result_creation(self, sample_detection_result):
        """Test DetectionResult can be created with all components."""
        assert sample_detection_result.text == "這個人真的很討厭，我很生氣"
        assert sample_detection_result.toxicity.prediction == ToxicityLevel.TOXIC
        assert sample_detection_result.emotion.prediction == EmotionType.NEGATIVE
        assert sample_detection_result.bullying.prediction == BullyingType.HARASSMENT
        assert sample_detection_result.role.prediction == RoleType.VICTIM
        assert len(sample_detection_result.explanations) == 1
        assert len(sample_detection_result.model_predictions) == 1

    def test_detection_result_high_risk_assessment(
        self,
        sample_detection_result
    ):
        """Test high risk assessment logic."""
        # Should be high risk due to toxic + harassment + strong negative emotion
        assert sample_detection_result.is_high_risk() is True

    def test_detection_result_serialization(self, sample_detection_result):
        """Test complete result serialization."""
        result_dict = sample_detection_result.to_dict()

        assert result_dict['text'] == sample_detection_result.text
        assert 'toxicity' in result_dict
        assert 'emotion' in result_dict
        assert 'bullying' in result_dict
        assert 'role' in result_dict
        assert 'explanations' in result_dict
        assert 'model_predictions' in result_dict
        assert 'ensemble_weights' in result_dict

    def test_detection_result_json_round_trip(self, sample_detection_result):
        """Test JSON serialization and deserialization."""
        json_str = sample_detection_result.to_json()
        restored_result = DetectionResult.from_json(json_str)

        assert restored_result.text == sample_detection_result.text
        assert restored_result.toxicity.prediction == sample_detection_result.toxicity.prediction
        assert restored_result.emotion.strength == sample_detection_result.emotion.strength

    def test_detection_result_summary(self, sample_detection_result):
        """Test result summary generation."""
        summary = sample_detection_result.get_summary()

        assert 'text_preview' in summary
        assert 'toxicity' in summary
        assert 'emotion' in summary
        assert 'high_risk' in summary
        assert summary['high_risk'] is True


class TestResultAggregator:
    """Test ResultAggregator utility class."""

    def test_aggregate_batch_results(self, sample_detection_result):
        """Test batch result aggregation."""
        results = [sample_detection_result] * 5  # 5 identical results

        stats = ResultAggregator.aggregate_batch_results(results)

        assert stats['total_results'] == 5
        assert stats['high_risk_count'] == 5
        assert stats['high_risk_percentage'] == 100.0
        assert 'prediction_counts' in stats
        assert 'confidence_statistics' in stats
        assert 'processing_time_statistics' in stats

    def test_filter_results_by_confidence(self, sample_detection_result):
        """Test filtering results by confidence threshold."""
        results = [sample_detection_result] * 3

        # Filter by toxicity confidence >= 0.9 (should return 0 results)
        filtered = ResultAggregator.filter_results_by_confidence(
            results, 'toxicity', 0.9
        )
        assert len(filtered) == 0

        # Filter by toxicity confidence >= 0.8 (should return all 3)
        filtered = ResultAggregator.filter_results_by_confidence(
            results, 'toxicity', 0.8
        )
        assert len(filtered) == 3

    def test_get_top_risk_results(self, sample_detection_result):
        """Test getting top risk results."""
        # Create results with different risk levels
        low_risk_result = DetectionResult(
            text="正常文本",
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

        results = [sample_detection_result, low_risk_result]

        top_risks = ResultAggregator.get_top_risk_results(results, top_k=1)

        assert len(top_risks) == 1
        assert top_risks[0] == sample_detection_result  # High risk result should be first


class TestConfidenceThresholds:
    """Test ConfidenceThresholds utility class."""

    def test_get_threshold(self):
        """Test getting threshold for specific task and prediction."""
        threshold = ConfidenceThresholds.get_threshold('toxicity', 'severe')
        assert threshold == 0.8

        threshold = ConfidenceThresholds.get_threshold('emotion', 'pos')
        assert threshold == 0.6

    def test_update_thresholds(self):
        """Test updating default thresholds."""
        original_threshold = ConfidenceThresholds.get_threshold(
            'toxicity',
            'toxic'
        )

        new_thresholds = {
            'toxicity': {'toxic': 0.9}
        }
        ConfidenceThresholds.update_thresholds(new_thresholds)

        updated_threshold = ConfidenceThresholds.get_threshold(
            'toxicity',
            'toxic'
        )
        assert updated_threshold == 0.9

        # Restore original for other tests
        ConfidenceThresholds.DEFAULT_THRESHOLDS['toxicity']['toxic'] = original_threshold

    def test_validate_thresholds(self):
        """Test threshold validation."""
        valid_thresholds = {
            'toxicity': {'none': 0.5, 'toxic': 0.7, 'severe': 0.8}
        }
        assert ConfidenceThresholds.validate_thresholds(valid_thresholds) is True

        invalid_thresholds = {
            'toxicity': {'none': 0.5, 'toxic': 1.5}  # Invalid: > 1.0
        }
        assert ConfidenceThresholds.validate_thresholds(invalid_thresholds) is False


class TestCyberPuppyDetectorSimple:
    """Simplified tests for CyberPuppyDetector focusing on key behaviors."""

    @pytest.fixture
    def detector_config(self):
        """Simple detector configuration."""
        return {
            'model_paths': {
                'baseline': 'models/baseline.pt',
                'contextual': 'models/contextual.pt',
                'weak_supervision': 'models/weak_supervision.pt'
            },
            'ensemble_weights': {
                'baseline': 0.4,
                'contextual': 0.35,
                'weak_supervision': 0.25
            },
            'confidence_thresholds': ConfidenceThresholds.DEFAULT_THRESHOLDS,
            'preprocessing': {
                'max_length': 512,
                'normalize_unicode': True,
                'convert_traditional': True
            }
        }

    def test_detector_initialization_without_models(self, detector_config):
        """Test detector can be initialized without loading models."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config, load_models=False)

            assert detector.config == detector_config
            assert not detector._models_loaded
            assert not detector.is_ready()

    def test_detector_input_validation(self, detector_config):
        """Test detector validates inputs correctly."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config, load_models=False)

            # Test None input
            with pytest.raises(ValueError):
                detector._validate_input(None)

            # Test empty string
            with pytest.raises(ValueError):
                detector._validate_input("")

            # Test non-string input
            with pytest.raises(TypeError):
                detector._validate_input(123)

            # Test valid input
            detector._validate_input("這是有效的文本")  # Should not raise

    def test_detector_preprocessing(self, detector_config):
        """Test text preprocessing functionality."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config, load_models=False)

            # Test normal preprocessing
            text = "  這是測試文本  "
            processed = detector._preprocess_text(text)
            assert processed == "這是測試文本"

            # Test length truncation
            long_text = "長文本" * 200  # Should exceed max_length
            processed = detector._preprocess_text(long_text)
            assert len(processed) <= detector_config['preprocessing']['max_length']

    def test_detector_ensemble_weights_normalization(self, detector_config):
        """Test ensemble weights are normalized correctly."""
        # Test weights that don't sum to 1.0
        unnormalized_config = detector_config.copy()
        unnormalized_config['ensemble_weights'] = {
            'baseline': 0.5,
            'contextual': 0.3,
            'weak_supervision': 0.3  # Sum = 1.1
        }

        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(
                unnormalized_config,
                load_models=False
            )

            # Weights should be normalized to sum to 1.0
            total_weight = sum(detector.ensemble_weights.values())
            assert abs(total_weight - 1.0) < 1e-6

    def test_detector_confidence_calibration(self, detector_config):
        """Test confidence calibration logic."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config, load_models=False)

            # High confidence prediction
            high_conf_pred = torch.tensor([0.05, 0.9, 0.05])
            high_conf = detector._calibrate_confidence(
                high_conf_pred,
                'toxicity'
            )

            # Low confidence prediction
            low_conf_pred = torch.tensor([0.4, 0.35, 0.25])
            low_conf = detector._calibrate_confidence(
                low_conf_pred,
                'toxicity'
            )

            # High confidence should be greater than low confidence
            assert high_conf > low_conf
            assert 0.0 <= high_conf <= 1.0
            assert 0.0 <= low_conf <= 1.0

    def test_detector_performance_tracking(self, detector_config):
        """Test performance statistics tracking."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config, load_models=False)

            # Initially no stats
            stats = detector.get_performance_stats()
            assert stats['total_predictions'] == 0
            assert stats['average_processing_time'] == 0.0

            # Reset stats should work
            detector.reset_performance_stats()
            stats = detector.get_performance_stats()
            assert stats['total_predictions'] == 0

    def test_detector_context_manager(self, detector_config):
        """Test detector can be used as context manager."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            with CyberPuppyDetector(
                detector_config,
                load_models=False
            ) as detector:
                assert detector is not None

            # Should exit cleanly without errors

    def test_detector_model_info(self, detector_config):
        """Test model information retrieval."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config, load_models=False)

            info = detector.get_model_info()
            assert info['status'] == 'models_not_loaded'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
