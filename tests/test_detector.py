"""
Test module for CyberPuppyDetector and DetectionResult classes.

This module follows London School TDD principles with comprehensive
behavior verification using mocks and outside-in development.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import torch

# Mock the imports since we're testing before implementation


@pytest.fixture
def mock_detection_result():
    """Mock DetectionResult for testing."""

    class MockDetectionResult:
        def __init__(self, text: str):
            self.text = text
            self.toxicity_prediction = "none"
            self.toxicity_confidence = 0.85
            self.emotion_prediction = "neu"
            self.emotion_confidence = 0.78
            self.emotion_strength = 2
            self.bullying_prediction = "none"
            self.bullying_confidence = 0.92
            self.role_prediction = "none"
            self.role_confidence = 0.67
            self.explanations = {}
            self.attribution_scores = {}
            self.model_predictions = {}
            self.ensemble_weights = {}
            self.processing_time = 0.15

        def to_dict(self):
            return {
                "text": self.text,
                "toxicity": {
                    "prediction": self.toxicity_prediction,
                    "confidence": self.toxicity_confidence
                },
                "emotion": {
                    "prediction": self.emotion_prediction,
                    "confidence": self.emotion_confidence,
                    "strength": self.emotion_strength
                },
                "bullying": {
                    "prediction": self.bullying_prediction,
                    "confidence": self.bullying_confidence
                },
                "role": {
                    "prediction": self.role_prediction,
                    "confidence": self.role_confidence
                },
                "explanations": self.explanations,
                "attribution_scores": self.attribution_scores,
                "processing_time": self.processing_time
            }

    return MockDetectionResult


@pytest.fixture
def mock_models():
    """Mock models for dependency injection."""
    baseline_model = Mock()
    contextual_model = Mock()
    weak_supervision_model = Mock()
    explainer = Mock()

    # Configure baseline model mock
    baseline_model.predict.return_value = {
        'toxicity': torch.tensor([0.1, 0.8, 0.1]),  # toxic
        'emotion': torch.tensor([0.2, 0.7, 0.1]),   # neutral
        'bullying': torch.tensor([0.9, 0.1, 0.0]),  # none
        'role': torch.tensor([0.8, 0.1, 0.1])       # none
    }

    # Configure contextual model mock
    contextual_model.predict.return_value = {
        'toxicity': torch.tensor([0.2, 0.7, 0.1]),
        'emotion': torch.tensor([0.3, 0.6, 0.1]),
        'bullying': torch.tensor([0.8, 0.2, 0.0]),
        'role': torch.tensor([0.7, 0.2, 0.1])
    }

    # Configure weak supervision model mock
    weak_supervision_model.predict.return_value = {
        'toxicity': torch.tensor([0.15, 0.75, 0.1]),
        'emotion': torch.tensor([0.25, 0.65, 0.1]),
        'bullying': torch.tensor([0.85, 0.15, 0.0]),
        'role': torch.tensor([0.75, 0.15, 0.1])
    }

    # Configure explainer mock
    explainer.explain.return_value = {
        'attributions': [0.1, 0.3, -0.2, 0.4, -0.1],
        'tokens': ['這', '是', '很', '糟糕', '的'],
        'explanation': "Word '糟糕' contributes most to toxicity detection"
    }

    return {
        'baseline': baseline_model,
        'contextual': contextual_model,
        'weak_supervision': weak_supervision_model,
        'explainer': explainer
    }


@pytest.fixture
def detector_config():
    """Mock detector configuration."""
    return {
        'model_paths': {
            'baseline': 'models/baseline_model.pt',
            'contextual': 'models/contextual_model.pt',
            'weak_supervision': 'models/weak_supervision_model.pt'
        },
        'ensemble_weights': {
            'baseline': 0.4,
            'contextual': 0.35,
            'weak_supervision': 0.25
        },
        'confidence_thresholds': {
            'toxicity': {'none': 0.6, 'toxic': 0.7, 'severe': 0.8},
            'emotion': {'pos': 0.6, 'neu': 0.5, 'neg': 0.6},
            'bullying': {'none': 0.6, 'harassment': 0.7, 'threat': 0.8},
            'role': {'none': 0.5, 'perpetrator': 0.7, 'victim': 0.6,
                'bystander': 0.6}
        },
        'preprocessing': {
            'max_length': 512,
            'normalize_unicode': True,
            'convert_traditional': True
        },
        'explanation': {
            'method': 'integrated_gradients',
            'n_steps': 50,
            'internal_batch_size': 8
        }
    }


class TestDetectionResult:
    """Test cases for DetectionResult class."""

    def test_detection_result_initialization(self, mock_detection_result):
        """Test DetectionResult can be initialized with required fields."""
        result = mock_detection_result("測試文本")

        assert result.text == "測試文本"
        assert result.toxicity_prediction == "none"
        assert result.toxicity_confidence == 0.85
        assert result.emotion_prediction == "neu"
        assert result.emotion_confidence == 0.78
        assert result.emotion_strength == 2
        assert result.bullying_prediction == "none"
        assert result.bullying_confidence == 0.92
        assert result.role_prediction == "none"
        assert result.role_confidence == 0.67
        assert isinstance(result.explanations, dict)
        assert isinstance(result.attribution_scores, dict)
        assert isinstance(result.model_predictions, dict)
        assert isinstance(result.ensemble_weights, dict)
        assert isinstance(result.processing_time, float)

    def test_detection_result_to_dict_serialization(
        self,
        mock_detection_result
    ):
        """Test DetectionResult can be serialized to dictionary."""
        result = mock_detection_result("測試文本")
        result_dict = result.to_dict()

        assert result_dict["text"] == "測試文本"
        assert "toxicity" in result_dict
        assert "emotion" in result_dict
        assert "bullying" in result_dict
        assert "role" in result_dict
        assert "explanations" in result_dict
        assert "attribution_scores" in result_dict
        assert "processing_time" in result_dict

        # Test nested structure
        assert result_dict["toxicity"]["prediction"] == "none"
        assert result_dict["toxicity"]["confidence"] == 0.85
        assert result_dict["emotion"]["strength"] == 2

    def test_detection_result_confidence_validation(
        self,
        mock_detection_result
    ):
        """Test detection result confidence validation."""
        result = mock_detection_result("測試文本")

        # Valid confidence scores
        assert 0.0 <= result.toxicity_confidence <= 1.0
        assert 0.0 <= result.emotion_confidence <= 1.0
        assert 0.0 <= result.bullying_confidence <= 1.0
        assert 0.0 <= result.role_confidence <= 1.0

    def test_detection_result_emotion_strength_validation(
        self,
        mock_detection_result
    ):
        """Test detection result emotion strength validation."""
        result = mock_detection_result("測試文本")

        # Valid emotion strength (0-4)
        assert 0 <= result.emotion_strength <= 4


class TestCyberPuppyDetector:
    """Test cases for CyberPuppyDetector class."""

    def test_detector_initialization(self, detector_config):
        """Test CyberPuppyDetector initializes with proper configuration."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models') as mock_load:
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config, load_models=False)

            # Verify configuration is stored
            assert detector.config == detector_config
            assert detector.ensemble_weights == detector_config['ensemble_weights']
            assert not detector._models_loaded

            # Verify _load_models was not called
            mock_load.assert_not_called()

    def test_detector_model_loading_error_handling(self, detector_config):
        """Test CyberPuppyDetector handles model loading errors gracefully."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models') as mock_load:
            mock_load.side_effect = FileNotFoundError("Model file not found")

            from src.cyberpuppy.models.detector import CyberPuppyDetector

            with pytest.raises(RuntimeError, match="Model loading failed"):
                CyberPuppyDetector(detector_config)

    def test_text_preprocessing_pipeline(self, detector_config):
        """Test text preprocessing handles various input formats."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config, load_models=False)

            # Test cases for preprocessing
            test_cases = [
                "正常文本",
                "包含表情符號的文本 😊",
                "繁体字文本",
                "Mixed English and 中文",
                "   有空格的文本   ",
                "重複重複的文字",
            ]

            for text in test_cases:
                preprocessed = detector._preprocess_text(text)
                assert isinstance(preprocessed, str)
                assert len(preprocessed) <= detector_config['preprocessing']['max_length']

            # Test empty text raises error
            with pytest.raises(ValueError):
                detector._preprocess_text("")

    def test_ensemble_prediction_logic(self, detector_config):
        """Test ensemble prediction combines model outputs correctly."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config, load_models=False)

            text = "測試文本"

            # Mock model predictions
            model_predictions = {
                'baseline': {
                    'predictions': {
                        'toxicity': torch.tensor([0.1, 0.8, 0.1]),
                        'emotion': torch.tensor([0.2, 0.7, 0.1]),
                        'bullying': torch.tensor([0.9, 0.1, 0.0]),
                        'role': torch.tensor([0.8, 0.1, 0.1, 0.0])
                    },
                    'processing_time': 0.1
                },
                'contextual': {
                    'predictions': {
                        'toxicity': torch.tensor([0.2, 0.7, 0.1]),
                        'emotion': torch.tensor([0.3, 0.6, 0.1]),
                        'bullying': torch.tensor([0.8, 0.2, 0.0]),
                        'role': torch.tensor([0.7, 0.2, 0.1, 0.0])
                    },
                    'processing_time': 0.15
                }
            }

            result = detector._ensemble_predict(text, model_predictions)

            # Verify ensemble logic is applied
            assert 'toxicity' in result
            assert 'emotion' in result
            assert 'bullying' in result
            assert 'role' in result

            # Verify predictions are torch tensors
            for task_result in result.values():
                assert isinstance(task_result, torch.Tensor)

    def test_confidence_calibration(self, mock_models, detector_config):
        """Test confidence scores are properly calibrated."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config)
            detector.models = mock_models

            # Test confidence calibration for different prediction strengths
            high_confidence_logits = torch.tensor([0.1, 0.9, 0.0])
            low_confidence_logits = torch.tensor([0.4, 0.3, 0.3])

            high_conf = detector._calibrate_confidence(
                high_confidence_logits,
                'toxicity'
            )
            low_conf = detector._calibrate_confidence(
                low_confidence_logits,
                'toxicity'
            )

            assert high_conf > low_conf
            assert 0.0 <= high_conf <= 1.0
            assert 0.0 <= low_conf <= 1.0

    def test_explanation_generation(self, mock_models, detector_config):
        """Test explanation generation with attribution scores."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config)
            detector.models = mock_models

            text = "這是很糟糕的評論"
            predictions = {
                'toxicity': torch.tensor([0.1, 0.8, 0.1]),
                'emotion': torch.tensor([0.2, 0.1, 0.7])
            }

            explanations = detector._generate_explanations(text, predictions)

            # Verify explainer was called
            mock_models['explainer'].explain.assert_called()

            # Verify explanation structure
            assert isinstance(explanations, dict)
            assert 'attributions' in explanations
            assert 'tokens' in explanations
            assert 'explanation' in explanations

    def test_full_analysis_pipeline(
        self,
        mock_models,
        detector_config,
        mock_detection_result
    ):
        """Test complete text analysis pipeline integration."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            with patch(
                'src.cyberpuppy.models.detector.DetectionResult',
                mock_detection_result
            ):
                from src.cyberpuppy.models.detector import CyberPuppyDetector

                detector = CyberPuppyDetector(detector_config)
                detector.models = mock_models

                text = "這個人真的很討厭"
                result = detector.analyze(text)

                # Verify all models were called
                mock_models['baseline'].predict.assert_called_once()
                mock_models['contextual'].predict.assert_called_once()
                mock_models['weak_supervision'].predict.assert_called_once()
                mock_models['explainer'].explain.assert_called_once()

                # Verify result structure
                assert result.text == text
                assert hasattr(result, 'toxicity_prediction')
                assert hasattr(result, 'emotion_prediction')
                assert hasattr(result, 'bullying_prediction')
                assert hasattr(result, 'role_prediction')
                assert hasattr(result, 'explanations')

    def test_batch_analysis(
        self,
        mock_models,
        detector_config,
        mock_detection_result
    ):
        """Test batch processing of multiple texts."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            with patch(
                'src.cyberpuppy.models.detector.DetectionResult',
                mock_detection_result
            ):
                from src.cyberpuppy.models.detector import CyberPuppyDetector

                detector = CyberPuppyDetector(detector_config)
                detector.models = mock_models

                texts = [
                    "第一個文本",
                    "第二個文本",
                    "第三個文本"
                ]

                results = detector.analyze_batch(texts)

                assert len(results) == 3
                for i, result in enumerate(results):
                    assert result.text == texts[i]

    def test_invalid_input_handling(self, mock_models, detector_config):
        """Test error handling for invalid inputs."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config)
            detector.models = mock_models

            # Test None input
            with pytest.raises(ValueError):
                detector.analyze(None)

            # Test empty string
            with pytest.raises(ValueError):
                detector.analyze("")

            # Test non-string input
            with pytest.raises(TypeError):
                detector.analyze(123)

    def test_timeout_handling(self, mock_models, detector_config):
        """Test timeout handling for long-running predictions."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector
            import time

            detector = CyberPuppyDetector(detector_config)
            detector.models = mock_models

            # Mock slow model prediction
            def slow_predict(*args, **kwargs):
                time.sleep(2)  # Simulate slow prediction
                return mock_models['baseline'].predict.return_value

            mock_models['baseline'].predict.side_effect = slow_predict

            with pytest.raises(TimeoutError):
                detector.analyze("測試文本", timeout=1.0)

    def test_performance_requirements(self, mock_models, detector_config):
        """Test performance requirements are met."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector
            import time

            detector = CyberPuppyDetector(detector_config)
            detector.models = mock_models

            text = "測試文本" * 50  # Longer text

            start_time = time.time()
            result = detector.analyze(text)
            end_time = time.time()

            processing_time = end_time - start_time

            # Should complete within reasonable time (< 5 seconds for mock)
            assert processing_time < 5.0
            assert result.processing_time > 0

    def test_context_handling(self, mock_models, detector_config):
        """Test contextual information handling."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config)
            detector.models = mock_models

            text = "你好嗎？"
            context = [
                "昨天的對話內容",
                "今天的心情不好"
            ]

            result = detector.analyze(text, context=context)

            # Verify contextual model was called with context
            mock_models['contextual'].predict.assert_called_once()

            # Context should influence predictions
            assert hasattr(result, 'model_predictions')

    def test_memory_usage_optimization(self, mock_models, detector_config):
        """Test memory usage is optimized for large inputs."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            detector = CyberPuppyDetector(detector_config)
            detector.models = mock_models

            # Process multiple batches to test memory cleanup
            for batch_idx in range(5):
                texts = [f"批次 {batch_idx} 文本 {i}" for i in range(10)]
                results = detector.analyze_batch(texts)

                # Verify batch processing completed
                assert len(results) == 10

                # Memory should be managed between batches
                # This is more of a conceptual test - actual memory monitoring
                # would require memory profiling tools

    def test_model_ensemble_weights_configuration(
        self,
        mock_models,
        detector_config
    ):
        """Test ensemble weights can be configured and applied correctly."""
        with patch('src.cyberpuppy.models.detector.CyberPuppyDetector._load_models'):
            from src.cyberpuppy.models.detector import CyberPuppyDetector

            # Custom weights configuration
            custom_config = detector_config.copy()
            custom_config['ensemble_weights'] = {
                'baseline': 0.5,
                'contextual': 0.3,
                'weak_supervision': 0.2
            }

            detector = CyberPuppyDetector(custom_config)
            detector.models = mock_models

            assert detector.ensemble_weights == custom_config['ensemble_weights']

            # Test weight validation
            invalid_config = detector_config.copy()
            invalid_config['ensemble_weights'] = {
                'baseline': 0.5,
                'contextual': 0.3,
                'weak_supervision': 0.3  # Sums to 1.1, should be normalized
            }

            detector_invalid = CyberPuppyDetector(invalid_config)
            # Should normalize weights to sum to 1.0
            total_weight = sum(detector_invalid.ensemble_weights.values())
            assert abs(total_weight - 1.0) < 1e-6


class TestDetectorIntegration:
    """Integration tests for detector with actual model components."""

    @pytest.mark.integration
    def test_real_model_integration(self, detector_config):
        """Test integration with real model components (when available)."""
        # This test would run only when actual models are available
        # It's marked as integration test to be run separately
        pytest.skip("Integration test - requires actual model files")

    @pytest.mark.performance
    def test_performance_benchmarks(self, detector_config):
        """Test performance benchmarks meet requirements."""
        # This test would measure actual performance metrics
        pytest.skip("Performance test - requires benchmarking setup")

    @pytest.mark.gpu
    def test_gpu_acceleration(self, detector_config):
        """Test GPU acceleration when available."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        # Test GPU utilization
        pytest.skip("GPU test - requires CUDA setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
