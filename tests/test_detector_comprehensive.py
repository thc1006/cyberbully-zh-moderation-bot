"""
Comprehensive test suite for CyberPuppyDetector covering missing areas.

This test file specifically targets the uncovered code sections to boost
coverage from 3.77% to 70%+. Focuses on:
- Device setup logic
- Model loading error scenarios
- Individual model predictions
- Ensemble prediction logic
- Label conversion and task result creation
- Emotion strength calculation
- Explanation generation
- Full analysis pipeline
- Batch processing
- Configuration classes
- Chinese text edge cases
"""

import time
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from cyberpuppy.models.detector import (
    CyberPuppyDetector,
    EnsembleConfig,
    PreprocessingConfig,
)
from cyberpuppy.models.result import (
    BullyingResult,
    BullyingType,
    DetectionResult,
    EmotionResult,
    EmotionType,
    ExplanationResult,
    ModelPrediction,
    RoleResult,
    RoleType,
    ToxicityLevel,
    ToxicityResult,
)


@pytest.fixture
def basic_config():
    """Basic configuration for testing."""
    return {
        "base_model": "hfl/chinese-macbert-base",
        "model_paths": {
            "baseline": "models/baseline.pt",
            "contextual": "models/contextual.pt",
            "weak_supervision": "models/weak.pt",
        },
        "ensemble_weights": {"baseline": 0.4, "contextual": 0.35, "weak_supervision": 0.25},
        "confidence_thresholds": {
            "toxicity": {"none": 0.6, "toxic": 0.7, "severe": 0.8},
            "emotion": {"pos": 0.6, "neu": 0.5, "neg": 0.6},
            "bullying": {"none": 0.6, "harassment": 0.7, "threat": 0.8},
            "role": {"none": 0.5, "perpetrator": 0.7, "victim": 0.6, "bystander": 0.6},
        },
        "preprocessing": {
            "max_length": 512,
            "normalize_unicode": True,
            "convert_traditional": True,
        },
        "model_version": "1.0.0",
    }


@pytest.fixture
def mock_models():
    """Create mock models for testing."""
    baseline = MagicMock()
    baseline.predict.return_value = {
        "toxicity": torch.tensor([0.2, 0.7, 0.1]),
        "emotion": torch.tensor([0.1, 0.3, 0.6]),
        "bullying": torch.tensor([0.8, 0.15, 0.05]),
        "role": torch.tensor([0.7, 0.1, 0.15, 0.05]),
    }
    baseline.eval.return_value = None
    baseline.parameters.return_value = [torch.tensor([1.0])]

    contextual = MagicMock()
    contextual.predict.return_value = {
        "toxicity": torch.tensor([0.3, 0.6, 0.1]),
        "emotion": torch.tensor([0.15, 0.25, 0.6]),
        "bullying": torch.tensor([0.75, 0.2, 0.05]),
        "role": torch.tensor([0.65, 0.15, 0.15, 0.05]),
    }
    contextual.eval.return_value = None
    contextual.parameters.return_value = [torch.tensor([1.0])]

    weak = MagicMock()
    weak.predict.return_value = {
        "toxicity": torch.tensor([0.25, 0.65, 0.1]),
        "emotion": torch.tensor([0.12, 0.28, 0.6]),
        "bullying": torch.tensor([0.77, 0.18, 0.05]),
        "role": torch.tensor([0.68, 0.12, 0.15, 0.05]),
    }
    weak.eval.return_value = None
    weak.parameters.return_value = [torch.tensor([1.0])]

    return {"baseline": baseline, "contextual": contextual, "weak_supervision": weak}


@pytest.fixture
def mock_explainer():
    """Create mock explainer."""
    explainer = MagicMock()
    explainer.explain.return_value = {
        "attributions": [0.1, 0.5, -0.2, 0.8, 0.3],
        "tokens": ["這", "個", "人", "很", "討厭"],
        "explanation": "Key words contribute to classification",
    }
    return explainer


class TestDeviceSetup:
    """Test device selection logic."""

    def test_setup_device_auto_cuda_available(self, basic_config):
        """Test auto device selection when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch.object(CyberPuppyDetector, "_load_models"):
                detector = CyberPuppyDetector(basic_config, device="auto", load_models=False)
                assert detector.device == "cuda"

    def test_setup_device_auto_mps_available(self, basic_config):
        """Test auto device selection when MPS is available."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                with patch.object(CyberPuppyDetector, "_load_models"):
                    detector = CyberPuppyDetector(basic_config, device="auto", load_models=False)
                    assert detector.device == "mps"

    def test_setup_device_auto_cpu_fallback(self, basic_config):
        """Test auto device selection falls back to CPU."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(CyberPuppyDetector, "_load_models"):
                detector = CyberPuppyDetector(basic_config, device="auto", load_models=False)
                assert detector.device == "cpu"

    def test_setup_device_explicit(self, basic_config):
        """Test explicit device specification."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, device="cpu", load_models=False)
            assert detector.device == "cpu"

    def test_setup_device_none_defaults_to_auto(self, basic_config):
        """Test device=None triggers auto selection."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch.object(CyberPuppyDetector, "_load_models"):
                detector = CyberPuppyDetector(basic_config, device=None, load_models=False)
                assert detector.device == "cpu"


class TestWeightNormalization:
    """Test ensemble weight normalization."""

    def test_normalize_weights_empty_uses_defaults(self, basic_config):
        """Test empty weights dict uses default equal weights."""
        basic_config["ensemble_weights"] = {}
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            assert "baseline" in detector.ensemble_weights
            assert "contextual" in detector.ensemble_weights
            assert "weak_supervision" in detector.ensemble_weights
            assert abs(sum(detector.ensemble_weights.values()) - 1.0) < 1e-6

    def test_normalize_weights_sums_to_one(self, basic_config):
        """Test weights are normalized to sum to 1.0."""
        basic_config["ensemble_weights"] = {"baseline": 2.0, "contextual": 1.0, "weak_supervision": 1.0}
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            assert abs(sum(detector.ensemble_weights.values()) - 1.0) < 1e-6
            assert abs(detector.ensemble_weights["baseline"] - 0.5) < 1e-6


class TestPreprocessing:
    """Test text preprocessing functionality."""

    def test_preprocess_text_none_raises_error(self, basic_config):
        """Test None input raises ValueError."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            with pytest.raises(ValueError, match="Text cannot be None"):
                detector._preprocess_text(None)

    def test_preprocess_text_non_string_raises_error(self, basic_config):
        """Test non-string input raises TypeError."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            with pytest.raises(TypeError, match="Text must be string"):
                detector._preprocess_text(123)

    def test_preprocess_text_empty_raises_error(self, basic_config):
        """Test empty string raises ValueError."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            with pytest.raises(ValueError, match="Text cannot be empty"):
                detector._preprocess_text("")

    def test_preprocess_text_whitespace_only_raises_error(self, basic_config):
        """Test whitespace-only string raises ValueError."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            with pytest.raises(ValueError, match="Text cannot be empty"):
                detector._preprocess_text("   ")

    def test_preprocess_text_unicode_normalization(self, basic_config):
        """Test Unicode normalization is applied."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            text = "測試文本"
            processed = detector._preprocess_text(text)
            assert isinstance(processed, str)

    def test_preprocess_text_truncation(self, basic_config):
        """Test text truncation at max_length."""
        basic_config["preprocessing"]["max_length"] = 10
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            long_text = "這是一個很長的測試文本" * 10
            processed = detector._preprocess_text(long_text)
            assert len(processed) <= 10

    def test_preprocess_text_whitespace_cleanup(self, basic_config):
        """Test whitespace cleanup."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            text = "  多餘   空格   測試  "
            processed = detector._preprocess_text(text)
            assert "  " not in processed
            assert processed == processed.strip()


class TestIndividualPredictions:
    """Test individual model prediction logic."""

    def test_get_individual_predictions_success(self, basic_config, mock_models):
        """Test successful predictions from all models."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            detector.models = mock_models
            detector._models_loaded = True

            predictions = detector._get_individual_predictions("測試文本")

            assert "baseline" in predictions
            assert "contextual" in predictions
            assert "weak_supervision" in predictions

            for model_name, pred in predictions.items():
                assert "predictions" in pred
                assert "processing_time" in pred
                assert "toxicity" in pred["predictions"]
                assert "emotion" in pred["predictions"]

    def test_get_individual_predictions_with_context(self, basic_config, mock_models):
        """Test predictions with conversation context."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            detector.models = mock_models
            detector._models_loaded = True

            context = ["前面的對話", "之前說過的話"]
            predictions = detector._get_individual_predictions("測試文本", context=context)

            mock_models["contextual"].predict.assert_called_once()
            assert "contextual" in predictions

    def test_get_individual_predictions_baseline_error(self, basic_config, mock_models):
        """Test baseline model error is raised."""
        mock_models["baseline"].predict.side_effect = RuntimeError("Model error")

        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            detector.models = mock_models
            detector._models_loaded = True

            with pytest.raises(RuntimeError):
                detector._get_individual_predictions("測試文本")

    def test_get_individual_predictions_contextual_fallback(self, basic_config, mock_models):
        """Test contextual model falls back to baseline on error."""
        mock_models["contextual"].predict.side_effect = RuntimeError("Context error")

        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            detector.models = mock_models
            detector._models_loaded = True

            predictions = detector._get_individual_predictions("測試文本")

            assert "contextual" in predictions
            assert predictions["contextual"] == predictions["baseline"]


class TestEnsemblePrediction:
    """Test ensemble prediction logic."""

    def test_ensemble_predict_combines_models(self, basic_config):
        """Test ensemble combines predictions from multiple models."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            model_predictions = {
                "baseline": {
                    "predictions": {
                        "toxicity": torch.tensor([0.2, 0.7, 0.1]),
                        "emotion": torch.tensor([0.1, 0.3, 0.6]),
                        "bullying": torch.tensor([0.8, 0.15, 0.05]),
                        "role": torch.tensor([0.7, 0.1, 0.15, 0.05]),
                    }
                },
                "contextual": {
                    "predictions": {
                        "toxicity": torch.tensor([0.3, 0.6, 0.1]),
                        "emotion": torch.tensor([0.15, 0.25, 0.6]),
                        "bullying": torch.tensor([0.75, 0.2, 0.05]),
                        "role": torch.tensor([0.65, 0.15, 0.15, 0.05]),
                    }
                },
            }

            result = detector._ensemble_predict("測試", model_predictions)

            assert "toxicity" in result
            assert "emotion" in result
            assert "bullying" in result
            assert "role" in result

    def test_ensemble_predict_fallback_on_missing_predictions(self, basic_config):
        """Test ensemble falls back to baseline if predictions missing."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            model_predictions = {
                "baseline": {
                    "predictions": {
                        "toxicity": torch.tensor([0.2, 0.7, 0.1]),
                        "emotion": torch.tensor([0.1, 0.3, 0.6]),
                        "bullying": torch.tensor([0.8, 0.15, 0.05]),
                        "role": torch.tensor([0.7, 0.1, 0.15, 0.05]),
                    }
                }
            }

            result = detector._ensemble_predict("測試", model_predictions)

            assert "toxicity" in result
            assert isinstance(result["toxicity"], torch.Tensor)


class TestConfidenceCalibration:
    """Test confidence calibration logic."""

    def test_calibrate_confidence_high_confidence(self, basic_config):
        """Test calibration for high confidence predictions."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            high_conf_pred = torch.tensor([0.05, 0.9, 0.05])
            confidence = detector._calibrate_confidence(high_conf_pred, "toxicity")

            assert 0.0 <= confidence <= 1.0
            assert confidence > 0.5

    def test_calibrate_confidence_low_confidence(self, basic_config):
        """Test calibration for low confidence predictions."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            low_conf_pred = torch.tensor([0.4, 0.35, 0.25])
            confidence = detector._calibrate_confidence(low_conf_pred, "toxicity")

            assert 0.0 <= confidence <= 1.0
            assert confidence < 0.7

    def test_calibrate_confidence_entropy_adjustment(self, basic_config):
        """Test entropy is used for uncertainty adjustment."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            uniform_pred = torch.tensor([0.33, 0.33, 0.34])
            peaked_pred = torch.tensor([0.05, 0.9, 0.05])

            uniform_conf = detector._calibrate_confidence(uniform_pred, "emotion")
            peaked_conf = detector._calibrate_confidence(peaked_pred, "emotion")

            assert peaked_conf > uniform_conf


class TestLabelConversion:
    """Test prediction to label conversion."""

    def test_convert_prediction_to_label_toxicity(self, basic_config):
        """Test toxicity label conversion."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            prediction = torch.tensor([0.1, 0.8, 0.1])
            label, scores = detector._convert_prediction_to_label(prediction, "toxicity")

            assert label == "toxic"
            assert "none" in scores
            assert "toxic" in scores
            assert "severe" in scores
            assert scores["toxic"] > scores["none"]

    def test_convert_prediction_to_label_emotion(self, basic_config):
        """Test emotion label conversion."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            prediction = torch.tensor([0.1, 0.2, 0.7])
            label, scores = detector._convert_prediction_to_label(prediction, "emotion")

            assert label == "neg"
            assert "pos" in scores
            assert "neu" in scores
            assert "neg" in scores

    def test_convert_prediction_to_label_role(self, basic_config):
        """Test role label conversion."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            prediction = torch.tensor([0.1, 0.1, 0.7, 0.1])
            label, scores = detector._convert_prediction_to_label(prediction, "role")

            assert label == "victim"
            assert len(scores) == 4


class TestTaskResultCreation:
    """Test task-specific result creation."""

    def test_create_toxicity_result(self, basic_config):
        """Test creating ToxicityResult."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            prediction = torch.tensor([0.1, 0.8, 0.1])
            thresholds = {"toxic": 0.3}

            result = detector._create_task_result(prediction, "toxicity", thresholds)

            assert isinstance(result, ToxicityResult)
            assert result.prediction == ToxicityLevel.TOXIC
            assert result.threshold_met

    def test_create_emotion_result(self, basic_config):
        """Test creating EmotionResult with strength."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            prediction = torch.tensor([0.1, 0.2, 0.7])
            thresholds = {"neg": 0.3}

            result = detector._create_task_result(prediction, "emotion", thresholds)

            assert isinstance(result, EmotionResult)
            assert result.prediction == EmotionType.NEGATIVE
            assert 0 <= result.strength <= 4

    def test_create_bullying_result(self, basic_config):
        """Test creating BullyingResult."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            prediction = torch.tensor([0.2, 0.7, 0.1])
            thresholds = {"harassment": 0.3}

            result = detector._create_task_result(prediction, "bullying", thresholds)

            assert isinstance(result, BullyingResult)
            assert result.prediction == BullyingType.HARASSMENT

    def test_create_role_result(self, basic_config):
        """Test creating RoleResult."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            prediction = torch.tensor([0.2, 0.6, 0.15, 0.05])
            thresholds = {"perpetrator": 0.3}

            result = detector._create_task_result(prediction, "role", thresholds)

            assert isinstance(result, RoleResult)
            assert result.prediction == RoleType.PERPETRATOR


class TestEmotionStrength:
    """Test emotion strength calculation."""

    def test_calculate_emotion_strength_neutral(self, basic_config):
        """Test neutral emotion has strength 0."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            strength = detector._calculate_emotion_strength(torch.tensor([0.2, 0.7, 0.1]), "neu", 0.8)

            assert strength == 0

    def test_calculate_emotion_strength_weak(self, basic_config):
        """Test weak emotion strength."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            strength = detector._calculate_emotion_strength(torch.tensor([0.6, 0.3, 0.1]), "pos", 0.4)

            assert 1 <= strength <= 2

    def test_calculate_emotion_strength_strong(self, basic_config):
        """Test strong emotion strength."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            strength = detector._calculate_emotion_strength(torch.tensor([0.9, 0.05, 0.05]), "pos", 0.9)

            assert 3 <= strength <= 4


class TestExplanationGeneration:
    """Test explanation generation."""

    def test_generate_explanations_success(self, basic_config, mock_explainer):
        """Test successful explanation generation."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            detector.explainer = mock_explainer

            predictions = {"toxicity": torch.tensor([0.1, 0.8, 0.1])}
            explanations = detector._generate_explanations("測試文本", predictions)

            assert "toxicity" in explanations
            assert isinstance(explanations["toxicity"], ExplanationResult)

    def test_generate_explanations_no_explainer(self, basic_config):
        """Test explanation generation without explainer."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            detector.explainer = None

            predictions = {"toxicity": torch.tensor([0.1, 0.8, 0.1])}
            explanations = detector._generate_explanations("測試文本", predictions)

            assert len(explanations) == 0


class TestFullAnalysisPipeline:
    """Test complete analysis pipeline."""

    def test_analyze_raises_error_if_models_not_loaded(self, basic_config):
        """Test analyze raises RuntimeError if models not loaded."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            with pytest.raises(RuntimeError, match="Models not loaded"):
                detector.analyze("測試文本")

    def test_analyze_validates_input(self, basic_config, mock_models, mock_explainer):
        """Test analyze validates input before processing."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            detector.models = mock_models
            detector.explainer = mock_explainer
            detector._models_loaded = True

            with pytest.raises(ValueError):
                detector.analyze(None)

            with pytest.raises(TypeError):
                detector.analyze(123)

            with pytest.raises(ValueError):
                detector.analyze("")


class TestBatchAnalysis:
    """Test batch processing."""

    def test_analyze_batch_empty_list_raises_error(self, basic_config, mock_models):
        """Test empty batch raises ValueError."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            detector.models = mock_models
            detector._models_loaded = True

            with pytest.raises(ValueError, match="Texts list cannot be empty"):
                detector.analyze_batch([])

    def test_analyze_batch_context_length_mismatch(self, basic_config, mock_models):
        """Test context length mismatch raises ValueError."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)
            detector.models = mock_models
            detector._models_loaded = True

            texts = ["文本1", "文本2"]
            context = [["上下文1"]]

            with pytest.raises(ValueError, match="Context list must have same length"):
                detector.analyze_batch(texts, context=context)


class TestPerformanceTracking:
    """Test performance statistics tracking."""

    def test_get_performance_stats_initial(self, basic_config):
        """Test initial performance stats."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            stats = detector.get_performance_stats()

            assert stats["total_predictions"] == 0
            assert stats["average_processing_time"] == 0.0

    def test_reset_performance_stats(self, basic_config):
        """Test resetting performance stats."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            detector._prediction_count = 10
            detector._total_processing_time = 5.0

            detector.reset_performance_stats()

            stats = detector.get_performance_stats()
            assert stats["total_predictions"] == 0


class TestEnsembleConfig:
    """Test EnsembleConfig dataclass."""

    def test_ensemble_config_defaults(self):
        """Test default EnsembleConfig values."""
        config = EnsembleConfig()

        assert config.models == ["baseline", "contextual", "weak_supervision"]
        assert len(config.weights) == 3
        assert abs(sum(config.weights) - 1.0) < 1e-6

    def test_ensemble_config_normalization(self):
        """Test weight normalization in EnsembleConfig."""
        config = EnsembleConfig(models=["m1", "m2"], weights=[2.0, 1.0])

        assert abs(config.weights[0] - 2.0 / 3.0) < 1e-6
        assert abs(config.weights[1] - 1.0 / 3.0) < 1e-6

    def test_ensemble_config_validation_error(self):
        """Test EnsembleConfig raises error for mismatched lengths."""
        with pytest.raises(ValueError, match="Number of models.*must match"):
            EnsembleConfig(models=["m1", "m2"], weights=[1.0])

    def test_ensemble_config_to_dict(self):
        """Test EnsembleConfig to_dict method."""
        config = EnsembleConfig(models=["m1", "m2"], weights=[0.6, 0.4])
        d = config.to_dict()

        assert "models" in d
        assert "weights" in d
        assert "voting_strategy" in d

    def test_ensemble_config_from_dict(self):
        """Test EnsembleConfig from_dict method."""
        data = {"models": ["m1", "m2"], "weights": [0.6, 0.4], "voting_strategy": "weighted"}

        config = EnsembleConfig.from_dict(data)

        assert config.models == ["m1", "m2"]
        assert len(config.weights) == 2

    def test_ensemble_config_is_valid(self):
        """Test EnsembleConfig is_valid method."""
        valid_config = EnsembleConfig(models=["m1", "m2"], weights=[0.5, 0.5])
        assert valid_config.is_valid()

        invalid_config = EnsembleConfig(models=[], weights=[])
        assert not invalid_config.is_valid()


class TestPreprocessingConfig:
    """Test PreprocessingConfig dataclass."""

    def test_preprocessing_config_defaults(self):
        """Test default PreprocessingConfig values."""
        config = PreprocessingConfig()

        assert config.normalize_unicode is True
        assert config.max_length == 512
        assert config.convert_traditional_to_simplified is True

    def test_preprocessing_config_to_dict(self):
        """Test PreprocessingConfig to_dict method."""
        config = PreprocessingConfig(max_length=256, lowercase=True)
        d = config.to_dict()

        assert d["max_length"] == 256
        assert d["lowercase"] is True

    def test_preprocessing_config_from_dict(self):
        """Test PreprocessingConfig from_dict method."""
        data = {"max_length": 128, "remove_urls": True, "lowercase": False}

        config = PreprocessingConfig.from_dict(data)

        assert config.max_length == 128
        assert config.remove_urls is True

    def test_preprocessing_config_is_valid(self):
        """Test PreprocessingConfig is_valid method."""
        valid_config = PreprocessingConfig(max_length=512)
        assert valid_config.is_valid()

        invalid_config = PreprocessingConfig(max_length=-1)
        assert not invalid_config.is_valid()


class TestModelInfo:
    """Test model information retrieval."""

    def test_is_ready_false_when_not_loaded(self, basic_config):
        """Test is_ready returns False when models not loaded."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            assert not detector.is_ready()

    def test_get_model_info_not_loaded(self, basic_config):
        """Test get_model_info when models not loaded."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            info = detector.get_model_info()

            assert info["status"] == "models_not_loaded"


class TestContextManager:
    """Test context manager functionality."""

    def test_context_manager_enter_exit(self, basic_config):
        """Test detector can be used as context manager."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            with CyberPuppyDetector(basic_config, load_models=False) as detector:
                assert detector is not None

    def test_context_manager_cuda_cleanup(self, basic_config):
        """Test CUDA cache is cleared on exit."""
        with patch("torch.cuda.empty_cache") as mock_empty:
            with patch.object(CyberPuppyDetector, "_load_models"):
                detector = CyberPuppyDetector(basic_config, device="cuda", load_models=False)
                with detector:
                    pass

                mock_empty.assert_called_once()


class TestUpdateWeights:
    """Test ensemble weight updates."""

    def test_update_ensemble_weights(self, basic_config):
        """Test updating ensemble weights."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            new_weights = {"baseline": 0.5, "contextual": 0.3, "weak_supervision": 0.2}
            detector.update_ensemble_weights(new_weights)

            assert detector.ensemble_weights == new_weights

    def test_update_ensemble_weights_normalization(self, basic_config):
        """Test weight normalization during update."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            new_weights = {"baseline": 2.0, "contextual": 1.0, "weak_supervision": 1.0}
            detector.update_ensemble_weights(new_weights)

            assert abs(sum(detector.ensemble_weights.values()) - 1.0) < 1e-6


class TestChineseTextEdgeCases:
    """Test edge cases specific to Chinese text."""

    def test_preprocess_chinese_punctuation(self, basic_config):
        """Test Chinese punctuation handling."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            text = "你好！這是測試。真的嗎？"
            processed = detector._preprocess_text(text)

            assert isinstance(processed, str)

    def test_preprocess_mixed_chinese_english(self, basic_config):
        """Test mixed Chinese-English text."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            text = "這是 test 測試 with mixed 語言"
            processed = detector._preprocess_text(text)

            assert isinstance(processed, str)

    def test_preprocess_emoji_handling(self, basic_config):
        """Test emoji in Chinese text."""
        with patch.object(CyberPuppyDetector, "_load_models"):
            detector = CyberPuppyDetector(basic_config, load_models=False)

            text = "測試文本 😊 包含表情符號"
            processed = detector._preprocess_text(text)

            assert isinstance(processed, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.cyberpuppy.models.detector", "--cov-report=term-missing"])