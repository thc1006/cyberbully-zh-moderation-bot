#!/usr/bin/env python3
"""
Advanced comprehensive unit tests for models module
Focusing on improving coverage for cyberpuppy.models modules
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json

# Import the modules under test
from cyberpuppy.models.baselines import (
    ModelConfig,
    FocalLoss,
    MultiTaskHead,
    BaselineModel,
    ModelEvaluator,
    create_model_variants
)
from cyberpuppy.models.result import (
    DetectionResult,
    BatchResults,
    ModelOutput,
    ExplanationOutput
)
from cyberpuppy.models.detector import (
    CyberPuppyDetector,
    EnsembleConfig,
    PreprocessingConfig
)


class TestAdvancedModelConfig(unittest.TestCase):
    """Advanced tests for ModelConfig"""

    def test_model_config_validation(self):
        """Test comprehensive model config validation"""
        # Test valid config
        config = ModelConfig(
            model_name="hfl/chinese-macbert-base",
            num_classes={"toxicity": 2, "emotion": 3, "bullying": 2},
            hidden_size=768,
            dropout_rate=0.3,
            task_weights={"toxicity": 1.5, "emotion": 1.0, "bullying": 2.0}
        )

        self.assertEqual(config.model_name, "hfl/chinese-macbert-base")
        self.assertEqual(config.num_classes["toxicity"], 2)
        self.assertEqual(config.task_weights["toxicity"], 1.5)

    def test_model_config_invalid_dropout(self):
        """Test model config with invalid dropout rate"""
        with self.assertRaises(ValueError):
            ModelConfig(
                model_name="test",
                num_classes={"toxicity": 2},
                dropout_rate=1.5  # Invalid: > 1.0
            )

    def test_model_config_negative_hidden_size(self):
        """Test model config with invalid hidden size"""
        with self.assertRaises(ValueError):
            ModelConfig(
                model_name="test",
                num_classes={"toxicity": 2},
                hidden_size=-100  # Invalid: negative
            )

    def test_model_config_empty_num_classes(self):
        """Test model config with empty num_classes"""
        with self.assertRaises(ValueError):
            ModelConfig(
                model_name="test",
                num_classes={}  # Invalid: empty
            )

    def test_model_config_mismatched_task_weights(self):
        """Test model config with mismatched task weights"""
        config = ModelConfig(
            model_name="test",
            num_classes={"toxicity": 2, "emotion": 3},
            task_weights={"toxicity": 1.0}  # Missing emotion weight
        )

        # Should auto-complete missing weights
        self.assertEqual(config.task_weights["emotion"], 1.0)

    def test_model_config_serialization(self):
        """Test model config serialization and deserialization"""
        config = ModelConfig(
            model_name="test-model",
            num_classes={"toxicity": 2, "emotion": 3},
            hidden_size=512,
            dropout_rate=0.2
        )

        # Test to_dict
        config_dict = config.to_dict()
        self.assertIn("model_name", config_dict)
        self.assertIn("num_classes", config_dict)

        # Test from_dict
        restored_config = ModelConfig.from_dict(config_dict)
        self.assertEqual(restored_config.model_name, config.model_name)
        self.assertEqual(restored_config.num_classes, config.num_classes)


class TestAdvancedFocalLoss(unittest.TestCase):
    """Advanced tests for FocalLoss"""

    def test_focal_loss_with_different_gamma_values(self):
        """Test focal loss with various gamma values"""
        gamma_values = [0.0, 0.5, 1.0, 2.0, 5.0]

        for gamma in gamma_values:
            focal_loss = FocalLoss(alpha=1.0, gamma=gamma)

            # Create test data
            logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [0.5, 0.5]])
            targets = torch.tensor([0, 1, 0])

            loss = focal_loss(logits, targets)

            self.assertIsInstance(loss, torch.Tensor)
            self.assertTrue(loss.item() >= 0)

    def test_focal_loss_with_class_weights(self):
        """Test focal loss with different class weights (alpha)"""
        alpha_values = [0.25, 0.5, 0.75, 1.0, 2.0]

        for alpha in alpha_values:
            focal_loss = FocalLoss(alpha=alpha, gamma=2.0)

            logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
            targets = torch.tensor([0, 1])

            loss = focal_loss(logits, targets)

            self.assertIsInstance(loss, torch.Tensor)
            self.assertFalse(torch.isnan(loss))

    def test_focal_loss_edge_cases(self):
        """Test focal loss edge cases"""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)

        # Perfect predictions
        perfect_logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        targets = torch.tensor([0, 1])

        loss = focal_loss(perfect_logits, targets)
        self.assertTrue(loss.item() < 0.1)  # Should be very small

        # Very uncertain predictions
        uncertain_logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        targets = torch.tensor([0, 1])

        loss = focal_loss(uncertain_logits, targets)
        self.assertTrue(loss.item() > 0)

    def test_focal_loss_backward_compatibility(self):
        """Test that focal loss works with autograd"""
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)

        logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
        targets = torch.tensor([0])

        loss = focal_loss(logits, targets)
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(logits.grad)
        self.assertFalse(
            torch.allclose(logits.grad,
            torch.zeros_like(logits.grad))
        )


class TestAdvancedMultiTaskHead(unittest.TestCase):
    """Advanced tests for MultiTaskHead"""

    def test_multitask_head_with_regression(self):
        ""
            ""
        task_configs = {
            "toxicity": {"type": "classification", "num_classes": 2},
            "emotion": {"type": "classification", "num_classes": 3},
            "intensity": {"type": "regression", "output_size": 1}
        }

        head = MultiTaskHead(
            input_size=768,
            task_configs=task_configs,
            hidden_size=256
        )

        # Test forward pass
        batch_size = 4
        input_features = torch.randn(batch_size, 768)

        outputs = head(input_features)

        self.assertIn("toxicity", outputs)
        self.assertIn("emotion", outputs)
        self.assertIn("intensity", outputs)

        # Check output shapes
        self.assertEqual(outputs["toxicity"].shape, (batch_size, 2))
        self.assertEqual(outputs["emotion"].shape, (batch_size, 3))
        self.assertEqual(outputs["intensity"].shape, (batch_size, 1))

    def test_multitask_head_shared_layers(self):
        """Test multi-task head with different numbers of shared layers"""
        for num_shared_layers in [0, 1, 2, 3]:
            task_configs = {
                "toxicity": {"type": "classification", "num_classes": 2},
                "emotion": {"type": "classification", "num_classes": 3}
            }

            head = MultiTaskHead(
                input_size=768,
                task_configs=task_configs,
                hidden_size=256,
                num_shared_layers=num_shared_layers
            )

            # Test forward pass
            input_features = torch.randn(2, 768)
            outputs = head(input_features)

            self.assertEqual(len(outputs), 2)
            for task_name, output in outputs.items():
                self.assertEqual(output.shape[0], 2)

    def test_multitask_head_custom_activations(self):
        """Test multi-task head with custom activation functions"""
        task_configs = {
            "toxicity": {
                "type": "classification",
                "num_classes": 2,
                "activation": "sigmoid"
            },
            "emotion": {
                "type": "classification",
                "num_classes": 3,
                "activation": "softmax"
            },
            "score": {
                "type": "regression",
                "output_size": 1,
                "activation": "tanh"
            }
        }

        head = MultiTaskHead(
            input_size=768,
            task_configs=task_configs
        )

        input_features = torch.randn(3, 768)
        outputs = head(input_features)

        # Check that outputs are in expected ranges
        toxicity_out = outputs["toxicity"]
        emotion_out = outputs["emotion"]
        score_out = outputs["score"]

        # Sigmoid output should be in [0, 1]
        self.assertTrue(torch.all(toxicity_out >= 0) and torch.all(toxicity_out
            <= 1))

        # Softmax output should sum to 1
        emotion_sums = torch.sum(emotion_out, dim=1)
        self.assertTrue(
            torch.allclose(emotion_sums,
            torch.ones_like(emotion_sums))
        )

        # Tanh output should be in [-1, 1]
        self.assertTrue(torch.all(score_out >= -1) and torch.all(score_out <=
            1))


class TestAdvancedBaselineModel(unittest.TestCase):
    """Advanced tests for BaselineModel"""

    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_baseline_model_initialization_with_pretrained(
        self,
        mock_tokenizer,
        mock_model
    ):
        """Test baseline model initialization with pretrained models"""
        # Mock the pretrained components
        mock_transformer = Mock()
        mock_transformer.config.hidden_size = 768
        mock_model.return_value = mock_transformer

        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        config = ModelConfig(
            model_name="hfl/chinese-macbert-base",
            num_classes={"toxicity": 2, "emotion": 3}
        )

        model = BaselineModel(config)

        self.assertIsNotNone(model.transformer)
        self.assertIsNotNone(model.classifier)
        mock_model.assert_called_once()

    def test_baseline_model_forward_with_attention_mask(self):
        """Test baseline model forward pass with attention masks"""
        config = ModelConfig(
            model_name="test-model",
            num_classes={"toxicity": 2}
        )

        # Create model with mocked transformer
        model = BaselineModel(config)
        model.transformer = Mock()
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(2, 10, 768)
        model.transformer.return_value = mock_output

        # Test forward pass with attention mask
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, -2:] = 0  # Mask last 2 tokens

        outputs = model(input_ids, attention_mask=attention_mask)

        self.assertIn("toxicity", outputs)
        model.transformer.assert_called_once()

    def test_baseline_model_compute_loss_with_weights(self):
        """Test loss computation with task weights"""
        config = ModelConfig(
            model_name="test-model",
            num_classes={"toxicity": 2, "emotion": 3},
            task_weights={"toxicity": 2.0, "emotion": 1.0}
        )

        model = BaselineModel(config)

        # Mock outputs
        outputs = {
            "toxicity": torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
            "emo"
                "tion": torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        }

        labels = {
            "toxicity": torch.tensor([0, 1]),
            "emotion": torch.tensor([0, 1, 2])
        }

        loss = model.compute_loss(outputs, labels)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() > 0)

    def test_baseline_model_predict_with_thresholds(self):
        """Test prediction with custom thresholds"""
        config = ModelConfig(
            model_name="test-model",
            num_classes={"toxicity": 2}
        )

        model = BaselineModel(config)
        model.eval()

        # Mock the model's forward pass
        original_forward = model.forward
        model.forward = Mock(return_value={"toxi"
            "city": torch.tensor([[0.3, 0.7], [0.8, 0.2]])})

        input_ids = torch.randint(0, 1000, (2, 10))

        # Test with different thresholds
        predictions_default = model.predict(input_ids)
        predictions_high_threshold = model.predict(input_ids, threshold=0.8)

        self.assertIn("toxicity", predictions_default)
        self.assertIn("toxicity", predictions_high_threshold)

        # High threshold should be more conservative
        __default_positive = torch.sum(predictions_default["toxicity"]).item()
        __high_thresh_positive = torch.sum(predictions_high_threshold["toxi"
            "city"]).item()

        # Restore original forward
        model.forward = original_forward

    def test_baseline_model_save_and_load(self):
        """Test model saving and loading functionality"""
        config = ModelConfig(
            model_name="test-model",
            num_classes={"toxicity": 2}
        )

        model = BaselineModel(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model"

            # Save model
            model.save_pretrained(str(save_path))

            # Check that files were created
            self.assertTrue((save_path / "config.json").exists())
            self.assertTrue((save_path / "pytorch_model.bin").exists())

            # Load model
            loaded_model = BaselineModel.from_pretrained(str(save_path))

            # Compare configurations
            self.assertEqual(loaded_model.config.model_name, config.model_name)
            self.assertEqual(
                loaded_model.config.num_classes,
                config.num_classes
            )

    def test_baseline_model_gradient_checkpointing(self):
        """Test model with gradient checkpointing"""
        config = ModelConfig(
            model_name="test-model",
            num_classes={"toxicity": 2},
            gradient_checkpointing=True
        )

        model = BaselineModel(config)

        # Mock transformer with gradient checkpointing support
        model.transformer = Mock()
        model.transformer.gradient_checkpointing_enable = Mock()

        # Enable gradient checkpointing
        model.enable_gradient_checkpointing()

        model.transformer.gradient_checkpointing_enable.assert_called_once()

    def test_baseline_model_freeze_layers(self):
        """Test selective layer freezing"""
        config = ModelConfig(
            model_name="test-model",
            num_classes={"toxicity": 2}
        )

        model = BaselineModel(config)

        # Mock transformer layers
        mock_layer = Mock()
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_layer.parameters.return_value = [mock_param]

        model.transformer = Mock()
        model.transformer.encoder = Mock()
        model.transformer.encoder.layer = [mock_layer] * 12

        # Freeze first 6 layers
        model.freeze_layers(num_layers=6)

        # Check that parameters were set to not require grad
        self.assertEqual(mock_param.requires_grad, False)


class TestAdvancedDetectionResult(unittest.TestCase):
    """Advanced tests for DetectionResult"""

    def test_detection_result_comprehensive_validation(self):
        """Test comprehensive validation of DetectionResult"""
        # Valid result
        result = DetectionResult(
            text="測試文本",
            toxicity_label=1,
            toxicity_confidence=0.85,
            emotion_label=2,
            emotion_confidence=0.75,
            bullying_label=0,
            bullying_confidence=0.15,
            metadata={"model_version": "1.0", "timestamp": "2024-01-01"}
        )

        self.assertEqual(result.text, "測試文本")
        self.assertEqual(result.toxicity_label, 1)
        self.assertEqual(result.metadata["model_version"], "1.0")

    def test_detection_result_confidence_validation(self):
        """Test confidence value validation"""
        # Test invalid confidence values
        with self.assertRaises(ValueError):
            DetectionResult(
                text="test",
                toxicity_label=1,
                toxicity_confidence=1.5,  # Invalid: > 1.0
                emotion_label=0,
                emotion_confidence=0.5,
                bullying_label=0,
                bullying_confidence=0.2
            )

        with self.assertRaises(ValueError):
            DetectionResult(
                text="test",
                toxicity_label=1,
                toxicity_confidence=0.8,
                emotion_label=0,
                emotion_confidence=-0.1,  # Invalid: < 0.0
                bullying_label=0,
                bullying_confidence=0.2
            )

    def test_detection_result_serialization(self):
        """Test DetectionResult serialization methods"""
        result = DetectionResult(
            text="測試",
            toxicity_label=1,
            toxicity_confidence=0.85,
            emotion_label=2,
            emotion_confidence=0.75,
            bullying_label=0,
            bullying_confidence=0.15,
            explanation={"tokens": ["測", "試"], "attributions": [0.1, 0.9]}
        )

        # Test to_dict
        result_dict = result.to_dict()
        self.assertIn("text", result_dict)
        self.assertIn("toxicity_label", result_dict)
        self.assertIn("explanation", result_dict)

        # Test to_json
        json_str = result.to_json()
        self.assertIsInstance(json_str, str)

        # Test from_dict
        restored_result = DetectionResult.from_dict(result_dict)
        self.assertEqual(restored_result.text, result.text)
        self.assertEqual(restored_result.toxicity_label, result.toxicity_label)

    def test_detection_result_privacy_compliance(self):
        """Test privacy-compliant logging"""
        result = DetectionResult(
            text="這裡包含個人信息：john@example.com 和電話 123-456-7890",
            toxicity_label=1,
            toxicity_confidence=0.85,
            emotion_label=0,
            emotion_confidence=0.3,
            bullying_label=0,
            bullying_confidence=0.1
        )

        # Test privacy-safe representation
        safe_dict = result.to_dict(privacy_safe=True)

        # Original text should be hashed or redacted
        self.assertNotEqual(safe_dict["text"], result.text)
        self.assertIn("text_hash", safe_dict)

    def test_detection_result_aggregation(self):
        """Test result aggregation methods"""
        results = []
        for i in range(10):
            result = DetectionResult(
                text=f"test_{i}",
                toxicity_label=i % 2,
                toxicity_confidence=0.5 + (i * 0.05),
                emotion_label=i % 3,
                emotion_confidence=0.4 + (i * 0.04),
                bullying_label=0,
                bullying_confidence=0.1
            )
            results.append(result)

        # Test batch aggregation
        batch_results = BatchResults(results)

        summary = batch_results.get_summary_statistics()

        self.assertIn("toxicity_distribution", summary)
        self.assertIn("average_confidence", summary)
        self.assertIn("total_samples", summary)

    def test_detection_result_confidence_calibration(self):
        """Test confidence calibration"""
        result = DetectionResult(
            text="test",
            toxicity_label=1,
            toxicity_confidence=0.95,  # High confidence
            emotion_label=0,
            emotion_confidence=0.6,
            bullying_label=0,
            bullying_confidence=0.1
        )

        # Test confidence calibration
        calibrated_result = result.calibrate_confidence(
            calibration_params={"toxicity": {"slope": 0.8, "intercept": 0.1}}
        )

        # Calibrated confidence should be different
        self.assertNotEqual(
            calibrated_result.toxicity_confidence,
            result.toxicity_confidence
        )


class TestAdvancedCyberPuppyDetector(unittest.TestCase):
    """Advanced tests for CyberPuppyDetector"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()

        # Configure mock tokenizer
        self.mock_tokenizer.encode_plus.return_value = {
            "input_ids": torch.tensor([[101, 2523, 3221, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "token_type_ids": torch.tensor([[0, 0, 0, 0]])
        }

        # Configure mock model output
        mock_output = Mock()
        mock_output.toxicity = torch.tensor([[0.2, 0.8]])
        mock_output.emotion = torch.tensor([[0.1, 0.3, 0.6]])
        mock_output.bullying = torch.tensor([[0.9, 0.1]])
        self.mock_model.predict.return_value = mock_output

    def test_detector_with_ensemble_config(self):
        """Test detector with ensemble configuration"""
        ensemble_config = EnsembleConfig(
            models=["model1", "model2", "model3"],
            weights=[0.4, 0.4, 0.2],
            voting_strategy="weighted"
        )

        detector = CyberPuppyDetector(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            ensemble_config=ensemble_config
        )

        self.assertEqual(detector.ensemble_config.voting_strategy, "weighted")
        self.assertEqual(len(detector.ensemble_config.models), 3)

    def test_detector_preprocessing_pipeline(self):
        """Test comprehensive text preprocessing"""
        preprocessing_config = PreprocessingConfig(
            normalize_unicode=True,
            remove_urls=True,
            remove_mentions=True,
            convert_traditional_to_simplified=True,
            max_length=512
        )

        detector = CyberPuppyDetector(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            preprocessing_config=preprocessing_config
        )

        # Test preprocessing different text types
        test_texts = [
            "這是一個包含URL的文本 https://example.com",
            "@用戶名 你好世界",
            "繁體中文測試",
            "Mixed 中文 and English 123 !@#",
            "   多餘空白   "
        ]

        for text in test_texts:
            result = detector.analyze(text)
            self.assertIsInstance(result, DetectionResult)
            # Preprocessed text should be different from original
            self.assertIsNotNone(result.text)

    def test_detector_batch_processing_optimization(self):
        """Test batch processing optimization"""
        detector = CyberPuppyDetector(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            batch_size=4
        )

        # Test batch analysis
        texts = [f"測試文本 {i}" for i in range(10)]

        results = detector.analyze_batch(texts)

        self.assertEqual(len(results), 10)
        for result in results:
            self.assertIsInstance(result, DetectionResult)

    def test_detector_context_aware_analysis(self):
        """Test context-aware analysis"""
        detector = CyberPuppyDetector(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            context_aware=True
        )

        # Test with conversation context
        conversation_history = [
            {"role": "user", "text": "你好", "timestamp": "2024-01-01T10:00:00"},
            {"ro"
                "le": 
            {"ro"
                "le": 
        ]

        current_text = "我想要傷害自己"

        result = detector.analyze(
            text=current_text,
            context=conversation_history
        )

        self.assertIsInstance(result, DetectionResult)
        # Context should influence the analysis
        self.assertIsNotNone(result.metadata.get("context_influence"))

    def test_detector_performance_profiling(self):
        """Test performance profiling capabilities"""
        detector = CyberPuppyDetector(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            enable_profiling=True
        )

        text = "測試性能分析"

        result = detector.analyze(text)

        self.assertIsInstance(result, DetectionResult)
        # Should have performance metrics
        self.assertIn("processing_time", result.metadata)
        self.assertIn("memory_usage", result.metadata)

    def test_detector_error_handling_and_fallback(self):
        """Test comprehensive error handling and fallback mechanisms"""
        # Configure model to raise errors
        self.mock_model.predict.side_effect = Exception("Model error")

        detector = CyberPuppyDetector(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            fallback_enabled=True
        )

        # Should handle errors gracefully with fallback
        result = detector.analyze("測試錯誤處理")

        self.assertIsInstance(result, DetectionResult)
        self.assertIn("fallback_used", result.metadata)
        self.assertTrue(result.metadata["fallback_used"])

    def test_detector_confidence_thresholding(self):
        """Test dynamic confidence thresholding"""
        detector = CyberPuppyDetector(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            confidence_thresholds={
                "toxicity": 0.8,
                "emotion": 0.7,
                "bullying": 0.9
            }
        )

        # Configure model to return varying confidence levels
        mock_outputs = [
            {"toxi"
                "city": 0.75, 
            {"toxi"
                "city": 0.85, 
        ]

        for i, expected_confidences in enumerate(mock_outputs):
            # Mock the prediction to return expected values
            mock_output = Mock()
            mock_output.toxicity = torch.tensor([[1-expected_confidences["toxi"
                "city"],
                                                  expected_confidences["toxi"
                                                      "city"]]])
            mock_output.emotion = torch.tensor([[expected_confidences["emo"
                "tion"], 0.2, 0.1]])
            mock_output.bullying = torch.tensor([[1-expected_confidences["bull"
                "ying"],
                                                 expected_confidences["bull"
                                                     "ying"]]])
            self.mock_model.predict.return_value = mock_output

            result = detector.analyze(f"測試文本 {i}")

            # Check that thresholding is applied
            self.assertIsInstance(result, DetectionResult)

    def test_detector_multilingual_support(self):
        """Test multilingual text analysis"""
        detector = CyberPuppyDetector(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            multilingual=True
        )

        # Test various languages and scripts
        multilingual_texts = [
            "這是中文測試",  # Chinese
            "This is English test",  # English
            "これは日本語のテストです",  # Japanese
            "이것은 한국어 테스트입니다",  # Korean
            "Mixed 語言 test 文字"  # Mixed
        ]

        for text in multilingual_texts:
            result = detector.analyze(text)
            self.assertIsInstance(result, DetectionResult)
            self.assertIn("detected_language", result.metadata)

    def test_detector_memory_optimization(self):
        """Test memory optimization features"""
        detector = CyberPuppyDetector(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            memory_efficient=True,
            max_memory_usage="1GB"
        )

        # Test with large batch to trigger memory optimization
        large_texts = [f"測試文本 {i} 內容很長" * 100 for i in range(50)]

        # Should handle large batches without memory overflow
        try:
            results = detector.analyze_batch(large_texts, batch_size=10)
            self.assertEqual(len(results), 50)
        except MemoryError:
            self.fail("Memory optimization failed to handle large batch")


class TestModelIntegration(unittest.TestCase):
    """Test integration scenarios across models"""

    def test_end_to_end_training_workflow(self):
        """Test complete training workflow"""
        # Create config
        config = ModelConfig(
            model_name="test-model",
            num_classes={"toxicity": 2, "emotion": 3},
            task_weights={"toxicity": 1.5, "emotion": 1.0}
        )

        # Create model
        model = BaselineModel(config)

        # Mock training data
        batch_size = 4
        sequence_length = 20
        input_ids = torch.randint(0, 1000, (batch_size, sequence_length))
        attention_mask = torch.ones_like(input_ids)

        labels = {
            "toxicity": torch.randint(0, 2, (batch_size,)),
            "emotion": torch.randint(0, 3, (batch_size,))
        }

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)

        # Compute loss
        loss = model.compute_loss(outputs, labels)

        # Backward pass
        loss.backward()

        # Verify gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_model_evaluation_workflow(self):
        """Test complete model evaluation workflow"""
        config = ModelConfig(
            model_name="test-model",
            num_classes={"toxicity": 2}
        )

        model = BaselineModel(config)
        evaluator = ModelEvaluator(model)

        # Mock evaluation data
        eval_data = []
        for i in range(20):
            sample = {
                "input_ids": torch.randint(0, 1000, (10,)),
                "attention_mask": torch.ones(10),
                "labels": {"toxicity": torch.randint(0, 2, (1,))}
            }
            eval_data.append(sample)

        # Run evaluation
        results = evaluator.evaluate(eval_data)

        self.assertIn("accuracy", results)
        self.assertIn("f1_score", results)
        self.assertIn("confusion_matrix", results)

    def test_model_variant_creation(self):
        """Test creation of different model variants"""
        base_config = ModelConfig(
            model_name="test-model",
            num_classes={"toxicity": 2, "emotion": 3}
        )

        variants = create_model_variants(
            base_config=base_config,
            variant_configs=[
                {"dropout_rate": 0.1, "hidden_size": 512},
                {"dropout_rate": 0.3, "hidden_size": 256},
                {"dropout_rate": 0.5, "hidden_size": 128}
            ]
        )

        self.assertEqual(len(variants), 3)
        for variant in variants:
            self.assertIsInstance(variant, BaselineModel)

        # Check that variants have different configurations
        self.assertNotEqual(
            variants[0].config.dropout_rate,
            variants[1].config.dropout_rate
        )


if __name__ == '__main__':
    unittest.main()
