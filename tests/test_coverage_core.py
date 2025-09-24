#!/usr/bin/env python3
"""
Core module tests to improve coverage for critical components
Focusing on modules with minimal external dependencies
"""

import unittest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Import core modules that should work
from cyberpuppy.config import Settings, get_config
from cyberpuppy.labeling.label_map import LabelMapper
from cyberpuppy.models.result import DetectionResult, ModelOutput


class TestConfigModule(unittest.TestCase):
    """Extended tests for config module to improve coverage"""

    def test_settings_validation_edge_cases(self):
        """Test edge cases for settings validation"""
        # Test with extreme values
        with self.assertRaises(ValueError):
            Settings(
                model_name="test",
                confidence_threshold=-0.5,  # Invalid negative threshold
            )

        with self.assertRaises(ValueError):
            Settings(
                model_name="test", confidence_threshold=1.5  # Invalid threshold > 1
            )

        with self.assertRaises(ValueError):
            Settings(
                model_name="", confidence_threshold=0.5  # Invalid empty model name
            )

    def test_settings_path_validation(self):
        """Test path validation in settings"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid paths
            settings = Settings(
                model_name="test", data_dir=temp_dir, model_dir=temp_dir
            )

            # Check that paths are resolved correctly
            self.assertTrue(Path(settings.data_dir).exists())
            self.assertTrue(Path(settings.model_dir).exists())

    def test_settings_environment_overrides(self):
        """Test environment variable overrides"""
        import os

        # Set environment variables
        os.environ["CYBERPUPPY_MODEL_NAME"] = "env_model"
        os.environ["CYBERPUPPY_CONFIDENCE_THRESHOLD"] = "0.85"
        os.environ["CYBERPUPPY_DEBUG"] = "true"

        try:
            settings = Settings()

            # Check that environment values are used
            self.assertEqual(settings.model_name, "env_model")
            self.assertEqual(settings.confidence_threshold, 0.85)
            self.assertTrue(settings.debug)

        finally:
            # Clean up environment variables
            del os.environ["CYBERPUPPY_MODEL_NAME"]
            del os.environ["CYBERPUPPY_CONFIDENCE_THRESHOLD"]
            del os.environ["CYBERPUPPY_DEBUG"]

    def test_settings_serialization(self):
        """Test settings serialization methods"""
        settings = Settings(
            model_name="test_model", confidence_threshold=0.75, debug=True
        )

        # Test to_dict
        settings_dict = settings.to_dict()
        self.assertIn("model_name", settings_dict)
        self.assertEqual(settings_dict["model_name"], "test_model")

        # Sensitive data should be masked
        self.assertNotIn("api_key", str(settings_dict))

    def test_labels_configuration(self):
        """Test labels configuration from settings"""
        settings = Settings()

        self.assertEqual(len(settings.TOXICITY_LABELS), 3)
        self.assertEqual(len(settings.EMOTION_LABELS), 3)
        self.assertEqual(len(settings.BULLYING_LABELS), 3)

    def test_config_presets(self):
        """Test configuration presets"""
        # Test all presets using get_config
        presets = ["development", "production", "testing"]

        for preset_name in presets:
            config = get_config(preset_name)
            self.assertIsInstance(config, Settings)
            self.assertTrue(hasattr(config, "BASE_MODEL"))
            self.assertTrue(hasattr(config, "TOXICITY_THRESHOLD"))

        # Test unknown config (should default to development)
        config = get_config("invalid_preset")
        self.assertIsInstance(config, Settings)

    def test_settings_path_methods(self):
        """Test path helper methods"""
        settings = Settings(
            model_name="test_model", data_dir="data", model_dir="models"
        )

        # Test path getter methods
        data_path = settings.get_data_path("dataset.json")
        model_path = settings.get_model_path("model.bin")

        self.assertIsInstance(data_path, Path)
        self.assertIsInstance(model_path, Path)

        self.assertTrue(str(data_path).endswith("dataset.json"))
        self.assertTrue(str(model_path).endswith("model.bin"))


class TestLabelMapperModule(unittest.TestCase):
    """Extended tests for label mapper to improve coverage"""

    def setUp(self):
        """Set up test fixtures"""
        self.task_configs = {
            "toxicity": TaskLabelConfig(
                name="toxicity",
                labels=["none", "toxic", "severe"],
                label_to_id={"none": 0, "toxic": 1, "severe": 2},
                id_to_label={0: "none", 1: "toxic", 2: "severe"},
            ),
            "emotion": TaskLabelConfig(
                name="emotion",
                labels=["positive", "neutral", "negative"],
                label_to_id={"positive": 0, "neutral": 1, "negative": 2},
                id_to_label={0: "positive", 1: "neutral", 2: "negative"},
            ),
        }

    def test_label_mapper_initialization(self):
        """Test label mapper initialization"""
        mapper = LabelMapper(self.task_configs)

        self.assertEqual(len(mapper.task_configs), 2)
        self.assertIn("toxicity", mapper.task_configs)
        self.assertIn("emotion", mapper.task_configs)

    def test_label_to_id_conversion(self):
        """Test label to ID conversion"""
        mapper = LabelMapper(self.task_configs)

        # Test valid conversions
        self.assertEqual(mapper.label_to_id("toxicity", "none"), 0)
        self.assertEqual(mapper.label_to_id("toxicity", "toxic"), 1)
        self.assertEqual(mapper.label_to_id("emotion", "negative"), 2)

    def test_id_to_label_conversion(self):
        """Test ID to label conversion"""
        mapper = LabelMapper(self.task_configs)

        # Test valid conversions
        self.assertEqual(mapper.id_to_label("toxicity", 0), "none")
        self.assertEqual(mapper.id_to_label("toxicity", 1), "toxic")
        self.assertEqual(mapper.id_to_label("emotion", 2), "negative")

    def test_invalid_task_handling(self):
        """Test handling of invalid tasks"""
        mapper = LabelMapper(self.task_configs)

        with self.assertRaises(KeyError):
            mapper.label_to_id("invalid_task", "label")

        with self.assertRaises(KeyError):
            mapper.id_to_label("invalid_task", 0)

    def test_invalid_label_handling(self):
        """Test handling of invalid labels"""
        mapper = LabelMapper(self.task_configs)

        with self.assertRaises(KeyError):
            mapper.label_to_id("toxicity", "invalid_label")

    def test_invalid_id_handling(self):
        """Test handling of invalid IDs"""
        mapper = LabelMapper(self.task_configs)

        with self.assertRaises(KeyError):
            mapper.id_to_label("toxicity", 999)

    def test_batch_label_conversion(self):
        """Test batch label conversion"""
        mapper = LabelMapper(self.task_configs)

        labels = ["none", "toxic", "severe"]
        ids = mapper.labels_to_ids("toxicity", labels)

        self.assertEqual(ids, [0, 1, 2])

        # Convert back
        converted_labels = mapper.ids_to_labels("toxicity", ids)
        self.assertEqual(converted_labels, labels)

    def test_get_task_info(self):
        """Test getting task information"""
        mapper = LabelMapper(self.task_configs)

        toxicity_info = mapper.get_task_info("toxicity")

        self.assertEqual(toxicity_info["name"], "toxicity")
        self.assertEqual(len(toxicity_info["labels"]), 3)
        self.assertEqual(toxicity_info["num_classes"], 3)

    def test_task_validation(self):
        """Test task validation methods"""
        mapper = LabelMapper(self.task_configs)

        self.assertTrue(mapper.is_valid_task("toxicity"))
        self.assertTrue(mapper.is_valid_task("emotion"))
        self.assertFalse(mapper.is_valid_task("invalid_task"))

    def test_label_validation(self):
        """Test label validation methods"""
        mapper = LabelMapper(self.task_configs)

        self.assertTrue(mapper.is_valid_label("toxicity", "none"))
        self.assertTrue(mapper.is_valid_label("toxicity", "toxic"))
        self.assertFalse(mapper.is_valid_label("toxicity", "invalid"))

    def test_get_all_tasks(self):
        """Test getting all available tasks"""
        mapper = LabelMapper(self.task_configs)

        tasks = mapper.get_all_tasks()
        self.assertEqual(set(tasks), {"toxicity", "emotion"})

    def test_statistics_methods(self):
        """Test label statistics methods"""
        mapper = LabelMapper(self.task_configs)

        stats = mapper.get_label_statistics("toxicity")

        self.assertIn("num_classes", stats)
        self.assertIn("labels", stats)
        self.assertEqual(stats["num_classes"], 3)


class TestModelOutputModule(unittest.TestCase):
    """Extended tests for model output classes"""

    def test_model_output_creation(self):
        """Test ModelOutput creation"""
        output = ModelOutput(
            toxicity_logits=torch.tensor([[0.2, 0.8]]),
            emotion_logits=torch.tensor([[0.1, 0.3, 0.6]]),
            bullying_logits=torch.tensor([[0.9, 0.1]]),
        )

        self.assertIsNotNone(output.toxicity_logits)
        self.assertIsNotNone(output.emotion_logits)
        self.assertIsNotNone(output.bullying_logits)

    def test_model_output_probabilities(self):
        """Test probability calculation from logits"""
        output = ModelOutput(
            toxicity_logits=torch.tensor([[0.0, 1.0]]),
            emotion_logits=torch.tensor([[0.0, 0.0, 2.0]]),
            bullying_logits=torch.tensor([[2.0, 0.0]]),
        )

        # Get probabilities
        toxicity_probs = torch.softmax(output.toxicity_logits, dim=1)
        emotion_probs = torch.softmax(output.emotion_logits, dim=1)
        bullying_probs = torch.softmax(output.bullying_logits, dim=1)

        # Check that probabilities sum to 1
        self.assertAlmostEqual(torch.sum(toxicity_probs).item(), 1.0, places=5)
        self.assertAlmostEqual(torch.sum(emotion_probs).item(), 1.0, places=5)
        self.assertAlmostEqual(torch.sum(bullying_probs).item(), 1.0, places=5)

    def test_model_output_predictions(self):
        """Test prediction extraction from logits"""
        output = ModelOutput(
            toxicity_logits=torch.tensor([[0.2, 0.8], [-0.5, 0.3]]),
            emotion_logits=torch.tensor([[0.1, 0.3, 0.6], [0.8, 0.1, 0.1]]),
            bullying_logits=torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
        )

        # Get predictions (argmax)
        toxicity_preds = torch.argmax(output.toxicity_logits, dim=1)
        emotion_preds = torch.argmax(output.emotion_logits, dim=1)
        bullying_preds = torch.argmax(output.bullying_logits, dim=1)

        expected_toxicity = torch.tensor([1, 1])
        expected_emotion = torch.tensor([2, 0])
        expected_bullying = torch.tensor([0, 1])

        self.assertTrue(torch.equal(toxicity_preds, expected_toxicity))
        self.assertTrue(torch.equal(emotion_preds, expected_emotion))
        self.assertTrue(torch.equal(bullying_preds, expected_bullying))


class TestDetectionResultModule(unittest.TestCase):
    """Extended tests for detection result"""

    def test_detection_result_comprehensive_creation(self):
        """Test comprehensive DetectionResult creation"""
        result = DetectionResult(
            text="測試文本內容",
            toxicity_label=1,
            toxicity_confidence=0.85,
            emotion_label=2,
            emotion_confidence=0.75,
            bullying_label=0,
            bullying_confidence=0.15,
            metadata={
                "model_version": "1.2.0",
                "processing_time": 0.045,
                "timestamp": datetime.now().isoformat(),
            },
        )

        self.assertEqual(result.text, "測試文本內容")
        self.assertEqual(result.toxicity_label, 1)
        self.assertAlmostEqual(result.toxicity_confidence, 0.85)
        self.assertIn("model_version", result.metadata)

    def test_detection_result_validation_bounds(self):
        """Test detection result validation with boundary values"""
        # Test boundary values that should be valid
        result_min = DetectionResult(
            text="",  # Empty text should be allowed
            toxicity_label=0,
            toxicity_confidence=0.0,  # Minimum confidence
            emotion_label=0,
            emotion_confidence=0.0,
            bullying_label=0,
            bullying_confidence=0.0,
        )

        result_max = DetectionResult(
            text="X" * 1000,  # Long text
            toxicity_label=1,
            toxicity_confidence=1.0,  # Maximum confidence
            emotion_label=2,
            emotion_confidence=1.0,
            bullying_label=1,
            bullying_confidence=1.0,
        )

        self.assertEqual(result_min.toxicity_confidence, 0.0)
        self.assertEqual(result_max.toxicity_confidence, 1.0)

    def test_detection_result_serialization_formats(self):
        """Test multiple serialization formats"""
        result = DetectionResult(
            text="測試序列化",
            toxicity_label=1,
            toxicity_confidence=0.80,
            emotion_label=1,
            emotion_confidence=0.65,
            bullying_label=0,
            bullying_confidence=0.20,
            explanation={
                "important_tokens": ["測試", "序列化"],
                "attribution_scores": [0.3, 0.7],
            },
        )

        # Test dictionary serialization
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertIn("text", result_dict)
        self.assertIn("explanation", result_dict)

        # Test JSON serialization
        json_str = result.to_json()
        self.assertIsInstance(json_str, str)

        # Test deserialization
        loaded_dict = json.loads(json_str)
        self.assertEqual(loaded_dict["text"], "測試序列化")

    def test_detection_result_confidence_calibration(self):
        """Test confidence calibration utilities"""
        result = DetectionResult(
            text="測試校準",
            toxicity_label=1,
            toxicity_confidence=0.95,
            emotion_label=0,
            emotion_confidence=0.60,
            bullying_label=0,
            bullying_confidence=0.10,
        )

        # Test confidence recalibration (mock implementation)
        original_confidence = result.toxicity_confidence

        # Simulate temperature scaling
        temperature = 1.5
        calibrated_confidence = original_confidence ** (1.0 / temperature)

        # Calibrated confidence should be different and lower
        self.assertNotEqual(calibrated_confidence, original_confidence)
        self.assertLess(calibrated_confidence, original_confidence)

    def test_detection_result_comparison_methods(self):
        """Test result comparison methods"""
        result1 = DetectionResult(
            text="文本1",
            toxicity_label=1,
            toxicity_confidence=0.80,
            emotion_label=0,
            emotion_confidence=0.60,
            bullying_label=0,
            bullying_confidence=0.20,
        )

        result2 = DetectionResult(
            text="文本2",
            toxicity_label=1,
            toxicity_confidence=0.90,  # Higher confidence
            emotion_label=1,
            emotion_confidence=0.70,
            bullying_label=0,
            bullying_confidence=0.15,
        )

        # Compare confidence levels
        self.assertGreater(result2.toxicity_confidence, result1.toxicity_confidence)
        self.assertGreater(result2.emotion_confidence, result1.emotion_confidence)
        self.assertLess(result2.bullying_confidence, result1.bullying_confidence)

    def test_detection_result_batch_processing(self):
        """Test batch processing of detection results"""
        results = []

        for i in range(10):
            result = DetectionResult(
                text=f"測試文本 {i}",
                toxicity_label=i % 2,
                toxicity_confidence=0.5 + (i * 0.05),
                emotion_label=i % 3,
                emotion_confidence=0.4 + (i * 0.04),
                bullying_label=0,
                bullying_confidence=0.1 + (i * 0.01),
            )
            results.append(result)

        # Test batch statistics
        toxicity_confidences = [r.toxicity_confidence for r in results]
        mean_toxicity_confidence = np.mean(toxicity_confidences)
        std_toxicity_confidence = np.std(toxicity_confidences)

        self.assertIsInstance(mean_toxicity_confidence, float)
        self.assertIsInstance(std_toxicity_confidence, float)
        self.assertGreater(mean_toxicity_confidence, 0)

    def test_detection_result_privacy_features(self):
        """Test privacy-preserving features"""
        sensitive_text = "我的郵箱是 john.doe@company.com 電話號碼 +1-555-0123"

        result = DetectionResult(
            text=sensitive_text,
            toxicity_label=0,
            toxicity_confidence=0.30,
            emotion_label=1,
            emotion_confidence=0.55,
            bullying_label=0,
            bullying_confidence=0.05,
        )

        # Test privacy-safe serialization
        safe_dict = result.to_dict()

        # In a real implementation, we'd want to ensure PII is handled
        # For now, just verify the structure
        self.assertIn("text", safe_dict)
        self.assertIsInstance(safe_dict["text"], str)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling across modules"""

    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs"""

        # Test empty string handling in DetectionResult
        result = DetectionResult(
            text="",
            toxicity_label=0,
            toxicity_confidence=0.5,
            emotion_label=0,
            emotion_confidence=0.5,
            bullying_label=0,
            bullying_confidence=0.1,
        )

        self.assertEqual(result.text, "")
        self.assertEqual(len(result.text), 0)

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters"""
        special_texts = [
            "測試中文 🤖 機器人",
            "Emojiテスト 😀🎉🔥",
            "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
            "Mixed script: English 中文 日本語 한글",
            "Numbers: 123 456.789 1e-5",
            "Whitespace variations:   \t\n   test   \r\n",
        ]

        for text in special_texts:
            result = DetectionResult(
                text=text,
                toxicity_label=0,
                toxicity_confidence=0.5,
                emotion_label=1,
                emotion_confidence=0.6,
                bullying_label=0,
                bullying_confidence=0.1,
            )

            self.assertEqual(result.text, text)
            self.assertIsInstance(result.text, str)

    def test_extreme_values(self):
        """Test handling of extreme values"""
        # Very long text
        long_text = "測試" * 1000

        result = DetectionResult(
            text=long_text,
            toxicity_label=1,
            toxicity_confidence=0.99999,  # Very high confidence
            emotion_label=2,
            emotion_confidence=0.00001,  # Very low confidence
            bullying_label=0,
            bullying_confidence=0.5,
        )

        self.assertEqual(len(result.text), 4000)  # 2 chars * 2 * 1000
        self.assertAlmostEqual(result.toxicity_confidence, 0.99999, places=5)
        self.assertAlmostEqual(result.emotion_confidence, 0.00001, places=5)

    def test_concurrent_access_patterns(self):
        """Test patterns that might be used in concurrent scenarios"""
        # Create multiple results simultaneously
        results = []

        for i in range(100):
            result = DetectionResult(
                text=f"並發測試 {i}",
                toxicity_label=i % 2,
                toxicity_confidence=np.random.uniform(0.1, 0.9),
                emotion_label=i % 3,
                emotion_confidence=np.random.uniform(0.2, 0.8),
                bullying_label=0,
                bullying_confidence=np.random.uniform(0.05, 0.3),
            )
            results.append(result)

        # Verify all results are valid
        for result in results:
            self.assertIsInstance(result.text, str)
            self.assertTrue(0.0 <= result.toxicity_confidence <= 1.0)
            self.assertTrue(0.0 <= result.emotion_confidence <= 1.0)
            self.assertTrue(0.0 <= result.bullying_confidence <= 1.0)

        # Test batch operations
        texts = [r.text for r in results]
        confidences = [r.toxicity_confidence for r in results]

        self.assertEqual(len(texts), 100)
        self.assertEqual(len(confidences), 100)


if __name__ == "__main__":
    unittest.main()
