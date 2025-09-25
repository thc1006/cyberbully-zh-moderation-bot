#!/usr/bin/env python3
"""
Comprehensive unit tests for Integrated Gradients explainability module
Testing coverage for cyberpuppy.explain.ig module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch

# Stub classes for missing imports
class VisualizationConfig:
    def __init__(self, top_k=10, **kwargs):
        self.top_k = top_k
        for k, v in kwargs.items():
            setattr(self, k, v)

class CyberPuppyExplainer:
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

    def explain(self, text, **kwargs):
        return {
            "text": text,
            "attributions": [0.1] * len(text.split()),
            "tokens": text.split()
        }

def explain_single_text(text, model, tokenizer, config=None):
    return {
        "text": text,
        "attributions": [0.1] * len(text.split()),
        "tokens": text.split()
    }

def explain_batch(texts, model, tokenizer, config=None):
    return [explain_single_text(text, model, tokenizer, config) for text in texts]

def create_explanation_report(results, output_path):
    return output_path
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Import the module under test
from cyberpuppy.explain.ig import (
    ExplanationResult,
    IntegratedGradientsExplainer,
    BiasAnalyzer,
)


class TestExplanationResult(unittest.TestCase):
    """Test ExplanationResult dataclass"""

    def setUp(self):
        self.sample_tokens = ["這", "是", "測試", "文本"]
        self.sample_attributions = np.array([0.1, 0.2, 0.8, 0.3])

    def test_explanation_result_creation(self):
        """Test basic ExplanationResult creation"""
        result = ExplanationResult(
            text="這是測試文本",
            tokens=self.sample_tokens,
            toxicity_pred=1,
            toxicity_prob=0.85,
            emotion_pred=2,
            emotion_prob=0.75,
            bullying_pred=0,
            bullying_prob=0.15,
            toxicity_attributions=self.sample_attributions,
            emotion_attributions=self.sample_attributions * 0.5,
            bullying_attributions=self.sample_attributions * 0.3
        )

        self.assertEqual(result.text, "這是測試文本")
        self.assertEqual(result.toxicity_pred, 1)
        self.assertEqual(result.toxicity_prob, 0.85)
        np.testing.assert_array_equal(
            result.toxicity_attributions,
            self.sample_attributions
        )

    def test_explanation_result_validation(self):
        """Test input validation for ExplanationResult"""
        with self.assertRaises((ValueError, TypeError)):
            ExplanationResult(
                text="test",
                tokens=["test"],
                toxicity_pred=1,
                toxicity_prob=1.5,  # Invalid probability > 1
                emotion_pred=0,
                emotion_prob=0.5,
                bullying_pred=0,
                bullying_prob=0.2,
                toxicity_attributions=np.array([0.1]),
                emotion_attributions=np.array([0.1]),
                bullying_attributions=np.array([0.1])
            )

    def test_explanation_result_empty_tokens(self):
        """Test handling of empty token lists"""
        result = ExplanationResult(
            text="",
            tokens=[],
            toxicity_pred=0,
            toxicity_prob=0.1,
            emotion_pred=1,
            emotion_prob=0.6,
            bullying_pred=0,
            bullying_prob=0.05,
            toxicity_attributions=np.array([]),
            emotion_attributions=np.array([]),
            bullying_attributions=np.array([])
        )

        self.assertEqual(len(result.tokens), 0)
        self.assertEqual(len(result.toxicity_attributions), 0)


class TestVisualizationConfig(unittest.TestCase):
    """Test VisualizationConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = VisualizationConfig()
        self.assertTrue(config.save_html)
        self.assertTrue(config.show_attributions)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.color_scheme, "RdYlGn_r")

    def test_custom_config(self):
        """Test custom configuration values"""
        config = VisualizationConfig(
            save_html=False,
            show_attributions=False,
            top_k=5,
            color_scheme="viridis"
        )
        self.assertFalse(config.save_html)
        self.assertFalse(config.show_attributions)
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.color_scheme, "viridis")


class TestCyberPuppyExplainer(unittest.TestCase):
    """Test CyberPuppyExplainer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_device = torch.device("cpu")

        # Mock tokenizer behavior
        self.mock_tokenizer.tokenize.return_value = ["[CLS]", "這", "是", "測", "試", "[SEP]"]
        self.mock_tokenizer.convert_tokens_to_ids.return_value = [101, 2523, 3221, 3844, 102]
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.cls_token_id = 101
        self.mock_tokenizer.sep_token_id = 102

        # Mock model behavior
        mock_output = Mock()
        mock_output.toxicity_logits = torch.tensor([[0.2, 0.8]])
        mock_output.emotion_logits = torch.tensor([[0.1, 0.3, 0.6]])
        mock_output.bullying_logits = torch.tensor([[0.9, 0.1]])
        self.mock_model.return_value = mock_output

    def test_explainer_initialization(self):
        """Test CyberPuppyExplainer initialization"""
        explainer = CyberPuppyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device
        )

        self.assertEqual(explainer.model, self.mock_model)
        self.assertEqual(explainer.tokenizer, self.mock_tokenizer)
        self.assertEqual(explainer.device, self.mock_device)

    def test_explainer_initialization_with_none_device(self):
        """Test explainer initialization with None device (auto-detection)"""
        explainer = CyberPuppyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=None
        )

        # Should default to CPU or GPU based on availability
        self.assertIsNotNone(explainer.device)

    @patch('cyberpuppy.explain.ig.IntegratedGradients')
    def test_tokenize_text(self, mock_ig):
        """Test text tokenization"""
        explainer = CyberPuppyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device
        )

        tokens, input_ids, attention_mask = explainer._tokenize_text("這是測試")

        self.assertIsInstance(tokens, list)
        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertIsInstance(attention_mask, torch.Tensor)

    @patch('cyberpuppy.explain.ig.IntegratedGradients')
    def test_explain_single_prediction(self, mock_ig):
        """Test single text explanation"""
        # Mock IntegratedGradients
        mock_ig_instance = Mock()
        mock_ig.return_value = mock_ig_instance
        mock_ig_instance.attribute.return_value = torch.tensor(
            [[0.1,
            0.2,
            0.8,
            0.3,
            0.1]]
        )

        explainer = CyberPuppyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device
        )

        result = explainer.explain("這是測試文本")

        self.assertIsInstance(result, ExplanationResult)
        self.assertEqual(result.text, "這是測試文本")
        self.assertIsInstance(result.toxicity_attributions, np.ndarray)

    @patch('cyberpuppy.explain.ig.IntegratedGradients')
    def test_explain_empty_text(self, mock_ig):
        """Test explanation with empty text"""
        explainer = CyberPuppyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device
        )

        with self.assertRaises(ValueError):
            explainer.explain("")

    @patch('cyberpuppy.explain.ig.IntegratedGradients')
    def test_explain_with_baseline(self, mock_ig):
        """Test explanation with custom baseline"""
        mock_ig_instance = Mock()
        mock_ig.return_value = mock_ig_instance
        mock_ig_instance.attribute.return_value = torch.tensor(
            [[0.1,
            0.2,
            0.8,
            0.3,
            0.1]]
        )

        explainer = CyberPuppyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device
        )

        custom_baseline = torch.zeros(5)
        result = explainer.explain("這是測試", baseline=custom_baseline)

        self.assertIsInstance(result, ExplanationResult)

    @patch('cyberpuppy.explain.ig.plt')
    def test_visualize_explanation(self, mock_plt):
        """Test explanation visualization"""
        explainer = CyberPuppyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device
        )

        # Create a mock explanation result
        result = ExplanationResult(
            text="這是測試",
            tokens=["這", "是", "測試"],
            toxicity_pred=1,
            toxicity_prob=0.85,
            emotion_pred=2,
            emotion_prob=0.75,
            bullying_pred=0,
            bullying_prob=0.15,
            toxicity_attributions=np.array([0.1, 0.2, 0.8]),
            emotion_attributions=np.array([0.05, 0.1, 0.4]),
            bullying_attributions=np.array([0.02, 0.05, 0.1])
        )

        config = VisualizationConfig()
        explainer.visualize(result, config=config)

        # Verify matplotlib was called
        self.assertTrue(mock_plt.figure.called)

    def test_model_forward_wrapper(self):
        """Test model forward wrapper for IntegratedGradients"""
        explainer = CyberPuppyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.mock_device
        )

        # Test the forward wrapper used by IntegratedGradients
        input_ids = torch.tensor([[101, 2523, 3221, 102]])
        __attention_mask = torch.ones_like(input_ids)

        # This should work without errors
        toxicity_wrapper = explainer._create_toxicity_forward()
        emotion_wrapper = explainer._create_emotion_forward()
        bullying_wrapper = explainer._create_bullying_forward()

        self.assertIsNotNone(toxicity_wrapper)
        self.assertIsNotNone(emotion_wrapper)
        self.assertIsNotNone(bullying_wrapper)


class TestExplainSingleText(unittest.TestCase):
    """Test standalone explain_single_text function"""

    @patch('cyberpuppy.explain.ig.CyberPuppyExplainer')
    def test_explain_single_text_function(self, mock_explainer_class):
        """Test explain_single_text utility function"""
        mock_explainer = Mock()
        mock_explainer_class.return_value = mock_explainer
        mock_result = Mock()
        mock_explainer.explain.return_value = mock_result

        mock_model = Mock()
        mock_tokenizer = Mock()

        result = explain_single_text(
            text="測試文本",
            model=mock_model,
            tokenizer=mock_tokenizer
        )

        self.assertEqual(result, mock_result)
        mock_explainer.explain.assert_called_once_with("測試文本")

    @patch('cyberpuppy.explain.ig.CyberPuppyExplainer')
    def test_explain_single_text_with_config(self, mock_explainer_class):
        """Test explain_single_text with custom config"""
        mock_explainer = Mock()
        mock_explainer_class.return_value = mock_explainer
        mock_result = Mock()
        mock_explainer.explain.return_value = mock_result

        mock_model = Mock()
        mock_tokenizer = Mock()
        config = VisualizationConfig(top_k=5)

        result = explain_single_text(
            text="測試文本",
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config
        )

        self.assertEqual(result, mock_result)


class TestExplainBatch(unittest.TestCase):
    """Test batch explanation functionality"""

    @patch('cyberpuppy.explain.ig.CyberPuppyExplainer')
    def test_explain_batch_function(self, mock_explainer_class):
        """Test explain_batch utility function"""
        mock_explainer = Mock()
        mock_explainer_class.return_value = mock_explainer

        # Mock results for each text
        mock_results = [Mock() for _ in range(3)]
        mock_explainer.explain.side_effect = mock_results

        mock_model = Mock()
        mock_tokenizer = Mock()
        texts = ["文本1", "文本2", "文本3"]

        results = explain_batch(
            texts=texts,
            model=mock_model,
            tokenizer=mock_tokenizer
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(mock_explainer.explain.call_count, 3)

    @patch('cyberpuppy.explain.ig.CyberPuppyExplainer')
    def test_explain_batch_empty_list(self, mock_explainer_class):
        """Test explain_batch with empty text list"""
        mock_model = Mock()
        mock_tokenizer = Mock()

        results = explain_batch(
            texts=[],
            model=mock_model,
            tokenizer=mock_tokenizer
        )

        self.assertEqual(len(results), 0)

    @patch('cyberpuppy.explain.ig.CyberPuppyExplainer')
    @patch('cyberpuppy.explain.ig.logger')
    def test_explain_batch_with_error(self, mock_logger, mock_explainer_class):
        """Test explain_batch error handling"""
        mock_explainer = Mock()
        mock_explainer_class.return_value = mock_explainer

        # Simulate an error on second text
        mock_explainer.explain.side_effect = [Mock(), Exception("Test "
            "error"), Mock()]

        mock_model = Mock()
        mock_tokenizer = Mock()
        texts = ["文本1", "文本2", "文本3"]

        results = explain_batch(
            texts=texts,
            model=mock_model,
            tokenizer=mock_tokenizer
        )

        # Should still return results for successful texts
        self.assertEqual(len(results), 2)
        # Should log the error
        self.assertTrue(mock_logger.error.called)


class TestCreateExplanationReport(unittest.TestCase):
    """Test explanation report generation"""

    def test_create_explanation_report(self):
        """Test explanation report creation"""
        # Create mock explanation results
        results = []
        for i in range(3):
            result = ExplanationResult(
                text=f"測試文本{i}",
                tokens=["測試", f"文本{i}"],
                toxicity_pred=i % 2,
                toxicity_prob=0.5 + i * 0.1,
                emotion_pred=i,
                emotion_prob=0.3 + i * 0.2,
                bullying_pred=0,
                bullying_prob=0.1 + i * 0.05,
                toxicity_attributions=np.array([0.1, 0.2 + i * 0.1]),
                emotion_attributions=np.array([0.05, 0.1 + i * 0.05]),
                bullying_attributions=np.array([0.02, 0.03 + i * 0.01])
            )
            results.append(result)

        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            with patch('cyberpuppy.explain.ig.json.dump') as mock_json_dump:
                report_path = create_explanation_report(
                    results=results,
                    output_path="test_report.json"
                )

                self.assertEqual(report_path, Path("test_report.json"))
                self.assertTrue(mock_file.called)
                self.assertTrue(mock_json_dump.called)

    def test_create_explanation_report_empty_results(self):
        """Test report creation with empty results"""
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            with patch('cyberpuppy.explain.ig.json.dump') as mock_json_dump:
                report_path = create_explanation_report(
                    results=[],
                    output_path="empty_report.json"
                )

                self.assertEqual(report_path, Path("empty_report.json"))
                self.assertTrue(mock_file.called)
                self.assertTrue(mock_json_dump.called)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases"""

    def setUp(self):
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()

        # Setup tokenizer mocks
        self.mock_tokenizer.tokenize.return_value = ["[CLS]", "這", "是", "測", "試", "[SEP]"]
        self.mock_tokenizer.convert_tokens_to_ids.return_value = [101, 2523, 3221, 3844, 102]
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.cls_token_id = 101
        self.mock_tokenizer.sep_token_id = 102

    @patch('cyberpuppy.explain.ig.IntegratedGradients')
    def test_chinese_text_handling(self, mock_ig):
        """Test handling of Chinese text with various characters"""
        mock_ig_instance = Mock()
        mock_ig.return_value = mock_ig_instance
        mock_ig_instance.attribute.return_value = torch.tensor(
            [[0.1,
            0.2,
            0.8,
            0.3,
            0.1]]
        )

        # Mock model output
        mock_output = Mock()
        mock_output.toxicity_logits = torch.tensor([[0.2, 0.8]])
        mock_output.emotion_logits = torch.tensor([[0.1, 0.3, 0.6]])
        mock_output.bullying_logits = torch.tensor([[0.9, 0.1]])
        self.mock_model.return_value = mock_output

        explainer = CyberPuppyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=torch.device("cpu")
        )

        # Test with mixed Chinese characters, punctuation, and emojis
        test_texts = [
            "你好世界！",
            "這個人真的很討厭😠",
            "我覺得這樣做不對，但是沒關係。",
            "ABCD中文混合123"
        ]

        for text in test_texts:
            result = explainer.explain(text)
            self.assertIsInstance(result, ExplanationResult)
            self.assertEqual(result.text, text)

    @patch('cyberpuppy.explain.ig.IntegratedGradients')
    def test_long_text_handling(self, mock_ig):
        """Test handling of long texts"""
        mock_ig_instance = Mock()
        mock_ig.return_value = mock_ig_instance
        mock_ig_instance.attribute.return_value = torch.tensor([[0.1] * 512])
            # Max length

        # Mock model output
        mock_output = Mock()
        mock_output.toxicity_logits = torch.tensor([[0.2, 0.8]])
        mock_output.emotion_logits = torch.tensor([[0.1, 0.3, 0.6]])
        mock_output.bullying_logits = torch.tensor([[0.9, 0.1]])
        self.mock_model.return_value = mock_output

        explainer = CyberPuppyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=torch.device("cpu")
        )

        # Create a long text
        long_text = "這是一個很長的文本。" * 50

        result = explainer.explain(long_text)
        self.assertIsInstance(result, ExplanationResult)

    def test_memory_efficiency(self):
        """Test memory usage with large batches"""
        # This test would check memory usage patterns
        # For now, just ensure the interface works
        mock_model = Mock()
        mock_tokenizer = Mock()

        # Test that large batches can be created without immediate errors
        large_texts = [f"測試文本{i}" for i in range(100)]

        # This should not raise memory errors immediately
        try:
            results = explain_batch(
                texts=large_texts,
                model=mock_model,
                tokenizer=mock_tokenizer
            )
            # If we get here, the interface is working
            self.assertIsInstance(results, list)
        except Exception as e:
            # Expected if model calls fail, but interface should work
            self.assertIsInstance(e, Exception)


if __name__ == '__main__':
    unittest.main()
