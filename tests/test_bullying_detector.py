#!/usr/bin/env python3
"""
Comprehensive test suite for cyberbullying detection model.
Tests cover functionality, edge cases, performance, and robustness.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import os
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.cyberpuppy.models.baselines import MultiTaskBullyingDetector
from src.cyberpuppy.models.contextual import ContextualBullyingDetector
from src.cyberpuppy.config import Config
from src.cyberpuppy.eval.continuous_eval import ContinuousEvaluator
from scripts.evaluate_bullying_detection import BullyingDetectionEvaluator


class TestBullyingDetector:
    """Test suite for bullying detection models."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=Config)
        config.MODEL_SAVE_PATH = "test_model"
        config.MAX_LENGTH = 512
        config.BATCH_SIZE = 8
        config.NUM_CLASSES = {
            'toxicity': 3,
            'bullying': 3,
            'role': 4,
            'emotion': 3
        }
        return config

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "你真的很笨，什麼都不會",  # Toxic/bullying
            "今天天氣很好",  # Neutral
            "我覺得你說得很有道理",  # Positive
            "滾出去，沒人需要你",  # Severe toxic/threat
            "謝謝你的幫助",  # Positive
            "",  # Empty text (edge case)
            "a" * 1000,  # Very long text (edge case)
            "😊😊😊",  # Only emojis
            "123456",  # Only numbers
            "AAAAAAA"  # Repeated characters
        ]

    @pytest.fixture
    def sample_labels(self):
        """Sample labels corresponding to sample texts."""
        return [
            {'toxicity': 'toxic', 'bullying': 'harassment', 'role': 'perpetrator', 'emotion': 'neg', 'emotion_strength': 3},
            {'toxicity': 'none', 'bullying': 'none', 'role': 'none', 'emotion': 'neu', 'emotion_strength': 1},
            {'toxicity': 'none', 'bullying': 'none', 'role': 'none', 'emotion': 'pos', 'emotion_strength': 2},
            {'toxicity': 'severe', 'bullying': 'threat', 'role': 'perpetrator', 'emotion': 'neg', 'emotion_strength': 4},
            {'toxicity': 'none', 'bullying': 'none', 'role': 'none', 'emotion': 'pos', 'emotion_strength': 2},
            {'toxicity': 'none', 'bullying': 'none', 'role': 'none', 'emotion': 'neu', 'emotion_strength': 0},
            {'toxicity': 'none', 'bullying': 'none', 'role': 'none', 'emotion': 'neu', 'emotion_strength': 0},
            {'toxicity': 'none', 'bullying': 'none', 'role': 'none', 'emotion': 'pos', 'emotion_strength': 1},
            {'toxicity': 'none', 'bullying': 'none', 'role': 'none', 'emotion': 'neu', 'emotion_strength': 0},
            {'toxicity': 'none', 'bullying': 'none', 'role': 'none', 'emotion': 'neu', 'emotion_strength': 0}
        ]

    @pytest.fixture
    def mock_model(self):
        """Mock PyTorch model for testing."""
        model = Mock()
        model.eval = Mock()
        model.to = Mock(return_value=model)

        # Mock forward pass
        def mock_forward(**kwargs):
            batch_size = kwargs['input_ids'].shape[0]
            # Return mock logits for multi-task output
            logits = torch.randn(batch_size, 14)  # Total classes across all tasks
            return Mock(logits=logits)

        model.forward = mock_forward
        model.__call__ = mock_forward
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        tokenizer = Mock()

        def mock_tokenize(text, **kwargs):
            return {
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
            }

        tokenizer.__call__ = mock_tokenize
        tokenizer.from_pretrained = Mock(return_value=tokenizer)
        return tokenizer


class TestModelFunctionality(TestBullyingDetector):
    """Test core model functionality."""

    def test_model_loading(self, config, mock_model, mock_tokenizer):
        """Test model and tokenizer loading."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)

            assert evaluator.model is not None
            assert evaluator.tokenizer is not None
            mock_model.to.assert_called_once()
            mock_model.eval.assert_called_once()

    def test_prediction_output_format(self, config, mock_model, mock_tokenizer, sample_texts):
        """Test that predictions have correct format."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)
            predictions = evaluator.predict_batch(sample_texts[:3])

            assert len(predictions) == 3
            for pred in predictions:
                assert 'toxicity' in pred
                assert 'bullying' in pred
                assert 'role' in pred
                assert 'emotion' in pred
                assert 'emotion_strength' in pred
                assert 'confidence' in pred

                # Check value ranges
                assert pred['toxicity'] in ['none', 'toxic', 'severe']
                assert pred['bullying'] in ['none', 'harassment', 'threat']
                assert pred['role'] in ['none', 'perpetrator', 'victim', 'bystander']
                assert pred['emotion'] in ['pos', 'neu', 'neg']
                assert 0 <= pred['emotion_strength'] <= 4

    def test_batch_prediction_consistency(self, config, mock_model, mock_tokenizer, sample_texts):
        """Test that batch predictions are consistent."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)

            # Predict same text multiple times
            text = sample_texts[0]
            pred1 = evaluator.predict_batch([text])[0]
            pred2 = evaluator.predict_batch([text])[0]

            # Results should be identical (deterministic)
            assert pred1 == pred2


class TestEdgeCases(TestBullyingDetector):
    """Test edge cases and boundary conditions."""

    def test_empty_text_handling(self, config, mock_model, mock_tokenizer):
        """Test handling of empty text."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)
            predictions = evaluator.predict_batch([""])

            assert len(predictions) == 1
            assert all(key in predictions[0] for key in ['toxicity', 'bullying', 'role', 'emotion', 'emotion_strength'])

    def test_very_long_text_handling(self, config, mock_model, mock_tokenizer):
        """Test handling of very long text."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)
            long_text = "這是一個很長的文本 " * 200  # Very long text
            predictions = evaluator.predict_batch([long_text])

            assert len(predictions) == 1
            # Should not crash and should return valid prediction

    def test_special_characters_handling(self, config, mock_model, mock_tokenizer):
        """Test handling of special characters and emojis."""
        special_texts = [
            "😊😢😡🤬",  # Emojis
            "!@#$%^&*()",  # Special characters
            "你好！！！！！",  # Repeated punctuation
            "AAAAAAAAAA",  # Repeated characters
            "123456789",  # Numbers only
            "    ",  # Whitespace only
        ]

        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)
            predictions = evaluator.predict_batch(special_texts)

            assert len(predictions) == len(special_texts)
            for pred in predictions:
                assert all(key in pred for key in ['toxicity', 'bullying', 'role', 'emotion', 'emotion_strength'])

    def test_mixed_language_handling(self, config, mock_model, mock_tokenizer):
        """Test handling of mixed language text."""
        mixed_texts = [
            "Hello 你好",  # English + Chinese
            "こんにちは 안녕하세요",  # Japanese + Korean
            "Bonjour 你好 Hello",  # Multiple languages
        ]

        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)
            predictions = evaluator.predict_batch(mixed_texts)

            assert len(predictions) == len(mixed_texts)
            # Should handle gracefully without errors


class TestPerformanceRequirements(TestBullyingDetector):
    """Test performance requirements and constraints."""

    def test_prediction_speed(self, config, mock_model, mock_tokenizer, sample_texts):
        """Test that predictions complete within reasonable time."""
        import time

        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)

            start_time = time.time()
            predictions = evaluator.predict_batch(sample_texts)
            end_time = time.time()

            # Should complete within 5 seconds for 10 samples
            assert end_time - start_time < 5.0
            assert len(predictions) == len(sample_texts)

    def test_memory_usage(self, config, mock_model, mock_tokenizer):
        """Test memory usage remains reasonable."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)

            # Process large batch
            large_batch = ["測試文本 " * 50] * 100
            predictions = evaluator.predict_batch(large_batch)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (< 500MB for this test)
            assert memory_increase < 500
            assert len(predictions) == 100

    def test_batch_size_scaling(self, config, mock_model, mock_tokenizer):
        """Test performance with different batch sizes."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)

            batch_sizes = [1, 5, 10, 20, 50]
            for batch_size in batch_sizes:
                batch = ["測試文本"] * batch_size
                predictions = evaluator.predict_batch(batch)

                assert len(predictions) == batch_size
                # Should handle different batch sizes without errors


class TestRobustness(TestBullyingDetector):
    """Test model robustness and error handling."""

    def test_model_loading_failure(self, config):
        """Test graceful handling of model loading failure."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', side_effect=Exception("Model not found")):
            with pytest.raises(Exception):
                BullyingDetectionEvaluator(config)

    def test_tokenizer_loading_failure(self, config, mock_model):
        """Test graceful handling of tokenizer loading failure."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', side_effect=Exception("Tokenizer not found")):
            with pytest.raises(Exception):
                BullyingDetectionEvaluator(config)

    def test_prediction_with_corrupted_input(self, config, mock_model, mock_tokenizer):
        """Test handling of corrupted or malformed input."""
        corrupted_inputs = [
            None,  # None input
            123,   # Non-string input
            [],    # List input
            {},    # Dict input
        ]

        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)

            # Should handle gracefully (convert to string or filter out)
            try:
                predictions = evaluator.predict_batch(corrupted_inputs)
                # If it doesn't raise an exception, check the output
                assert isinstance(predictions, list)
            except (TypeError, AttributeError):
                # Expected behavior for corrupted input
                pass

    def test_device_compatibility(self, config, mock_model, mock_tokenizer):
        """Test compatibility with different devices (CPU/GPU)."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            # Test with CPU
            with patch('torch.cuda.is_available', return_value=False):
                evaluator = BullyingDetectionEvaluator(config)
                assert evaluator.device.type == 'cpu'

            # Test with GPU (if available)
            with patch('torch.cuda.is_available', return_value=True):
                evaluator = BullyingDetectionEvaluator(config)
                # Should not crash even if GPU not actually available


class TestEvaluationMetrics(TestBullyingDetector):
    """Test evaluation metrics calculation."""

    def test_comprehensive_evaluation(self, config, mock_model, mock_tokenizer, sample_texts, sample_labels):
        """Test comprehensive evaluation pipeline."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)

            with tempfile.TemporaryDirectory() as temp_dir:
                # Mock the dependent components
                with patch.object(evaluator, 'metrics_calculator') as mock_metrics, \
                     patch.object(evaluator, 'error_analyzer') as mock_error, \
                     patch.object(evaluator, 'robustness_evaluator') as mock_robustness, \
                     patch.object(evaluator, 'bias_evaluator') as mock_bias, \
                     patch.object(evaluator, 'report_generator') as mock_report:

                    # Configure mocks
                    mock_metrics.calculate_all_metrics.return_value = {'toxicity': {'f1_macro': 0.85}}
                    mock_error.analyze_errors.return_value = {'error_types': {'false_positive': 10}}
                    mock_robustness.evaluate_robustness.return_value = {'perturbation_results': {}}
                    mock_bias.evaluate_bias.return_value = {'fairness_metrics': {}}
                    mock_report.generate_report.return_value = None

                    results = evaluator.evaluate_comprehensive(
                        sample_texts, sample_labels, temp_dir
                    )

                    # Check result structure
                    assert 'basic_metrics' in results
                    assert 'error_analysis' in results
                    assert 'robustness' in results
                    assert 'bias_evaluation' in results
                    assert 'metadata' in results

                    # Check that results file is created
                    results_file = os.path.join(temp_dir, 'evaluation_results.json')
                    assert os.path.exists(results_file)

    def test_model_comparison(self, config, mock_model, mock_tokenizer):
        """Test model comparison functionality."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)

            baseline_results = {
                'basic_metrics': {
                    'toxicity': {'f1_macro': 0.75},
                    'bullying': {'f1_macro': 0.70}
                }
            }

            current_results = {
                'basic_metrics': {
                    'toxicity': {'f1_macro': 0.85},
                    'bullying': {'f1_macro': 0.75}
                }
            }

            with tempfile.TemporaryDirectory() as temp_dir:
                comparison = evaluator.compare_models(
                    baseline_results, current_results, temp_dir
                )

                # Check improvements are calculated correctly
                assert comparison['toxicity']['improvement'] == 0.10
                assert comparison['bullying']['improvement'] == 0.05
                assert comparison['toxicity']['improvement_pct'] > 0
                assert comparison['bullying']['improvement_pct'] > 0

                # Check comparison file is created
                comparison_file = os.path.join(temp_dir, 'model_comparison.json')
                assert os.path.exists(comparison_file)


class TestDataLoading(TestBullyingDetector):
    """Test data loading functionality."""

    def test_load_test_data(self, config, mock_model, mock_tokenizer):
        """Test loading test data from file."""
        # Create sample test data
        test_data = [
            {
                'text': '你真笨',
                'toxicity': 'toxic',
                'bullying': 'harassment',
                'role': 'perpetrator',
                'emotion': 'neg',
                'emotion_strength': 3
            },
            {
                'text': '今天天氣很好',
                'toxicity': 'none',
                'bullying': 'none',
                'role': 'none',
                'emotion': 'neu',
                'emotion_strength': 1
            }
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_file = f.name

        try:
            with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

                evaluator = BullyingDetectionEvaluator(config)
                texts, labels = evaluator.load_test_data(temp_file)

                assert len(texts) == 2
                assert len(labels) == 2
                assert texts[0] == '你真笨'
                assert labels[0]['toxicity'] == 'toxic'
                assert labels[1]['emotion_strength'] == 1

        finally:
            os.unlink(temp_file)

    def test_malformed_data_handling(self, config, mock_model, mock_tokenizer):
        """Test handling of malformed test data."""
        # Create malformed test data
        malformed_data = [
            {'text': '測試'},  # Missing labels
            {'toxicity': 'toxic'},  # Missing text
            {},  # Empty record
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            for item in malformed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_file = f.name

        try:
            with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

                evaluator = BullyingDetectionEvaluator(config)

                # Should handle gracefully with default values
                texts, labels = evaluator.load_test_data(temp_file)

                assert len(texts) == 3
                assert len(labels) == 3

                # Check default values are applied
                for label_dict in labels:
                    assert 'toxicity' in label_dict
                    assert 'emotion' in label_dict
                    assert 'emotion_strength' in label_dict

        finally:
            os.unlink(temp_file)


# Integration tests
class TestIntegration(TestBullyingDetector):
    """Integration tests for the full evaluation pipeline."""

    @pytest.mark.slow
    def test_full_evaluation_pipeline(self, config, sample_texts, sample_labels):
        """Test the complete evaluation pipeline end-to-end."""
        # This test requires actual model files and is marked as slow
        # Skip if model files don't exist
        if not os.path.exists(config.MODEL_SAVE_PATH):
            pytest.skip("Model files not available for integration test")

        evaluator = BullyingDetectionEvaluator(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            results = evaluator.evaluate_comprehensive(
                sample_texts, sample_labels, temp_dir
            )

            # Check all expected files are created
            expected_files = [
                'evaluation_results.json',
                'evaluation_report.html',
                'confusion_matrices.png',
                'performance_comparison.png',
                'error_distribution.png',
                'robustness_results.png',
                'fairness_metrics.png'
            ]

            for filename in expected_files:
                filepath = os.path.join(temp_dir, filename)
                assert os.path.exists(filepath), f"Expected file {filename} not created"

            # Check result structure and content
            assert isinstance(results, dict)
            assert all(key in results for key in [
                'basic_metrics', 'error_analysis', 'robustness',
                'bias_evaluation', 'metadata'
            ])


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.mark.benchmark
    def test_prediction_throughput(self, benchmark, config, mock_model, mock_tokenizer):
        """Benchmark prediction throughput."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)
            test_texts = ["測試文本"] * 100

            def predict():
                return evaluator.predict_batch(test_texts)

            result = benchmark(predict)
            assert len(result) == 100

    @pytest.mark.benchmark
    def test_evaluation_performance(self, benchmark, config, mock_model, mock_tokenizer):
        """Benchmark full evaluation performance."""
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            evaluator = BullyingDetectionEvaluator(config)
            test_texts = ["測試文本"] * 50
            test_labels = [{'toxicity': 'none', 'bullying': 'none', 'role': 'none', 'emotion': 'neu', 'emotion_strength': 0}] * 50

            with tempfile.TemporaryDirectory() as temp_dir:
                # Mock the expensive components for benchmarking
                with patch.object(evaluator, 'metrics_calculator') as mock_metrics, \
                     patch.object(evaluator, 'error_analyzer') as mock_error, \
                     patch.object(evaluator, 'robustness_evaluator') as mock_robustness, \
                     patch.object(evaluator, 'bias_evaluator') as mock_bias, \
                     patch.object(evaluator, 'report_generator') as mock_report:

                    mock_metrics.calculate_all_metrics.return_value = {}
                    mock_error.analyze_errors.return_value = {}
                    mock_robustness.evaluate_robustness.return_value = {}
                    mock_bias.evaluate_bias.return_value = {}
                    mock_report.generate_report.return_value = None

                    def evaluate():
                        return evaluator.evaluate_comprehensive(test_texts, test_labels, temp_dir)

                    result = benchmark(evaluate)
                    assert isinstance(result, dict)


if __name__ == '__main__':
    # Run tests with coverage
    pytest.main([
        __file__,
        '-v',
        '--cov=src.cyberpuppy',
        '--cov-report=html',
        '--cov-report=term-missing',
        '--benchmark-only',
        '--benchmark-sort=mean'
    ])