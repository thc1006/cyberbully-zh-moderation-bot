"""
Comprehensive Tests for Data Augmentation Module

Tests all augmentation strategies and pipeline functionality:
- SynonymAugmenter with NTUSD sentiment dictionary
- BackTranslationAugmenter for Chinese-English diversity
- ContextualAugmenter with MacBERT masking
- EDAugmenter with random operations
- AugmentationPipeline orchestration
- Label consistency and quality validation
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import random
from typing import List, Dict, Any

import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberpuppy.data_augmentation import (
    SynonymAugmenter,
    BackTranslationAugmenter,
    ContextualAugmenter,
    EDAugmenter,
    AugmentationPipeline,
    AugmentationConfig,
    PipelineConfig,
    create_augmentation_pipeline,
    validate_augmentation_quality,
    calculate_text_similarity
)


class TestAugmentationConfig(unittest.TestCase):
    """Test AugmentationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AugmentationConfig()
        self.assertEqual(config.synonym_prob, 0.1)
        self.assertEqual(config.backtrans_prob, 0.3)
        self.assertEqual(config.contextual_prob, 0.15)
        self.assertEqual(config.eda_prob, 0.1)
        self.assertEqual(config.max_augmentations, 5)
        self.assertTrue(config.preserve_entities)
        self.assertEqual(config.quality_threshold, 0.7)

    def test_custom_config(self):
        """Test custom configuration."""
        config = AugmentationConfig(
            synonym_prob=0.2,
            quality_threshold=0.5
        )
        self.assertEqual(config.synonym_prob, 0.2)
        self.assertEqual(config.quality_threshold, 0.5)


class TestSynonymAugmenter(unittest.TestCase):
    """Test SynonymAugmenter functionality."""

    def setUp(self):
        self.config = AugmentationConfig(synonym_prob=1.0)  # Always replace for testing
        self.augmenter = SynonymAugmenter(self.config)

    def test_initialization(self):
        """Test augmenter initialization."""
        self.assertIsInstance(self.augmenter.synonym_dict, dict)
        self.assertGreater(len(self.augmenter.synonym_dict), 0)

    def test_augment_basic(self):
        """Test basic synonym augmentation."""
        text = "我很好"
        augmented = self.augmenter.augment(text, num_augmentations=1)

        self.assertEqual(len(augmented), 1)
        self.assertIsInstance(augmented[0], str)
        self.assertGreater(len(augmented[0]), 0)

    def test_augment_multiple(self):
        """Test multiple augmentations."""
        text = "我很好"
        augmented = self.augmenter.augment(text, num_augmentations=3)

        self.assertEqual(len(augmented), 3)
        for aug_text in augmented:
            self.assertIsInstance(aug_text, str)

    def test_empty_text(self):
        """Test handling of empty text."""
        augmented = self.augmenter.augment("", num_augmentations=2)
        self.assertEqual(len(augmented), 2)
        self.assertEqual(augmented[0], "")

    def test_preserve_special_tokens(self):
        """Test preservation of special tokens."""
        text = "你好 https://example.com @username #hashtag"
        processed, preserved = self.augmenter._preserve_special_tokens(text)

        self.assertIn("___URL_0___", processed)
        self.assertIn("___MENTION_0___", processed)
        self.assertIn("___HASHTAG_0___", processed)
        self.assertIn("https://example.com", preserved.values())

    def test_restore_special_tokens(self):
        """Test restoration of special tokens."""
        original = "你好 https://example.com"
        processed, preserved = self.augmenter._preserve_special_tokens(original)
        restored = self.augmenter._restore_special_tokens(processed, preserved)

        self.assertEqual(original, restored)

    def test_batch_augment(self):
        """Test batch augmentation."""
        texts = ["我很好", "你在哪裡", "今天天氣不錯"]
        batch_result = self.augmenter.batch_augment(texts, num_augmentations=2)

        self.assertEqual(len(batch_result), 3)
        for result in batch_result:
            self.assertEqual(len(result), 2)


class TestBackTranslationAugmenter(unittest.TestCase):
    """Test BackTranslationAugmenter functionality."""

    def setUp(self):
        self.config = AugmentationConfig(backtrans_prob=1.0)  # Always translate for testing
        self.augmenter = BackTranslationAugmenter(self.config)

    @patch('cyberpuppy.data_augmentation.augmenters.MarianMTModel')
    @patch('cyberpuppy.data_augmentation.augmenters.MarianTokenizer')
    def test_initialization(self, mock_tokenizer, mock_model):
        """Test augmenter initialization with mocked models."""
        augmenter = BackTranslationAugmenter()
        self.assertIsNone(augmenter._zh_en_model)
        self.assertIsNone(augmenter._en_zh_model)

    def test_translate_text_mock(self):
        """Test translation with mocked models."""
        # Mock the translation models
        with patch.object(self.augmenter, 'zh_en_model') as mock_zh_en, \
             patch.object(self.augmenter, 'en_zh_model') as mock_en_zh:

            # Setup mocks
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_zh_en.__get__ = Mock(return_value=(mock_model, mock_tokenizer))

            mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
            mock_model.generate.return_value = [[4, 5, 6]]
            mock_tokenizer.decode.return_value = "translated text"

            result = self.augmenter._translate_text("測試文本", "zh", "en")
            self.assertEqual(result, "translated text")

    def test_augment_with_mock(self):
        """Test augmentation with mocked translation."""
        with patch.object(self.augmenter, '_translate_text') as mock_translate:
            mock_translate.side_effect = ["test text", "測試文本2"]

            text = "測試文本"
            augmented = self.augmenter.augment(text, num_augmentations=1)

            self.assertEqual(len(augmented), 1)
            self.assertEqual(mock_translate.call_count, 2)  # zh->en, en->zh

    def test_translation_failure_handling(self):
        """Test handling of translation failures."""
        with patch.object(self.augmenter, '_translate_text', side_effect=Exception("Translation failed")):
            text = "測試文本"
            augmented = self.augmenter.augment(text, num_augmentations=1)

            # Should return original text on failure
            self.assertEqual(len(augmented), 1)


class TestContextualAugmenter(unittest.TestCase):
    """Test ContextualAugmenter functionality."""

    def setUp(self):
        self.config = AugmentationConfig(contextual_prob=0.5)
        self.augmenter = ContextualAugmenter(self.config)

    @patch('cyberpuppy.data_augmentation.augmenters.AutoModelForMaskedLM')
    @patch('cyberpuppy.data_augmentation.augmenters.AutoTokenizer')
    def test_initialization(self, mock_tokenizer, mock_model):
        """Test augmenter initialization with mocked models."""
        augmenter = ContextualAugmenter()
        self.assertIsNone(augmenter._model)
        self.assertIsNone(augmenter._tokenizer)

    def test_augment_with_mock(self):
        """Test contextual augmentation with mocked model."""
        with patch.object(self.augmenter, 'model') as mock_model_prop, \
             patch.object(self.augmenter, '_get_masked_predictions') as mock_predictions:

            # Setup mocks
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_model_prop.__get__ = Mock(return_value=(mock_model, mock_tokenizer))

            mock_tokenizer.tokenize.return_value = ['我', '很', '好']
            mock_tokenizer.convert_tokens_to_string.return_value = "我很棒"
            mock_predictions.return_value = [['棒', '讚', '優秀']]

            text = "我很好"
            augmented = self.augmenter.augment(text, num_augmentations=1)

            self.assertEqual(len(augmented), 1)
            self.assertIsInstance(augmented[0], str)

    def test_short_text_handling(self):
        """Test handling of very short texts."""
        with patch.object(self.augmenter, 'model') as mock_model_prop:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_model_prop.__get__ = Mock(return_value=(mock_model, mock_tokenizer))
            mock_tokenizer.tokenize.return_value = ['好']  # Too short

            text = "好"
            augmented = self.augmenter.augment(text, num_augmentations=1)

            self.assertEqual(len(augmented), 1)
            self.assertEqual(augmented[0], text)  # Should return original


class TestEDAugmenter(unittest.TestCase):
    """Test EDAugmenter functionality."""

    def setUp(self):
        self.config = AugmentationConfig(eda_prob=0.5)
        self.augmenter = EDAugmenter(self.config)

    def test_initialization(self):
        """Test augmenter initialization."""
        self.assertIsInstance(self.augmenter.function_words, list)
        self.assertGreater(len(self.augmenter.function_words), 0)

    def test_random_insertion(self):
        """Test random insertion operation."""
        words = ['我', '很', '好']
        augmented = self.augmenter._random_insertion(words, 1)

        self.assertEqual(len(augmented), 4)  # Original + 1 insertion
        self.assertTrue(any(word in self.augmenter.function_words for word in augmented))

    def test_random_deletion(self):
        """Test random deletion operation."""
        words = ['我', '很', '好', '的', '今天']
        augmented = self.augmenter._random_deletion(words, 0.5)

        self.assertLessEqual(len(augmented), len(words))
        self.assertGreater(len(augmented), 0)  # Should not delete all

    def test_random_deletion_single_word(self):
        """Test random deletion with single word."""
        words = ['好']
        augmented = self.augmenter._random_deletion(words, 0.9)

        self.assertEqual(len(augmented), 1)  # Should preserve single word

    def test_random_swap(self):
        """Test random swap operation."""
        words = ['我', '很', '好', '的', '今天']
        original_set = set(words)
        augmented = self.augmenter._random_swap(words, 1)

        self.assertEqual(len(augmented), len(words))
        self.assertEqual(set(augmented), original_set)  # Same words, different order

    def test_augment_basic(self):
        """Test basic EDA augmentation."""
        text = "我很好的今天"
        augmented = self.augmenter.augment(text, num_augmentations=1)

        self.assertEqual(len(augmented), 1)
        self.assertIsInstance(augmented[0], str)

    def test_augment_short_text(self):
        """Test EDA with very short text."""
        text = "好"
        augmented = self.augmenter.augment(text, num_augmentations=1)

        self.assertEqual(len(augmented), 1)
        self.assertEqual(augmented[0], text)  # Should return original


class TestQualityValidation(unittest.TestCase):
    """Test quality validation functions."""

    def test_calculate_text_similarity(self):
        """Test text similarity calculation."""
        # Identical texts
        sim = calculate_text_similarity("我很好", "我很好")
        self.assertEqual(sim, 1.0)

        # Completely different texts
        sim = calculate_text_similarity("我很好", "天氣不錯")
        self.assertLess(sim, 1.0)

        # Partially similar texts
        sim = calculate_text_similarity("我很好", "我很棒")
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_validate_augmentation_quality(self):
        """Test augmentation quality validation."""
        original = "我很好的今天"

        # Good quality augmentation
        good_aug = "我很棒的今天"
        self.assertTrue(validate_augmentation_quality(original, good_aug, threshold=0.2))

        # Poor quality augmentation (empty)
        self.assertFalse(validate_augmentation_quality(original, "", threshold=0.2))

        # Poor quality augmentation (too different)
        poor_aug = "完全不同的文本內容"
        self.assertFalse(validate_augmentation_quality(original, poor_aug, threshold=0.5))


class TestPipelineConfig(unittest.TestCase):
    """Test PipelineConfig dataclass."""

    def test_default_config(self):
        """Test default pipeline configuration."""
        config = PipelineConfig()
        self.assertTrue(config.use_synonym)
        self.assertTrue(config.use_backtranslation)
        self.assertTrue(config.use_contextual)
        self.assertTrue(config.use_eda)
        self.assertEqual(config.augmentation_ratio, 0.3)
        self.assertEqual(config.augmentations_per_text, 2)


class TestAugmentationPipeline(unittest.TestCase):
    """Test AugmentationPipeline functionality."""

    def setUp(self):
        # Create pipeline with mocked augmenters for testing
        self.config = PipelineConfig(
            use_backtranslation=False,  # Disable heavy augmenters for testing
            use_contextual=False,
            augmentation_ratio=0.5,
            augmentations_per_text=1
        )
        self.pipeline = AugmentationPipeline(self.config)

    def test_initialization(self):
        """Test pipeline initialization."""
        self.assertIsInstance(self.pipeline.augmenters, dict)
        self.assertIn('synonym', self.pipeline.augmenters)
        self.assertIn('eda', self.pipeline.augmenters)

    def test_select_augmentation_strategy(self):
        """Test strategy selection."""
        strategy = self.pipeline._select_augmentation_strategy()
        self.assertIn(strategy, self.pipeline.augmenters.keys())

    def test_augment_single_text(self):
        """Test single text augmentation."""
        text = "我很好"
        label = {'toxicity': 'none', 'emotion': 'pos'}

        with patch.object(self.pipeline, '_validate_augmented_text', return_value=True):
            augmented = self.pipeline._augment_single_text(text, label, 1)

            self.assertIsInstance(augmented, list)
            for aug_text, aug_label in augmented:
                self.assertIsInstance(aug_text, str)
                self.assertEqual(aug_label, label)

    def test_validate_augmented_text(self):
        """Test augmented text validation."""
        original = "我很好"

        # Valid augmentation
        self.assertTrue(self.pipeline._validate_augmented_text(original, "我很棒"))

        # Invalid augmentation (empty)
        self.assertFalse(self.pipeline._validate_augmented_text(original, ""))

        # Invalid augmentation (too long)
        long_text = "我" * 1000
        self.assertFalse(self.pipeline._validate_augmented_text(original, long_text))

    def test_augment_basic(self):
        """Test basic pipeline augmentation."""
        texts = ["我很好", "今天天氣不錯"]
        labels = [
            {'toxicity': 'none', 'emotion': 'pos'},
            {'toxicity': 'none', 'emotion': 'neu'}
        ]

        # Mock augmenters to avoid model loading
        with patch.object(self.pipeline.augmenters['synonym'], 'augment', return_value=["我很棒"]), \
             patch.object(self.pipeline.augmenters['eda'], 'augment', return_value=["今天很不錯的天氣"]):

            aug_texts, aug_labels = self.pipeline.augment(texts, labels, verbose=False)

            self.assertGreaterEqual(len(aug_texts), len(texts))
            self.assertEqual(len(aug_texts), len(aug_labels))

    def test_augment_dataframe(self):
        """Test DataFrame augmentation."""
        df = pd.DataFrame({
            'text': ["我很好", "今天天氣不錯"],
            'toxicity': ['none', 'none'],
            'emotion': ['pos', 'neu']
        })

        # Mock augmenters
        with patch.object(self.pipeline.augmenters['synonym'], 'augment', return_value=["我很棒"]), \
             patch.object(self.pipeline.augmenters['eda'], 'augment', return_value=["今天很不錯的天氣"]):

            result_df = self.pipeline.augment_dataframe(
                df, 'text', ['toxicity', 'emotion'], verbose=False
            )

            self.assertGreaterEqual(len(result_df), len(df))
            self.assertIn('is_augmented', result_df.columns)
            self.assertTrue(result_df['is_augmented'].any())

    def test_balance_dataset(self):
        """Test dataset balancing."""
        texts = ["好文本"] * 10 + ["壞文本"] * 2  # Imbalanced
        labels = [{'toxicity': 'none'}] * 10 + [{'toxicity': 'toxic'}] * 2

        balanced_texts, balanced_labels = self.pipeline._balance_dataset(texts, labels)

        # Should have more samples due to upsampling
        self.assertGreater(len(balanced_texts), len(texts))

    def test_get_statistics(self):
        """Test statistics collection."""
        stats = self.pipeline.get_statistics()

        required_keys = [
            'total_processed', 'total_augmented', 'quality_filtered',
            'augmentation_ratio', 'strategy_usage', 'strategy_percentages',
            'quality_pass_rate'
        ]

        for key in required_keys:
            self.assertIn(key, stats)

    def test_reset_statistics(self):
        """Test statistics reset."""
        # Set some stats
        self.pipeline.stats['total_processed'] = 100

        self.pipeline.reset_statistics()
        self.assertEqual(self.pipeline.stats['total_processed'], 0)


class TestFactoryFunction(unittest.TestCase):
    """Test factory function for creating pipelines."""

    def test_create_augmentation_pipeline_light(self):
        """Test creating light intensity pipeline."""
        pipeline = create_augmentation_pipeline('light')

        self.assertEqual(pipeline.config.augmentation_ratio, 0.1)
        self.assertEqual(pipeline.config.augmentations_per_text, 1)
        self.assertEqual(pipeline.config.quality_threshold, 0.5)

    def test_create_augmentation_pipeline_medium(self):
        """Test creating medium intensity pipeline."""
        pipeline = create_augmentation_pipeline('medium')

        self.assertEqual(pipeline.config.augmentation_ratio, 0.3)
        self.assertEqual(pipeline.config.augmentations_per_text, 2)
        self.assertEqual(pipeline.config.quality_threshold, 0.3)

    def test_create_augmentation_pipeline_heavy(self):
        """Test creating heavy intensity pipeline."""
        pipeline = create_augmentation_pipeline('heavy')

        self.assertEqual(pipeline.config.augmentation_ratio, 0.5)
        self.assertEqual(pipeline.config.augmentations_per_text, 3)
        self.assertEqual(pipeline.config.quality_threshold, 0.2)

    def test_create_with_specific_strategies(self):
        """Test creating pipeline with specific strategies."""
        pipeline = create_augmentation_pipeline('medium', ['synonym', 'eda'])

        self.assertTrue(pipeline.config.use_synonym)
        self.assertFalse(pipeline.config.use_backtranslation)
        self.assertFalse(pipeline.config.use_contextual)
        self.assertTrue(pipeline.config.use_eda)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete augmentation system."""

    def test_end_to_end_augmentation(self):
        """Test complete end-to-end augmentation workflow."""
        # Create sample data
        texts = ["我很好", "今天天氣不錯", "這個很有趣"]
        labels = [
            {'toxicity': 'none', 'emotion': 'pos'},
            {'toxicity': 'none', 'emotion': 'neu'},
            {'toxicity': 'none', 'emotion': 'pos'}
        ]

        # Create pipeline with minimal models to avoid loading issues
        config = PipelineConfig(
            use_backtranslation=False,
            use_contextual=False,
            augmentation_ratio=0.3,
            augmentations_per_text=1,
            use_multiprocessing=False
        )
        pipeline = AugmentationPipeline(config)

        # Mock the heavy augmenters to avoid model loading
        with patch.object(pipeline.augmenters['synonym'], 'augment') as mock_syn, \
             patch.object(pipeline.augmenters['eda'], 'augment') as mock_eda:

            mock_syn.return_value = ["我很棒"]
            mock_eda.return_value = ["今天很好的天氣"]

            # Run augmentation
            aug_texts, aug_labels = pipeline.augment(texts, labels, verbose=False)

            # Verify results
            self.assertGreaterEqual(len(aug_texts), len(texts))
            self.assertEqual(len(aug_texts), len(aug_labels))

            # Verify statistics
            stats = pipeline.get_statistics()
            self.assertGreater(stats['total_processed'], 0)

    def test_dataframe_workflow(self):
        """Test DataFrame-based workflow."""
        # Create sample DataFrame
        df = pd.DataFrame({
            'text': ["我很好", "今天天氣不錯"],
            'toxicity': ['none', 'none'],
            'bullying': ['none', 'none'],
            'emotion': ['pos', 'neu']
        })

        # Create simple pipeline
        config = PipelineConfig(
            use_backtranslation=False,
            use_contextual=False,
            augmentation_ratio=0.5,
            use_multiprocessing=False
        )
        pipeline = AugmentationPipeline(config)

        # Mock augmenters
        with patch.object(pipeline.augmenters['synonym'], 'augment', return_value=["我很棒"]), \
             patch.object(pipeline.augmenters['eda'], 'augment', return_value=["今天很好的天氣"]):

            result_df = pipeline.augment_dataframe(
                df, 'text', ['toxicity', 'bullying', 'emotion'], verbose=False
            )

            # Verify structure
            self.assertGreaterEqual(len(result_df), len(df))
            self.assertTrue(all(col in result_df.columns for col in df.columns))
            self.assertIn('is_augmented', result_df.columns)


if __name__ == '__main__':
    # Set random seeds for reproducible tests
    random.seed(42)
    np.random.seed(42)

    # Run tests
    unittest.main(verbosity=2)