"""
Comprehensive Data Augmentation Strategies for Chinese Cyberbullying Detection

This module implements four core augmentation strategies:
1. SynonymAugmenter - NTUSD sentiment dictionary-based synonym replacement
2. BackTranslationAugmenter - Chinese-English back-translation for diversity
3. ContextualAugmenter - MacBERT [MASK] prediction for contextual perturbation
4. EDAugmenter - Easy Data Augmentation (random insert/delete/swap)

All augmenters maintain label consistency and support configurable intensity.
"""

import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jieba
import opencc
import torch
from torch.nn.functional import softmax
from transformers import (AutoModelForMaskedLM, AutoTokenizer, MarianMTModel,
                          MarianTokenizer)

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for augmentation intensity and behavior."""

    synonym_prob: float = 0.1  # Probability of replacing each word with synonym
    backtrans_prob: float = 0.3  # Probability of applying back-translation
    contextual_prob: float = 0.15  # Probability of masking each token
    eda_prob: float = 0.1  # Probability for each EDA operation
    max_augmentations: int = 5  # Maximum augmentations per text
    preserve_entities: bool = True  # Preserve named entities and special tokens
    quality_threshold: float = 0.7  # Minimum quality score for augmented text


class BaseAugmenter(ABC):
    """Base class for all text augmenters."""

    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        self.converter = opencc.OpenCC("s2t")  # Simplified to Traditional Chinese

    @abstractmethod
    def augment(self, text: str, **kwargs) -> List[str]:
        """Augment a single text and return list of augmented versions."""
        pass

    def batch_augment(self, texts: List[str], **kwargs) -> List[List[str]]:
        """Augment a batch of texts."""
        return [self.augment(text, **kwargs) for text in texts]

    def _preserve_special_tokens(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Extract and preserve special tokens (URLs, mentions, etc.)."""
        special_patterns = {
            "url": r"https?://[^\s]+",
            "mention": r"@[a-zA-Z0-9_]+",
            "hashtag": r"#[^\s]+",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        }

        preserved = {}
        processed_text = text

        for token_type, pattern in special_patterns.items():
            matches = re.findall(pattern, text)
            for i, match in enumerate(matches):
                placeholder = f"___{token_type.upper()}_{i}___"
                preserved[placeholder] = match
                processed_text = processed_text.replace(match, placeholder, 1)

        return processed_text, preserved

    def _restore_special_tokens(self, text: str, preserved: Dict[str, str]) -> str:
        """Restore preserved special tokens."""
        for placeholder, original in preserved.items():
            text = text.replace(placeholder, original)
        return text


class SynonymAugmenter(BaseAugmenter):
    """
    Synonym replacement augmenter using NTUSD sentiment dictionary.
    Maintains semantic consistency by replacing words with sentiment-aware synonyms.
    """

    def __init__(self, config: AugmentationConfig = None, ntusd_path: str = None):
        super().__init__(config)
        self.ntusd_path = ntusd_path
        self.synonym_dict = self._load_ntusd_synonyms()

    def _load_ntusd_synonyms(self) -> Dict[str, List[str]]:
        """Load NTUSD sentiment dictionary and build synonym mappings."""
        # This would load the actual NTUSD dictionary
        # For now, we'll use a sample structure
        sample_synonyms = {
            # Positive synonyms
            "好": ["棒", "讚", "優秀", "良好", "不錯"],
            "喜歡": ["愛", "喜愛", "鍾愛", "偏愛"],
            "開心": ["快樂", "高興", "愉快", "歡喜"],
            # Negative synonyms (important for cyberbullying detection)
            "討厭": ["厭惡", "憎恨", "痛恨", "反感"],
            "笨": ["蠢", "愚笨", "愚蠢", "無腦"],
            "醜": ["難看", "不好看", "醜陋"],
            "垃圾": ["廢物", "渣滓", "廢料"],
            # Neutral synonyms
            "說": ["講", "表示", "提到", "談到"],
            "看": ["瞧", "觀看", "注視", "察看"],
            "想": ["認為", "覺得", "思考", "考慮"],
        }

        logger.info(f"Loaded {len(sample_synonyms)} synonym groups")
        return sample_synonyms

    def augment(self, text: str, num_augmentations: int = 1) -> List[str]:
        """Augment text using synonym replacement."""
        if not text.strip():
            return [text] * num_augmentations

        # Preserve special tokens
        processed_text, preserved = self._preserve_special_tokens(text)

        augmented_texts = []
        for _ in range(num_augmentations):
            # Segment text using jieba
            words = list(jieba.cut(processed_text))
            augmented_words = []

            for word in words:
                if word in self.synonym_dict and random.random() < self.config.synonym_prob:
                    # Replace with random synonym
                    synonyms = self.synonym_dict[word]
                    replacement = random.choice(synonyms)
                    augmented_words.append(replacement)
                    logger.debug(f"Replaced '{word}' with '{replacement}'")
                else:
                    augmented_words.append(word)

            augmented_text = "".join(augmented_words)
            # Restore special tokens
            augmented_text = self._restore_special_tokens(augmented_text, preserved)
            augmented_texts.append(augmented_text)

        return augmented_texts


class BackTranslationAugmenter(BaseAugmenter):
    """
    Back-translation augmenter for Chinese-English diversity.
    Translates Chinese -> English -> Chinese to introduce natural variations.
    """

    def __init__(self, config: AugmentationConfig = None):
        super().__init__(config)
        self.zh_en_model_name = "Helsinki-NLP/opus-mt-zh-en"
        self.en_zh_model_name = "Helsinki-NLP/opus-mt-en-zh"

        # Load models lazily to save memory
        self._zh_en_model = None
        self._zh_en_tokenizer = None
        self._en_zh_model = None
        self._en_zh_tokenizer = None

    @property
    def zh_en_model(self):
        if self._zh_en_model is None:
            self._zh_en_model = MarianMTModel.from_pretrained(self.zh_en_model_name)
            self._zh_en_tokenizer = MarianTokenizer.from_pretrained(self.zh_en_model_name)
        return self._zh_en_model, self._zh_en_tokenizer

    @property
    def en_zh_model(self):
        if self._en_zh_model is None:
            self._en_zh_model = MarianMTModel.from_pretrained(self.en_zh_model_name)
            self._en_zh_tokenizer = MarianTokenizer.from_pretrained(self.en_zh_model_name)
        return self._en_zh_model, self._en_zh_tokenizer

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between languages."""
        if source_lang == "zh" and target_lang == "en":
            model, tokenizer = self.zh_en_model
        elif source_lang == "en" and target_lang == "zh":
            model, tokenizer = self.en_zh_model
        else:
            raise ValueError(f"Unsupported translation: {source_lang} -> {target_lang}")

        try:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            # Generate translation
            with torch.no_grad():
                translated_tokens = model.generate(**inputs, max_length=512, num_beams=4)

            # Decode
            translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            return translated_text.strip()

        except Exception as e:
            logger.warning(f"Translation failed: {e}, returning original text")
            return text

    def augment(self, text: str, num_augmentations: int = 1) -> List[str]:
        """Augment text using back-translation."""
        if not text.strip():
            return [text] * num_augmentations

        augmented_texts = []

        for _ in range(num_augmentations):
            if random.random() < self.config.backtrans_prob:
                # Preserve special tokens
                processed_text, preserved = self._preserve_special_tokens(text)

                # Chinese -> English -> Chinese
                english_text = self._translate_text(processed_text, "zh", "en")
                back_translated = self._translate_text(english_text, "en", "zh")

                # Restore special tokens
                back_translated = self._restore_special_tokens(back_translated, preserved)

                logger.debug(f"Back-translated: {text[:50]}... -> {back_translated[:50]}...")
                augmented_texts.append(back_translated)
            else:
                augmented_texts.append(text)

        return augmented_texts


class ContextualAugmenter(BaseAugmenter):
    """
    Contextual perturbation using MacBERT [MASK] prediction.
    Uses contextual understanding to suggest semantically appropriate replacements.
    """

    def __init__(self, config: AugmentationConfig = None):
        super().__init__(config)
        self.model_name = "hfl/chinese-macbert-base"

        # Load model lazily
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.eval()
        return self._model, self._tokenizer

    def _get_masked_predictions(
        self, text: str, mask_positions: List[int], top_k: int = 5
    ) -> List[List[str]]:
        """Get top-k predictions for masked positions."""
        model, tokenizer = self.model

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        predictions = []
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            for pos in mask_positions:
                if pos < logits.shape[1]:
                    # Get top-k predictions for this position
                    probs = softmax(logits[0, pos], dim=-1)
                    top_k_indices = torch.topk(probs, k=top_k).indices

                    # Decode tokens
                    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
                    # Filter out special tokens
                    valid_tokens = [
                        token
                        for token in top_k_tokens
                        if not token.startswith("[") and token.strip()
                    ]
                    predictions.append(valid_tokens[:3])  # Keep top 3 valid
                else:
                    predictions.append([])

        return predictions

    def augment(self, text: str, num_augmentations: int = 1) -> List[str]:
        """Augment text using contextual masking and prediction."""
        if not text.strip():
            return [text] * num_augmentations

        # Preserve special tokens
        processed_text, preserved = self._preserve_special_tokens(text)

        model, tokenizer = self.model

        augmented_texts = []
        for _ in range(num_augmentations):
            # Tokenize for masking
            tokens = tokenizer.tokenize(processed_text)

            if len(tokens) < 3:  # Too short to mask meaningfully
                augmented_texts.append(text)
                continue

            # Determine positions to mask
            mask_positions = []
            for i, token in enumerate(tokens):
                if (
                    random.random() < self.config.contextual_prob
                    and not token.startswith("[")
                    and len(token.strip()) > 0
                ):
                    mask_positions.append(i)

            if not mask_positions:
                augmented_texts.append(text)
                continue

            # Create masked text
            masked_tokens = tokens.copy()
            for pos in mask_positions:
                masked_tokens[pos] = "[MASK]"

            masked_text = tokenizer.convert_tokens_to_string(masked_tokens)

            # Get predictions
            predictions = self._get_masked_predictions(masked_text, mask_positions)

            # Replace masks with predictions
            augmented_tokens = masked_tokens.copy()
            for i, (pos, preds) in enumerate(zip(mask_positions, predictions)):
                if preds:
                    replacement = random.choice(preds)
                    augmented_tokens[pos] = replacement
                    logger.debug(f"Replaced mask at {pos} with '{replacement}'")
                else:
                    # Restore original token if no good predictions
                    augmented_tokens[pos] = tokens[pos]

            augmented_text = tokenizer.convert_tokens_to_string(augmented_tokens)

            # Restore special tokens
            augmented_text = self._restore_special_tokens(augmented_text, preserved)
            augmented_texts.append(augmented_text)

        return augmented_texts


class EDAugmenter(BaseAugmenter):
    """
    Easy Data Augmentation (EDA) for Chinese text.
    Implements: Random Insertion, Random Deletion, Random Swap.
    """

    def __init__(self, config: AugmentationConfig = None):
        super().__init__(config)

        # Common Chinese function words for insertion
        self.function_words = [
            "的",
            "了",
            "在",
            "是",
            "我",
            "有",
            "和",
            "就",
            "不",
            "人",
            "都",
            "一",
            "個",
            "上",
            "也",
            "很",
            "到",
            "說",
            "要",
            "去",
            "你",
            "會",
            "著",
            "沒",
            "看",
            "好",
            "自己",
            "這樣",
            "能夠",
            "而且",
        ]

    def _random_insertion(self, words: List[str], n: int) -> List[str]:
        """Randomly insert n function words."""
        augmented_words = words.copy()

        for _ in range(n):
            # Choose random function word and position
            new_word = random.choice(self.function_words)
            random_idx = random.randint(0, len(augmented_words))
            augmented_words.insert(random_idx, new_word)

        return augmented_words

    def _random_deletion(self, words: List[str], p: float) -> List[str]:
        """Randomly delete words with probability p."""
        if len(words) == 1:
            return words

        augmented_words = []
        for word in words:
            if random.random() > p:
                augmented_words.append(word)

        # If all words deleted, return random word
        if not augmented_words:
            augmented_words = [random.choice(words)]

        return augmented_words

    def _random_swap(self, words: List[str], n: int) -> List[str]:
        """Randomly swap n pairs of words."""
        augmented_words = words.copy()

        for _ in range(n):
            if len(augmented_words) < 2:
                break

            # Choose two random positions
            idx1, idx2 = random.sample(range(len(augmented_words)), 2)

            # Swap
            augmented_words[idx1], augmented_words[idx2] = (
                augmented_words[idx2],
                augmented_words[idx1],
            )

        return augmented_words

    def augment(self, text: str, num_augmentations: int = 1) -> List[str]:
        """Augment text using EDA strategies."""
        if not text.strip():
            return [text] * num_augmentations

        # Preserve special tokens
        processed_text, preserved = self._preserve_special_tokens(text)

        # Segment text
        words = list(jieba.cut(processed_text))

        if len(words) < 2:
            return [text] * num_augmentations

        augmented_texts = []

        for _ in range(num_augmentations):
            augmented_words = words.copy()

            # Calculate number of operations based on text length
            n_ops = max(1, int(self.config.eda_prob * len(words)))

            # Apply random operations
            operations = ["insert", "delete", "swap"]
            chosen_op = random.choice(operations)

            if chosen_op == "insert":
                augmented_words = self._random_insertion(augmented_words, n_ops)
            elif chosen_op == "delete":
                augmented_words = self._random_deletion(augmented_words, self.config.eda_prob)
            elif chosen_op == "swap":
                augmented_words = self._random_swap(augmented_words, n_ops)

            augmented_text = "".join(augmented_words)

            # Restore special tokens
            augmented_text = self._restore_special_tokens(augmented_text, preserved)
            augmented_texts.append(augmented_text)

        return augmented_texts


# Quality assessment utilities
def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts (simplified version)."""
    # This is a simplified implementation
    # In practice, you might use sentence transformers or other embedding models

    words1 = set(jieba.cut(text1))
    words2 = set(jieba.cut(text2))

    if not words1 or not words2:
        return 0.0

    # Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0.0


def validate_augmentation_quality(original: str, augmented: str, threshold: float = 0.3) -> bool:
    """Validate that augmented text maintains minimum quality."""
    # Check similarity
    similarity = calculate_text_similarity(original, augmented)

    # Check length ratio
    length_ratio = len(augmented) / len(original) if len(original) > 0 else 0

    # Quality checks
    has_content = len(augmented.strip()) > 0
    reasonable_length = 0.5 <= length_ratio <= 2.0
    sufficient_similarity = similarity >= threshold

    return has_content and reasonable_length and sufficient_similarity
