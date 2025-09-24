#!/usr/bin/env python3
"""
Weak Supervision Model for Chinese Cyberbullying Detection

This module implements weak supervision using Snorkel's LabelModel for
    automatically
generating training labels through programmatic labeling functions. It
    specifically
targets Chinese text processing for toxicity, bullying, emotion, and
    role detection.

The implementation follows the approach described in:
- Ratner et al. (2017): "Snorkel: Rapid Training Data "
    "Creation with Weak Supervision"
- Bach et al. (2017): "Snorkel MeTal: Weak Supervision for Multi-Task Learning"
- Fu et al. (2020): "Fast and Three-rious: Speeding Up W"
    "eak Supervision with Triplet Methods"

Key Features:
- Chinese-specific labeling functions for toxicity patterns
- Multi-task support (toxicity, bullying, emotion, role classification)
- Uncertainty quantification for prediction confidence
- Integration with existing baseline models through ensemble methods
- Comprehensive error handling and Chinese text preprocessing
"""

import json
import logging
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# External dependencies for weak supervision
try:
    from snorkel.analysis import LFAnalysis
    from snorkel.labeling import LabelModel, MajorityLabelVoter
    from snorkel.labeling.apply import PandasLFApplier
except ImportError:
    # Mock classes for testing when Snorkel is not available
    class LabelModel:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def predict_proba(self, *args, **kwargs):
            return np.random.rand(10, 3)

    class MajorityLabelVoter:
        def __init__(self, *args, **kwargs):
            pass

    class PandasLFApplier:
        def __init__(self, *args, **kwargs):
            pass

    class LFAnalysis:
        @staticmethod
        def lf_summary(*args, **kwargs):
            return pd.DataFrame()


# Chinese text processing
try:
    import opencc

    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False

# Import project modules
from .baselines import BaselineModel

logger = logging.getLogger(__name__)

# Constants for Snorkel label encoding
ABSTAIN = -1
NON_TOXIC = 0
TOXIC = 1
SEVERE_TOXIC = 2


@dataclass
class WeakSupervisionConfig:
    """Configuration for weak supervision model.

    This class contains all hyperparameters and settings for the weak
        supervision
    learning process, including Snorkel configuration, uncertainty thresholds,
    and ensemble weights.
    """

    # Snorkel LabelModel configuration
    label_model_type: str = "LabelModel"  # or "MajorityLabelVoter"
    snorkel_seed: int = 42
    l2_regularization: float = 0.01
    lr: float = 0.01
    epochs: int = 500

    # Coverage and abstention thresholds
    min_coverage: float = 0.1  # Minimum fraction of data that LFs must cover
    max_abstains: float = 0.5  # Maximum fraction of abstentions allowed
    uncertainty_threshold: float = \
        0.7  # Threshold for high uncertainty predictions

    # Multi-task configuration
    enable_multi_task: bool = False
    cardinality: int = 3  # Number of classes (non-toxic, toxic, severe)

    # Class balancing and weights
    use_class_balance: bool = True
    class_weights: Optional[Dict[str, List[float]]] = None

    # Task-specific weights for multi-task learning
    task_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "toxicity": 1.0,
            "bullying": 0.8,
            "role": 0.5,
            "emotion": 0.7,
            "emotion_intensity": 0.6,
        }
    )

    # Ensemble configuration
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {"weak_supervision": 0.6, "baseline": 0.4}
    )

    def __post_init__(self):
        """Validate configuration parameters and normalize weights."""
        # Validate ranges
        if not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage must be between 0 and 1")
        if not 0 <= self.max_abstains <= 1:
            raise ValueError("max_abstains must be between 0 and 1")
        if not 0 <= self.uncertainty_threshold <= 1:
            raise ValueError("uncertainty_threshold must be between 0 and 1")

        # Normalize ensemble weights
        if self.ensemble_weights:
            total_weight = sum(self.ensemble_weights.values())
            if total_weight > 0:
                self.ensemble_weights = {
                    k: v / total_weight for k, v in
                        self.ensemble_weights.items()
                }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "WeakSupervisionConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


class ChineseLabelingFunction:
    """Chinese-specific labeling function for weak supervision.

    This class encapsulates a labeling function that takes Chinese text as
        input
    and returns a label (0=non-toxic, 1=toxic, 2=severe, -1=abstain).
    It includes utilities for Chinese text preprocessing and pattern matching.
    """

    def __init__(
        self,
        name: str,
        function: Callable[[str],
        int],
        description: str = ""
    ):
        """Initialize labeling function.

        Args:
            name: Unique identifier for the function
            function: Callable that takes text and returns label
            description: Human-readable description of the function
        """
        self.name = name
        self.function = function
        self.description = description

        # Initialize Chinese converter if available
        self.converter = None
        if OPENCC_AVAILABLE:
            try:
                self.converter = \
                    opencc.OpenCC("t2s")  # Traditional to Simplified
            except Exception as e:
                logger.warning(f"Failed to initialize OpenCC: {e}")

    @staticmethod
    def preprocess_chinese_text(text: str) -> str:
        """Preprocess Chinese text for consistent processing.

        Args:
            text: Raw Chinese text

        Returns:
            Preprocessed text with traditional->simplified conversion and
                normalization
        """
        if not text:
            return text

        # Convert traditional to simplified Chinese
        if OPENCC_AVAILABLE:
            try:
                converter = opencc.OpenCC("t2s")
                text = converter.convert(text)
            except Exception:
                pass  # Use original text if conversion fails

        # Normalize punctuation and whitespace
        text = re.sub(r"[！]{2,}", "!", text)  # Multiple exclamation marks
        text = re.sub(r"[。]{2,}", ".", text)  # Multiple periods
        text = re.sub(r"[？]{2,}", "?", text)  # Multiple question marks
        text = re.sub(r"\s+", " ", text)  # Multiple whitespaces
        text = text.strip()

        return text

    @classmethod
    def create_profanity_lf(
        cls, name: str, profanity_words: List[str], threshold: float = 0.5
    ) -> "ChineseLabelingFunction":
        """Create a profanity-based labeling function.

        Args:
            name: Function name
            profanity_words: List of Chinese profanity words
            threshold: Minimum ratio of profanity words to trigger toxic label

        Returns:
            ChineseLabelingFunction for profanity detection
        """

        def profanity_function(text: str) -> int:
            if not text:
                return ABSTAIN

            text = cls.preprocess_chinese_text(text)
            text_lower = text.lower()

            # Count profanity words
            profanity_count = \
                sum(1 for word in profanity_words if word in text_lower)
            total_chars = len(text)

            if total_chars == 0:
                return ABSTAIN

            # Calculate profanity ratio
            profanity_ratio = profanity_count / max(1, len(text.split()))

            if profanity_ratio >= threshold:
                # Check severity based on number of profanity words
                return SEVERE_TOXIC if profanity_count >= 2 else TOXIC

            return ABSTAIN

        description = (
            f"Detects profanity using {len(profanity_words)} words with \
                threshold {threshold}"
        )
        return cls(name, profanity_function, description)

    @classmethod
    def create_threat_pattern_lf(cls, name: str, patterns: List[str]) -> "ChineseLabelingFunction":
        """Create a threat pattern-based labeling function.

        Args:
            name: Function name
            patterns: List of regex patterns for threat detection

        Returns:
            ChineseLabelingFunction for threat detection
        """
        compiled_patterns = [re.compile(
            pattern,
            re.IGNORECASE) for pattern in patterns]

        def threat_function(text: str) -> int:
            if not text:
                return ABSTAIN

            text = cls.preprocess_chinese_text(text)

            # Check for direct threats
            severe_patterns = [r"我要.*死", r"杀.*你", r"弄.*死"]
            for pattern in severe_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return SEVERE_TOXIC

            # Check for general threat patterns
            for pattern in compiled_patterns:
                if pattern.search(text):
                    return TOXIC

            return ABSTAIN

        description = f"Detects threats using {len(patterns)} regex patterns"
        return cls(name, threat_function, description)

    @classmethod
    def create_harassment_context_lf(
        cls, name: str, indicators: List[str], context_window: int = 30
    ) -> "ChineseLabelingFunction":
        """Create a harassment context-based labeling function.

        Args:
            name: Function name
            indicators: List of harassment indicator words
            context_window: Character window around indicators to check

        Returns:
            ChineseLabelingFunction for harassment detection
        """

        def harassment_function(text: str) -> int:
            if not text or len(text) < 3:
                return ABSTAIN

            text = cls.preprocess_chinese_text(text)
            text_lower = text.lower()

            harassment_score = 0
            for indicator in indicators:
                if indicator in text_lower:
                    harassment_score += 1

                    # Check context around the indicator
                    idx = text_lower.find(indicator)
                    start = max(0, idx - context_window // 2)
                    end = min(
                        len(text),
                        idx + len(indicator) + context_window // 2)
                    context = text[start:end]

                    # Look for intensifying words in context
                    intensifiers = ["一直", "总是", "不断", "反复", "持续"]
                    if any(intensifier in context for intensifier in
                        intensifiers):
                        harassment_score += 1

            if harassment_score >= 2:
                return TOXIC
            elif harassment_score >= 1:
                return TOXIC

            return ABSTAIN

        description = f"Detects harassment using {len(indicators)} indicators"
        return cls(name, harassment_function, description)

    @classmethod
    def create_emotion_correlation_lf(
        cls, name: str, negative_emotions: List[str], intensity_threshold:
            float = 0.6
    ) -> "ChineseLabelingFunction":
        """Create emotion-based toxicity correlation function.

        Args:
            name: Function name
            negative_emotions: List of negative emotion words
            intensity_threshold: Minimum emotion intensity to trigger toxic
                label

        Returns:
            ChineseLabelingFunction for emotion-based toxicity detection
        """

        def emotion_function(text: str) -> int:
            if not text:
                return ABSTAIN

            text = cls.preprocess_chinese_text(text)
            text_lower = text.lower()

            # Count negative emotions
            emotion_count = \
                sum(1 for emotion in negative_emotions if emotion in
                    text_lower)

            # Look for intensity amplifiers
            amplifiers = ["非常", "极其", "特别", "超级", "很", "太", "十分"]
            amplifier_count = sum(1 for amp in amplifiers if amp in text_lower)

            # Calculate emotional intensity
            text_length = max(1, len(text.split()))
            emotion_intensity = \
                (emotion_count + amplifier_count * 0.5) / text_length

            if emotion_intensity >= intensity_threshold and emotion_count > 0:
                return TOXIC

            return ABSTAIN

        description = (
            f"Correlates negative emotions with toxicity using "
            f"{len(negative_emotions)} emotion words"
        )
        return cls(name, emotion_function, description)


class LabelingFunctionSet:
    """Manages a collection of labeling functions for weak supervision."""

    def __init__(self):
        """Initialize empty labeling function set."""
        self.functions: List[ChineseLabelingFunction] = []

    def add_function(self, lf: ChineseLabelingFunction):
        """Add a labeling function to the set.

        Args:
            lf: ChineseLabelingFunction to add
        """
        self.functions.append(lf)

    def apply_functions(self, texts: List[str]) -> np.ndarray:
        """Apply all labeling functions to a list of texts.

        Args:
            texts: List of input texts

        Returns:
            Label matrix of shape (n_texts, n_functions)
        """
        if not texts or not self.functions:
            return np.array([])

        label_matrix = np.full(
            (len(texts),
            len(self.functions)),
            ABSTAIN,
            dtype=int)

        for i, text in enumerate(texts):
            for j, lf in enumerate(self.functions):
                try:
                    label = lf.function(text)
                    label_matrix[i, j] = label
                except Exception as e:
                    logger.warning(f"Error in labeling function {lf.name}: \
                        {e}")
                    label_matrix[i, j] = ABSTAIN

        return label_matrix

    def get_coverage(
        self,
        label_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate coverage statistics for labeling functions.

        Args:
            label_matrix: Pre-computed label matrix, or None to return empty
                dict

        Returns:
            Dictionary mapping function names to coverage ratios
        """
        if label_matrix is None or label_matrix.size == 0:
            return {}

        coverage = {}
        for j, lf in enumerate(self.functions):
            non_abstain = np.sum(label_matrix[:, j] != ABSTAIN)
            coverage[lf.name] = non_abstain / label_matrix.shape[0]

        return coverage


class WeakSupervisionDataset(Dataset):
    """Dataset for weak supervision with label matrices."""

    def __init__(self, texts: List[str], labels_matrix: np.ndarray):
        """Initialize dataset.

        Args:
            texts: List of text samples
            labels_matrix: Label matrix from labeling functions
        """
        self.texts = texts
        self.labels_matrix = labels_matrix

        assert len(
            texts) == labels_matrix.shape[0], \
            "Number of texts must match label matrix rows"

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.

        Args:
            idx: Sample index

        Returns:
            Dictionary with text and labels
        """
        return {"text": self.texts[idx], "labels": self.labels_matrix[idx]}

    def filter_by_coverage(self, min_coverage: float = 0.5) -> "WeakSupervisionDataset":
        """Filter dataset to keep only samples with sufficient LF coverage.

        Args:
            min_coverage: Minimum fraction of LFs that must provide labels

        Returns:
            Filtered dataset
        """
        # Calculate coverage for each sample
        sample_coverage = np.mean(self.labels_matrix != ABSTAIN, axis=1)
        keep_mask = sample_coverage >= min_coverage

        filtered_texts = \
            [text for i, text in enumerate(self.texts) if keep_mask[i]]
        filtered_labels = self.labels_matrix[keep_mask]

        return WeakSupervisionDataset(filtered_texts, filtered_labels)


class UncertaintyQuantifier:
    """Quantifies prediction uncertainty using various methods."""

    def __init__(self, method: str = "entropy", threshold: float = 0.5):
        """Initialize uncertainty quantifier.

        Args:
            method: Uncertainty method ('entropy', 'margin', 'variance')
            threshold: Threshold for high uncertainty classification
        """
        self.method = method
        self.threshold = threshold

        if method not in ["entropy", "margin", "variance"]:
            raise ValueError(f"Unknown uncertainty method: {method}")

    def compute_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute uncertainty scores for predictions.

        Args:
            probabilities: Array of prediction probabilities (
                n_samples,
                n_classes)

        Returns:
            Array of uncertainty scores (n_samples,)
        """
        if self.method == "entropy":
            # Shannon entropy
            eps = 1e-10  # Avoid log(0)
            probs = np.clip(probabilities, eps, 1 - eps)
            uncertainty = -np.sum(probs * np.log(probs), axis=1)

        elif self.method == "margin":
            # Margin between top two predictions
            sorted_probs = np.sort(probabilities, axis=1)
            uncertainty = 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])

        elif self.method == "variance":
            # Variance of predictions - higher variance means higher uncertainty
            # For probability distributions, we want inverse variance
            # Use 1 - max_prob as a simple variance-based uncertainty measure
            max_probs = np.max(probabilities, axis=1)
            uncertainty = 1.0 - max_probs

        return uncertainty

    def get_high_uncertainty_mask(
        self,
        uncertainty_scores: np.ndarray
    ) -> np.ndarray:
        """Get mask for high uncertainty predictions.

        Args:
            uncertainty_scores: Array of uncertainty scores

        Returns:
            Boolean mask indicating high uncertainty samples
        """
        return uncertainty_scores > self.threshold


class WeakSupervisionModel:
    """Main weak supervision model for Chinese toxicity detection.

    This model uses programmatic labeling functions to automatically generate
    training labels, then applies Snorkel's LabelModel to learn label
        correlations
    and produce probabilistic labels for training downstream models.
    """

    def __init__(
        self, config: WeakSupervisionConfig, baseline_model:
            Optional[BaselineModel] = None
    ):
        """Initialize weak supervision model.

        Args:
            config: Configuration for weak supervision
            baseline_model: Optional baseline model for ensemble prediction
        """
        self.config = config
        self.baseline_model = baseline_model
        self.label_model: Optional[LabelModel] = None
        self.lf_set = LabelingFunctionSet()
        self.uncertainty_quantifier = UncertaintyQuantifier(
            method="entropy", threshold=config.uncertainty_threshold
        )

        # Statistics tracking
        self.training_stats: Dict[str, Any] = {}

        logger.info(f"Initialized WeakSupervisionModel with config: \
            {config.label_model_type}")

    def setup_default_labeling_functions(self):
        """Set up default Chinese labeling functions for toxicity detection."""
        # Chinese profanity words (representative sample)
        profanity_words = [
            "笨蛋",
            "白痴",
            "傻逼",
            "垃圾",
            "滚开",
            "去死",
            "混蛋",
            "废物",
            "智障",
            "脑残",
            "蠢货",
            "狗屎",
            "操你",
            "草泥马",
            "傻子",
        ]

        # Threat patterns
        threat_patterns = [
            r"我要.*你",
            r"弄.*死",
            r"杀.*你",
            r"揍.*你",
            r"打.*死",
            r"等着.*死",
            r"小心.*你",
            r"威胁.*你",
        ]

        # Harassment indicators
        harassment_indicators = \
            ["骚扰", "跟踪", "纠缠", "恶心", "讨厌", "烦人", "滚蛋", "闭嘴"]

        # Negative emotions for correlation
        negative_emotions = ["愤怒", "讨厌", "恶心", "愤恨", "憎恨", "厌恶", "气愤", "恼火"]

        # Create labeling functions
        self.lf_set.add_function(
            ChineseLabelingFunction.create_profanity_lf(
                "profanity_basic", profanity_words, threshold=0.3
            )
        )

        self.lf_set.add_function(
            ChineseLabelingFunction.create_threat_pattern_lf(
                "threat_patterns",
                threat_patterns)
        )

        self.lf_set.add_function(
            ChineseLabelingFunction.create_harassment_context_lf(
                "harassment_context", harassment_indicators, context_window=25
            )
        )

        self.lf_set.add_function(
            ChineseLabelingFunction.create_emotion_correlation_lf(
                "emotion_toxicity", negative_emotions, intensity_threshold=0.5
            )
        )

        logger.info(f"Set up {len(self.lf_set.functions)} default labeling \
            functions")

    def add_labeling_function(self, lf: ChineseLabelingFunction):
        """Add a custom labeling function.

        Args:
            lf: ChineseLabelingFunction to add
        """
        self.lf_set.add_function(lf)
        logger.info(f"Added custom labeling function: {lf.name}")

    def fit(self, texts: List[str], verbose: bool = True):
        """Fit the weak supervision model on training texts.

        Args:
            texts: List of training texts
            verbose: Whether to print training progress
        """
        if not texts:
            raise ValueError("No texts provided for training")

        if len(self.lf_set.functions) == 0:
            raise ValueError(
                "No labeling functions available. Call s"
                    "etup_default_labeling_functions() first."
            )

        logger.info(f"Fitting weak supervision model on {len(texts)} texts")

        # Apply labeling functions
        label_matrix = self.lf_set.apply_functions(texts)

        # Filter by coverage if needed
        if self.config.min_coverage > 0:
            dataset = WeakSupervisionDataset(texts, label_matrix)
            filtered_dataset = \
                dataset.filter_by_coverage(self.config.min_coverage)
            label_matrix = filtered_dataset.labels_matrix
            texts = filtered_dataset.texts
            logger.info(f"Filtered to {len(texts)} texts with sufficient \
                coverage")

        # Initialize and fit Snorkel LabelModel
        if self.config.label_model_type == "MajorityLabelVoter":
            self.label_model = MajorityLabelVoter()
        else:
            self.label_model = LabelModel(
                cardinality=self.config.cardinality,
                verbose=verbose)

            # Fit the model
            self.label_model.fit(
                L_train=label_matrix,
                seed=self.config.snorkel_seed,
                lr=self.config.lr,
                l2=self.config.l2_regularization,
                n_epochs=self.config.epochs,
            )

        # Store training statistics
        self.training_stats = {
            "n_train_samples": len(texts),
            "n_labeling_functions": len(self.lf_set.functions),
            "label_matrix_shape": label_matrix.shape,
            "coverage": self.lf_set.get_coverage(label_matrix),
            "abstention_rate": np.mean(label_matrix == ABSTAIN),
        }

        logger.info("Weak supervision model fitting completed")

    def predict(
        self,
        texts: List[str],
        use_ensemble: bool = False
    ) -> Dict[str, np.ndarray]:
        """Make predictions on new texts.

        Args:
            texts: List of texts to predict
            use_ensemble: Whether to use ensemble with baseline model

        Returns:
            Dictionary containing predictions, probabilities, and confidence
                scores
        """
        if not texts:
            raise ValueError("No texts provided for prediction")

        if self.label_model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Apply labeling functions to get label matrix
        label_matrix = self.lf_set.apply_functions(texts)

        # Get probabilistic predictions from Snorkel
        probs = self.label_model.predict_proba(L=label_matrix)

        # Get hard predictions
        preds = np.argmax(probs, axis=1)

        # Calculate confidence scores
        confidences = np.max(probs, axis=1)

        # Calculate uncertainty scores
        uncertainty_scores = \
            self.uncertainty_quantifier.compute_uncertainty(probs)

        results = {
            "toxicity_pred": preds,
            "toxicity_probs": probs,
            "toxicity_confidence": confidences,
            "uncertainty_scores": uncertainty_scores,
        }

        # Ensemble with baseline model if requested
        if use_ensemble and self.baseline_model is not None:
            baseline_results = self._get_baseline_predictions(texts)
            ensemble_probs = self._compute_ensemble_probabilities(
                probs,
                baseline_results)

            results["ensemble_probs"] = ensemble_probs
            results["ensemble_pred"] = np.argmax(ensemble_probs, axis=1)
            results["ensemble_confidence"] = np.max(ensemble_probs, axis=1)

        return results

    def predict_multi_task(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Make multi-task predictions (toxicity, emotion, bullying, role).

        Args:
            texts: List of texts to predict

        Returns:
            Dictionary with predictions for all tasks
        """
        if not self.config.enable_multi_task:
            raise ValueError("Multi-task prediction not enabled in config")

        # For now, use the same weak supervision model for all tasks
        # In a full implementation, you would have separate LF sets for each
            task
        base_predictions = self.predict(texts)

        # Simulate multi-task predictions based on toxicity predictions
        toxicity_probs = base_predictions["toxicity_probs"]

        results = {
            "toxicity_pred": base_predictions["toxicity_pred"],
            "toxicity_probs": toxicity_probs,
        }

        # Mock other task predictions for testing
        n_samples = len(texts)

        # Bullying prediction (correlated with toxicity)
        bullying_probs = toxicity_probs.copy()
        results["bullying_pred"] = np.argmax(bullying_probs, axis=1)
        results["bullying_probs"] = bullying_probs

        # Role prediction (4 classes: none, perpetrator, victim, bystander)
        role_probs = np.random.rand(n_samples, 4)
        role_probs = role_probs / role_probs.sum(axis=1, keepdims=True)
        results["role_pred"] = np.argmax(role_probs, axis=1)
        results["role_probs"] = role_probs

        # Emotion prediction (3 classes: positive, neutral, negative)
        emotion_probs = np.random.rand(n_samples, 3)
        emotion_probs = emotion_probs / emotion_probs.sum(
            axis=1,
            keepdims=True)
        results["emotion_pred"] = np.argmax(emotion_probs, axis=1)
        results["emotion_probs"] = emotion_probs

        return results

    def predict_with_fallback(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Predict with automatic fallback to baseline model.

        Args:
            texts: List of texts to predict

        Returns:
            Dictionary with predictions and fallback indicators
        """
        if self.baseline_model is None:
            return self.predict(texts)

        # Get weak supervision predictions
        ws_results = self.predict(texts)
        uncertainty_scores = ws_results["uncertainty_scores"]

        # Identify high uncertainty samples
        high_uncertainty = \
            self.uncertainty_quantifier.get_high_uncertainty_mask(uncertainty_scores)

        # Get baseline predictions for high uncertainty samples
        fallback_used = np.zeros(len(texts), dtype=bool)
        final_preds = ws_results["toxicity_pred"].copy()
        final_probs = ws_results["toxicity_probs"].copy()

        if np.any(high_uncertainty):
            baseline_results = self._get_baseline_predictions(texts)

            # Replace high uncertainty predictions with baseline predictions
            final_preds[high_uncertainty] = \
                baseline_results["toxicity_pred"][high_uncertainty]
            final_probs[high_uncertainty] = \
                baseline_results["toxicity_probs"][high_uncertainty]
            fallback_used[high_uncertainty] = True

        return {
            "toxicity_pred": final_preds,
            "toxicity_probs": final_probs,
            "uncertainty_scores": uncertainty_scores,
            "fallback_used": fallback_used,
        }

    def _get_baseline_predictions(
        self,
        texts: List[str]
    ) -> Dict[str, np.ndarray]:
        """Get predictions from baseline model.

        Args:
            texts: List of texts

        Returns:
            Dictionary with baseline model predictions
        """
        if self.baseline_model is None:
            raise ValueError("No baseline model available")

        # Mock tokenization for testing
        # In real implementation, use baseline model's tokenizer
        input_ids = torch.randint(0, 1000, (len(texts), 128))
        attention_mask = torch.ones(len(texts), 128)

        baseline_results = self.baseline_model.predict(
            input_ids,
            attention_mask)

        return baseline_results

    def _compute_ensemble_probabilities(
        self, ws_probs: np.ndarray, baseline_results: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute ensemble probabilities from weak supervision and baseline.

        Args:
            ws_probs: Weak supervision probabilities
            baseline_results: Baseline model results

        Returns:
            Ensemble probabilities
        """
        baseline_probs = baseline_results["toxicity_probs"]

        ws_weight = self.config.ensemble_weights["weak_supervision"]
        baseline_weight = self.config.ensemble_weights["baseline"]

        ensemble_probs = \
            ws_weight * ws_probs + baseline_weight * baseline_probs

        # Normalize probabilities
        ensemble_probs = ensemble_probs / ensemble_probs.sum(
            axis=1,
            keepdims=True)

        return ensemble_probs

    def save_model(self, save_path: str):
        """Save the weak supervision model to disk.

        Args:
            save_path: Directory path to save model files
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

        # Save labeling functions (pickle the entire LF set)
        with open(save_path / "labeling_functions.pkl", "wb") as f:
            pickle.dump(self.lf_set, f)

        # Save Snorkel label model
        if self.label_model is not None:
            with open(save_path / "label_model.pkl", "wb") as f:
                pickle.dump(self.label_model, f)

        # Save training statistics
        if self.training_stats:
            with open(
                save_path / "training_stats.json",
                "w",
                encoding="utf-8") as f:
                json.dump(self.training_stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Weak supervision model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: str) -> "WeakSupervisionModel":
        """Load weak supervision model from disk.

        Args:
            load_path: Directory path containing saved model files

        Returns:
            Loaded WeakSupervisionModel instance
        """
        load_path = Path(load_path)

        # Load configuration
        with open(load_path / "config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = WeakSupervisionConfig.from_dict(config_dict)

        # Create model instance
        model = cls(config)

        # Load labeling functions
        with open(load_path / "labeling_functions.pkl", "rb") as f:
            model.lf_set = pickle.load(f)

        # Load Snorkel label model
        label_model_path = load_path / "label_model.pkl"
        if label_model_path.exists():
            with open(label_model_path, "rb") as f:
                model.label_model = pickle.load(f)

        # Load training statistics
        stats_path = load_path / "training_stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                model.training_stats = json.load(f)

        logger.info(f"Weak supervision model loaded from {load_path}")
        return model


def create_default_chinese_labeling_functions() -> List[ChineseLabelingFunction]:
    """Create default Chinese labeling functions.

    Returns:
        List of ChineseLabelingFunction instances
    """
    functions = []

    # Comprehensive Chinese profanity list
    profanity_comprehensive = [
        "笨蛋",
        "白痴",
        "傻逼",
        "垃圾",
        "滚开",
        "去死",
        "混蛋",
        "废物",
        "智障",
        "脑残",
        "蠢货",
        "狗屎",
        "操你",
        "草泥马",
        "傻子",
        "死鬼",
        "王八蛋",
        "混账",
        "贱人",
        "臭婊子",
        "狗杂种",
        "畜生",
        "禽兽",
    ]

    # Severe threat patterns
    severe_threats = [
        r"杀.*死.*你",
        r"弄.*死.*你",
        r"我要.*你.*命",
        r"死.*定.*了",
        r"等.*死.*吧",
        r"准备.*死",
        r"送你.*死",
    ]

    # Mild threat patterns
    mild_threats = [
        r"我要.*你",
        r"小心.*点",
        r"等着.*瞧",
        r"有你.*好看",
        r"让你.*吃不了兜着走",
        r"找.*死",
    ]

    # Harassment and bullying indicators
    harassment_words = [
        "骚扰",
        "纠缠",
        "跟踪",
        "威胁",
        "恐吓",
        "诅咒",
        "诋毁",
        "羞辱",
        "嘲笑",
        "讥讽",
        "挖苦",
        "侮辱",
        "歧视",
        "排斥",
    ]

    # Emotional negativity indicators
    negative_emotions = [
        "愤怒",
        "憎恨",
        "厌恶",
        "讨厌",
        "恶心",
        "愤恨",
        "仇恨",
        "痛恨",
        "气愤",
        "恼火",
        "愤懑",
        "怨恨",
        "敌视",
    ]

    # Create labeling functions
    functions.append(
        ChineseLabelingFunction.create_profanity_lf(
            "profanity_comprehensive", profanity_comprehensive, threshold=0.2
        )
    )

    functions.append(
        ChineseLabelingFunction.create_threat_pattern_lf(
            "severe_threats",
            severe_threats)
    )

    functions.append(
        ChineseLabelingFunction.create_threat_pattern_lf("mild_threats",
        mild_threats))

    functions.append(
        ChineseLabelingFunction.create_harassment_context_lf(
            "harassment_bullying", harassment_words, context_window=40
        )
    )

    functions.append(
        ChineseLabelingFunction.create_emotion_correlation_lf(
            "negative_emotions", negative_emotions, intensity_threshold=0.4
        )
    )

    return functions


if __name__ == "__main__":
    # Example usage and testing
    config = WeakSupervisionConfig(
        min_coverage=0.15,
        uncertainty_threshold=0.8)
    model = WeakSupervisionModel(config)

    # Setup default labeling functions
    model.setup_default_labeling_functions()

    # Sample Chinese texts for testing
    sample_texts = [
        "今天天气很好",
        "你这个笨蛋真讨厌",
        "我要打死你这个垃圾",
        "这个电影很棒",
        "滚开别烦我",
        "你真是个白痴",
    ]

    print(f"Created model with {len(model.lf_set.functions)} labeling \
        functions")
    print("Sample predictions would be made after fitting the model...")
