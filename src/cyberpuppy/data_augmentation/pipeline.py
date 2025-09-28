"""
Augmentation Pipeline for Cyberbullying Detection Data

Orchestrates multiple augmentation strategies with configurable intensity,
label consistency validation, and batch processing support.

Features:
- Multi-strategy augmentation pipeline
- Label consistency preservation
- Quality validation and filtering
- Batch processing with memory optimization
- Configurable augmentation intensity
- Progress tracking and logging
"""

import logging
import multiprocessing as mp
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .augmenters import (AugmentationConfig, BackTranslationAugmenter,
                         BaseAugmenter, ContextualAugmenter, EDAugmenter,
                         SynonymAugmenter, validate_augmentation_quality)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the augmentation pipeline."""

    # Augmentation strategies to use
    use_synonym: bool = True
    use_backtranslation: bool = True
    use_contextual: bool = True
    use_eda: bool = True

    # Intensity control
    augmentation_ratio: float = 0.3  # Proportion of original data to augment
    augmentations_per_text: int = 2  # Number of augmentations per selected text
    max_total_augmentations: int = 10000  # Maximum total augmentations

    # Quality control
    quality_threshold: float = 0.3  # Minimum similarity to original
    max_length_ratio: float = 2.0  # Maximum length increase ratio
    min_length_ratio: float = 0.5  # Minimum length decrease ratio

    # Label distribution control
    preserve_label_distribution: bool = True
    target_balance_ratio: float = 1.0  # For minority class upsampling

    # Processing options
    batch_size: int = 32
    num_workers: int = 4
    use_multiprocessing: bool = True
    random_seed: int = 42

    # Strategy weights (for random selection)
    strategy_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "synonym": 0.3,
            "backtranslation": 0.2,
            "contextual": 0.3,
            "eda": 0.2,
        }
    )


class AugmentationPipeline:
    """
    Main augmentation pipeline that orchestrates multiple augmentation strategies.
    """

    def __init__(
        self, config: PipelineConfig = None, augmentation_config: AugmentationConfig = None
    ):
        self.config = config or PipelineConfig()
        self.augmentation_config = augmentation_config or AugmentationConfig()

        # Set random seed for reproducibility
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # Initialize augmenters
        self.augmenters = self._initialize_augmenters()

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "total_augmented": 0,
            "quality_filtered": 0,
            "strategy_usage": defaultdict(int),
            "processing_time": 0,
        }

    def _initialize_augmenters(self) -> Dict[str, BaseAugmenter]:
        """Initialize all augmentation strategies."""
        augmenters = {}

        if self.config.use_synonym:
            augmenters["synonym"] = SynonymAugmenter(self.augmentation_config)
            logger.info("Initialized SynonymAugmenter")

        if self.config.use_backtranslation:
            augmenters["backtranslation"] = BackTranslationAugmenter(self.augmentation_config)
            logger.info("Initialized BackTranslationAugmenter")

        if self.config.use_contextual:
            augmenters["contextual"] = ContextualAugmenter(self.augmentation_config)
            logger.info("Initialized ContextualAugmenter")

        if self.config.use_eda:
            augmenters["eda"] = EDAugmenter(self.augmentation_config)
            logger.info("Initialized EDAugmenter")

        logger.info(f"Initialized {len(augmenters)} augmentation strategies")
        return augmenters

    def _select_augmentation_strategy(self) -> str:
        """Randomly select an augmentation strategy based on weights."""
        strategies = list(self.config.strategy_weights.keys())
        [self.config.strategy_weights.get(s, 0) for s in strategies]

        # Only consider available strategies
        available_strategies = [s for s in strategies if s in self.augmenters]
        available_weights = [self.config.strategy_weights[s] for s in available_strategies]

        if not available_strategies:
            raise ValueError("No augmentation strategies available")

        # Normalize weights
        total_weight = sum(available_weights)
        if total_weight == 0:
            return random.choice(available_strategies)

        probabilities = [w / total_weight for w in available_weights]
        return np.random.choice(available_strategies, p=probabilities)

    def _augment_single_text(
        self, text: str, label: Dict[str, Any], num_augmentations: int
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Augment a single text with label preservation."""
        augmented_pairs = []

        for _ in range(num_augmentations):
            # Select augmentation strategy
            strategy = self._select_augmentation_strategy()
            augmenter = self.augmenters[strategy]

            # Track strategy usage
            self.stats["strategy_usage"][strategy] += 1

            try:
                # Generate augmentation
                augmented_texts = augmenter.augment(text, num_augmentations=1)

                for aug_text in augmented_texts:
                    # Validate quality
                    if self._validate_augmented_text(text, aug_text):
                        # Preserve original labels
                        augmented_pairs.append((aug_text, label.copy()))
                    else:
                        self.stats["quality_filtered"] += 1
                        logger.debug(f"Filtered low-quality augmentation: {aug_text[:50]}...")

            except Exception as e:
                logger.warning(f"Augmentation failed with {strategy}: {e}")
                continue

        return augmented_pairs

    def _validate_augmented_text(self, original: str, augmented: str) -> bool:
        """Validate augmented text quality."""
        # Basic quality checks
        if not augmented.strip():
            return False

        # Length ratio check
        length_ratio = len(augmented) / len(original) if len(original) > 0 else 0
        if not (self.config.min_length_ratio <= length_ratio <= self.config.max_length_ratio):
            return False

        # Semantic similarity check
        return validate_augmentation_quality(
            original, augmented, threshold=self.config.quality_threshold
        )

    def _process_batch(
        self, batch_data: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Process a batch of texts for augmentation."""
        augmented_batch = []

        for text, label in batch_data:
            # Determine number of augmentations for this text
            num_augs = min(
                self.config.augmentations_per_text,
                self.config.max_total_augmentations - len(augmented_batch),
            )

            if num_augs <= 0:
                break

            # Augment text
            augmented_pairs = self._augment_single_text(text, label, num_augs)
            augmented_batch.extend(augmented_pairs)

            self.stats["total_processed"] += 1

        return augmented_batch

    def _balance_dataset(
        self, texts: List[str], labels: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Balance dataset by upsampling minority classes."""
        if not self.config.preserve_label_distribution:
            return texts, labels

        # Group by toxicity label (primary label for cyberbullying)
        label_groups = defaultdict(list)
        for i, (_text, label) in enumerate(zip(texts, labels)):
            toxicity = label.get("toxicity", "none")
            label_groups[toxicity].append(i)

        # Find majority class size
        max_size = max(len(indices) for indices in label_groups.values())
        target_size = int(max_size * self.config.target_balance_ratio)

        balanced_indices = []
        for toxicity_level, indices in label_groups.items():
            if len(indices) < target_size:
                # Upsample minority class
                upsampled_indices = np.random.choice(
                    indices, size=target_size, replace=True
                ).tolist()
                balanced_indices.extend(upsampled_indices)
                logger.info(f"Upsampled {toxicity_level} from {len(indices)} to {target_size}")
            else:
                balanced_indices.extend(indices)

        # Create balanced dataset
        balanced_texts = [texts[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]

        return balanced_texts, balanced_labels

    def augment(
        self, texts: List[str], labels: List[Dict[str, Any]], verbose: bool = True
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Augment a dataset of texts with their labels.

        Args:
            texts: List of input texts
            labels: List of label dictionaries
            verbose: Whether to show progress bars

        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length")

        logger.info(f"Starting augmentation of {len(texts)} texts")

        # Balance dataset if requested
        if self.config.preserve_label_distribution:
            texts, labels = self._balance_dataset(texts, labels)

        # Select texts for augmentation
        num_to_augment = int(len(texts) * self.config.augmentation_ratio)
        selected_indices = random.sample(range(len(texts)), min(num_to_augment, len(texts)))

        selected_data = [(texts[i], labels[i]) for i in selected_indices]
        logger.info(f"Selected {len(selected_data)} texts for augmentation")

        # Process in batches
        all_augmented = []
        batches = [
            selected_data[i : i + self.config.batch_size]
            for i in range(0, len(selected_data), self.config.batch_size)
        ]

        if verbose:
            batches = tqdm(batches, desc="Augmenting batches")

        if self.config.use_multiprocessing and len(batches) > 1:
            # Multiprocessing for large datasets
            with mp.Pool(processes=self.config.num_workers) as pool:
                batch_results = pool.map(self._process_batch, batches)
                for batch_result in batch_results:
                    all_augmented.extend(batch_result)
        else:
            # Sequential processing
            for batch in batches:
                batch_result = self._process_batch(batch)
                all_augmented.extend(batch_result)

        # Combine original and augmented data
        final_texts = texts + [pair[0] for pair in all_augmented]
        final_labels = labels + [pair[1] for pair in all_augmented]

        # Update statistics
        self.stats["total_augmented"] = len(all_augmented)

        logger.info(f"Augmentation complete. Generated {len(all_augmented)} new samples")
        logger.info(f"Final dataset size: {len(final_texts)} (was {len(texts)})")

        return final_texts, final_labels

    def augment_dataframe(
        self, df: pd.DataFrame, text_column: str, label_columns: List[str], verbose: bool = True
    ) -> pd.DataFrame:
        """
        Augment a pandas DataFrame containing text and labels.

        Args:
            df: Input DataFrame
            text_column: Name of the text column
            label_columns: List of label column names
            verbose: Whether to show progress

        Returns:
            Augmented DataFrame
        """
        # Extract texts and labels
        texts = df[text_column].tolist()
        labels = df[label_columns].to_dict("records")

        # Augment
        aug_texts, aug_labels = self.augment(texts, labels, verbose=verbose)

        # Create new DataFrame
        result_data = {text_column: aug_texts}
        for col in label_columns:
            result_data[col] = [label[col] for label in aug_labels]

        # Add augmentation flag
        is_augmented = [False] * len(texts) + [True] * (len(aug_texts) - len(texts))
        result_data["is_augmented"] = is_augmented

        return pd.DataFrame(result_data)

    def get_statistics(self) -> Dict[str, Any]:
        """Get augmentation statistics."""
        total_strategy_usage = sum(self.stats["strategy_usage"].values())
        strategy_percentages = {
            strategy: (count / total_strategy_usage * 100) if total_strategy_usage > 0 else 0
            for strategy, count in self.stats["strategy_usage"].items()
        }

        return {
            "total_processed": self.stats["total_processed"],
            "total_augmented": self.stats["total_augmented"],
            "quality_filtered": self.stats["quality_filtered"],
            "augmentation_ratio": (
                (self.stats["total_augmented"] / self.stats["total_processed"])
                if self.stats["total_processed"] > 0
                else 0
            ),
            "strategy_usage": dict(self.stats["strategy_usage"]),
            "strategy_percentages": strategy_percentages,
            "quality_pass_rate": (
                (
                    (self.stats["total_augmented"])
                    / (self.stats["total_augmented"] + self.stats["quality_filtered"])
                )
                if (self.stats["total_augmented"] + self.stats["quality_filtered"]) > 0
                else 0
            ),
        }

    def reset_statistics(self):
        """Reset augmentation statistics."""
        self.stats = {
            "total_processed": 0,
            "total_augmented": 0,
            "quality_filtered": 0,
            "strategy_usage": defaultdict(int),
            "processing_time": 0,
        }


def create_augmentation_pipeline(
    intensity: str = "medium", strategies: Optional[List[str]] = None
) -> AugmentationPipeline:
    """
    Factory function to create pre-configured augmentation pipelines.

    Args:
        intensity: 'light', 'medium', or 'heavy'
        strategies: List of strategies to use ['synonym', 'backtranslation', 'contextual', 'eda']

    Returns:
        Configured AugmentationPipeline
    """
    # Intensity presets
    intensity_configs = {
        "light": PipelineConfig(
            augmentation_ratio=0.1, augmentations_per_text=1, quality_threshold=0.5
        ),
        "medium": PipelineConfig(
            augmentation_ratio=0.3, augmentations_per_text=2, quality_threshold=0.3
        ),
        "heavy": PipelineConfig(
            augmentation_ratio=0.5, augmentations_per_text=3, quality_threshold=0.2
        ),
    }

    config = intensity_configs.get(intensity, intensity_configs["medium"])

    # Configure strategies
    if strategies:
        config.use_synonym = "synonym" in strategies
        config.use_backtranslation = "backtranslation" in strategies
        config.use_contextual = "contextual" in strategies
        config.use_eda = "eda" in strategies

    return AugmentationPipeline(config)
