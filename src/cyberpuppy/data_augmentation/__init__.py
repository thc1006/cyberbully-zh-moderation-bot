"""
CyberPuppy Data Augmentation Package

Comprehensive Chinese text augmentation strategies for cyberbullying detection.
Addresses synthetic label problems and improves data diversity and authenticity.

Available Augmenters:
- SynonymAugmenter: NTUSD-based synonym replacement
- BackTranslationAugmenter: Chinese-English back-translation
- ContextualAugmenter: MacBERT contextual perturbation
- EDAugmenter: Easy Data Augmentation (insert/delete/swap)

Usage:
    from cyberpuppy.data_augmentation import AugmentationPipeline

    pipeline = AugmentationPipeline()
    augmented_data = pipeline.augment(texts, labels, num_augmentations=3)
"""

from .augmenters import (
    SynonymAugmenter,
    BackTranslationAugmenter,
    ContextualAugmenter,
    EDAugmenter,
    AugmentationConfig,
    validate_augmentation_quality,
    calculate_text_similarity
)
from .pipeline import AugmentationPipeline, PipelineConfig, create_augmentation_pipeline
from .validation import (
    LabelConsistencyValidator,
    LabelConsistencyConfig,
    ValidationResult,
    QualityAssuranceReport,
    validate_augmented_dataset
)

__all__ = [
    'SynonymAugmenter',
    'BackTranslationAugmenter',
    'ContextualAugmenter',
    'EDAugmenter',
    'AugmentationConfig',
    'validate_augmentation_quality',
    'calculate_text_similarity',
    'AugmentationPipeline',
    'PipelineConfig',
    'create_augmentation_pipeline',
    'LabelConsistencyValidator',
    'LabelConsistencyConfig',
    'ValidationResult',
    'QualityAssuranceReport',
    'validate_augmented_dataset'
]

__version__ = '1.0.0'