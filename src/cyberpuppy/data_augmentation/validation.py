"""
Label Consistency Validation and Quality Checks

Ensures that data augmentation maintains label consistency and quality
for cyberbullying detection tasks. Implements comprehensive validation
strategies to prevent label drift and maintain dataset integrity.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

import jieba
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation checks."""
    is_valid: bool
    confidence: float
    violations: List[str]
    warnings: List[str]
    metrics: Dict[str, float]


@dataclass
class LabelConsistencyConfig:
    """Configuration for label consistency validation."""
    # Toxicity validation
    min_toxicity_confidence: float = 0.7  # Minimum confidence for toxicity labels
    toxicity_keywords: Dict[str, Set[str]] = None  # Keywords for each toxicity level

    # Bullying validation
    min_bullying_confidence: float = 0.7
    bullying_patterns: Dict[str, List[str]] = None  # Regex patterns for bullying types

    # Emotion validation
    min_emotion_confidence: float = 0.6
    emotion_keywords: Dict[str, Set[str]] = None

    # Role validation
    role_indicators: Dict[str, Set[str]] = None  # Indicators for perpetrator/victim/bystander

    # Quality thresholds
    max_length_change: float = 0.5  # Maximum relative length change
    min_semantic_similarity: float = 0.3  # Minimum semantic similarity
    max_label_drift: float = 0.1  # Maximum allowed label distribution drift

    def __post_init__(self):
        """Initialize default keyword sets if not provided."""
        if self.toxicity_keywords is None:
            self.toxicity_keywords = {
                'none': {'好', '棒', '讚', '優秀', '美好', '喜歡', '愛', '開心', '快樂'},
                'toxic': {'笨', '蠢', '討厭', '垃圾', '廢物', '醜', '噁心', '滾'},
                'severe': {'死', '殺', '打死', '去死', '自殺', '滾蛋', '混蛋', '王八蛋'}
            }

        if self.bullying_patterns is None:
            self.bullying_patterns = {
                'none': [],
                'harassment': [
                    r'一直.*說',  # "一直說你..."
                    r'每天.*罵',  # "每天罵你..."
                    r'故意.*欺負'  # "故意欺負..."
                ],
                'threat': [
                    r'我要.*你',  # "我要打你", "我要殺你"
                    r'等著.*吧',  # "等著瞧吧"
                    r'小心.*你'  # "小心你的..."
                ]
            }

        if self.emotion_keywords is None:
            self.emotion_keywords = {
                'pos': {'開心', '快樂', '高興', '愉快', '滿意', '喜歡', '愛', '棒', '好'},
                'neu': {'還好', '普通', '一般', '可以', '尚可', '算是', '應該', '大概'},
                'neg': {'難過', '生氣', '憤怒', '失望', '沮喪', '傷心', '討厭', '害怕'}
            }

        if self.role_indicators is None:
            self.role_indicators = {
                'perpetrator': {'我罵', '我打', '我欺負', '我要', '你給我'},
                'victim': {'我被', '欺負我', '罵我', '打我', '讓我很'},
                'bystander': {'看到', '聽到', '發現', '覺得', '他們', '大家'},
                'none': set()
            }


class LabelConsistencyValidator:
    """
    Validates label consistency for augmented cyberbullying detection data.
    """

    def __init__(self, config: LabelConsistencyConfig = None):
        self.config = config or LabelConsistencyConfig()

    def validate_single_sample(self, original_text: str, augmented_text: str,
                             original_labels: Dict[str, Any],
                             augmented_labels: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single augmented sample for label consistency.

        Args:
            original_text: Original text
            augmented_text: Augmented text
            original_labels: Original labels
            augmented_labels: Augmented labels (should be same as original)

        Returns:
            ValidationResult with validation status and details
        """
        violations = []
        warnings = []
        metrics = {}

        # Check if labels are actually the same (they should be)
        if original_labels != augmented_labels:
            violations.append("Labels have changed during augmentation")

        # Text quality checks
        length_ratio = len(augmented_text) / len(original_text) if len(original_text) > 0 else 0
        if abs(length_ratio - 1.0) > self.config.max_length_change:
            violations.append(f"Text length changed too much: {length_ratio:.2f}")

        metrics['length_ratio'] = length_ratio

        # Semantic consistency checks
        try:
            semantic_sim = self._calculate_semantic_consistency(original_text, augmented_text)
            metrics['semantic_similarity'] = semantic_sim

            if semantic_sim < self.config.min_semantic_similarity:
                violations.append(f"Semantic similarity too low: {semantic_sim:.3f}")
        except Exception as e:
            warnings.append(f"Could not calculate semantic similarity: {e}")

        # Label-specific validation
        if 'toxicity' in original_labels:
            toxicity_valid, toxicity_conf = self._validate_toxicity_consistency(
                augmented_text, original_labels['toxicity']
            )
            metrics['toxicity_confidence'] = toxicity_conf

            if not toxicity_valid:
                violations.append(f"Toxicity label inconsistent with text content")

        if 'bullying' in original_labels:
            bullying_valid, bullying_conf = self._validate_bullying_consistency(
                augmented_text, original_labels['bullying']
            )
            metrics['bullying_confidence'] = bullying_conf

            if not bullying_valid:
                violations.append(f"Bullying label inconsistent with text content")

        if 'emotion' in original_labels:
            emotion_valid, emotion_conf = self._validate_emotion_consistency(
                augmented_text, original_labels['emotion']
            )
            metrics['emotion_confidence'] = emotion_conf

            if not emotion_valid:
                violations.append(f"Emotion label inconsistent with text content")

        if 'role' in original_labels:
            role_valid, role_conf = self._validate_role_consistency(
                augmented_text, original_labels['role']
            )
            metrics['role_confidence'] = role_conf

            if not role_valid:
                violations.append(f"Role label inconsistent with text content")

        # Calculate overall confidence
        confidence_scores = [v for k, v in metrics.items() if k.endswith('_confidence')]
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5

        is_valid = len(violations) == 0

        return ValidationResult(
            is_valid=is_valid,
            confidence=overall_confidence,
            violations=violations,
            warnings=warnings,
            metrics=metrics
        )

    def validate_batch(self, original_texts: List[str], augmented_texts: List[str],
                      original_labels: List[Dict[str, Any]],
                      augmented_labels: List[Dict[str, Any]]) -> Tuple[List[ValidationResult], Dict[str, Any]]:
        """
        Validate a batch of augmented samples.

        Returns:
            Tuple of (individual_results, batch_statistics)
        """
        if not (len(original_texts) == len(augmented_texts) == len(original_labels) == len(augmented_labels)):
            raise ValueError("All input lists must have the same length")

        individual_results = []
        batch_stats = {
            'total_samples': len(original_texts),
            'valid_samples': 0,
            'violation_counts': defaultdict(int),
            'warning_counts': defaultdict(int),
            'average_confidence': 0.0,
            'label_distribution_drift': {}
        }

        for orig_text, aug_text, orig_labels, aug_labels in zip(
            original_texts, augmented_texts, original_labels, augmented_labels
        ):
            result = self.validate_single_sample(orig_text, aug_text, orig_labels, aug_labels)
            individual_results.append(result)

            if result.is_valid:
                batch_stats['valid_samples'] += 1

            for violation in result.violations:
                batch_stats['violation_counts'][violation] += 1

            for warning in result.warnings:
                batch_stats['warning_counts'][warning] += 1

        # Calculate batch statistics
        confidences = [r.confidence for r in individual_results]
        batch_stats['average_confidence'] = np.mean(confidences) if confidences else 0.0

        # Check for label distribution drift
        batch_stats['label_distribution_drift'] = self._calculate_distribution_drift(
            original_labels, augmented_labels
        )

        return individual_results, batch_stats

    def _calculate_semantic_consistency(self, text1: str, text2: str) -> float:
        """Calculate semantic consistency between two texts."""
        # Simple implementation using word overlap
        # In practice, you might use sentence embeddings
        words1 = set(jieba.cut(text1))
        words2 = set(jieba.cut(text2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _validate_toxicity_consistency(self, text: str, toxicity_label: str) -> Tuple[bool, float]:
        """Validate toxicity label consistency with text content."""
        words = set(jieba.cut(text.lower()))
        keyword_matches = defaultdict(int)

        # Count keyword matches for each toxicity level
        for level, keywords in self.config.toxicity_keywords.items():
            matches = len(words.intersection(keywords))
            keyword_matches[level] = matches

        total_matches = sum(keyword_matches.values())
        if total_matches == 0:
            return True, 0.5  # No clear indicators, assume valid

        # Calculate confidence based on keyword distribution
        expected_level_matches = keyword_matches[toxicity_label]
        confidence = expected_level_matches / total_matches

        is_valid = confidence >= self.config.min_toxicity_confidence

        return is_valid, confidence

    def _validate_bullying_consistency(self, text: str, bullying_label: str) -> Tuple[bool, float]:
        """Validate bullying label consistency with text content."""
        pattern_matches = defaultdict(int)

        # Check pattern matches for each bullying type
        for btype, patterns in self.config.bullying_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    pattern_matches[btype] += 1

        total_matches = sum(pattern_matches.values())
        if total_matches == 0:
            return True, 0.5  # No clear patterns, assume valid

        expected_matches = pattern_matches[bullying_label]
        confidence = expected_matches / total_matches

        is_valid = confidence >= self.config.min_bullying_confidence

        return is_valid, confidence

    def _validate_emotion_consistency(self, text: str, emotion_label: str) -> Tuple[bool, float]:
        """Validate emotion label consistency with text content."""
        words = set(jieba.cut(text.lower()))
        emotion_matches = defaultdict(int)

        # Count emotion keyword matches
        for emotion, keywords in self.config.emotion_keywords.items():
            matches = len(words.intersection(keywords))
            emotion_matches[emotion] = matches

        total_matches = sum(emotion_matches.values())
        if total_matches == 0:
            return True, 0.5  # No clear emotion indicators

        expected_matches = emotion_matches[emotion_label]
        confidence = expected_matches / total_matches

        is_valid = confidence >= self.config.min_emotion_confidence

        return is_valid, confidence

    def _validate_role_consistency(self, text: str, role_label: str) -> Tuple[bool, float]:
        """Validate role label consistency with text content."""
        role_matches = defaultdict(int)

        # Check for role indicators
        for role, indicators in self.config.role_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    role_matches[role] += 1

        total_matches = sum(role_matches.values())
        if total_matches == 0:
            return True, 0.5  # No clear role indicators

        expected_matches = role_matches[role_label]
        confidence = expected_matches / total_matches if total_matches > 0 else 0.0

        is_valid = confidence >= 0.5  # Lower threshold for role validation

        return is_valid, confidence

    def _calculate_distribution_drift(self, original_labels: List[Dict[str, Any]],
                                    augmented_labels: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate label distribution drift between original and augmented data."""
        drift_metrics = {}

        # Get all label keys
        all_keys = set()
        for labels in original_labels + augmented_labels:
            all_keys.update(labels.keys())

        for key in all_keys:
            # Extract values for this label key
            orig_values = [labels.get(key) for labels in original_labels if key in labels]
            aug_values = [labels.get(key) for labels in augmented_labels if key in labels]

            if not orig_values or not aug_values:
                continue

            # Calculate distribution difference
            orig_dist = Counter(orig_values)
            aug_dist = Counter(aug_values)

            # Normalize distributions
            orig_total = sum(orig_dist.values())
            aug_total = sum(aug_dist.values())

            orig_probs = {k: v/orig_total for k, v in orig_dist.items()}
            aug_probs = {k: v/aug_total for k, v in aug_dist.items()}

            # Calculate KL divergence or chi-square statistic
            all_labels = set(orig_probs.keys()) | set(aug_probs.keys())

            # Use chi-square test for independence
            try:
                orig_counts = [orig_dist.get(label, 0) for label in all_labels]
                aug_counts = [aug_dist.get(label, 0) for label in all_labels]

                if sum(orig_counts) > 0 and sum(aug_counts) > 0:
                    chi2, p_value = chi2_contingency([orig_counts, aug_counts])[:2]
                    drift_metrics[f'{key}_chi2'] = chi2
                    drift_metrics[f'{key}_p_value'] = p_value
            except Exception as e:
                logger.warning(f"Could not calculate drift for {key}: {e}")

        return drift_metrics


class QualityAssuranceReport:
    """Generate comprehensive quality assurance reports."""

    @staticmethod
    def generate_validation_report(validation_results: List[ValidationResult],
                                 batch_stats: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report."""
        report_lines = []

        report_lines.append("=" * 60)
        report_lines.append("DATA AUGMENTATION QUALITY ASSURANCE REPORT")
        report_lines.append("=" * 60)

        # Overall statistics
        total_samples = batch_stats['total_samples']
        valid_samples = batch_stats['valid_samples']
        validation_rate = valid_samples / total_samples if total_samples > 0 else 0

        report_lines.append(f"\nOVERALL STATISTICS:")
        report_lines.append(f"  Total samples: {total_samples:,}")
        report_lines.append(f"  Valid samples: {valid_samples:,}")
        report_lines.append(f"  Validation rate: {validation_rate:.2%}")
        report_lines.append(f"  Average confidence: {batch_stats['average_confidence']:.3f}")

        # Violation summary
        if batch_stats['violation_counts']:
            report_lines.append(f"\nVIOLATION SUMMARY:")
            for violation, count in sorted(batch_stats['violation_counts'].items(),
                                         key=lambda x: x[1], reverse=True):
                rate = count / total_samples
                report_lines.append(f"  {violation}: {count} ({rate:.1%})")

        # Warning summary
        if batch_stats['warning_counts']:
            report_lines.append(f"\nWARNING SUMMARY:")
            for warning, count in sorted(batch_stats['warning_counts'].items(),
                                       key=lambda x: x[1], reverse=True):
                rate = count / total_samples
                report_lines.append(f"  {warning}: {count} ({rate:.1%})")

        # Label distribution drift
        if batch_stats['label_distribution_drift']:
            report_lines.append(f"\nLABEL DISTRIBUTION ANALYSIS:")
            drift_data = batch_stats['label_distribution_drift']
            for key, value in drift_data.items():
                if key.endswith('_p_value'):
                    label = key.replace('_p_value', '')
                    chi2_key = f"{label}_chi2"
                    if chi2_key in drift_data:
                        report_lines.append(f"  {label}:")
                        report_lines.append(f"    Chi-square: {drift_data[chi2_key]:.3f}")
                        report_lines.append(f"    P-value: {value:.3f}")
                        significance = "significant" if value < 0.05 else "not significant"
                        report_lines.append(f"    Drift: {significance}")

        # Confidence distribution
        confidences = [r.confidence for r in validation_results]
        if confidences:
            report_lines.append(f"\nCONFIDENCE DISTRIBUTION:")
            report_lines.append(f"  Min: {min(confidences):.3f}")
            report_lines.append(f"  Mean: {np.mean(confidences):.3f}")
            report_lines.append(f"  Median: {np.median(confidences):.3f}")
            report_lines.append(f"  Max: {max(confidences):.3f}")
            report_lines.append(f"  Std: {np.std(confidences):.3f}")

        # Recommendations
        report_lines.append(f"\nRECOMMENDATIONS:")
        if validation_rate < 0.8:
            report_lines.append("  - Validation rate is low. Consider adjusting augmentation parameters.")
        if batch_stats['average_confidence'] < 0.6:
            report_lines.append("  - Average confidence is low. Review augmentation quality.")

        for key, p_value in batch_stats['label_distribution_drift'].items():
            if key.endswith('_p_value') and p_value < 0.05:
                label = key.replace('_p_value', '')
                report_lines.append(f"  - Significant distribution drift detected for {label}. Review augmentation strategy.")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    @staticmethod
    def save_report(report: str, output_path: str):
        """Save validation report to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Validation report saved to {output_path}")


def validate_augmented_dataset(original_texts: List[str], augmented_texts: List[str],
                             original_labels: List[Dict[str, Any]],
                             augmented_labels: List[Dict[str, Any]],
                             config: LabelConsistencyConfig = None) -> Tuple[bool, str]:
    """
    Convenience function to validate an entire augmented dataset.

    Returns:
        Tuple of (is_valid, report_text)
    """
    validator = LabelConsistencyValidator(config)
    results, batch_stats = validator.validate_batch(
        original_texts, augmented_texts, original_labels, augmented_labels
    )

    report = QualityAssuranceReport.generate_validation_report(results, batch_stats)

    # Dataset is considered valid if validation rate is above threshold
    validation_rate = batch_stats['valid_samples'] / batch_stats['total_samples']
    is_valid = validation_rate >= 0.8 and batch_stats['average_confidence'] >= 0.6

    return is_valid, report