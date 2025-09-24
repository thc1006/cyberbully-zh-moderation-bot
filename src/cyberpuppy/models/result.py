"""
Result classes for CyberPuppy detection system.

This module provides structured result classes for different prediction types,
JSON serialization/deserialization, confidence thresholding utilities,
and result aggregation methods.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import torch
import numpy as np
from datetime import datetime


class ToxicityLevel(Enum):
    """Toxicity level enumeration."""

    NONE = "none"
    TOXIC = "toxic"
    SEVERE = "severe"


class EmotionType(Enum):
    """Emotion type enumeration."""

    POSITIVE = "pos"
    NEUTRAL = "neu"
    NEGATIVE = "neg"


class BullyingType(Enum):
    """Bullying type enumeration."""

    NONE = "none"
    HARASSMENT = "harassment"
    THREAT = "threat"


class RoleType(Enum):
    """Role type enumeration."""

    NONE = "none"
    PERPETRATOR = "perpetrator"
    VICTIM = "victim"
    BYSTANDER = "bystander"


@dataclass
class ToxicityResult:
    """Result for toxicity detection task."""

    prediction: ToxicityLevel
    confidence: float
    raw_scores: Dict[str, float]
    threshold_met: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction.value,
            "confidence": self.confidence,
            "raw_scores": self.raw_scores,
            "threshold_met": self.threshold_met,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToxicityResult":
        """Create from dictionary."""
        return cls(
            prediction=ToxicityLevel(data["prediction"]),
            confidence=data["confidence"],
            raw_scores=data["raw_scores"],
            threshold_met=data["threshold_met"],
        )


@dataclass
class EmotionResult:
    """Result for emotion classification task."""

    prediction: EmotionType
    confidence: float
    strength: int  # 0-4 scale
    raw_scores: Dict[str, float]
    threshold_met: bool

    def __post_init__(self):
        """Validate emotion strength range."""
        if not 0 <= self.strength <= 4:
            raise ValueError(f"Emotion strength must be" " 0-4, got {self.strength}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction.value,
            "confidence": self.confidence,
            "strength": self.strength,
            "raw_scores": self.raw_scores,
            "threshold_met": self.threshold_met,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionResult":
        """Create from dictionary."""
        return cls(
            prediction=EmotionType(data["prediction"]),
            confidence=data["confidence"],
            strength=data["strength"],
            raw_scores=data["raw_scores"],
            threshold_met=data["threshold_met"],
        )


@dataclass
class BullyingResult:
    """Result for bullying detection task."""

    prediction: BullyingType
    confidence: float
    raw_scores: Dict[str, float]
    threshold_met: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction.value,
            "confidence": self.confidence,
            "raw_scores": self.raw_scores,
            "threshold_met": self.threshold_met,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BullyingResult":
        """Create from dictionary."""
        return cls(
            prediction=BullyingType(data["prediction"]),
            confidence=data["confidence"],
            raw_scores=data["raw_scores"],
            threshold_met=data["threshold_met"],
        )


@dataclass
class RoleResult:
    """Result for role classification task."""

    prediction: RoleType
    confidence: float
    raw_scores: Dict[str, float]
    threshold_met: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction.value,
            "confidence": self.confidence,
            "raw_scores": self.raw_scores,
            "threshold_met": self.threshold_met,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoleResult":
        """Create from dictionary."""
        return cls(
            prediction=RoleType(data["prediction"]),
            confidence=data["confidence"],
            raw_scores=data["raw_scores"],
            threshold_met=data["threshold_met"],
        )


@dataclass
class ExplanationResult:
    """Result for explanation generation."""

    attributions: List[float]
    tokens: List[str]
    explanation_text: str
    top_contributing_words: List[tuple]  # (word, attribution_score)
    method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attributions": self.attributions,
            "tokens": self.tokens,
            "explanation_text": self.explanation_text,
            "top_contributing_words": self.top_contributing_words,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplanationResult":
        """Create from dictionary."""
        return cls(
            attributions=data["attributions"],
            tokens=data["tokens"],
            explanation_text=data["explanation_text"],
            top_contributing_words=data["top_contributing_words"],
            method=data["method"],
        )


@dataclass
class ModelPrediction:
    """Individual model prediction result."""

    model_name: str
    predictions: Dict[str, torch.Tensor]
    confidence_scores: Dict[str, float]
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "predictions": {
                task: pred.tolist() if isinstance(pred, torch.Tensor) else pred
                for task, pred in self.predictions.items()
            },
            "confidence_scores": self.confidence_scores,
            "processing_time": self.processing_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelPrediction":
        """Create from dictionary."""
        predictions = {}
        for task, pred in data["predictions"].items():
            if isinstance(pred, list):
                predictions[task] = torch.tensor(pred)
            else:
                predictions[task] = pred

        return cls(
            model_name=data["model_name"],
            predictions=predictions,
            confidence_scores=data["confidence_scores"],
            processing_time=data["processing_time"],
        )


@dataclass
class DetectionResult:
    """Complete detection result containing all predictions and metadata."""

    # Input text and metadata
    text: str
    timestamp: datetime
    processing_time: float

    # Task-specific results
    toxicity: ToxicityResult
    emotion: EmotionResult
    bullying: BullyingResult
    role: RoleResult

    # Explanations and interpretability
    explanations: Dict[str, ExplanationResult]

    # Model ensemble information
    model_predictions: Dict[str, ModelPrediction]
    ensemble_weights: Dict[str, float]

    # Additional metadata
    context_used: bool = False
    context_length: int = 0
    model_version: str = "1.0.0"

    def __post_init__(self):
        """Validate detection result after initialization."""
        # Validate confidence scores
        for result in [self.toxicity, self.emotion, self.bullying, self.role]:
            if not 0.0 <= result.confidence <= 1.0:
                raise ValueError(
                    f"Confidence must be in [0,1" "], got {result.confidence}"
                )

        # Validate processing time
        if self.processing_time < 0:
            raise ValueError(
                f"Processing time must be non-nega" "tive, got {self.processing_time}"
            )

    def is_high_risk(self, thresholds: Optional[Dict[str, float]] = None) -> bool:
        """
        Determine if the result indicates high risk based on thresholds.

        Args:
            thresholds: Custom thresholds for risk assessment

        Returns:
            True if any high-risk condition is met
        """
        if thresholds is None:
            thresholds = {
                "toxicity_severe": 0.8,
                "toxicity_toxic": 0.7,
                "bullying_threat": 0.8,
                "bullying_harassment": 0.7,
                "emotion_negative_strong": 0.8,
            }

        # Check severe toxicity
        if (
            self.toxicity.prediction == ToxicityLevel.SEVERE
            and self.toxicity.confidence >= thresholds.get("toxicity_severe", 0.8)
        ):
            return True

        # Check toxic content
        if (
            self.toxicity.prediction == ToxicityLevel.TOXIC
            and self.toxicity.confidence >= thresholds.get("toxicity_toxic", 0.7)
        ):
            return True

        # Check threats
        if (
            self.bullying.prediction == BullyingType.THREAT
            and self.bullying.confidence >= thresholds.get("bullying_threat", 0.8)
        ):
            return True

        # Check harassment
        if (
            self.bullying.prediction == BullyingType.HARASSMENT
            and self.bullying.confidence >= thresholds.get("bullying_harassment", 0.7)
        ):
            return True

        # Check strong negative emotion
        if (
            self.emotion.prediction == EmotionType.NEGATIVE
            and self.emotion.strength >= 3
            and self.emotion.confidence
            >= thresholds.get("emotion_negative_strong", 0.8)
        ):
            return True

        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the detection results."""
        return {
            "text_preview": self.text[:100] + ("..." if len(self.text) > 100 else ""),
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time,
            "toxicity": self.toxicity.prediction.value,
            "toxicity_confidence": self.toxicity.confidence,
            "emotion": self.emotion.prediction.value,
            "emotion_strength": self.emotion.strength,
            "bullying": self.bullying.prediction.value,
            "role": self.role.prediction.value,
            "high_risk": self.is_high_risk(),
            "context_used": self.context_used,
            "model_version": self.model_version,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete result to dictionary."""
        return {
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time,
            "toxicity": self.toxicity.to_dict(),
            "emotion": self.emotion.to_dict(),
            "bullying": self.bullying.to_dict(),
            "role": self.role.to_dict(),
            "explanations": {
                task: explanation.to_dict()
                for task, explanation in self.explanations.items()
            },
            "model_predictions": {
                model: prediction.to_dict()
                for model, prediction in self.model_predictions.items()
            },
            "ensemble_weights": self.ensemble_weights,
            "context_used": self.context_used,
            "context_length": self.context_length,
            "model_version": self.model_version,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResult":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            processing_time=data["processing_time"],
            toxicity=ToxicityResult.from_dict(data["toxicity"]),
            emotion=EmotionResult.from_dict(data["emotion"]),
            bullying=BullyingResult.from_dict(data["bullying"]),
            role=RoleResult.from_dict(data["role"]),
            explanations={
                task: ExplanationResult.from_dict(explanation)
                for task, explanation in data["explanations"].items()
            },
            model_predictions={
                model: ModelPrediction.from_dict(prediction)
                for model, prediction in data["model_predictions"].items()
            },
            ensemble_weights=data["ensemble_weights"],
            context_used=data.get("context_used", False),
            context_length=data.get("context_length", 0),
            model_version=data.get("model_version", "1.0.0"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DetectionResult":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class ResultAggregator:
    """Utility class for aggregating detection results."""

    @staticmethod
    def aggregate_batch_results(results: List[DetectionResult]) -> Dict[str, Any]:
        """
        Aggregate statistics from a batch of results.

        Args:
            results: List of detection results

        Returns:
            Aggregated statistics
        """
        if not results:
            return {}

        total_results = len(results)

        # Count predictions by type
        toxicity_counts = {level.value: 0 for level in ToxicityLevel}
        emotion_counts = {emotion.value: 0 for emotion in EmotionType}
        bullying_counts = {bully.value: 0 for bully in BullyingType}
        role_counts = {role.value: 0 for role in RoleType}

        # Aggregate confidence scores
        toxicity_confidences = []
        emotion_confidences = []
        bullying_confidences = []
        role_confidences = []

        # Processing time statistics
        processing_times = []
        high_risk_count = 0

        for result in results:
            # Count predictions
            toxicity_counts[result.toxicity.prediction.value] += 1
            emotion_counts[result.emotion.prediction.value] += 1
            bullying_counts[result.bullying.prediction.value] += 1
            role_counts[result.role.prediction.value] += 1

            # Collect confidence scores
            toxicity_confidences.append(result.toxicity.confidence)
            emotion_confidences.append(result.emotion.confidence)
            bullying_confidences.append(result.bullying.confidence)
            role_confidences.append(result.role.confidence)

            # Processing time
            processing_times.append(result.processing_time)

            # High risk count
            if result.is_high_risk():
                high_risk_count += 1

        return {
            "total_results": total_results,
            "prediction_counts": {
                "toxicity": toxicity_counts,
                "emotion": emotion_counts,
                "bullying": bullying_counts,
                "role": role_counts,
            },
            "confidence_statistics": {
                "toxicity": {
                    "mean": np.mean(toxicity_confidences),
                    "std": np.std(toxicity_confidences),
                    "min": np.min(toxicity_confidences),
                    "max": np.max(toxicity_confidences),
                },
                "emotion": {
                    "mean": np.mean(emotion_confidences),
                    "std": np.std(emotion_confidences),
                    "min": np.min(emotion_confidences),
                    "max": np.max(emotion_confidences),
                },
                "bullying": {
                    "mean": np.mean(bullying_confidences),
                    "std": np.std(bullying_confidences),
                    "min": np.min(bullying_confidences),
                    "max": np.max(bullying_confidences),
                },
                "role": {
                    "mean": np.mean(role_confidences),
                    "std": np.std(role_confidences),
                    "min": np.min(role_confidences),
                    "max": np.max(role_confidences),
                },
            },
            "processing_time_statistics": {
                "mean": np.mean(processing_times),
                "std": np.std(processing_times),
                "min": np.min(processing_times),
                "max": np.max(processing_times),
            },
            "high_risk_count": high_risk_count,
            "high_risk_percentage": (high_risk_count / total_results) * 100,
        }

    @staticmethod
    def filter_results_by_confidence(
        results: List[DetectionResult], task: str, min_confidence: float
    ) -> List[DetectionResult]:
        """
        Filter results by minimum confidence for a specific task.

        Args:
            results: List of detection results
            task: Task name ('toxicity', 'emotion', 'bullying', 'role')
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered results
        """
        filtered = []
        valid_tasks = {"toxicity", "emotion", "bullying", "role"}

        if task not in valid_tasks:
            return filtered

        for result in results:
            task_result = getattr(result, task)
            if task_result.confidence >= min_confidence:
                filtered.append(result)

        return filtered

    @staticmethod
    def get_top_risk_results(
        results: List[DetectionResult], top_k: int = 10
    ) -> List[DetectionResult]:
        """
        Get top K highest risk results.

        Args:
            results: List of detection results
            top_k: Number of top results to return

        Returns:
            Top K highest risk results
        """

        # Calculate risk scores
        def calculate_risk_score(result: DetectionResult) -> float:
            score = 0.0

            # Toxicity contribution
            if result.toxicity.prediction == ToxicityLevel.SEVERE:
                score += result.toxicity.confidence * 3.0
            elif result.toxicity.prediction == ToxicityLevel.TOXIC:
                score += result.toxicity.confidence * 2.0

            # Bullying contribution
            if result.bullying.prediction == BullyingType.THREAT:
                score += result.bullying.confidence * 2.5
            elif result.bullying.prediction == BullyingType.HARASSMENT:
                score += result.bullying.confidence * 2.0

            # Strong negative emotion contribution
            if (
                result.emotion.prediction == EmotionType.NEGATIVE
                and result.emotion.strength >= 3
            ):
                score += result.emotion.confidence * 1.5

            return score

        # Sort by risk score
        results_with_scores = [
            (result, calculate_risk_score(result)) for result in results
        ]
        results_with_scores.sort(key=lambda x: x[1], reverse=True)

        return [result for result, _ in results_with_scores[:top_k]]


class ConfidenceThresholds:
    """Utility class for managing confidence thresholds."""

    DEFAULT_THRESHOLDS = {
        "toxicity": {"none": 0.6, "toxic": 0.7, "severe": 0.8},
        "emotion": {"pos": 0.6, "neu": 0.5, "neg": 0.6},
        "bullying": {"none": 0.6, "harassment": 0.7, "threat": 0.8},
        "role": {"none": 0.5, "perpetrator": 0.7, "victim": 0.6, "bystander": 0.6},
    }

    @classmethod
    def get_threshold(cls, task: str, prediction: str) -> float:
        """Get threshold for a specific task and prediction."""
        return cls.DEFAULT_THRESHOLDS.get(task, {}).get(prediction, 0.5)

    @classmethod
    def update_thresholds(cls, new_thresholds: Dict[str, Dict[str, float]]) -> None:
        """Update default thresholds."""
        for task, thresholds in new_thresholds.items():
            if task in cls.DEFAULT_THRESHOLDS:
                cls.DEFAULT_THRESHOLDS[task].update(thresholds)
            else:
                cls.DEFAULT_THRESHOLDS[task] = thresholds

    @classmethod
    def validate_thresholds(cls, thresholds: Dict[str, Dict[str, float]]) -> bool:
        """Validate threshold values are in valid range."""
        for task, task_thresholds in thresholds.items():
            for prediction, threshold in task_thresholds.items():
                if not 0.0 <= threshold <= 1.0:
                    return False
        return True
