"""
Result classes for CyberPuppy detection system.

This module provides structured result classes for different prediction types,
JSON serialization/deserialization, confidence thresholding utilities,
and result aggregation methods.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch


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
            raise ValueError("Emotion strength must be" " 0-4, got {self.strength}")

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
    timestamp: Optional[datetime] = None
    processing_time: float = 0.0

    # Task-specific results (can be None for backwards compatibility)
    toxicity: Optional[ToxicityResult] = None
    emotion: Optional[EmotionResult] = None
    bullying: Optional[BullyingResult] = None
    role: Optional[RoleResult] = None

    # Explanations and interpretability
    explanations: Optional[Dict[str, ExplanationResult]] = None

    # Model ensemble information
    model_predictions: Optional[Dict[str, ModelPrediction]] = None
    ensemble_weights: Optional[Dict[str, float]] = None

    # Additional metadata
    context_used: bool = False
    context_length: int = 0
    model_version: str = "1.0.0"
    metadata: Optional[Dict[str, Any]] = None

    # Simple parameters for test compatibility
    toxicity_label: Optional[int] = None
    toxicity_confidence: Optional[float] = None
    emotion_label: Optional[int] = None
    emotion_confidence: Optional[float] = None
    bullying_label: Optional[int] = None
    bullying_confidence: Optional[float] = None
    role_label: Optional[int] = None
    role_confidence: Optional[float] = None
    explanation: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        text: str,
        # Traditional detailed parameters
        toxicity: Optional[ToxicityResult] = None,
        emotion: Optional[EmotionResult] = None,
        bullying: Optional[BullyingResult] = None,
        role: Optional[RoleResult] = None,
        timestamp: Optional[datetime] = None,
        processing_time: float = 0.0,
        explanations: Optional[Dict[str, ExplanationResult]] = None,
        model_predictions: Optional[Dict[str, ModelPrediction]] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
        context_used: bool = False,
        context_length: int = 0,
        model_version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
        # Simple parameters for test compatibility
        toxicity_label: Optional[int] = None,
        toxicity_confidence: Optional[float] = None,
        emotion_label: Optional[int] = None,
        emotion_confidence: Optional[float] = None,
        bullying_label: Optional[int] = None,
        bullying_confidence: Optional[float] = None,
        role_label: Optional[int] = None,
        role_confidence: Optional[float] = None,
        explanation: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize DetectionResult with flexible parameter support.

        Supports both the original detailed format and simplified test format.
        """
        self.text = text
        self.timestamp = timestamp or datetime.now()
        self.processing_time = processing_time
        self.context_used = context_used
        self.context_length = context_length
        self.model_version = model_version
        self.metadata = metadata or {}
        self.explanation = explanation

        # Store simple parameters for test compatibility
        self.toxicity_label = toxicity_label
        self.toxicity_confidence = toxicity_confidence
        self.emotion_label = emotion_label
        self.emotion_confidence = emotion_confidence
        self.bullying_label = bullying_label
        self.bullying_confidence = bullying_confidence
        self.role_label = role_label
        self.role_confidence = role_confidence

        # Handle complex objects or create from simple parameters
        if toxicity is not None:
            self.toxicity = toxicity
        elif toxicity_label is not None and toxicity_confidence is not None:
            # Create ToxicityResult from simple parameters
            prediction = ToxicityLevel.NONE
            if toxicity_label == 1:
                prediction = ToxicityLevel.TOXIC
            elif toxicity_label == 2:
                prediction = ToxicityLevel.SEVERE

            self.toxicity = ToxicityResult(
                prediction=prediction,
                confidence=toxicity_confidence,
                raw_scores={prediction.value: toxicity_confidence},
                threshold_met=toxicity_confidence > 0.5,
            )
        else:
            self.toxicity = None

        if emotion is not None:
            self.emotion = emotion
        elif emotion_label is not None and emotion_confidence is not None:
            # Create EmotionResult from simple parameters
            prediction = EmotionType.NEUTRAL
            if emotion_label == 0:
                prediction = EmotionType.POSITIVE
            elif emotion_label == 2:
                prediction = EmotionType.NEGATIVE

            self.emotion = EmotionResult(
                prediction=prediction,
                confidence=emotion_confidence,
                strength=2,  # Default neutral strength
                raw_scores={prediction.value: emotion_confidence},
                threshold_met=emotion_confidence > 0.5,
            )
        else:
            self.emotion = None

        if bullying is not None:
            self.bullying = bullying
        elif bullying_label is not None and bullying_confidence is not None:
            # Create BullyingResult from simple parameters
            prediction = BullyingType.NONE
            if bullying_label == 1:
                prediction = BullyingType.HARASSMENT
            elif bullying_label == 2:
                prediction = BullyingType.THREAT

            self.bullying = BullyingResult(
                prediction=prediction,
                confidence=bullying_confidence,
                raw_scores={prediction.value: bullying_confidence},
                threshold_met=bullying_confidence > 0.5,
            )
        else:
            self.bullying = None

        if role is not None:
            self.role = role
        elif role_label is not None and role_confidence is not None:
            # Create RoleResult from simple parameters
            prediction = RoleType.NONE
            if role_label == 1:
                prediction = RoleType.PERPETRATOR
            elif role_label == 2:
                prediction = RoleType.VICTIM
            elif role_label == 3:
                prediction = RoleType.BYSTANDER

            self.role = RoleResult(
                prediction=prediction,
                confidence=role_confidence,
                raw_scores={prediction.value: role_confidence},
                threshold_met=role_confidence > 0.5,
            )
        else:
            self.role = None

        # Set other attributes
        self.explanations = explanations or {}
        self.model_predictions = model_predictions or {}
        self.ensemble_weights = ensemble_weights or {}

        # Call validation after construction
        self._validate()

    def _validate(self):
        """Validate detection result after initialization."""
        # Validate confidence scores for non-None results
        for result in [self.toxicity, self.emotion, self.bullying, self.role]:
            if result is not None and not 0.0 <= result.confidence <= 1.0:
                raise ValueError(f"Confidence must be in [0,1], got {result.confidence}")

        # Validate processing time
        if self.processing_time < 0:
            raise ValueError(f"Processing time must be non-negative, got {self.processing_time}")

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
            self.toxicity is not None
            and self.toxicity.prediction == ToxicityLevel.SEVERE
            and self.toxicity.confidence >= thresholds.get("toxicity_severe", 0.8)
        ):
            return True

        # Check toxic content
        if (
            self.toxicity is not None
            and self.toxicity.prediction == ToxicityLevel.TOXIC
            and self.toxicity.confidence >= thresholds.get("toxicity_toxic", 0.7)
        ):
            return True

        # Check threats
        if (
            self.bullying is not None
            and self.bullying.prediction == BullyingType.THREAT
            and self.bullying.confidence >= thresholds.get("bullying_threat", 0.8)
        ):
            return True

        # Check harassment
        if (
            self.bullying is not None
            and self.bullying.prediction == BullyingType.HARASSMENT
            and self.bullying.confidence >= thresholds.get("bullying_harassment", 0.7)
        ):
            return True

        # Check strong negative emotion
        if (
            self.emotion is not None
            and self.emotion.prediction == EmotionType.NEGATIVE
            and self.emotion.strength >= 3
            and self.emotion.confidence >= thresholds.get("emotion_negative_strong", 0.8)
        ):
            return True

        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the detection results."""
        summary = {
            "text_preview": self.text[:100] + ("..." if len(self.text) > 100 else ""),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "processing_time": self.processing_time,
            "high_risk": self.is_high_risk(),
            "context_used": self.context_used,
            "model_version": self.model_version,
        }

        # Add task results if available
        if self.toxicity is not None:
            summary["toxicity"] = self.toxicity.prediction.value
            summary["toxicity_confidence"] = self.toxicity.confidence

        if self.emotion is not None:
            summary["emotion"] = self.emotion.prediction.value
            summary["emotion_strength"] = self.emotion.strength

        if self.bullying is not None:
            summary["bullying"] = self.bullying.prediction.value

        if self.role is not None:
            summary["role"] = self.role.prediction.value

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete result to dictionary."""
        result = {
            "text": self.text,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "processing_time": self.processing_time,
            "context_used": self.context_used,
            "context_length": self.context_length,
            "model_version": self.model_version,
        }

        # Add simple test-compatible fields at top level
        if self.toxicity_label is not None:
            result["toxicity_label"] = self.toxicity_label
        if self.toxicity_confidence is not None:
            result["toxicity_confidence"] = self.toxicity_confidence
        if self.emotion_label is not None:
            result["emotion_label"] = self.emotion_label
        if self.emotion_confidence is not None:
            result["emotion_confidence"] = self.emotion_confidence
        if self.bullying_label is not None:
            result["bullying_label"] = self.bullying_label
        if self.bullying_confidence is not None:
            result["bullying_confidence"] = self.bullying_confidence
        if self.role_label is not None:
            result["role_label"] = self.role_label
        if self.role_confidence is not None:
            result["role_confidence"] = self.role_confidence
        if self.explanation is not None:
            result["explanation"] = self.explanation

        # Add task results if available
        if self.toxicity is not None:
            result["toxicity"] = self.toxicity.to_dict()
        if self.emotion is not None:
            result["emotion"] = self.emotion.to_dict()
        if self.bullying is not None:
            result["bullying"] = self.bullying.to_dict()
        if self.role is not None:
            result["role"] = self.role.to_dict()

        # Add optional fields
        if self.explanations:
            result["explanations"] = {
                task: explanation.to_dict() for task, explanation in self.explanations.items()
            }
        if self.model_predictions:
            result["model_predictions"] = {
                model: prediction.to_dict() for model, prediction in self.model_predictions.items()
            }
        if self.ensemble_weights:
            result["ensemble_weights"] = self.ensemble_weights
        if self.metadata:
            result["metadata"] = self.metadata
        if self.explanation:
            result["explanation"] = self.explanation

        return result

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
            if result.emotion.prediction == EmotionType.NEGATIVE and result.emotion.strength >= 3:
                score += result.emotion.confidence * 1.5

            return score

        # Sort by risk score
        results_with_scores = [(result, calculate_risk_score(result)) for result in results]
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
        for _task, task_thresholds in thresholds.items():
            for _prediction, threshold in task_thresholds.items():
                if not 0.0 <= threshold <= 1.0:
                    return False
        return True


"""
Missing classes for CyberPuppy models module.

These classes are needed for test compatibility but were missing from result.py.
This is a temporary file - these should be integrated into the main result.py file.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class ModelOutput:
    """Raw model output containing logits for all tasks.

    This class represents the direct output from neural network models,
    containing raw logits before softmax normalization for different tasks.
    """

    toxicity_logits: Optional[torch.Tensor] = None
    emotion_logits: Optional[torch.Tensor] = None
    bullying_logits: Optional[torch.Tensor] = None
    role_logits: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Validate tensor shapes and dimensions."""
        tensors = [
            ("toxicity", self.toxicity_logits),
            ("emotion", self.emotion_logits),
            ("bullying", self.bullying_logits),
            ("role", self.role_logits),
        ]

        for name, tensor in tensors:
            if tensor is not None:
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"{name}_logits must be torch.Tensor, got {type(tensor)}")
                if len(tensor.shape) != 2:
                    raise ValueError(
                        f"{name}_logits must be 2D tensor (batch_size, num_classes), got shape {tensor.shape}"
                    )

    def get_probabilities(self) -> Dict[str, torch.Tensor]:
        """Convert logits to probabilities using softmax."""
        probs = {}

        if self.toxicity_logits is not None:
            probs["toxicity"] = F.softmax(self.toxicity_logits, dim=1)
        if self.emotion_logits is not None:
            probs["emotion"] = F.softmax(self.emotion_logits, dim=1)
        if self.bullying_logits is not None:
            probs["bullying"] = F.softmax(self.bullying_logits, dim=1)
        if self.role_logits is not None:
            probs["role"] = F.softmax(self.role_logits, dim=1)

        return probs

    def get_predictions(self) -> Dict[str, torch.Tensor]:
        """Get class predictions from logits (argmax)."""
        predictions = {}

        if self.toxicity_logits is not None:
            predictions["toxicity"] = torch.argmax(self.toxicity_logits, dim=1)
        if self.emotion_logits is not None:
            predictions["emotion"] = torch.argmax(self.emotion_logits, dim=1)
        if self.bullying_logits is not None:
            predictions["bullying"] = torch.argmax(self.bullying_logits, dim=1)
        if self.role_logits is not None:
            predictions["role"] = torch.argmax(self.role_logits, dim=1)

        return predictions

    def get_confidence_scores(self) -> Dict[str, torch.Tensor]:
        """Get maximum probability as confidence score for each task."""
        probs = self.get_probabilities()
        confidences = {}

        for task, prob_tensor in probs.items():
            # Get max probability along class dimension
            confidences[task] = torch.max(prob_tensor, dim=1)[0]

        return confidences

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "toxicity_logits": (
                self.toxicity_logits.tolist() if self.toxicity_logits is not None else None
            ),
            "emotion_logits": (
                self.emotion_logits.tolist() if self.emotion_logits is not None else None
            ),
            "bullying_logits": (
                self.bullying_logits.tolist() if self.bullying_logits is not None else None
            ),
            "role_logits": self.role_logits.tolist() if self.role_logits is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelOutput":
        """Create ModelOutput from dictionary."""
        return cls(
            toxicity_logits=(
                torch.tensor(data["toxicity_logits"])
                if data.get("toxicity_logits") is not None
                else None
            ),
            emotion_logits=(
                torch.tensor(data["emotion_logits"])
                if data.get("emotion_logits") is not None
                else None
            ),
            bullying_logits=(
                torch.tensor(data["bullying_logits"])
                if data.get("bullying_logits") is not None
                else None
            ),
            role_logits=(
                torch.tensor(data["role_logits"]) if data.get("role_logits") is not None else None
            ),
        )


@dataclass
class ExplanationOutput:
    """Output from explainability methods like Integrated Gradients or SHAP.

    This class encapsulates the results of model explanation techniques,
    providing both raw attribution scores and human-readable explanations.
    """

    method: str  # 'integrated_gradients', 'shap', 'attention', etc.
    attributions: List[float]  # Attribution scores per token
    tokens: List[str]  # Corresponding tokens
    target_class: Optional[str] = None  # Which class was explained
    target_task: Optional[str] = None  # Which task was explained
    baseline_score: Optional[float] = None  # Baseline prediction score
    explained_score: Optional[float] = None  # Score with current input
    convergence_delta: Optional[float] = None  # For IG, convergence check

    def __post_init__(self):
        """Validate explanation output."""
        if len(self.attributions) != len(self.tokens):
            raise ValueError(
                f"Attributions length ({len(self.attributions)}) must match "
                f"tokens length ({len(self.tokens)})"
            )

        # Validate method
        valid_methods = {"integrated_gradients", "shap", "attention", "gradient", "lime"}
        if self.method not in valid_methods:
            print(
                f"Warning: Unknown explanation method '{self.method}'. "
                f"Valid methods: {valid_methods}"
            )

    def get_top_contributing_tokens(self, k: int = 5, absolute: bool = True) -> List[tuple]:
        """Get top K contributing tokens.

        Args:
            k: Number of top tokens to return
            absolute: If True, use absolute values for ranking

        Returns:
            List of (token, attribution_score) tuples sorted by importance
        """
        if not self.tokens or not self.attributions:
            return []

        token_attr_pairs = list(zip(self.tokens, self.attributions))

        if absolute:
            # Sort by absolute attribution values
            token_attr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        else:
            # Sort by raw attribution values (positive first)
            token_attr_pairs.sort(key=lambda x: x[1], reverse=True)

        return token_attr_pairs[:k]

    def get_explanation_text(self, threshold: float = 0.1) -> str:
        """Generate human-readable explanation text.

        Args:
            threshold: Minimum attribution threshold to include in explanation

        Returns:
            Human-readable explanation string
        """
        if not self.tokens or not self.attributions:
            return "No explanation available."

        # Get significant contributions
        significant = [
            (token, attr)
            for token, attr in zip(self.tokens, self.attributions)
            if abs(attr) >= threshold
        ]

        if not significant:
            return f"No tokens met the significance threshold of {threshold}."

        # Sort by absolute contribution
        significant.sort(key=lambda x: abs(x[1]), reverse=True)

        explanation_parts = []

        # Separate positive and negative contributions
        positive_tokens = [(token, attr) for token, attr in significant if attr > 0]
        negative_tokens = [(token, attr) for token, attr in significant if attr < 0]

        if positive_tokens:
            pos_tokens_str = ", ".join([f"'{token}'" for token, _ in positive_tokens[:3]])
            explanation_parts.append(f"Positive contributors: {pos_tokens_str}")

        if negative_tokens:
            neg_tokens_str = ", ".join([f"'{token}'" for token, _ in negative_tokens[:3]])
            explanation_parts.append(f"Negative contributors: {neg_tokens_str}")

        # Add method and target information
        method_info = f"Method: {self.method}"
        if self.target_task and self.target_class:
            method_info += f" (Task: {self.target_task}, Class: {self.target_class})"
        explanation_parts.insert(0, method_info)

        return "; ".join(explanation_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method,
            "attributions": self.attributions,
            "tokens": self.tokens,
            "target_class": self.target_class,
            "target_task": self.target_task,
            "baseline_score": self.baseline_score,
            "explained_score": self.explained_score,
            "convergence_delta": self.convergence_delta,
            "top_contributing_tokens": self.get_top_contributing_tokens(),
            "explanation_text": self.get_explanation_text(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplanationOutput":
        """Create ExplanationOutput from dictionary."""
        return cls(
            method=data["method"],
            attributions=data["attributions"],
            tokens=data["tokens"],
            target_class=data.get("target_class"),
            target_task=data.get("target_task"),
            baseline_score=data.get("baseline_score"),
            explained_score=data.get("explained_score"),
            convergence_delta=data.get("convergence_delta"),
        )


class BatchResults:
    """Container for batch processing results with aggregation capabilities.

    This class manages collections of DetectionResult objects and provides
    methods for filtering, aggregation, and batch-level statistics.
    """

    def __init__(self, results=None):
        """Initialize BatchResults.

        Args:
            results: Initial list of detection results
        """
        self.results = results or []
        self._cached_stats = None
        self._stats_dirty = True

    def add_result(self, result) -> None:
        """Add a single result to the batch."""
        self.results.append(result)
        self._invalidate_cache()

    def add_results(self, results) -> None:
        """Add multiple results to the batch."""
        self.results.extend(results)
        self._invalidate_cache()

    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate cached statistics."""
        self._stats_dirty = True
        self._cached_stats = None

    def __len__(self) -> int:
        """Return number of results."""
        return len(self.results)

    def __iter__(self):
        """Iterate over results."""
        return iter(self.results)

    def __getitem__(self, index):
        """Get result by index or slice."""
        return self.results[index]

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics for the batch.

        Returns:
            Dictionary containing various statistics about the batch
        """
        if not self._stats_dirty and self._cached_stats is not None:
            return self._cached_stats

        if not self.results:
            return {
                "total_samples": 0,
                "toxicity_distribution": {},
                "emotion_distribution": {},
                "bullying_distribution": {},
                "role_distribution": {},
                "average_confidence": {},
                "high_risk_count": 0,
                "high_risk_percentage": 0.0,
                "processing_time_stats": {},
            }

        # Basic statistics
        stats = {"total_samples": len(self.results), "batch_size": len(self.results)}

        # If we have results, compute more detailed statistics
        if self.results:
            # Try to get timestamps if available
            try:
                timestamps = [r.timestamp for r in self.results if hasattr(r, "timestamp")]
                if timestamps:
                    stats["timestamp_range"] = {
                        "earliest": min(timestamps).isoformat(),
                        "latest": max(timestamps).isoformat(),
                    }
            except:
                pass

            # Try to get confidence distributions
            try:
                toxicity_confidences = []
                for r in self.results:
                    if hasattr(r, "toxicity") and hasattr(r.toxicity, "confidence"):
                        toxicity_confidences.append(r.toxicity.confidence)

                if toxicity_confidences:
                    stats["toxicity_distribution"] = {
                        "mean": float(sum(toxicity_confidences) / len(toxicity_confidences)),
                        "min": float(min(toxicity_confidences)),
                        "max": float(max(toxicity_confidences)),
                    }
            except:
                pass

        # Cache the statistics
        self._cached_stats = stats
        self._stats_dirty = False

        return stats

    def filter_by_confidence(self, task: str, min_confidence: float) -> "BatchResults":
        """Filter results by minimum confidence threshold for a specific task.

        Args:
            task: Task name (e.g., 'toxicity', 'emotion', 'bullying')
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            New BatchResults instance with filtered results
        """
        filtered_results = []

        for result in self.results:
            try:
                # Handle different result formats
                if isinstance(result, dict):
                    if result.get("confidence", 0) >= min_confidence:
                        filtered_results.append(result)
                elif hasattr(result, task):
                    task_result = getattr(result, task)
                    if (
                        hasattr(task_result, "confidence")
                        and task_result.confidence >= min_confidence
                    ):
                        filtered_results.append(result)
                elif hasattr(result, "confidence") and result.confidence >= min_confidence:
                    filtered_results.append(result)
            except (AttributeError, TypeError):
                # Skip malformed results
                continue

        return BatchResults(filtered_results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary."""
        return {
            "results": [
                result.to_dict() if hasattr(result, "to_dict") else str(result)
                for result in self.results
            ],
            "summary_statistics": self.get_summary_statistics(),
            "batch_metadata": {"size": len(self.results), "created_at": datetime.now().isoformat()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchResults":
        """Create BatchResults from dictionary."""
        # This is a simplified implementation - would need proper DetectionResult deserialization
        results = data.get("results", [])
        return cls(results)
