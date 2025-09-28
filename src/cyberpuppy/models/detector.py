"""
CyberPuppy main detection system.

This module provides the CyberPuppyDetector class that orchestrates
all models (baseline, contextual, weak supervision) and integrates
explanation generation for comprehensive text analysis.
"""

import logging
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import opencc
# from pathlib import Path  # Unused import commented out
import torch

from ..explain.ig import IntegratedGradientsExplainer
from .baselines import BaselineModel, ModelConfig
from .contextual import ContextualModel
from .result import (BullyingResult, BullyingType, ConfidenceThresholds,
                     DetectionResult, EmotionResult, EmotionType,
                     ExplanationResult, ModelPrediction, RoleResult, RoleType,
                     ToxicityLevel, ToxicityResult)
from .weak_supervision import WeakSupervisionModel

# from ..config import Config  # Unused import commented out

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class CyberPuppyDetector:
    """
    Main detection system that orchestrates all models for comprehensive
        analysis.

    This class follows the London School TDD approach with clear separation of
    concerns, dependency injection, and behavior-driven design.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
        load_models: bool = True,
    ):
        """
        Initialize the CyberPuppy detector.

        Args:
            config: Configuration dictionary containing model paths,
                thresholds, etc.
            device: Device to run models on ('cpu', 'cuda', 'auto')
            load_models: Whether to load models immediately
        """
        self.config = config
        self.device = self._setup_device(device)
        self.ensemble_weights = self._normalize_weights(config.get("ensemble_weights", {}))

        # Initialize text preprocessing components
        self._setup_preprocessing()

        # Model storage
        self.models = {}
        self.explainer = None
        self._models_loaded = False

        # Performance tracking
        self._lock = threading.Lock()
        self._prediction_count = 0
        self._total_processing_time = 0.0

        # Load models if requested
        if load_models:
            self._load_models()

        logger.info(f"CyberPuppyDetector initialized on device: {self.device}")

    def _setup_device(self, device: Optional[str] = None) -> str:
        """Setup computation device."""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize ensemble weights to sum to 1.0."""
        if not weights:
            # Default equal weights
            return {"baseline": 0.4, "contextual": 0.35, "weak_supervision": 0.25}

        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Ensemble weights sum to {total_weight}, normalizing to 1.0")
            return {model: weight / total_weight for model, weight in weights.items()}

        return weights

    def _setup_preprocessing(self):
        """Initialize text preprocessing components."""
        try:
            # Traditional to Simplified Chinese converter
            self.converter = opencc.OpenCC("t2s")
            self.preprocessing_config = self.config.get("preprocessing", {})
            self.max_length = self.preprocessing_config.get("max_length", 512)
            self.normalize_unicode = self.preprocessing_config.get("normalize_unicode", True)
            self.convert_traditional = self.preprocessing_config.get("convert_traditional", True)
            logger.info("Text preprocessing components initialized")
        except Exception as e:
            logger.warning(f"Error initializing preprocessing: {e}")
            # Fallback to minimal preprocessing
            self.converter = None

    def _load_models(self) -> None:
        """
        Load all models and explainer.

        Raises:
            FileNotFoundError: If model files are not found
            RuntimeError: If models fail to load
        """
        try:
            model_paths = self.config.get("model_paths", {})

            # Load baseline model
            logger.info("Loading baseline model...")
            model_config = ModelConfig(
                model_name=self.config.get("base_model", "hfl/chinese-macbert-base"),
                num_toxicity_classes=3,
                num_emotion_classes=3,
                num_bullying_classes=3,
                num_role_classes=4,
            )
            self.models["baseline"] = BaselineModel(model_config)
            if "baseline" in model_paths:
                self.models["baseline"].load_state_dict(
                    torch.load(model_paths["baseline"], map_location=self.device)
                )

            # Load contextual model
            logger.info("Loading contextual model...")
            self.models["contextual"] = ContextualModel(
                base_model_name=self.config.get("base_model", "hfl/chinese-macbert-base"),
                device=self.device,
            )
            if "contextual" in model_paths:
                self.models["contextual"].load_state_dict(
                    torch.load(model_paths["contextual"], map_location=self.device)
                )

            # Load weak supervision model
            logger.info("Loading weak supervision model...")
            self.models["weak_supervision"] = WeakSupervisionModel(
                base_model_name=self.config.get("base_model", "hfl/chinese-macbert-base"),
                device=self.device,
            )
            if "weak_supervision" in model_paths:
                self.models["weak_supervision"].load_state_dict(
                    torch.load(model_paths["weak_supervision"], map_location=self.device)
                )

            # Load explainer
            logger.info("Loading explainer...")
            self.explainer = IntegratedGradientsExplainer(
                model=self.models["baseline"],  # Use baseline for explanations
                device=self.device,
            )

            # Set models to evaluation mode
            for model in self.models.values():
                model.eval()

            self._models_loaded = True
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess input text for analysis.

        Args:
            text: Raw input text

        Returns:
            Preprocessed text

        Raises:
            ValueError: If text is None or empty
            TypeError: If text is not a string
        """
        if text is None:
            raise ValueError("Text cannot be None")

        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got {type(text)}")

        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Unicode normalization
        if self.normalize_unicode:
            import unicodedata

            text = unicodedata.normalize("NFKC", text)

        # Traditional to Simplified conversion
        if self.convert_traditional and self.converter:
            try:
                text = self.converter.convert(text)
            except Exception:
                logger.warning("Traditional to Simplifie" "d conversion failed: {e}")

        # Truncate if too long
        if len(text) > self.max_length:
            text = text[: self.max_length]
            logger.debug(f"Text truncated to {self.max_length} characters")

        # Clean whitespace
        text = " ".join(text.split())

        return text

    def _get_individual_predictions(
        self, text: str, context: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get predictions from individual models.

        Args:
            text: Preprocessed input text
            context: Optional conversation context

        Returns:
            Dictionary mapping model names to their predictions
        """
        predictions = {}

        # Get baseline predictions
        try:
            start_time = time.time()
            baseline_pred = self.models["baseline"].predict(text)
            baseline_time = time.time() - start_time

            predictions["baseline"] = {
                "predictions": baseline_pred,
                "processing_time": baseline_time,
            }
        except Exception as e:
            logger.error(f"Baseline model prediction failed: {e}")
            raise

        # Get contextual predictions
        try:
            start_time = time.time()
            contextual_pred = self.models["contextual"].predict(text, context=context)
            contextual_time = time.time() - start_time

            predictions["contextual"] = {
                "predictions": contextual_pred,
                "processing_time": contextual_time,
            }
        except Exception as e:
            logger.error(f"Contextual model prediction failed: {e}")
            # Fall back to baseline predictions
            predictions["contextual"] = predictions["baseline"].copy()

        # Get weak supervision predictions
        try:
            start_time = time.time()
            weak_pred = self.models["weak_supervision"].predict(text)
            weak_time = time.time() - start_time

            predictions["weak_supervision"] = {
                "predictions": weak_pred,
                "processing_time": weak_time,
            }
        except Exception as e:
            logger.error(f"Weak supervision model prediction failed: {e}")
            # Fall back to baseline predictions
            predictions["weak_supervision"] = predictions["baseline"].copy()

        return predictions

    def _ensemble_predict(
        self, text: str, model_predictions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Combine predictions from multiple models using ensemble weights.

        Args:
            text: Input text (for debugging)
            model_predictions: Individual model predictions

        Returns:
            Ensemble predictions for each task
        """
        tasks = ["toxicity", "emotion", "bullying", "role"]
        ensemble_results = {}

        for task in tasks:
            weighted_predictions = []
            total_weight = 0.0

            for model_name, result in model_predictions.items():
                weight = self.ensemble_weights.get(model_name, 0.0)
                if weight > 0 and task in result["predictions"]:
                    pred = result["predictions"][task]
                    if isinstance(pred, torch.Tensor):
                        weighted_predictions.append(pred * weight)
                        total_weight += weight

            if weighted_predictions and total_weight > 0:
                # Average weighted predictions
                ensemble_pred = sum(weighted_predictions)
                if total_weight != 1.0:
                    ensemble_pred = ensemble_pred / total_weight

                ensemble_results[task] = ensemble_pred
            else:
                logger.warning("No valid predictions for t" "ask {task}, using baseline")
                # Fallback to baseline
                if (
                    "baseline" in model_predictions
                    and task in model_predictions["baseline"]["predictions"]
                ):
                    ensemble_results[task] = model_predictions["baseline"]["predictions"][task]
                else:
                    # Generate dummy prediction
                    if task == "role":
                        ensemble_results[task] = torch.tensor([0.7, 0.1, 0.1, 0.1])
                    else:
                        ensemble_results[task] = torch.tensor([0.7, 0.2, 0.1])

        return ensemble_results

    def _calibrate_confidence(self, prediction: torch.Tensor, task: str) -> float:
        """
        Calibrate confidence score using temperature scaling and uncertainty.

        Args:
            prediction: Raw prediction logits/probabilities
            task: Task name for task-specific calibration

        Returns:
            Calibrated confidence score
        """
        # Apply softmax if not already probabilities
        if prediction.max() > 1.0 or prediction.min() < 0.0:
            prediction = torch.softmax(prediction, dim=-1)

        # Basic confidence as max probability
        max_prob = prediction.max().item()

        # Calculate entropy for uncertainty
        entropy = -(prediction * torch.log(prediction + 1e-8)).sum().item()
        max_entropy = np.log(len(prediction))
        normalized_entropy = entropy / max_entropy

        # Confidence adjusted by uncertainty
        uncertainty_adjusted_conf = max_prob * (1 - normalized_entropy)

        # Task-specific calibration (can be learned from validation data)
        task_calibration = {
            "toxicity": 0.95,  # Slightly conservative
            "emotion": 1.0,  # No adjustment
            "bullying": 0.9,  # Conservative for safety
            "role": 1.05,  # Slightly optimistic
        }

        calibrated_conf = uncertainty_adjusted_conf * task_calibration.get(task, 1.0)

        return max(0.0, min(1.0, calibrated_conf))  # Clamp to [0,1]

    def _convert_prediction_to_label(
        self, prediction: torch.Tensor, task: str
    ) -> Tuple[str, Dict[str, float]]:
        """
        Convert prediction tensor to label and raw scores.

        Args:
            prediction: Prediction tensor
            task: Task name

        Returns:
            Tuple of (predicted_label, raw_scores_dict)
        """
        # Apply softmax to get probabilities
        probabilities = torch.softmax(prediction, dim=-1)

        # Task-specific label mapping
        label_mappings = {
            "toxicity": ["none", "toxic", "severe"],
            "emotion": ["pos", "neu", "neg"],
            "bullying": ["none", "harassment", "threat"],
            "role": ["none", "perpetrator", "victim", "bystander"],
        }

        labels = label_mappings.get(task, [])
        predicted_idx = probabilities.argmax().item()

        if predicted_idx < len(labels):
            predicted_label = labels[predicted_idx]
        else:
            predicted_label = labels[0] if labels else "unknown"

        # Create raw scores dictionary
        raw_scores = {}
        for i, label in enumerate(labels):
            if i < len(probabilities):
                raw_scores[label] = probabilities[i].item()

        return predicted_label, raw_scores

    def _create_task_result(
        self, prediction: torch.Tensor, task: str, thresholds: Dict[str, float]
    ) -> Union[ToxicityResult, EmotionResult, BullyingResult, RoleResult]:
        """
        Create task-specific result object.

        Args:
            prediction: Prediction tensor
            task: Task name
            thresholds: Confidence thresholds

        Returns:
            Task-specific result object
        """
        predicted_label, raw_scores = self._convert_prediction_to_label(prediction, task)
        confidence = self._calibrate_confidence(prediction, task)
        threshold = thresholds.get(predicted_label, 0.5)
        threshold_met = confidence >= threshold

        if task == "toxicity":
            return ToxicityResult(
                prediction=ToxicityLevel(predicted_label),
                confidence=confidence,
                raw_scores=raw_scores,
                threshold_met=threshold_met,
            )
        elif task == "emotion":
            # Calculate emotion strength based on confidence and prediction
            strength = self._calculate_emotion_strength(prediction, predicted_label, confidence)
            return EmotionResult(
                prediction=EmotionType(predicted_label),
                confidence=confidence,
                strength=strength,
                raw_scores=raw_scores,
                threshold_met=threshold_met,
            )
        elif task == "bullying":
            return BullyingResult(
                prediction=BullyingType(predicted_label),
                confidence=confidence,
                raw_scores=raw_scores,
                threshold_met=threshold_met,
            )
        elif task == "role":
            return RoleResult(
                prediction=RoleType(predicted_label),
                confidence=confidence,
                raw_scores=raw_scores,
                threshold_met=threshold_met,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def _calculate_emotion_strength(
        self, prediction: torch.Tensor, predicted_label: str, confidence: float
    ) -> int:
        """
        Calculate emotion strength on 0-4 scale.

        Args:
            prediction: Raw prediction tensor
            predicted_label: Predicted emotion label
            confidence: Confidence score

        Returns:
            Emotion strength (0-4)
        """
        if predicted_label == "neu":
            return 0  # Neutral has no strength

        # Use max probability and confidence to determine strength
        probabilities = torch.softmax(prediction, dim=-1)
        max_prob = probabilities.max().item()

        # Combine confidence and max probability
        intensity = (confidence + max_prob) / 2.0

        # Map to 0-4 scale
        if intensity < 0.3:
            return 1
        elif intensity < 0.5:
            return 2
        elif intensity < 0.7:
            return 3
        else:
            return 4

    def _generate_explanations(
        self, text: str, predictions: Dict[str, torch.Tensor]
    ) -> Dict[str, ExplanationResult]:
        """
        Generate explanations for predictions using integrated gradients.

        Args:
            text: Input text
            predictions: Ensemble predictions

        Returns:
            Dictionary mapping tasks to explanation results
        """
        explanations = {}

        try:
            if self.explainer is None:
                logger.warning("Explainer not available" ", skipping explanations")
                return explanations

            for task, prediction in predictions.items():
                try:
                    explanation = self.explainer.explain(
                        text=text, task=task, target_class=prediction.argmax().item()
                    )

                    # Extract top contributing words
                    top_words = []
                    if "attributions" in explanation and "tokens" in explanation:
                        word_attrs = list(zip(explanation["tokens"], explanation["attributions"]))
                        word_attrs.sort(key=lambda x: abs(x[1]), reverse=True)
                        top_words = [(word, float(attr)) for word, attr in word_attrs[:5]]

                    explanations[task] = ExplanationResult(
                        attributions=explanation.get("attributions", []),
                        tokens=explanation.get("tokens", []),
                        explanation_text=explanation.get("explanation", ""),
                        top_contributing_words=top_words,
                        method="integrated_gradients",
                    )

                except Exception:
                    logger.error("Explanation generation fa" "iled for task {task}: {e}")
                    # Create empty explanation as fallback
                    explanations[task] = ExplanationResult(
                        attributions=[],
                        tokens=[],
                        explanation_text="Explanation generat" "ion failed: {str(e)}",
                        top_contributing_words=[],
                        method="integrated_gradients",
                    )

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")

        return explanations

    def analyze(
        self,
        text: str,
        context: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> DetectionResult:
        """
        Analyze text for toxicity, emotion, bullying, and role.

        Args:
            text: Input text to analyze
            context: Optional conversation context
            timeout: Optional timeout in seconds

        Returns:
            Complete detection result

        Raises:
            ValueError: If text is invalid
            TypeError: If text is not a string
            TimeoutError: If analysis times out
            RuntimeError: If models are not loaded
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call _load_models() first.")

        start_time = time.time()

        # Input validation
        self._validate_input(text)

        # Preprocess text
        preprocessed_text = self._preprocess_text(text)

        def _analyze_with_timeout():
            """Internal analysis function for timeout handling."""
            # Get individual model predictions
            model_predictions = self._get_individual_predictions(preprocessed_text, context)

            # Ensemble predictions
            ensemble_predictions = self._ensemble_predict(preprocessed_text, model_predictions)

            # Get confidence thresholds
            thresholds = self.config.get(
                "confidence_thresholds", ConfidenceThresholds.DEFAULT_THRESHOLDS
            )

            # Create task-specific results
            task_results = {}
            for task, prediction in ensemble_predictions.items():
                task_thresholds = thresholds.get(task, {})
                task_results[task] = self._create_task_result(prediction, task, task_thresholds)

            # Generate explanations
            explanations = self._generate_explanations(preprocessed_text, ensemble_predictions)

            # Create model prediction results
            model_prediction_results = {}
            for model_name, result in model_predictions.items():
                model_prediction_results[model_name] = ModelPrediction(
                    model_name=model_name,
                    predictions=result["predictions"],
                    confidence_scores={
                        task: self._calibrate_confidence(pred, task)
                        for task, pred in result["predictions"].items()
                    },
                    processing_time=result.get("processing_time", 0.0),
                )

            return task_results, explanations, model_prediction_results

        # Execute with optional timeout
        if timeout:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_analyze_with_timeout)
                    task_results, explanations, model_prediction_results = future.result(
                        timeout=timeout
                    )
            except FutureTimeoutError:
                raise TimeoutError(f"Analysis timed out after {timeout} seconds")
        else:
            task_results, explanations, model_prediction_results = _analyze_with_timeout()

        # Calculate processing time
        processing_time = time.time() - start_time

        # Update performance tracking
        with self._lock:
            self._prediction_count += 1
            self._total_processing_time += processing_time

        # Create final result
        result = DetectionResult(
            text=text,
            timestamp=datetime.now(),
            processing_time=processing_time,
            toxicity=task_results["toxicity"],
            emotion=task_results["emotion"],
            bullying=task_results["bullying"],
            role=task_results["role"],
            explanations=explanations,
            model_predictions=model_prediction_results,
            ensemble_weights=self.ensemble_weights,
            context_used=context is not None,
            context_length=len(context) if context else 0,
            model_version=self.config.get("model_version", "1.0.0"),
        )

        logger.debug(f"Analysis completed in {processing_time:.3f}s")
        return result

    def analyze_batch(
        self,
        texts: List[str],
        context: Optional[List[List[str]]] = None,
        batch_size: int = 8,
        timeout: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        Analyze multiple texts in batch.

        Args:
            texts: List of texts to analyze
            context: Optional list of context for each text
            batch_size: Batch size for processing
            timeout: Optional timeout per batch

        Returns:
            List of detection results

        Raises:
            ValueError: If inputs are invalid
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        if context and len(context) != len(texts):
            raise ValueError("Context list must have same length as texts")

        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))

            batch_texts = texts[start_idx:end_idx]
            batch_context = context[start_idx:end_idx] if context else None

            logger.debug(f"Processing batch {batch_idx + 1}/{total_batches}")

            for i, text in enumerate(batch_texts):
                text_context = batch_context[i] if batch_context else None
                result = self.analyze(text, context=text_context, timeout=timeout)
                results.append(result)

        return results

    def _validate_input(self, text: str) -> None:
        """Validate input text."""
        if text is None:
            raise ValueError("Text cannot be None")

        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got {type(text)}")

        if not text.strip():
            raise ValueError("Text cannot be empty")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if self._prediction_count == 0:
                return {
                    "total_predictions": 0,
                    "average_processing_time": 0.0,
                    "total_processing_time": 0.0,
                }

            return {
                "total_predictions": self._prediction_count,
                "average_processing_time": self._total_processing_time / self._prediction_count,
                "total_processing_time": self._total_processing_time,
                "device": self.device,
                "models_loaded": self._models_loaded,
            }

    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        with self._lock:
            self._prediction_count = 0
            self._total_processing_time = 0.0

    def update_ensemble_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update ensemble weights.

        Args:
            new_weights: New ensemble weights
        """
        self.ensemble_weights = self._normalize_weights(new_weights)
        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")

    def is_ready(self) -> bool:
        """Check if detector is ready for inference."""
        return (
            self._models_loaded
            and all(model is not None for model in self.models.values())
            and self.explainer is not None
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        if not self._models_loaded:
            return {"status": "models_not_loaded"}

        info = {
            "status": "ready",
            "device": self.device,
            "ensemble_weights": self.ensemble_weights,
            "models": {},
        }

        for name, model in self.models.items():
            info["models"][name] = {
                "type": type(model).__name__,
                "device": str(getattr(model, "device", "unknown")),
                "parameters": sum(
                    p.numel() for p in model.parameters() if hasattr(model, "parameters")
                ),
            }

        return info

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        # Clear GPU memory if using CUDA
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        logger.info("CyberPuppyDetector resources cleaned up")


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model setup."""

    models: List[str] = None
    weights: List[float] = None
    voting_strategy: str = "weighted"  # "weighted", "majority", "average"

    def __post_init__(self):
        """Validate ensemble configuration."""
        if self.models is None:
            self.models = ["baseline", "contextual", "weak_supervision"]

        if self.weights is None:
            # Default equal weights
            self.weights = [1.0 / len(self.models)] * len(self.models)

        if len(self.models) != len(self.weights):
            raise ValueError(
                f"Number of models ({len(self.models)}) must match number of weights ({len(self.weights)})"
            )

        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights)
        if abs(total_weight - 1.0) > 1e-6:
            self.weights = [w / total_weight for w in self.weights]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "models": self.models,
            "weights": self.weights,
            "voting_strategy": self.voting_strategy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnsembleConfig":
        """Create from dictionary."""
        config = cls()
        config.models = data.get("models", ["baseline", "contextual", "weak_supervision"])
        config.weights = data.get("weights", [1.0 / len(config.models)] * len(config.models))
        config.voting_strategy = data.get("voting_strategy", "weighted")
        config.__post_init__()  # Re-validate
        return config

    def is_valid(self) -> bool:
        """Validate configuration."""
        try:
            if not self.models:
                return False
            if not self.weights:
                return False
            if len(self.models) != len(self.weights):
                return False
            if abs(sum(self.weights) - 1.0) > 1e-6:
                return False
            if self.voting_strategy not in ["weighted", "majority", "average"]:
                return False
            return True
        except:
            return False


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""

    normalize_unicode: bool = True
    remove_urls: bool = False
    remove_mentions: bool = False
    convert_traditional_to_simplified: bool = True
    max_length: int = 512
    clean_whitespace: bool = True
    lowercase: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "normalize_unicode": self.normalize_unicode,
            "remove_urls": self.remove_urls,
            "remove_mentions": self.remove_mentions,
            "convert_traditional_to_simplified": self.convert_traditional_to_simplified,
            "max_length": self.max_length,
            "clean_whitespace": self.clean_whitespace,
            "lowercase": self.lowercase,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingConfig":
        """Create from dictionary."""
        return cls(
            normalize_unicode=data.get("normalize_unicode", True),
            remove_urls=data.get("remove_urls", False),
            remove_mentions=data.get("remove_mentions", False),
            convert_traditional_to_simplified=data.get("convert_traditional_to_simplified", True),
            max_length=data.get("max_length", 512),
            clean_whitespace=data.get("clean_whitespace", True),
            lowercase=data.get("lowercase", False),
        )

    def is_valid(self) -> bool:
        """Validate configuration."""
        try:
            if self.max_length <= 0 or self.max_length > 10000:
                return False
            return True
        except:
            return False
