"""
Model loader utilities for CyberPuppy API.

This module provides utilities for loading trained models, GPU/CPU detection,
model caching, and warm-up operations for the FastAPI application.
"""

import json
import logging
import os
# Import CyberPuppy components
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch

sys.path.append(str(Path(__file__).parent.parent))
from src.cyberpuppy.models.baselines import (BaselineModel,  # noqa: E402
                                             ModelConfig)
from src.cyberpuppy.models.detector import CyberPuppyDetector  # noqa: E402

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class ModelCache:
    """Thread-safe model cache with performance tracking."""

    def __init__(self):
        self._models = {}
        self._load_times = {}
        self._access_count = {}

    def get(self, key: str) -> Optional[Any]:
        """Get model from cache and track access."""
        if key in self._models:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._models[key]
        return None

    def put(self, key: str, model: Any, load_time: float) -> None:
        """Put model in cache with metadata."""
        self._models[key] = model
        self._load_times[key] = load_time
        self._access_count[key] = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_models": list(self._models.keys()),
            "load_times": self._load_times.copy(),
            "access_counts": self._access_count.copy(),
            "total_models": len(self._models),
        }

    def clear(self) -> None:
        """Clear cache."""
        # Clear GPU memory if needed
        for model in self._models.values():
            if hasattr(model, "device") and str(model.device).startswith("cuda"):
                del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._models.clear()
        self._load_times.clear()
        self._access_count.clear()


class ModelLoader:
    """Model loader with GPU/CPU detection and caching."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.cache = ModelCache()
        self.device = self._detect_device()
        self.detector = None
        self._warmup_complete = False

        logger.info(f"ModelLoader initialized with device: {self.device}")

    def _detect_device(self) -> str:
        """Detect optimal computation device."""
        try:
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                device_props = torch.cuda.get_device_properties(0)
                memory_gb = device_props.total_memory / (1024**3)
                logger.info(f"CUDA available: {gpu_name} ({memory_gb:.1f}GB)")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Apple MPS available")
            else:
                device = "cpu"
                logger.info("Using CPU device")

            return device
        except Exception as e:
            logger.warning(f"Device detection failed: {e}, defaulting to CPU")
            return "cpu"

    def _load_model_config(self, model_path: Path) -> Dict[str, Any]:
        """Load model configuration from JSON file."""
        config_path = model_path / "model_config.json"

        if not config_path.exists():
            logger.warning(f"No config found at {config_path}, using defaults")
            return self._get_default_config()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            "model_name": "hfl/chinese-macbert-base",
            "max_length": 256,
            "num_toxicity_classes": 3,
            "num_bullying_classes": 3,
            "num_role_classes": 4,
            "num_emotion_classes": 3,
        }

    def _create_model_config(self, config_dict: Dict[str, Any]) -> ModelConfig:
        """Create ModelConfig object from dictionary."""
        return ModelConfig(
            model_name=config_dict.get("model_name", "hfl/chinese-macbert-base"),
            num_toxicity_classes=config_dict.get("num_toxicity_classes", 3),
            num_emotion_classes=config_dict.get("num_emotion_classes", 3),
            num_bullying_classes=config_dict.get("num_bullying_classes", 3),
            num_role_classes=config_dict.get("num_role_classes", 4),
        )

    def _load_single_model(self, model_path: Path, model_type: str) -> BaselineModel:
        """Load a single trained model."""
        checkpoint_path = model_path / "best.ckpt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: \
                {checkpoint_path}"
            )

        # Load configuration
        config_dict = self._load_model_config(model_path)
        model_config = self._create_model_config(config_dict)

        # Create model instance
        model = BaselineModel(model_config)

        # Load trained weights
        start_time = time.time()
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            model = model.to(self.device)
            model.eval()

            load_time = time.time() - start_time
            logger.info(
                f"Loaded {model_type} model from {checkpoint_path} \
                in {load_time:.2f}s"
            )

            return model

        except Exception as e:
            logger.error(
                f"Failed to load model weights from \
                {checkpoint_path}: {e}"
            )
            raise RuntimeError(f"Model loading failed: {e}")

    def load_models(self) -> CyberPuppyDetector:
        """Load all trained models and create detector."""
        cache_key = "cyberpuppy_detector"
        cached_detector = self.cache.get(cache_key)

        if cached_detector is not None:
            logger.info("Using cached CyberPuppy detector")
            return cached_detector

        start_time = time.time()

        try:
            # Define model paths - using the toxicity specialist as primary
            toxicity_model_path = self.models_dir / "toxicity_only_demo"
            multitask_model_path = self.models_dir / "macbert_base_demo"

            # Check which models are available
            model_paths = {}

            if toxicity_model_path.exists():
                logger.info("Found toxicity specialist model")
                model_paths["toxicity_specialist"] = str(toxicity_model_path / "best.ckpt")

            if multitask_model_path.exists():
                logger.info("Found multitask model")
                model_paths["multitask"] = str(multitask_model_path / "best.ckpt")

            if not model_paths:
                raise FileNotFoundError("No trained models found in models directory")

            # Create detector configuration
            detector_config = {
                "base_model": "hfl/chinese-macbert-base",
                "model_paths": model_paths,
                "ensemble_weights": {
                    "baseline": 1.0,  # Use single model for now
                    "contextual": 0.0,
                    "weak_supervision": 0.0,
                },
                "preprocessing": {
                    "max_length": 256,
                    "normalize_unicode": True,
                    "convert_traditional": True,
                },
                "confidence_thresholds": {
                    "toxicity": {"none": 0.3, "toxic": 0.7, "severe": 0.8},
                    "emotion": {"pos": 0.6, "neu": 0.5, "neg": 0.6},
                    "bullying": {"none": 0.3, "harassment": 0.7, "threat": 0.8},
                    "role": {
                        "none": 0.3,
                        "perpetrator": 0.7,
                        "victim": 0.7,
                        "bystander": 0.6,
                    },
                },
                "model_version": "1.0.0",
            }

            # For now, load only the baseline model to avoid complexity
            # We'll use a simplified approach with the toxicity specialist
            model_path_to_use = (
                toxicity_model_path if toxicity_model_path.exists() else multitask_model_path
            )
            detector = SimplifiedDetector(
                device=self.device, model_path=model_path_to_use, config=detector_config
            )

            load_time = time.time() - start_time
            self.cache.put(cache_key, detector, load_time)
            self.detector = detector

            logger.info(f"Successfully loaded CyberPuppy detector in {load_time:.2f}s")
            return detector

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def warm_up_models(self, sample_texts: Optional[list] = None) -> Dict[str, Any]:
        """Warm up models with sample predictions."""
        if self.detector is None:
            self.detector = self.load_models()

        if sample_texts is None:
            sample_texts = [
                "你好",
                "这是一个测试",
                "笨蛋，去死吧",
                "我很开心今天的天气很好",
            ]

        warmup_stats = {
            "warmup_samples": len(sample_texts),
            "warmup_times": [],
            "success_count": 0,
            "error_count": 0,
        }

        logger.info(f"Starting model warm-up with {len(sample_texts)} samples")

        for i, text in enumerate(sample_texts):
            try:
                start_time = time.time()
                result = self.detector.analyze(text)
                warmup_time = time.time() - start_time

                warmup_stats["warmup_times"].append(warmup_time)
                warmup_stats["success_count"] += 1

                logger.debug(
                    f"Warmup {i+1}/{len(sample_texts)}: {warmup_time:.3f}s "
                    f"(toxicity: {result['toxicity']})"
                )

            except Exception as e:
                logger.warning(f"Warmup sample {i+1} failed: {e}")
                warmup_stats["error_count"] += 1

        warmup_stats["average_time"] = (
            sum(warmup_stats["warmup_times"]) / len(warmup_stats["warmup_times"])
            if warmup_stats["warmup_times"]
            else 0
        )

        self._warmup_complete = True
        success_ratio = f"{warmup_stats['success_count']}/{len(sample_texts)}"
        logger.info(f"Model warm-up complete: {success_ratio} successful")

        return warmup_stats

    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status."""
        return {
            "device": self.device,
            "models_loaded": self.detector is not None,
            "warmup_complete": self._warmup_complete,
            "cache_stats": self.cache.get_stats(),
            "detector_ready": (
                self.detector is not None and self.detector.is_ready()
                if hasattr(self.detector, "is_ready")
                else self.detector is not None
            ),
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory": (self._get_gpu_memory_info() if torch.cuda.is_available() else None),
        }

    def _get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return {}

        try:
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            return {
                "allocated_gb": round(memory_allocated, 2),
                "cached_gb": round(memory_cached, 2),
                "total_gb": round(memory_total, 2),
                "utilization_percent": round((memory_allocated / memory_total) * 100, 1),
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {}

    def clear_cache(self) -> None:
        """Clear model cache and free memory."""
        self.cache.clear()
        self.detector = None
        self._warmup_complete = False
        logger.info("Model cache cleared")


class SimplifiedDetector:
    """Simplified detector using a single trained model."""

    def __init__(self, device: str, model_path: Path, config: Dict[str, Any]):
        self.device = device
        self.config = config
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """Load the trained model."""
        try:
            # Load configuration
            config_path = model_path / "model_config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                model_config_dict = json.load(f)

            # Create model config
            model_config = ModelConfig(
                model_name=model_config_dict.get("model_name", "hfl/chinese-macbert-base"),
                num_toxicity_classes=model_config_dict.get("num_toxicity_classes", 3),
                num_emotion_classes=model_config_dict.get("num_emotion_classes", 3),
                num_bullying_classes=model_config_dict.get("num_bullying_classes", 3),
                num_role_classes=model_config_dict.get("num_role_classes", 4),
            )

            # Load model
            self.model = BaselineModel(model_config)
            checkpoint_path = model_path / "best.ckpt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"SimplifiedDetector loaded model from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load simplified model: {e}")
            raise

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text using the loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Get predictions
            predictions = self.model.predict(text)

            # Convert to API format
            result = self._convert_predictions_to_api_format(predictions, text)
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def _convert_predictions_to_api_format(
        self, predictions: Dict[str, torch.Tensor], text: str
    ) -> Dict[str, Any]:
        """Convert model predictions to API response format."""

        # Label mappings
        label_mappings = {
            "toxicity": ["none", "toxic", "severe"],
            "emotion": ["pos", "neu", "neg"],
            "bullying": ["none", "harassment", "threat"],
            "role": ["none", "perpetrator", "victim", "bystander"],
        }

        result = {
            "scores": {},
            "explanations": {
                "important_words": [],
                "method": "baseline_model",
                "confidence": 0.0,
            },
        }

        overall_confidence = 0.0
        confidence_count = 0

        # Process each task
        for task, prediction in predictions.items():
            if task not in label_mappings:
                continue

            # Apply softmax to get probabilities
            probabilities = torch.softmax(prediction, dim=-1)
            predicted_idx = probabilities.argmax().item()
            max_prob = probabilities.max().item()

            # Get labels and scores
            labels = label_mappings[task]
            if predicted_idx < len(labels):
                predicted_label = labels[predicted_idx]
            else:
                predicted_label = labels[0]

            # Create scores dict
            scores_dict = {}
            for i, label in enumerate(labels):
                if i < len(probabilities):
                    scores_dict[label] = probabilities[i].item()

            # Store results
            result[task] = predicted_label
            result["scores"][task] = scores_dict

            # Update overall confidence
            overall_confidence += max_prob
            confidence_count += 1

        # Calculate emotion strength for API compatibility
        if "emotion" in result:
            emotion_label = result["emotion"]
            if emotion_label == "neu":
                result["emotion_strength"] = 0
            else:
                # Map confidence to 1-4 scale
                emotion_conf = result["scores"]["emotion"].get(emotion_label, 0.5)
                if emotion_conf < 0.3:
                    result["emotion_strength"] = 1
                elif emotion_conf < 0.5:
                    result["emotion_strength"] = 2
                elif emotion_conf < 0.7:
                    result["emotion_strength"] = 3
                else:
                    result["emotion_strength"] = 4
        else:
            result["emotion"] = "neu"
            result["emotion_strength"] = 0
            result["scores"]["emotion"] = {"pos": 0.33, "neu": 0.34, "neg": 0.33}

        # Generate simple explanations based on keywords
        result["explanations"]["important_words"] = self._generate_simple_explanations(text, result)
        result["explanations"]["confidence"] = overall_confidence / max(confidence_count, 1)

        return result

    def _generate_simple_explanations(self, text: str, predictions: Dict[str, Any]) -> list:
        """Generate simple explanations based on keyword matching."""
        # Simple keyword-based explanations
        toxic_keywords = ["笨蛋", "白痴", "去死", "滚", "废物", "蠢", "傻"]
        severe_keywords = ["杀死", "自杀", "威胁", "死"]
        positive_keywords = ["开心", "高兴", "好棒", "谢谢", "感谢", "不错"]
        negative_keywords = ["难过", "生气", "讨厌", "糟糕", "愤怒"]

        important_words = []
        words = text.split()

        for word in words:
            importance = 0.1  # Base importance

            # Check for toxic keywords
            if any(kw in word for kw in toxic_keywords):
                importance = 0.8
            elif any(kw in word for kw in severe_keywords):
                importance = 0.9
            elif any(kw in word for kw in positive_keywords):
                importance = 0.6
            elif any(kw in word for kw in negative_keywords):
                importance = 0.7

            if importance > 0.5 or len(important_words) < 3:
                important_words.append({"word": word, "importance": importance})

        # Sort by importance and take top 5
        important_words.sort(key=lambda x: x["importance"], reverse=True)
        return important_words[:5]

    def is_ready(self) -> bool:
        """Check if detector is ready."""
        return self.model is not None


# Global model loader instance
_model_loader = None


def get_model_loader() -> ModelLoader:
    """Get global model loader instance."""
    global _model_loader
    if _model_loader is None:
        models_dir = os.getenv("MODELS_DIR", "models")
        _model_loader = ModelLoader(models_dir)
    return _model_loader
