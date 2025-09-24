"""
Simplified model loader for API testing without complex dependencies.
This version uses mock predictions but maintains the API contract.
"""

import os
import logging
import time
import random
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SimpleMockDetector:
    """Simple mock detector that mimics real model behavior."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.loaded = True

    def analyze(self, text: str) -> Dict[str, Any]:
        """Mock analysis that returns realistic predictions."""
        # Simple keyword-based analysis for testing
        text_lower = text.lower()

        # Toxicity detection
        toxic_keywords = ["笨蛋", "白痴", "去死", "滚", "废物", "蠢", "傻"]
        severe_keywords = ["杀死", "自杀", "威胁", "死"]

        has_toxic = any(keyword in text_lower for keyword in toxic_keywords)
        has_severe = any(keyword in text_lower for keyword in severe_keywords)

        if has_severe:
            toxicity = "severe"
            tox_scores = {"none": 0.1, "toxic": 0.2, "severe": 0.7}
        elif has_toxic:
            toxicity = "toxic"
            tox_scores = {"none": 0.2, "toxic": 0.7, "severe": 0.1}
        else:
            toxicity = "none"
            tox_scores = {"none": 0.8, "toxic": 0.15, "severe": 0.05}

        # Bullying detection
        harassment_keywords = ["骚扰", "烦人", "讨厌", "没用", "废物"]
        threat_keywords = ["威胁", "警告", "后果", "小心", "等着"]

        has_harassment = any(keyword in text_lower for keyword in harassment_keywords)
        has_threat = any(keyword in text_lower for keyword in threat_keywords)

        if has_threat or has_severe:
            bullying = "threat"
            bully_scores = {"none": 0.1, "harassment": 0.2, "threat": 0.7}
        elif has_harassment or has_toxic:
            bullying = "harassment"
            bully_scores = {"none": 0.2, "harassment": 0.7, "threat": 0.1}
        else:
            bullying = "none"
            bully_scores = {"none": 0.8, "harassment": 0.15, "threat": 0.05}

        # Role analysis
        victim_keywords = ["帮助", "救命", "被欺负", "难过", "伤心"]
        perpetrator_keywords = ["笨蛋", "废物"] + toxic_keywords

        has_victim = any(keyword in text_lower for keyword in victim_keywords)
        has_perpetrator = any(keyword in text_lower for keyword in perpetrator_keywords)

        if has_perpetrator and not has_victim:
            role = "perpetrator"
            role_scores = {
                "none": 0.1,
                "perpetrator": 0.7,
                "victim": 0.1,
                "bystander": 0.1,
            }
        elif has_victim:
            role = "victim"
            role_scores = {
                "none": 0.1,
                "perpetrator": 0.05,
                "victim": 0.75,
                "bystander": 0.1,
            }
        else:
            role = "none"
            role_scores = {
                "none": 0.7,
                "perpetrator": 0.1,
                "victim": 0.1,
                "bystander": 0.1,
            }

        # Emotion analysis
        positive_keywords = ["开心", "高兴", "好棒", "谢谢", "感谢", "不错", "棒"]
        negative_keywords = ["难过", "生气", "讨厌", "糟糕", "愤怒"] + toxic_keywords

        has_positive = any(keyword in text_lower for keyword in positive_keywords)
        has_negative = any(keyword in text_lower for keyword in negative_keywords)

        if has_positive and not has_negative:
            emotion = "pos"
            emotion_strength = 3
            emo_scores = {"positive": 0.7, "neutral": 0.2, "negative": 0.1}
        elif has_negative:
            emotion = "neg"
            emotion_strength = 4 if has_severe else 3
            emo_scores = {"positive": 0.1, "neutral": 0.2, "negative": 0.7}
        else:
            emotion = "neu"
            emotion_strength = 1
            emo_scores = {"positive": 0.3, "neutral": 0.5, "negative": 0.2}

        # Generate explanations based on keywords
        important_words = []
        words = text.split()
        for word in words[:5]:
            importance = 0.1  # Base importance

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

        # Sort by importance
        important_words.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "toxicity": toxicity,
            "bullying": bullying,
            "role": role,
            "emotion": emotion,
            "emotion_strength": emotion_strength,
            "scores": {
                "toxicity": tox_scores,
                "bullying": bully_scores,
                "role": role_scores,
                "emotion": emo_scores,
            },
            "explanations": {
                "important_words": important_words[:5],
                "method": "keyword_based_mock",
                "confidence": 0.75 + random.random() * 0.2,
            },
        }

    def is_ready(self) -> bool:
        return True


class SimpleModelLoader:
    """Simplified model loader for API testing."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.device = "cpu"
        self.detector = None
        self._warmup_complete = False
        self.cache_stats = {
            "cached_models": ["mock_detector"],
            "load_times": {"mock_detector": 0.1},
            "access_counts": {"mock_detector": 0},
            "total_models": 1,
        }

        logger.info(
            f"SimpleModelLoader initialized with device: \
            {self.device}"
        )

    def load_models(self) -> SimpleMockDetector:
        """Load mock detector."""
        if self.detector is not None:
            return self.detector

        start_time = time.time()

        # Create mock detector
        self.detector = SimpleMockDetector(device=self.device)

        load_time = time.time() - start_time
        self.cache_stats["load_times"]["mock_detector"] = load_time

        logger.info(f"Mock detector loaded in {load_time:.3f}s")
        return self.detector

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

        logger.info(
            f"Starting mock model warm-up with {len(sample_texts)} \
            samples"
        )

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
        logger.info(f"Mock model warm-up complete: {success_ratio} successful")

        return warmup_stats

    def get_model_status(self) -> Dict[str, Any]:
        """Get model status."""
        return {
            "device": self.device,
            "models_loaded": self.detector is not None,
            "warmup_complete": self._warmup_complete,
            "cache_stats": self.cache_stats,
            "detector_ready": self.detector is not None and self.detector.is_ready(),
            "gpu_available": False,
            "gpu_memory": None,
            "model_type": "mock_detector",
        }

    def clear_cache(self) -> None:
        """Clear model cache."""
        self.detector = None
        self._warmup_complete = False
        self.cache_stats["access_counts"] = {"mock_detector": 0}
        logger.info("Mock model cache cleared")


# Global instance
_simple_model_loader = None


def get_model_loader() -> SimpleModelLoader:
    """Get global simple model loader instance."""
    global _simple_model_loader
    if _simple_model_loader is None:
        models_dir = os.getenv("MODELS_DIR", "models")
        _simple_model_loader = SimpleModelLoader(models_dir)
    return _simple_model_loader
