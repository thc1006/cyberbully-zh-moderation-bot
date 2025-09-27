"""
Simplified model loader for testing purposes.
Provides a mock detector with the same interface as the full model loader.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MockDetector:
    """Mock detector for testing."""

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text and return mock results.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with toxicity, bullying, role, emotion analysis
        """
        return {
            "toxicity": {"level": "none", "confidence": 0.9},
            "bullying": {"level": "none", "confidence": 0.85},
            "role": {"type": "none", "confidence": 0.88},
            "emotion": {"label": "neu", "confidence": 0.82},
            "emotion_strength": 0,
            "scores": {
                "toxicity_score": 0.1,
                "bullying_score": 0.05,
                "emotion_score": 0.5
            },
            "explanations": {
                "important_words": [
                    {"word": "测试", "importance": 0.75},
                    {"word": "文本", "importance": 0.65}
                ],
                "method": "keyword_based_mock",
                "confidence": 0.75
            }
        }


class SimpleModelLoader:
    """Simplified model loader for testing."""

    def __init__(self):
        self.detector = None
        logger.info("SimpleModelLoader initialized")

    def load_models(self) -> MockDetector:
        """Load mock detector."""
        if self.detector is None:
            self.detector = MockDetector()
            logger.info("Mock detector loaded")
        return self.detector

    def get_status(self) -> Dict[str, Any]:
        """Get loader status."""
        return {
            "models_loaded": self.detector is not None,
            "device": "cpu",
            "warmup_complete": True
        }


def get_model_loader() -> SimpleModelLoader:
    """Get singleton model loader instance."""
    return SimpleModelLoader()