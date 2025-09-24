"""
CyberPuppy models package.

This package contains all machine learning models and result classes
for the CyberPuppy detection system.
"""

from .result import (
    DetectionResult,
    ToxicityResult,
    EmotionResult,
    BullyingResult,
    RoleResult,
    ExplanationResult,
    ModelPrediction,
    ToxicityLevel,
    EmotionType,
    BullyingType,
    RoleType,
    ResultAggregator,
    ConfidenceThresholds
)

# Import detector only if dependencies are available
try:
    from .detector import CyberPuppyDetector  # noqa: F401
    __all__ = [
        "DetectionResult",
        "ToxicityResult",
        "EmotionResult",
        "BullyingResult",
        "RoleResult",
        "ExplanationResult",
        "ModelPrediction",
        "ToxicityLevel",
        "EmotionType",
        "BullyingType",
        "RoleType",
        "ResultAggregator",
        "ConfidenceThresholds",
        "CyberPuppyDetector"
    ]
except ImportError:
    # Allow imports without detector if dependencies not available
    __all__ = [
        "DetectionResult",
        "ToxicityResult",
        "EmotionResult",
        "BullyingResult",
        "RoleResult",
        "ExplanationResult",
        "ModelPrediction",
        "ToxicityLevel",
        "EmotionType",
        "BullyingType",
        "RoleType",
        "ResultAggregator",
        "ConfidenceThresholds"
    ]
