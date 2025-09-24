"""
CyberPuppy - Chinese Cyberbullying Detection and Moderation Bot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A comprehensive solution for detecting and moderating cyberbullying
in Chinese text with explainable AI capabilities.

Basic usage:
    >>> from cyberpuppy import CyberPuppyDetector
    >>> detector = CyberPuppyDetector()
    >>> result = detector.analyze("你好世界")
    >>> print(result.toxicity_score)

:copyright: (c) 2024 CyberPuppy Team.
:license: MIT, see LICENSE for more details.
"""

__version__ = "0.1.0"
__author__ = "CyberPuppy Team"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models.detector import CyberPuppyDetector
    from .models.result import DetectionResult

__all__ = [
    "CyberPuppyDetector",
    "DetectionResult",
    "__version__",
]


def get_version() -> str:
    """Return the current version of CyberPuppy."""
    return __version__
