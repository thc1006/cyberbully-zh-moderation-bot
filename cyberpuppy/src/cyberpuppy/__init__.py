"""
CyberPuppy - Cyberbully ZH Moderation Bot

A modern Python framework for content moderation and analysis.
"""

__version__ = "0.1.0"
__author__ = "CyberPuppy Team"

from .config import Config, get_config

__all__ = [
    "Config",
    "get_config",
]
