"""
Configuration module for CyberPuppy
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

# Compatibility imports for different Pydantic versions
try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings
    PYDANTIC_V2 = True
except ImportError:
    # Fallback for older Pydantic versions
    from pydantic import Field
    try:
        from pydantic_settings import BaseSettings
    except ImportError:
        from pydantic import BaseSettings
    PYDANTIC_V2 = False

    # Create a compatibility wrapper for field_validator
    def field_validator(field_name, mode=None):
        def decorator(func):
            return func  # Skip validation in older versions for now
        return decorator


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "CyberPuppy"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Optional[Path] = Field(default=None)
    MODEL_DIR: Optional[Path] = Field(default=None)
    CACHE_DIR: Optional[Path] = Field(default=None)

    # Model Configuration
    BASE_MODEL: str = Field(default="hfl/chinese-macbert-base")
    EMOTION_MODEL: str = Field(default="hfl/chinese-roberta-wwm-ext")
    MAX_LENGTH: int = Field(default=512)
    BATCH_SIZE: int = Field(default=16)

    # Toxicity Labels
    TOXICITY_LABELS: List[str] = Field(default=["none", "toxic", "severe"])
    BULLYING_LABELS: List[str] = Field(default=["none", "harassment", "threat"])
    ROLE_LABELS: List[str] = Field(
        default=["none", "perpetrator", "victim", "bystander"]
    )
    EMOTION_LABELS: List[str] = Field(default=["positive", "neutral", "negative"])

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_PREFIX: str = Field(default="/api/v1")
    CORS_ORIGINS: List[str] = Field(default=["*"])

    # LINE Bot Configuration
    LINE_CHANNEL_ACCESS_TOKEN: Optional[str] = Field(default=None)
    LINE_CHANNEL_SECRET: Optional[str] = Field(default=None)
    LINE_WEBHOOK_URL: Optional[str] = Field(default=None)

    # Perspective API (Optional)
    PERSPECTIVE_API_KEY: Optional[str] = Field(default=None)
    USE_PERSPECTIVE_API: bool = Field(default=False)

    # Safety Thresholds
    TOXICITY_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)
    SEVERE_TOXICITY_THRESHOLD: float = Field(default=0.85, ge=0.0, le=1.0)
    EMOTION_NEGATIVE_THRESHOLD: float = Field(default=0.8, ge=0.0, le=1.0)

    # Performance
    USE_GPU: bool = Field(default=True)
    GPU_DEVICE: int = Field(default=0)
    NUM_WORKERS: int = Field(default=4)

    # Privacy
    LOG_USER_CONTENT: bool = Field(default=False)
    HASH_USER_IDS: bool = Field(default=True)

    @field_validator("DATA_DIR", mode="before")
    @classmethod
    def set_data_dir(cls, v):
        if v is None:
            project_root = Path(__file__).parent.parent.parent
            return project_root / "data"
        return Path(v)

    @field_validator("MODEL_DIR", mode="before")
    @classmethod
    def set_model_dir(cls, v):
        if v is None:
            project_root = Path(__file__).parent.parent.parent
            return project_root / "models"
        return Path(v)

    @field_validator("CACHE_DIR", mode="before")
    @classmethod
    def set_cache_dir(cls, v):
        if v is None:
            project_root = Path(__file__).parent.parent.parent
            return project_root / ".cache"
        return Path(v)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

    def get_data_path(self, subpath: str = "") -> Path:
        """Get path within data directory."""
        path = self.DATA_DIR / subpath
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_model_path(self, subpath: str = "") -> Path:
        """Get path within model directory."""
        path = self.MODEL_DIR / subpath
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_cache_path(self, subpath: str = "") -> Path:
        """Get path within cache directory."""
        path = self.CACHE_DIR / subpath
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary, excluding sensitive information."""
        data = self.model_dump()
        # Remove sensitive keys
        sensitive_keys = [
            "LINE_CHANNEL_ACCESS_TOKEN",
            "LINE_CHANNEL_SECRET",
            "PERSPECTIVE_API_KEY",
        ]
        for key in sensitive_keys:
            if key in data:
                data[key] = "***" if data[key] else None
        return data


# Global settings instance
settings = Settings()


# Configuration presets for different environments
class DevelopmentConfig:
    """Development environment configuration."""

    DEBUG = True
    LOG_LEVEL = "DEBUG"
    USE_GPU = False
    BATCH_SIZE = 4


class ProductionConfig:
    """Production environment configuration."""

    DEBUG = False
    LOG_LEVEL = "WARNING"
    USE_GPU = True
    BATCH_SIZE = 32
    LOG_USER_CONTENT = False


class TestingConfig:
    """Testing environment configuration."""

    DEBUG = True
    LOG_LEVEL = "DEBUG"
    USE_GPU = False
    BATCH_SIZE = 2
    DATA_DIR = Path("/tmp/cyberpuppy/test_data")
    MODEL_DIR = Path("/tmp/cyberpuppy/test_models")
    CACHE_DIR = Path("/tmp/cyberpuppy/test_cache")


def get_config(env: str = "development") -> Settings:
    """Get configuration based on environment."""
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig,
    }

    config_class = configs.get(env, DevelopmentConfig)

    # Override settings with environment-specific values
    for key, value in config_class.__dict__.items():
        if not key.startswith("_"):
            setattr(settings, key, value)

    return settings


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or return default configuration.

    Args:
        config_path: Path to configuration file (YAML or JSON)

    Returns:
        Configuration dictionary
    """
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            if config_path.endswith((".yaml", ".yml")):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    else:
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for CyberPuppy.

    Returns:
        Default configuration dictionary
    """
    return {
        "base_model": "hfl/chinese-macbert-base",
        "model_version": "1.0.0",
        "max_length": 512,
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 2e-5,
        "ensemble_weights": {
            "baseline": 0.4,
            "contextual": 0.35,
            "weak_supervision": 0.25,
        },
        "confidence_thresholds": {
            "toxicity": {"none": 0.5, "toxic": 0.7, "severe": 0.85},
            "bullying": {"none": 0.5, "harassment": 0.7, "threat": 0.85},
            "emotion": {"pos": 0.6, "neu": 0.5, "neg": 0.6},
            "role": {"none": 0.5, "perpetrator": 0.7, "victim": 0.6, "bystander": 0.6},
        },
        "preprocessing": {
            "max_length": 512,
            "normalize_unicode": True,
            "convert_traditional": True,
        },
        "model_paths": {},
        "device": "auto",
        "use_gpu": True,
        "num_workers": 4,
    }
