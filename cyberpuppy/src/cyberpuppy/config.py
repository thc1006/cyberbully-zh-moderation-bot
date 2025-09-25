"""
Configuration management for CyberPuppy.

This module handles all configuration settings, environment variables,
and path conventions for the application.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


class PathConfig(BaseModel):
    """Path configuration for data and models."""

    base_dir: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data")
    models_dir: Path = Field(default_factory=lambda: Path.cwd() / "models")
    logs_dir: Path = Field(default_factory=lambda: Path.cwd() / "logs")
    cache_dir: Path = Field(default_factory=lambda: Path.cwd() / "cache")

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for path in [self.data_dir, self.models_dir, self.logs_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)


class ModelConfig(BaseModel):
    """Model configuration settings."""

    default_model: str = "bert-base-chinese"
    model_cache_dir: Optional[Path] = None
    device: str = "cuda"  # cuda, cpu, or mps
    batch_size: int = 32
    max_sequence_length: int = 512


class APIConfig(BaseModel):
    """API configuration settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False
    cors_origins: list[str] = ["*"]


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    url: str = "sqlite:///./data/cyberpuppy.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


class LoggingConfig(BaseModel):
    """Logging configuration settings."""

    level: str = "INFO"
    format: str = "{time} | {level} | {message}"
    rotation: str = "10 MB"
    retention: str = "7 days"
    backtrace: bool = True
    diagnose: bool = True


class Config(BaseSettings):
    """Main configuration class for CyberPuppy."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="allow",
    )

    # Application settings
    app_name: str = "CyberPuppy"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", alias="ENV")
    debug: bool = Field(default=False, alias="DEBUG")

    # Sub-configurations
    paths: PathConfig = Field(default_factory=PathConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Feature flags
    enable_cache: bool = True
    enable_monitoring: bool = True
    enable_profiling: bool = False

    def __init__(self, **kwargs):
        """Initialize configuration with environment variables."""
        # Load .env file if it exists
        env_file = Path.cwd() / ".env"
        if env_file.exists():
            load_dotenv(env_file)

        super().__init__(**kwargs)

        # Ensure all required directories exist
        self.paths.ensure_directories()

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Config":
        """Create configuration from environment file."""
        if env_file and env_file.exists():
            load_dotenv(env_file)
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def save_env(self, path: Optional[Path] = None) -> None:
        """Save current configuration to .env file."""
        env_path = path or (Path.cwd() / ".env")
        with open(env_path, "w") as f:
            f.write("# CyberPuppy Configuration\n")
            f.write("# Generated automatically\n\n")

            f.write(f"ENV={self.environment}\n")
            f.write(f"DEBUG={self.debug}\n")
            f.write(f"APP_NAME={self.app_name}\n")
            f.write(f"APP_VERSION={self.app_version}\n\n")

            f.write("# Paths\n")
            f.write(f"DATA_DIR={self.paths.data_dir}\n")
            f.write(f"MODELS_DIR={self.paths.models_dir}\n")
            f.write(f"LOGS_DIR={self.paths.logs_dir}\n")
            f.write(f"CACHE_DIR={self.paths.cache_dir}\n\n")

            f.write("# API Settings\n")
            f.write(f"API__HOST={self.api.host}\n")
            f.write(f"API__PORT={self.api.port}\n")
            f.write(f"API__WORKERS={self.api.workers}\n\n")

            f.write("# Database\n")
            f.write(f"DATABASE__URL={self.database.url}\n")
            f.write(f"DATABASE__POOL_SIZE={self.database.pool_size}\n\n")

            f.write("# Logging\n")
            f.write(f"LOGGING__LEVEL={self.logging.level}\n")
            f.write(f"LOGGING__ROTATION={self.logging.rotation}\n")
            f.write(f"LOGGING__RETENTION={self.logging.retention}\n")


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset global configuration instance."""
    global _config
    _config = None
