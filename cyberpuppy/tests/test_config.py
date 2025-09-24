"""
Tests for configuration module.
"""

from pathlib import Path
import pytest
from cyberpuppy import Config, get_config


class TestConfig:
    """Test suite for Config class."""

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = Config()
        assert config.app_name == "CyberPuppy"
        assert config.app_version == "0.1.0"
        assert config.environment == "development"
        assert config.debug is False

    def test_path_configuration(self, temp_dir: Path):
        """Test path configuration."""
        config = Config(
            paths={
                "base_dir": temp_dir,
                "data_dir": temp_dir / "data",
                "models_dir": temp_dir / "models",
            }
        )

        assert config.paths.base_dir == temp_dir
        assert config.paths.data_dir == temp_dir / "data"
        assert config.paths.models_dir == temp_dir / "models"

    def test_ensure_directories(self, temp_dir: Path):
        """Test directory creation."""
        config = Config(
            paths={
                "base_dir": temp_dir,
                "data_dir": temp_dir / "data",
                "models_dir": temp_dir / "models",
                "logs_dir": temp_dir / "logs",
                "cache_dir": temp_dir / "cache",
            }
        )

        config.paths.ensure_directories()

        assert (temp_dir / "data").exists()
        assert (temp_dir / "models").exists()
        assert (temp_dir / "logs").exists()
        assert (temp_dir / "cache").exists()

    def test_config_from_env(self, mock_env_file: Path):
        """Test loading configuration from environment file."""
        config = Config.from_env(mock_env_file)

        assert config.environment == "testing"
        assert config.debug is True
        assert config.app_name == "CyberPuppyTest"

    def test_config_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = Config(environment="testing", debug=True)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["environment"] == "testing"
        assert config_dict["debug"] is True
        assert "paths" in config_dict
        assert "model" in config_dict

    def test_save_env(self, temp_dir: Path):
        """Test saving configuration to .env file."""
        config = Config(
            environment="production",
            debug=False,
            paths={"base_dir": temp_dir}
        )

        env_file = temp_dir / ".env"
        config.save_env(env_file)

        assert env_file.exists()
        content = env_file.read_text()
        assert "ENV=production" in content
        assert "DEBUG=False" in content

    def test_global_config_singleton(self):
        """Test global configuration singleton pattern."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_model_config_defaults(self):
        """Test model configuration defaults."""
        config = Config()

        assert config.model.default_model == "bert-base-chinese"
        assert config.model.device == "cuda"
        assert config.model.batch_size == 32
        assert config.model.max_sequence_length == 512

    def test_api_config_defaults(self):
        """Test API configuration defaults."""
        config = Config()

        assert config.api.host == "0.0.0.0"
        assert config.api.port == 8000
        assert config.api.workers == 1
        assert config.api.debug is False

    def test_database_config_defaults(self):
        """Test database configuration defaults."""
        config = Config()

        assert config.database.url == "sqlite:///./data/cyberpuppy.db"
        assert config.database.pool_size == 5
        assert config.database.echo is False

    def test_logging_config_defaults(self):
        """Test logging configuration defaults."""
        config = Config()

        assert config.logging.level == "INFO"
        assert config.logging.rotation == "10 MB"
        assert config.logging.retention == "7 days"
        assert config.logging.backtrace is True

    @pytest.mark.parametrize("env,debug", [
        ("development", True),
        ("production", False),
        ("staging", False),
    ])
    def test_environment_settings(self, env: str, debug: bool):
        """Test environment-specific settings."""
        config = Config(environment=env, debug=debug)

        assert config.environment == env
        assert config.debug == debug
