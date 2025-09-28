"""
Tests for configuration module
"""

from pathlib import Path

import pytest

from cyberpuppy.config import Settings, get_config


@pytest.mark.unit
class TestSettings:
    """Test Settings class."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()

        assert settings.APP_NAME == "CyberPuppy"
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "INFO"
        assert isinstance(settings.PROJECT_ROOT, Path)
        assert settings.BATCH_SIZE == 16

    def test_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("CYBERPUPPY_DEBUG", "true")
        monkeypatch.setenv("CYBERPUPPY_BATCH_SIZE", "32")
        monkeypatch.setenv("CYBERPUPPY_LOG_LEVEL", "DEBUG")

        settings = Settings()

        assert settings.DEBUG is True
        assert settings.BATCH_SIZE == 32
        assert settings.LOG_LEVEL == "DEBUG"

    def test_path_validation(self):
        """Test path validation and creation."""
        settings = Settings()

        # Check paths are Path objects
        assert isinstance(settings.DATA_DIR, Path)
        assert isinstance(settings.MODEL_DIR, Path)
        assert isinstance(settings.CACHE_DIR, Path)

        # Check default paths
        assert settings.DATA_DIR == settings.PROJECT_ROOT / "data"
        assert settings.MODEL_DIR == settings.PROJECT_ROOT / "models"
        assert settings.CACHE_DIR == settings.PROJECT_ROOT / ".cache"

    def test_threshold_validation(self):
        """Test threshold value validation."""
        # Valid thresholds
        settings = Settings(
            TOXICITY_THRESHOLD=0.5,
            SEVERE_TOXICITY_THRESHOLD=0.9,
            EMOTION_NEGATIVE_THRESHOLD=0.7,
        )

        assert settings.TOXICITY_THRESHOLD == 0.5
        assert settings.SEVERE_TOXICITY_THRESHOLD == 0.9
        assert settings.EMOTION_NEGATIVE_THRESHOLD == 0.7

        # Invalid thresholds should raise validation error
        with pytest.raises(ValueError):
            Settings(TOXICITY_THRESHOLD=1.5)  # > 1.0

    def test_to_dict_masks_sensitive(self):
        """Test that to_dict masks sensitive information."""
        settings = Settings(
            LINE_CHANNEL_ACCESS_TOKEN="secret_token",
            LINE_CHANNEL_SECRET="secret_key",
            PERSPECTIVE_API_KEY="api_key",
        )

        config_dict = settings.to_dict()

        assert config_dict["LINE_CHANNEL_ACCESS_TOKEN"] == "***"
        assert config_dict["LINE_CHANNEL_SECRET"] == "***"
        assert config_dict["PERSPECTIVE_API_KEY"] == "***"

    def test_get_path_methods(self, temp_dir):
        """Test get_path helper methods."""
        settings = Settings(DATA_DIR=temp_dir)

        # Test get_data_path
        data_path = settings.get_data_path("test/file.txt")
        assert data_path == temp_dir / "test/file.txt"
        assert data_path.parent.exists()  # Should create parent dir

        # Test get_model_path
        settings.MODEL_DIR = temp_dir / "models"
        model_path = settings.get_model_path("checkpoint.pth")
        assert model_path == temp_dir / "models/checkpoint.pth"

        # Test get_cache_path
        settings.CACHE_DIR = temp_dir / ".cache"
        cache_path = settings.get_cache_path("temp.pkl")
        assert cache_path == temp_dir / ".cache/temp.pkl"


@pytest.mark.unit
class TestConfigPresets:
    """Test configuration presets."""

    def test_development_config(self):
        """Test development configuration."""
        settings = get_config("development")

        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.USE_GPU is False
        assert settings.BATCH_SIZE == 4

    def test_production_config(self):
        """Test production configuration."""
        settings = get_config("production")

        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "WARNING"
        assert settings.USE_GPU is True
        assert settings.BATCH_SIZE == 32
        assert settings.LOG_USER_CONTENT is False

    def test_testing_config(self):
        """Test testing configuration."""
        settings = get_config("testing")

        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.USE_GPU is False
        assert settings.BATCH_SIZE == 2
        assert "cyberpuppy" in str(settings.DATA_DIR)
        assert "test_data" in str(settings.DATA_DIR)

    def test_unknown_config_defaults_to_development(self):
        """Test that unknown config defaults to development."""
        settings = get_config("unknown")

        # Should use development config
        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == "DEBUG"


@pytest.mark.unit
class TestLabels:
    """Test label configurations."""

    def test_default_labels(self):
        """Test default label configurations."""
        settings = Settings()

        assert settings.TOXICITY_LABELS == ["none", "toxic", "severe"]
        assert settings.BULLYING_LABELS == ["none", "harassment", "threat"]
        assert settings.ROLE_LABELS == ["none", "perpetrator", "victim", "bystander"]
        assert settings.EMOTION_LABELS == ["positive", "neutral", "negative"]

    def test_custom_labels(self, monkeypatch):
        """Test custom label configuration."""
        import json

        custom_labels = ["safe", "warning", "danger"]
        monkeypatch.setenv("CYBERPUPPY_TOXICITY_LABELS", json.dumps(custom_labels))

        settings = Settings()
        assert settings.TOXICITY_LABELS == custom_labels
