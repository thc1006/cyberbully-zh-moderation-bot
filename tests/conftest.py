"""
Pytest configuration and shared fixtures for CyberPuppy tests
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberpuppy.config import Settings, TestingConfig  # noqa: E402


@pytest.fixture(scope="session")
def test_config() -> Settings:
    """Create test configuration."""
    settings = Settings()
    # Apply testing configuration
    for key, value in TestingConfig.__dict__.items():
        if not key.startswith("_"):
            setattr(settings, key, value)
    return settings


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_texts() -> Dict[str, str]:
    """Sample Chinese texts for testing."""
    return {
        "normal": "今天天氣真好，我們去公園散步吧。",
        "positive": "太棒了！恭喜你獲得第一名！",
        "negative": "真是糟糕的一天，什麼都不順利。",
        "toxic_mild": "你真的很煩人，能不能閉嘴？",
        "toxic_severe": "我要讓你付出代價，你給我等著！",
        "bullying": "你這個廢物，什麼都做不好，滾出去！",
    }


@pytest.fixture
def sample_labels() -> Dict[str, Any]:
    """Sample labels for testing."""
    return {
        "toxicity_labels": ["none", "toxic", "severe"],
        "emotion_labels": ["positive", "neutral", "negative"],
        "bullying_labels": ["none", "harassment", "threat"],
        "role_labels": ["none", "perpetrator", "victim", "bystander"],
    }


@pytest.fixture
def mock_model_output() -> Dict[str, Any]:
    """Mock model output for testing."""
    return {
        "toxicity_scores": {
            "none": 0.8,
            "toxic": 0.15,
            "severe": 0.05,
        },
        "emotion_scores": {
            "positive": 0.1,
            "neutral": 0.3,
            "negative": 0.6,
        },
        "bullying_scores": {
            "none": 0.7,
            "harassment": 0.2,
            "threat": 0.1,
        },
        "confidence": 0.85,
        "attention_weights": [0.1, 0.2, 0.15, 0.25, 0.3],
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    # Store original env
    original_env = os.environ.copy()

    yield

    # Restore original env
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_line_event():
    """Mock LINE messaging event for testing."""
    return {
        "type": "message",
        "message": {
            "type": "text",
            "id": "test_message_id",
            "text": "測試訊息",
        },
        "timestamp": 1234567890,
        "source": {
            "type": "user",
            "userId": "test_user_id",
        },
        "replyToken": "test_reply_token",
    }


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "shap: marks tests that use SHAP (may conflict with coverage)")


@pytest.fixture(autouse=True)
def mock_line_bot_env(monkeypatch):
    """Mock LINE Bot environment variables to prevent configuration errors."""
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "test_channel_secret_1234567890")
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "test_access_token_1234567890")


def pytest_collection_modifyitems(config, items):
    """Skip SHAP tests when running with coverage due to numba/coverage conflict."""
    if config.getoption("--cov"):
        skip_shap = pytest.mark.skip(reason="SHAP tests skipped with coverage due to numba conflict")
        for item in items:
            if "shap" in item.nodeid.lower() or "test_explain_shap" in item.nodeid:
                item.add_marker(skip_shap)
