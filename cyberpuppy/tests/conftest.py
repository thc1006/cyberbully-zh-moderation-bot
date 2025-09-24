"""
Pytest configuration and fixtures for CyberPuppy tests.
"""

import sys
from pathlib import Path
import pytest
from typing import Generator, Any
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberpuppy import Config, reset_config  # noqa: E402


@pytest.fixture(autouse=True)
def reset_config_fixture():
    """Reset configuration before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create a test configuration with temporary directories."""
    config = Config(
        environment="testing",
        debug=True,
        paths={
            "base_dir": temp_dir,
            "data_dir": temp_dir / "data",
            "models_dir": temp_dir / "models",
            "logs_dir": temp_dir / "logs",
            "cache_dir": temp_dir / "cache",
        },
    )
    config.paths.ensure_directories()
    return config


@pytest.fixture
def mock_env_file(temp_dir: Path) -> Path:
    """Create a mock .env file for testing."""
    env_file = temp_dir / ".env"
    env_file.write_text(
        """
ENV=testing
DEBUG=true
APP_NAME=CyberPuppyTest
DATA_DIR=/tmp/test_data
MODELS_DIR=/tmp/test_models
"""
    )
    return env_file


@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Sample data for testing."""
    return {
        "texts": [
            "這是一段測試文字",
            "Another test text",
            "第三段測試內容",
        ],
        "labels": [0, 1, 0],
        "metadata": {
            "source": "test",
            "version": "1.0.0",
        },
    }


@pytest.fixture
def async_client():
    """Create an async test client (placeholder for future API testing)."""
    # This will be implemented when we add API functionality
    pass


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "mar" "kers",
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
