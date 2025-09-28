"""
Test critical module imports to verify dependency resolution.
"""

import importlib
import sys
from typing import Dict, Tuple


def test_critical_imports() -> Dict[str, Tuple[bool, str]]:
    """Test importing critical modules and return results."""

    critical_modules = [
        # Core cyberpuppy modules
        "cyberpuppy.config",
        "cyberpuppy.models",
        "cyberpuppy.eval",
        "cyberpuppy.explain",
        "cyberpuppy.safety",
        "cyberpuppy.labeling",
        # Core ML/DL libraries
        "torch",
        "transformers",
        "datasets",
        "sklearn",
        "numpy",
        "pandas",
        # Chinese NLP
        "jieba",
        "opencc",
        # API & Web
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic.v1",  # Test V1 compatibility
        "pydantic_settings",
        # Explainability
        "captum",
        "shap",
        # Testing
        "pytest",
        # Development tools
        "black",
        "mypy",
        # Utilities
        "requests",
        "tqdm",
        "rich",
        "loguru",
    ]

    results = {}

    for module_name in critical_modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            results[module_name] = (True, f"OK - version: {version}")
        except ImportError as e:
            results[module_name] = (False, f"ImportError: {str(e)}")
        except Exception as e:
            results[module_name] = (False, f"Error: {type(e).__name__}: {str(e)}")

    return results


def test_pydantic_v2_compatibility():
    """Test Pydantic V2 compatibility specifically."""
    try:
        import pydantic
        from pydantic import BaseModel, ConfigDict, Field
        from pydantic_settings import BaseSettings

        # Test basic V2 model
        class TestModel(BaseModel):
            model_config = ConfigDict(extra="forbid")

            name: str = Field(..., description="Test name")
            value: int = Field(default=0, ge=0)

        # Test settings
        class TestSettings(BaseSettings):
            model_config = ConfigDict(env_prefix="TEST_")

            debug: bool = False
            port: int = 8000

        # Test instantiation
        TestModel(name="test", value=42)
        TestSettings()

        return True, f"Pydantic V2 OK - version: {pydantic.__version__}"

    except Exception as e:
        return False, f"Pydantic V2 Error: {type(e).__name__}: {str(e)}"


def main():
    """Run import tests and display results."""
    print("=" * 60)
    print("CYBERPUPPY DEPENDENCY IMPORT TESTS")
    print("=" * 60)

    # Test critical imports
    print("\n[INFO] Testing Critical Module Imports...")
    results = test_critical_imports()

    success_count = 0
    total_count = len(results)

    for module_name, (success, message) in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {module_name}: {message}")
        if success:
            success_count += 1

    print(f"\n[STATS] Import Results: {success_count}/{total_count} modules imported successfully")

    # Test Pydantic V2 compatibility
    print("\n[INFO] Testing Pydantic V2 Compatibility...")
    pydantic_ok, pydantic_msg = test_pydantic_v2_compatibility()
    status = "[OK]" if pydantic_ok else "[FAIL]"
    print(f"{status} Pydantic V2: {pydantic_msg}")

    # Summary
    print("\n" + "=" * 60)
    if success_count == total_count and pydantic_ok:
        print("[SUCCESS] ALL TESTS PASSED! Dependencies are properly resolved.")
        return 0
    else:
        print("[WARNING] SOME TESTS FAILED! Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
