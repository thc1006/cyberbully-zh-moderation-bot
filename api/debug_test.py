"""
Direct test of the model integration without FastAPI to debug issues.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import the model loader directly
from model_loader_simple import get_model_loader  # noqa: E402


def test_model_loader():
    """Test model loader directly."""
    print("Testing model loader...")

    try:
        # Get model loader
        loader = get_model_loader()
        print("✓ Model loader created")

        # Load models
        detector = loader.load_models()
        print("✓ Models loaded")

        # Test analysis
        test_text = "test message"
        print(f"Testing analysis with: '{test_text}'")

        result = detector.analyze(test_text)
        print("✓ Analysis successful")
        print("Result:")
        print(f"  Toxicity: {result['toxicity']}")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Explanations: {result['explanations']['important_words']}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_function():
    """Test the API function directly."""
    print("\nTesting API function...")

    try:
        # Import the API function
        from app import analyze_text_content
        import asyncio

        async def run_test():
            result = await analyze_text_content("test message")
            return result

        result = asyncio.run(run_test())
        print("✓ API function successful")
        print(f"Result: {result}")
        return True

    except Exception as e:
        print(f"✗ API function error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("CyberPuppy API Debug Test")
    print("=" * 50)

    # Test model loader
    model_ok = test_model_loader()

    # Test API function
    api_ok = test_api_function()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Model loader: {'✓ OK' if model_ok else '✗ FAILED'}")
    print(f"API function: {'✓ OK' if api_ok else '✗ FAILED'}")

    if model_ok and api_ok:
        print("\n✓ All tests passed - API integration should work!")
    else:
        print("\n✗ Some tests failed - check errors above")
