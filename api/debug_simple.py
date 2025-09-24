"""
Simple debug test without special characters.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def test_model_loader():
    """Test model loader directly."""
    print("Testing model loader...")

    try:
        # Import the model loader directly
        from model_loader_simple import get_model_loader

        # Get model loader
        loader = get_model_loader()
        print("OK: Model loader created")

        # Load models
        detector = loader.load_models()
        print("OK: Models loaded")

        # Test analysis
        test_text = "test message"
        print(f"Testing analysis with: '{test_text}'")

        result = detector.analyze(test_text)
        print("OK: Analysis successful")
        print("Result:")
        print(f"  Toxicity: {result['toxicity']}")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Explanations: \
            {len(result['explanations']['important_words'])} words")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("CyberPuppy API Debug Test")
    print("=" * 40)

    model_ok = test_model_loader()

    print("=" * 40)
    if model_ok:
        print("SUCCESS: Model integration works!")
    else:
        print("FAILED: Check errors above")
