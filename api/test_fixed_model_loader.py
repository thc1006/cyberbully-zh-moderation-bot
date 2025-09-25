"""
Test script for the fixed model loader.
"""

import sys
import os
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.model_loader_fixed import get_fixed_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_model_loading():
    """Test basic model loading."""
    print("=" * 50)
    print("TESTING MODEL LOADING")
    print("=" * 50)

    try:
        loader = get_fixed_loader()

        # Check status
        status = loader.get_status()
        print("\n=== Loader Status ===")
        print(json.dumps(status, indent=2))

        # Check available models
        available_models = status["available_models"]
        print(f"\nAvailable models: {available_models}")

        if not available_models:
            print("No models found! Please ensure model files exist.")
            return False

        # Try to load the first available model
        model_name = available_models[0]
        print(f"\nLoading model: {model_name}")

        model = loader.load_model(model_name)
        print(f"✓ Model loaded successfully: {type(model).__name__}")

        return True

    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        logger.exception("Model loading error:")
        return False


def test_predictions():
    """Test model predictions."""
    print("\n" + "=" * 50)
    print("TESTING PREDICTIONS")
    print("=" * 50)

    try:
        loader = get_fixed_loader()

        # Load model
        available_models = loader.get_available_models()
        if not available_models:
            print("No models available for testing")
            return False

        model_name = available_models[0]
        model = loader.load_model(model_name)

        # Test texts
        test_texts = [
            "你好，今天天气不错",           # Neutral/positive
            "笨蛋，你真是太蠢了",           # Toxic
            "去死吧，我要杀了你",           # Severe/threat
            "谢谢你的帮助，我很感激",       # Positive
            "这个天气真糟糕，心情不好",     # Negative
            "滚开，别烦我",                 # Harassment
        ]

        print(f"\nTesting with {len(test_texts)} sample texts:")

        results = []
        for i, text in enumerate(test_texts, 1):
            try:
                print(f"\n--- Test {i} ---")
                print(f"Text: {text}")

                result = model.predict_text(text)

                print(f"Toxicity: {result['toxicity']} ({result['scores']['toxicity'][result['toxicity']]:.3f})")
                print(f"Emotion: {result['emotion']} (strength: {result['emotion_strength']})")
                print(f"Bullying: {result['bullying']} ({result['scores']['bullying'][result['bullying']]:.3f})")
                print(f"Role: {result['role']}")
                print(f"Confidence: {result['explanations']['confidence']:.3f}")

                if result['explanations']['important_words']:
                    words = [w['word'] for w in result['explanations']['important_words'][:3]]
                    print(f"Key words: {', '.join(words)}")

                results.append({
                    'text': text,
                    'result': result,
                    'success': True
                })

            except Exception as e:
                print(f"✗ Prediction failed: {e}")
                results.append({
                    'text': text,
                    'error': str(e),
                    'success': False
                })

        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n=== Prediction Results ===")
        print(f"Successful: {successful}/{len(test_texts)}")
        print(f"Success rate: {successful/len(test_texts)*100:.1f}%")

        return successful > 0

    except Exception as e:
        print(f"✗ Prediction testing failed: {e}")
        logger.exception("Prediction testing error:")
        return False


def test_warmup():
    """Test model warmup."""
    print("\n" + "=" * 50)
    print("TESTING MODEL WARMUP")
    print("=" * 50)

    try:
        loader = get_fixed_loader()

        available_models = loader.get_available_models()
        if not available_models:
            print("No models available for warmup testing")
            return False

        model_name = available_models[0]
        print(f"Warming up model: {model_name}")

        warmup_stats = loader.warm_up(model_name)

        print("\n=== Warmup Results ===")
        print(json.dumps(warmup_stats, indent=2))

        success_rate = warmup_stats['success_count'] / warmup_stats['warmup_samples']
        print(f"\nWarmup success rate: {success_rate*100:.1f}%")

        if warmup_stats['times']:
            avg_time = warmup_stats['average_time']
            print(f"Average prediction time: {avg_time:.3f}s")

        return warmup_stats['success_count'] > 0

    except Exception as e:
        print(f"✗ Warmup testing failed: {e}")
        logger.exception("Warmup testing error:")
        return False


def test_error_handling():
    """Test error handling with problematic inputs."""
    print("\n" + "=" * 50)
    print("TESTING ERROR HANDLING")
    print("=" * 50)

    try:
        loader = get_fixed_loader()

        available_models = loader.get_available_models()
        if not available_models:
            print("No models available for error testing")
            return False

        model_name = available_models[0]
        model = loader.load_model(model_name)

        # Test problematic inputs
        problematic_texts = [
            "",  # Empty text
            " " * 1000,  # Very long spaces
            "🔥" * 100,  # Emojis
            "a" * 1000,  # Very long text
            "\n\n\n",  # Only newlines
            "测试" * 200,  # Long Chinese text
        ]

        print(f"\nTesting error handling with {len(problematic_texts)} problematic inputs:")

        errors = 0
        successes = 0

        for i, text in enumerate(problematic_texts, 1):
            try:
                display_text = text[:50] + "..." if len(text) > 50 else text
                display_text = display_text.replace('\n', '\\n')
                print(f"\nTest {i}: '{display_text}'")

                result = model.predict_text(text)
                print(f"✓ Handled successfully: {result['toxicity']}")
                successes += 1

            except Exception as e:
                print(f"✗ Failed: {e}")
                errors += 1

        print(f"\n=== Error Handling Results ===")
        print(f"Successful: {successes}/{len(problematic_texts)}")
        print(f"Errors: {errors}/{len(problematic_texts)}")
        print(f"Success rate: {successes/len(problematic_texts)*100:.1f}%")

        return successes >= len(problematic_texts) // 2  # At least 50% should work

    except Exception as e:
        print(f"✗ Error handling testing failed: {e}")
        logger.exception("Error handling testing error:")
        return False


def main():
    """Run all tests."""
    print("Starting Fixed Model Loader Tests...")
    print(f"Working directory: {os.getcwd()}")

    tests = [
        ("Model Loading", test_model_loading),
        ("Predictions", test_predictions),
        ("Warmup", test_warmup),
        ("Error Handling", test_error_handling),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_name} Test...")
        print(f"{'='*60}")

        try:
            success = test_func()
            results.append((test_name, success))
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: ✗ CRASHED - {e}")
            results.append((test_name, False))
            logger.exception(f"{test_name} test crashed:")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:20}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 ALL TESTS PASSED! The fixed model loader is working correctly.")
    elif passed >= total // 2:
        print("⚠️  PARTIAL SUCCESS. Some issues need to be addressed.")
    else:
        print("❌ MULTIPLE FAILURES. The model loader needs significant fixes.")


if __name__ == "__main__":
    main()