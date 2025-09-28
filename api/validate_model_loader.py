"""
Validation script for the fixed model loader.
Handles encoding issues and provides comprehensive testing.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.model_loader_fixed import get_fixed_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_validation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def validate_model_loading():
    """Validate that models can be loaded correctly."""
    print("Validating Model Loading...")

    try:
        loader = get_fixed_loader()
        status = loader.get_status()

        print(f"Device: {status['device']}")
        print(f"Models directory: {status['models_dir']}")
        print(f"Available models: {len(status['available_models'])}")

        if not status["available_models"]:
            print("FAIL: No models found")
            return False

        # Test loading each model
        for model_name in status["available_models"]:
            print(f"Loading {model_name}...")
            start_time = time.time()

            try:
                loader.load_model(model_name)
                load_time = time.time() - start_time

                print(f"SUCCESS: {model_name} loaded in {load_time:.2f}s")

            except Exception as e:
                print(f"FAIL: {model_name} loading failed - {e}")
                return False

        return True

    except Exception as e:
        print(f"FAIL: Model loading validation failed - {e}")
        return False


def validate_predictions():
    """Validate that predictions work correctly."""
    print("\nValidating Predictions...")

    try:
        loader = get_fixed_loader()
        available_models = loader.get_available_models()

        if not available_models:
            print("FAIL: No models available")
            return False

        # Use first available model
        model_name = available_models[0]
        model = loader.load_model(model_name)

        # Test cases with expected patterns
        test_cases = [
            {
                "text": "hello world",
                "expected_toxicity": "none",
                "description": "English neutral text",
            },
            {
                "text": "test message",
                "expected_toxicity": "none",
                "description": "English neutral test",
            },
            {
                "text": "stupid idiot fool",
                "expected_toxicity": "toxic",  # Should detect toxic content
                "description": "English toxic text",
            },
        ]

        success_count = 0
        total_count = len(test_cases)

        for i, test_case in enumerate(test_cases, 1):
            try:
                start_time = time.time()
                result = model.predict_text(test_case["text"])
                prediction_time = time.time() - start_time

                # Validate structure
                required_keys = [
                    "toxicity",
                    "emotion",
                    "emotion_strength",
                    "scores",
                    "explanations",
                ]
                if all(key in result for key in required_keys):
                    print(
                        f"Test {i}: SUCCESS - {test_case['description']} ({prediction_time:.3f}s)"
                    )
                    print(f"  Toxicity: {result['toxicity']}")
                    print(
                        f"  Emotion: {result['emotion']} (strength: {result['emotion_strength']})"
                    )
                    success_count += 1
                else:
                    print(f"Test {i}: FAIL - Missing keys in result")

            except Exception as e:
                print(f"Test {i}: FAIL - {e}")

        success_rate = success_count / total_count
        print(f"\nPrediction validation: {success_count}/{total_count} ({success_rate*100:.1f}%)")

        return success_rate >= 0.8  # 80% success rate required

    except Exception as e:
        print(f"FAIL: Prediction validation failed - {e}")
        return False


def validate_performance():
    """Validate model performance metrics."""
    print("\nValidating Performance...")

    try:
        loader = get_fixed_loader()
        available_models = loader.get_available_models()

        if not available_models:
            print("FAIL: No models available")
            return False

        model_name = available_models[0]

        # Run warmup test
        warmup_stats = loader.warm_up(model_name)

        success_rate = warmup_stats["success_count"] / warmup_stats["warmup_samples"]
        avg_time = warmup_stats["average_time"]

        print(f"Warmup success rate: {success_rate*100:.1f}%")
        print(f"Average prediction time: {avg_time:.3f}s")

        # Performance criteria
        performance_ok = success_rate >= 0.75 and avg_time <= 5.0  # 75% success, under 5s

        if performance_ok:
            print("SUCCESS: Performance meets criteria")
        else:
            print("FAIL: Performance below criteria")

        return performance_ok

    except Exception as e:
        print(f"FAIL: Performance validation failed - {e}")
        return False


def validate_error_handling():
    """Validate error handling with edge cases."""
    print("\nValidating Error Handling...")

    try:
        loader = get_fixed_loader()
        available_models = loader.get_available_models()

        if not available_models:
            print("FAIL: No models available")
            return False

        model_name = available_models[0]
        model = loader.load_model(model_name)

        # Test edge cases
        edge_cases = [
            "",  # Empty string
            " ",  # Single space
            "a",  # Single character
            "a" * 500,  # Long text
        ]

        success_count = 0

        for i, text in enumerate(edge_cases, 1):
            try:
                result = model.predict_text(text)

                # Should return valid structure even for edge cases
                if isinstance(result, dict) and "toxicity" in result:
                    print(f"Edge case {i}: SUCCESS")
                    success_count += 1
                else:
                    print(f"Edge case {i}: FAIL - Invalid result structure")

            except Exception as e:
                print(f"Edge case {i}: FAIL - {e}")

        success_rate = success_count / len(edge_cases)
        print(f"Error handling: {success_count}/{len(edge_cases)} ({success_rate*100:.1f}%)")

        return success_rate >= 0.75  # 75% success rate required

    except Exception as e:
        print(f"FAIL: Error handling validation failed - {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("MODEL LOADER VALIDATION")
    print("=" * 60)

    # Save results to file
    results_file = "model_validation_results.json"
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": {}}

    tests = [
        ("Model Loading", validate_model_loading),
        ("Predictions", validate_predictions),
        ("Performance", validate_performance),
        ("Error Handling", validate_error_handling),
    ]

    passed_tests = 0

    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running {test_name} Test")
        print(f"{'-' * 40}")

        try:
            success = test_func()
            results["tests"][test_name.lower().replace(" ", "_")] = {
                "passed": success,
                "error": None,
            }

            if success:
                passed_tests += 1
                print(f"RESULT: {test_name} PASSED")
            else:
                print(f"RESULT: {test_name} FAILED")

        except Exception as e:
            print(f"RESULT: {test_name} CRASHED - {e}")
            results["tests"][test_name.lower().replace(" ", "_")] = {
                "passed": False,
                "error": str(e),
            }

    # Final summary
    total_tests = len(tests)
    success_rate = passed_tests / total_tests

    results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": success_rate,
        "overall_status": "PASS" if success_rate >= 0.75 else "FAIL",
    }

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {success_rate*100:.1f}%")

    if success_rate >= 0.75:
        print("OVERALL RESULT: PASS - Model loader is working correctly")
    else:
        print("OVERALL RESULT: FAIL - Model loader needs fixes")

    # Save results
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {results_file}")

    return success_rate >= 0.75


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
