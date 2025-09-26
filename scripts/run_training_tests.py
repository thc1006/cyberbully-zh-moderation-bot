#!/usr/bin/env python3
"""
Quick test runner for local training system
Runs both pre-flight check and comprehensive tests
"""

import sys
import subprocess
from pathlib import Path


def run_preflight_check():
    """Run the pre-flight check script"""
    print("🔍 Running pre-flight check...")

    script_path = Path(__file__).parent / "test_local_setup.py"
    try:
        result = subprocess.run([sys.executable, str(script_path)],
                              capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run pre-flight check: {e}")
        return False


def run_comprehensive_tests():
    """Run the comprehensive test suite"""
    print("\n🧪 Running comprehensive tests...")

    test_path = Path(__file__).parent.parent / "tests" / "test_local_training.py"
    try:
        result = subprocess.run([sys.executable, str(test_path)],
                              capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run comprehensive tests: {e}")
        return False


def main():
    """Main test runner"""
    print("="*80)
    print("CYBERPUPPY LOCAL TRAINING - FULL TEST SUITE")
    print("="*80)

    # Run pre-flight check first
    preflight_ok = run_preflight_check()

    if not preflight_ok:
        print("\n❌ Pre-flight check failed. Fix issues before running comprehensive tests.")
        return False

    # Run comprehensive tests
    tests_ok = run_comprehensive_tests()

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    if preflight_ok and tests_ok:
        print("🎉 ALL TESTS PASSED! Ready for training.")
        print("\nTo start training:")
        print("  python scripts/training/train.py")
        return True
    else:
        print("❌ Some tests failed. Check output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)