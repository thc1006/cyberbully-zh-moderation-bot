"""
Simple test for the fixed model loader without Unicode symbols.
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


def main():
    print("Testing Fixed Model Loader")
    print("=" * 40)

    try:
        # Get loader
        loader = get_fixed_loader()
        print("Loader created successfully")

        # Check status
        status = loader.get_status()
        print(f"Available models: {status['available_models']}")
        print(f"Device: {status['device']}")

        if not status['available_models']:
            print("ERROR: No models found")
            return

        # Load first model
        model_name = status['available_models'][0]
        print(f"Loading model: {model_name}")

        model = loader.load_model(model_name)
        print("Model loaded successfully")

        # Test predictions
        test_texts = [
            "你好",
            "笨蛋",
            "去死",
            "谢谢"
        ]

        print("\nTesting predictions:")
        for text in test_texts:
            try:
                result = model.predict_text(text)
                print(f"Text: '{text}' -> Toxicity: {result['toxicity']}, Emotion: {result['emotion']}")
            except Exception as e:
                print(f"Text: '{text}' -> ERROR: {e}")

        # Test warmup
        print("\nTesting warmup:")
        warmup_stats = loader.warm_up(model_name, ["测试"])
        print(f"Warmup success: {warmup_stats['success_count']}/{warmup_stats['warmup_samples']}")
        if warmup_stats['times']:
            print(f"Average time: {warmup_stats['average_time']:.3f}s")

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        logger.exception("Test error:")


if __name__ == "__main__":
    main()