"""
Simple API test script to verify functionality.
"""

import json
import os
import sys
import time

import requests

# Set UTF-8 encoding for Windows console
if os.name == "nt":  # Windows
    os.system("chcp 65001")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def test_api():
    base_url = "http://localhost:8000"

    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/healthz")
        print(f"Health Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        print()
    except Exception as e:
        print(f"Health test failed: {e}")
        return

    # Test analyze endpoint
    test_cases = [
        {"text": "你好世界", "description": "Friendly greeting (Chinese)"},
        {"text": "bendan", "description": "Mild insult (Pinyin)"},
        {"text": "qusi", "description": "Severe threat (Pinyin)"},
        {"text": "I need help", "description": "Neutral request (English)"},
    ]

    print("Testing analyze endpoint...")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Text: '{test_case['text']}'")

        try:
            payload = {"text": test_case["text"], "context": None}

            response = requests.post(
                f"{base_url}/analyze",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"Toxicity: {result['toxicity']}")
                print(f"Emotion: {result['emotion']}")
                print(f"Bullying: {result['bullying']}")
                print(f"Processing time: {result['processing_time_ms']}ms")
                print(
                    f"Important words: \
                    {result['explanations']['important_words']}"
                )
            else:
                print(f"Error: {response.text}")

        except Exception as e:
            print(f"Test failed: {e}")

        time.sleep(0.1)  # Small delay

    # Test metrics endpoint
    print("\nTesting metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics")
        print(f"Metrics Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Metrics test failed: {e}")


if __name__ == "__main__":
    test_api()
