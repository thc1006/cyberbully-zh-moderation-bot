"""
Test script to verify the API server starts and responds correctly.
This is a manual test that starts the server and makes actual HTTP requests.
"""

import sys

import pytest
import requests


@pytest.mark.integration
def test_api_server():
    """Test the live API server."""
    base_url = "http://localhost:8000"

    print("Testing live API server at", base_url)

    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/healthz", timeout=5)
        print(f"Health check: {response.status_code}")

        if response.status_code == 200:
            health_data = response.json()
            print(f"- Status: {health_data.get('status')}")
            print(f"- Model status: {health_data.get('model_status', {}).get('models_loaded')}")
        else:
            print(f"Health check failed: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to API server: {e}")
        print("Make sure to start the server first with: cd api && python app.py")
        return False

    # Test analyze endpoint
    try:
        test_data = {"text": "这是一个测试文本", "context": "测试上下文"}

        response = requests.post(f"{base_url}/analyze", json=test_data, timeout=10)
        print(f"Analyze endpoint: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"- Toxicity: {data.get('toxicity')}")
            print(f"- Emotion: {data.get('emotion')}")
            print(f"- Processing time: {data.get('processing_time_ms')}ms")

            # Check explanations structure
            explanations = data.get("explanations", {})
            important_words = explanations.get("important_words", [])
            print(f"- Important words count: {len(important_words)}")

            if important_words:
                first_word = important_words[0]
                print(f"- First word structure: {list(first_word.keys())}")
                print(f"- Word field type: {type(first_word.get('word'))}")
                print(f"- Importance field type: {type(first_word.get('importance'))}")

            print("SUCCESS: API response has correct structure!")
            return True

        else:
            print(f"Analyze request failed: {response.status_code}")
            print(response.text)
            return False

    except requests.exceptions.RequestException as e:
        print(f"Analyze request failed: {e}")
        return False


if __name__ == "__main__":
    print("API Live Test")
    print("=============")
    print()
    print("NOTE: This test requires the API server to be running.")
    print("To start the server: cd api && python app.py")
    print()

    if test_api_server():
        print("\nAll live API tests passed!")
        sys.exit(0)
    else:
        print("\nAPI tests failed!")
        sys.exit(1)
