#!/usr/bin/env python3
"""
Quick test for API on port 8001.
"""
import json

import requests


def test_single_request():
    """Test single API request."""
    url = "http://localhost:8001/analyze"
    payload = {"text": "你好，今天天气很好"}

    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("SUCCESS! API Response:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"Request failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing CyberPuppy API on port 8001...")
    success = test_single_request()

    if success:
        print("\n✅ API validation fix successful!")
    else:
        print("\n❌ API still has issues")
