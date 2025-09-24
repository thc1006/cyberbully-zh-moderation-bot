"""
Integration test for the API endpoint to verify the data validation fix.
"""

import sys
from pathlib import Path

# Add the api directory to the path
api_dir = Path(__file__).parent.parent / "api"
sys.path.insert(0, str(api_dir))

from fastapi.testclient import TestClient  # noqa: E402
from app import app  # noqa: E402
from model_loader_simple import get_model_loader  # noqa: E402


def test_api_analyze_endpoint():
    """Test the /analyze endpoint returns valid response structure."""
    # Initialize model loader manually for testing
    import app as app_module
    loader = get_model_loader()
    loader.load_models()
    app_module.model_loader = loader

    client = TestClient(app)

    # Test data
    test_request = {
        "text": "这是一个测试文本，包含一些笨蛋词汇",
        "context": "测试上下文",
        "thread_id": "test_thread_123"
    }

    # Make request
    response = client.post("/analyze", json=test_request)

    # Check response
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()

    # Check required fields
    required_fields = [
        "toxicity", "bullying", "role", "emotion", "emotion_strength",
        "scores",
    ]

    for field in required_fields:
        assert field in data, f"Missing field: {field}"

    # Check explanations structure
    explanations = data["explanations"]
    assert "important_words" in explanations
    assert "method" in explanations
    assert "confidence" in explanations

    # Check important_words structure
    important_words = explanations["important_words"]
    assert isinstance(important_words, list)

    if important_words:  # If there are any words
        for word_data in important_words:
            assert "word" in word_data, f"Missing 'word' field in {word_data}"
            assert "importance" in word_data, f"Missing 'importance' field in {word_data}"
            assert isinstance(word_data["word"], str), f"'word' should be str, got {type(word_data['word'])}"
            assert isinstance(word_data["importance"], (int, float)), f"'importance' should be numeric, got {type(word_data['importance'])}"
            assert 0 <= word_data["importance"] <= 1, f"'importance' should be between 0-1, got {word_data['importance']}" 

    # Check scores structure
    scores = data["scores"]
    assert "toxicity" in scores
    assert "bullying" in scores
    assert "role" in scores
    assert "emotion" in scores

    return data


def test_api_healthz_endpoint():
    """Test the /healthz endpoint."""
    client = TestClient(app)

    response = client.get("/healthz")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "uptime_seconds" in data


if __name__ == "__main__":
    print("Testing API endpoints...")

    try:
        # Test health endpoint
        test_api_healthz_endpoint()
        print("PASS: Health endpoint test passed")

        # Test analyze endpoint
        result = test_api_analyze_endpoint()
        print("PASS: Analyze endpoint test passed")

        print("\nSample API response structure:")
        print(f"- toxicity: {result['toxicity']}")
        print(f"- bullying: {result['bullying']}")
        print(f"- emotion: {result['emotion']}")
        print(f"- important_words count: {len(result['explanations']['important_words'])}")

        if result['explanations']['important_words']:
            first_word = result['explanations']['important_words'][0]
            try:
                print(f"- first important word: '{first_word['word']}' (importance: {first_word['importance']})")
            except UnicodeEncodeError:
                print(f"- first important word: [UNICODE] (importance: {first_word['importance']})")

        print("\nAll API integration tests passed!")

    except Exception as e:
        print(f"ERROR: API test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
