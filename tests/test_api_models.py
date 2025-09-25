"""
Test API Pydantic model validation to ensure proper data structures.
"""

import sys
from pathlib import Path

# Add the api directory to the path
api_dir = Path(__file__).parent.parent / "api"
sys.path.insert(0, str(api_dir))

from app import ExplanationData, ImportantWord, AnalyzeResponse  # noqa: E402
import pytest  # noqa: E402


def test_important_word_model():
    """Test ImportantWord model validation."""
    # Valid data
    word_data = {"word": "测试", "importance": 0.85}
    word_model = ImportantWord(**word_data)

    assert word_model.word == "测试"
    assert word_model.importance == 0.85

    # Test invalid importance (out of range)
    with pytest.raises(ValueError):
        ImportantWord(word="test", importance=1.5)

    with pytest.raises(ValueError):
        ImportantWord(word="test", importance=-0.1)


def test_explanation_data_model():
    """Test ExplanationData model validation."""
    explanation_data = {
        "important_words": [
            {"word": "测试", "importance": 0.85},
            {"word": "文本", "importance": 0.72},
        ],
        "method": "keyword_based_mock",
        "confidence": 0.75,
    }

    explanation_model = ExplanationData(**explanation_data)

    assert len(explanation_model.important_words) == 2
    assert explanation_model.important_words[0].word == "测试"
    assert explanation_model.important_words[0].importance == 0.85
    assert explanation_model.method == "keyword_based_mock"
    assert explanation_model.confidence == 0.75


def test_explanation_data_with_model_loader():
    """Test ExplanationData with actual model loader output."""
    from model_loader_simple import get_model_loader

    # Get mock analysis result
    loader = get_model_loader()
    detector = loader.load_models()
    result = detector.analyze("这是测试文本")

    # Test that we can create ExplanationData from the result
    explanation_model = ExplanationData(**result["explanations"])

    assert isinstance(explanation_model, ExplanationData)
    assert len(explanation_model.important_words) > 0
    assert all(
        isinstance(word, ImportantWord) for word in explanation_model.important_words
    )
    assert explanation_model.method == "keyword_based_mock"
    assert 0 <= explanation_model.confidence <= 1


def test_analyze_response_full():
    """Test complete AnalyzeResponse model with mock data."""
    from model_loader_simple import get_model_loader
    from datetime import datetime
    import hashlib

    # Get mock analysis result
    loader = get_model_loader()
    detector = loader.load_models()
    result = detector.analyze("测试文本")

    # Create full response
    text_hash = hashlib.sha256("测试文本".encode("utf-8")).hexdigest()[:16]

    response_data = {
        "toxicity": result["toxicity"],
        "bullying": result["bullying"],
        "role": result["role"],
        "emotion": result["emotion"],
        "emotion_strength": result["emotion_strength"],
        "scores": result["scores"],
        "explanations": result["explanations"],
        "text_hash": text_hash,
        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": 123.45,
    }

    response_model = AnalyzeResponse(**response_data)

    assert isinstance(response_model, AnalyzeResponse)
    assert isinstance(response_model.explanations, ExplanationData)
    assert len(response_model.explanations.important_words) > 0
    assert all(
        isinstance(word, ImportantWord)
        for word in response_model.explanations.important_words
    )


if __name__ == "__main__":
    # Run basic tests
    test_important_word_model()
    print("PASS: ImportantWord model test passed")

    test_explanation_data_model()
    print("PASS: ExplanationData model test passed")

    test_explanation_data_with_model_loader()
    print("PASS: ExplanationData with model loader test passed")

    test_analyze_response_full()
    print("PASS: Full AnalyzeResponse test passed")

    print("\nAll API model validation tests passed!")
