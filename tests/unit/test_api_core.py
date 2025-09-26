#!/usr/bin/env python3
"""
FastAPI 核心功能單元測試
Tests for FastAPI core functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
import json
import hashlib

# Import API components to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "api"))

from app import app, model_loader, get_model_loader
from model_loader import ModelLoader


class TestAPIHealthEndpoints:
    """測試 API 健康檢查端點"""

    def setup_method(self):
        """每個測試方法前的設置"""
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """測試健康檢查端點"""
        response = self.client.get("/healthz")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] in ["healthy", "starting", "error"]

    def test_health_endpoint_structure(self):
        """測試健康檢查端點結構"""
        response = self.client.get("/healthz")
        data = response.json()

        required_fields = ["status", "timestamp", "version", "model_status"]
        for field in required_fields:
            assert field in data

    def test_metrics_endpoint(self):
        """測試度量端點"""
        response = self.client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "request_count" in data
        assert "model_loaded" in data


class TestDetectionEndpoint:
    """測試毒性偵測端點"""

    def setup_method(self):
        """每個測試方法前的設置"""
        self.client = TestClient(app)

    @patch('app.get_model_loader')
    def test_detect_endpoint_valid_input(self, mock_get_loader):
        """測試有效輸入的偵測端點"""
        # Mock model loader
        mock_loader = Mock(spec=ModelLoader)
        mock_loader.predict.return_value = {
            "toxicity": {"label": "none", "confidence": 0.95},
            "bullying": {"label": "none", "confidence": 0.90},
            "emotion": {"label": "neutral", "confidence": 0.88},
            "role": {"label": "none", "confidence": 0.92},
            "overall_confidence": 0.91
        }
        mock_get_loader.return_value = mock_loader

        # Test request
        request_data = {
            "text": "今天天氣真好，我們去公園散步吧。",
            "include_explanation": False
        }

        response = self.client.post("/api/detect", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "text_hash" in data
        assert "results" in data
        assert "processing_time_ms" in data
        assert "timestamp" in data

        # Check results structure
        results = data["results"]
        assert "toxicity" in results
        assert "bullying" in results
        assert "emotion" in results
        assert "role" in results

    @patch('app.get_model_loader')
    def test_detect_endpoint_with_explanation(self, mock_get_loader):
        """測試包含解釋性的偵測端點"""
        # Mock model loader with explanation
        mock_loader = Mock(spec=ModelLoader)
        mock_loader.predict.return_value = {
            "toxicity": {"label": "toxic", "confidence": 0.85},
            "bullying": {"label": "harassment", "confidence": 0.82},
            "emotion": {"label": "negative", "confidence": 0.88},
            "role": {"label": "perpetrator", "confidence": 0.79},
            "overall_confidence": 0.84,
            "explanation": {
                "attention_weights": [0.1, 0.3, 0.6],
                "important_tokens": ["笨蛋", "廢物"]
            }
        }
        mock_get_loader.return_value = mock_loader

        request_data = {
            "text": "你這個笨蛋廢物",
            "include_explanation": True
        }

        response = self.client.post("/api/detect", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "explanation" in data
        explanation = data["explanation"]
        assert "attention_weights" in explanation
        assert "important_tokens" in explanation

    def test_detect_endpoint_empty_text(self):
        """測試空文本輸入"""
        request_data = {"text": ""}

        response = self.client.post("/api/detect", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_detect_endpoint_invalid_input(self):
        """測試無效輸入"""
        # Missing text field
        request_data = {"include_explanation": True}

        response = self.client.post("/api/detect", json=request_data)

        assert response.status_code == 422

    @pytest.mark.parametrize("text,expected_fields", [
        ("正常文字", ["toxicity", "bullying", "emotion", "role"]),
        ("這個笨蛋", ["toxicity", "bullying", "emotion", "role"]),
        ("我愛你", ["toxicity", "bullying", "emotion", "role"]),
    ])
    @patch('app.get_model_loader')
    def test_detect_various_texts(self, mock_get_loader, text, expected_fields):
        """測試各種文本輸入"""
        mock_loader = Mock(spec=ModelLoader)
        mock_loader.predict.return_value = {
            "toxicity": {"label": "none", "confidence": 0.95},
            "bullying": {"label": "none", "confidence": 0.90},
            "emotion": {"label": "positive", "confidence": 0.88},
            "role": {"label": "none", "confidence": 0.92},
            "overall_confidence": 0.91
        }
        mock_get_loader.return_value = mock_loader

        request_data = {"text": text}
        response = self.client.post("/api/detect", json=request_data)

        assert response.status_code == 200
        data = response.json()

        for field in expected_fields:
            assert field in data["results"]

    def test_detect_endpoint_privacy(self):
        """測試隱私保護 - 文本不應在響應中洩露"""
        request_data = {
            "text": "這是一個敏感的私人信息",
            "include_explanation": False
        }

        with patch('app.get_model_loader') as mock_get_loader:
            mock_loader = Mock(spec=ModelLoader)
            mock_loader.predict.return_value = {
                "toxicity": {"label": "none", "confidence": 0.95},
                "bullying": {"label": "none", "confidence": 0.90},
                "emotion": {"label": "neutral", "confidence": 0.88},
                "role": {"label": "none", "confidence": 0.92},
                "overall_confidence": 0.91
            }
            mock_get_loader.return_value = mock_loader

            response = self.client.post("/api/detect", json=request_data)

        data = response.json()

        # Original text should not appear in response
        response_str = json.dumps(data)
        assert "敏感的私人信息" not in response_str

        # But hash should be present for tracking
        assert "text_hash" in data


class TestRateLimiting:
    """測試速率限制"""

    def setup_method(self):
        """每個測試方法前的設置"""
        self.client = TestClient(app)

    @patch('app.get_model_loader')
    def test_rate_limiting(self, mock_get_loader):
        """測試速率限制功能"""
        mock_loader = Mock(spec=ModelLoader)
        mock_loader.predict.return_value = {
            "toxicity": {"label": "none", "confidence": 0.95},
            "bullying": {"label": "none", "confidence": 0.90},
            "emotion": {"label": "neutral", "confidence": 0.88},
            "role": {"label": "none", "confidence": 0.92},
            "overall_confidence": 0.91
        }
        mock_get_loader.return_value = mock_loader

        request_data = {"text": "測試文本"}

        # Make rapid requests to trigger rate limiting
        responses = []
        for i in range(15):  # Exceed typical rate limit
            response = self.client.post("/api/detect", json=request_data)
            responses.append(response)

        # Some requests should be rate limited
        status_codes = [r.status_code for r in responses]
        assert any(code == 429 for code in status_codes)  # Too Many Requests


class TestModelLoader:
    """測試模型載入器"""

    def test_model_loader_singleton(self):
        """測試模型載入器單例模式"""
        loader1 = get_model_loader()
        loader2 = get_model_loader()

        assert loader1 is loader2  # Should be the same instance

    @patch('model_loader.ImprovedBullyingDetector')
    def test_model_loader_initialization(self, mock_detector):
        """測試模型載入器初始化"""
        mock_model = Mock()
        mock_detector.from_pretrained.return_value = mock_model

        loader = ModelLoader()
        loader.load_model()

        assert loader.model is not None
        assert loader.is_loaded

    def test_model_loader_predict_not_loaded(self):
        """測試未載入模型時的預測"""
        loader = ModelLoader()
        loader.model = None
        loader.is_loaded = False

        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.predict("test text")


class TestCORSAndSecurity:
    """測試 CORS 和安全設置"""

    def setup_method(self):
        """每個測試方法前的設置"""
        self.client = TestClient(app)

    def test_cors_headers(self):
        """測試 CORS 標頭"""
        response = self.client.get("/healthz")

        # Check if CORS middleware is working
        assert response.status_code == 200

    def test_security_headers(self):
        """測試安全標頭"""
        response = self.client.get("/healthz")

        # Should not expose sensitive information
        headers = response.headers
        assert "x-powered-by" not in headers.lower()


class TestErrorHandling:
    """測試錯誤處理"""

    def setup_method(self):
        """每個測試方法前的設置"""
        self.client = TestClient(app)

    @patch('app.get_model_loader')
    def test_model_error_handling(self, mock_get_loader):
        """測試模型錯誤處理"""
        # Mock model that raises exception
        mock_loader = Mock(spec=ModelLoader)
        mock_loader.predict.side_effect = Exception("Model error")
        mock_get_loader.return_value = mock_loader

        request_data = {"text": "測試文本"}
        response = self.client.post("/api/detect", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "error" in data

    def test_invalid_endpoint(self):
        """測試無效端點"""
        response = self.client.get("/invalid/endpoint")
        assert response.status_code == 404

    def test_invalid_method(self):
        """測試無效方法"""
        response = self.client.get("/api/detect")  # Should be POST
        assert response.status_code == 405  # Method Not Allowed


class TestInputValidation:
    """測試輸入驗證"""

    def setup_method(self):
        """每個測試方法前的設置"""
        self.client = TestClient(app)

    @pytest.mark.parametrize("invalid_input", [
        {"text": None},
        {"text": 123},
        {"text": []},
        {"text": {}},
        {},
        {"invalid_field": "value"}
    ])
    def test_invalid_request_data(self, invalid_input):
        """測試無效請求資料"""
        response = self.client.post("/api/detect", json=invalid_input)
        assert response.status_code == 422  # Validation error

    def test_text_length_validation(self):
        """測試文本長度驗證"""
        # Very long text
        long_text = "測試" * 10000
        request_data = {"text": long_text}

        response = self.client.post("/api/detect", json=request_data)

        # Should handle or reject very long text
        assert response.status_code in [200, 413, 422]  # OK, Payload Too Large, or Validation Error

    def test_special_characters_handling(self):
        """測試特殊字符處理"""
        special_texts = [
            "👨‍💻🤖💻",  # Emojis
            "Hello 世界 🌍",  # Mixed languages
            "Test\n\t\r",  # Control characters
            "   \n   ",  # Only whitespace
        ]

        for text in special_texts:
            request_data = {"text": text}
            response = self.client.post("/api/detect", json=request_data)

            # Should handle gracefully
            assert response.status_code in [200, 422]


class TestPerformance:
    """測試效能相關功能"""

    def setup_method(self):
        """每個測試方法前的設置"""
        self.client = TestClient(app)

    @patch('app.get_model_loader')
    def test_response_time_tracking(self, mock_get_loader):
        """測試響應時間追蹤"""
        mock_loader = Mock(spec=ModelLoader)
        mock_loader.predict.return_value = {
            "toxicity": {"label": "none", "confidence": 0.95},
            "bullying": {"label": "none", "confidence": 0.90},
            "emotion": {"label": "neutral", "confidence": 0.88},
            "role": {"label": "none", "confidence": 0.92},
            "overall_confidence": 0.91
        }
        mock_get_loader.return_value = mock_loader

        request_data = {"text": "測試響應時間"}
        response = self.client.post("/api/detect", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], (int, float))
        assert data["processing_time_ms"] >= 0


if __name__ == "__main__":
    # Run basic smoke test
    client = TestClient(app)
    response = client.get("/healthz")
    print(f"健康檢查狀態: {response.status_code}")
    print("✅ API 核心功能測試準備完成")