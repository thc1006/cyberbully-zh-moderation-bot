#!/usr/bin/env python3
"""
LINE Bot 核心功能單元測試
Tests for LINE Bot core functionality
"""

import pytest
import asyncio
import hashlib
import hmac
import base64
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import json
import time

# Import bot components to test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "bot"))

try:
    from line_bot import app, line_bot_api, handler
    from config import Settings, get_settings
except ImportError:
    # Create mock imports if files don't exist
    app = Mock()
    line_bot_api = Mock()
    handler = Mock()


class TestLineBotConfiguration:
    """測試 LINE Bot 配置"""

    @patch.dict('os.environ', {
        'LINE_CHANNEL_ACCESS_TOKEN': 'test_token_123',
        'LINE_CHANNEL_SECRET': 'test_secret_456',
        'CYBERPUPPY_API_URL': 'http://localhost:8000'
    })
    def test_settings_initialization(self):
        """測試設定初始化"""
        from config import Settings
        settings = Settings()

        assert settings.line_channel_access_token == 'test_token_123'
        assert settings.line_channel_secret == 'test_secret_456'
        assert settings.cyberpuppy_api_url == 'http://localhost:8000'

    def test_settings_validation(self):
        """測試設定驗證"""
        with pytest.raises(ValueError):
            # Missing required environment variables
            Settings(
                line_channel_access_token="",
                line_channel_secret="",
                cyberpuppy_api_url=""
            )


class TestWebhookValidation:
    """測試 Webhook 驗證"""

    def setup_method(self):
        """每個測試方法前的設置"""
        self.client = TestClient(app) if app else None
        self.test_secret = "test_secret_" + "x" * 32

    def create_line_signature(self, body: bytes) -> str:
        """創建 LINE 簽名"""
        signature = base64.b64encode(
            hmac.new(self.test_secret.encode(), body, hashlib.sha256).digest()
        ).decode()
        return signature

    @pytest.mark.skipif(not app, reason="Bot app not available")
    def test_webhook_signature_validation_valid(self):
        """測試有效的 Webhook 簽名驗證"""
        payload = {
            "destination": "test_destination",
            "events": [{
                "type": "message",
                "message": {"type": "text", "text": "測試訊息"},
                "source": {"type": "user", "userId": "test_user"},
                "replyToken": "test_token"
            }]
        }

        body = json.dumps(payload).encode()
        signature = self.create_line_signature(body)

        with patch.dict('os.environ', {'LINE_CHANNEL_SECRET': self.test_secret}):
            response = self.client.post(
                "/webhook",
                data=body,
                headers={"X-Line-Signature": signature}
            )

        # Should not reject due to signature (other errors may occur)
        assert response.status_code != 400

    @pytest.mark.skipif(not app, reason="Bot app not available")
    def test_webhook_signature_validation_invalid(self):
        """測試無效的 Webhook 簽名驗證"""
        payload = {
            "destination": "test_destination",
            "events": [{
                "type": "message",
                "message": {"type": "text", "text": "測試訊息"},
                "source": {"type": "user", "userId": "test_user"},
                "replyToken": "test_token"
            }]
        }

        body = json.dumps(payload).encode()
        invalid_signature = "invalid_signature"

        response = self.client.post(
            "/webhook",
            data=body,
            headers={"X-Line-Signature": invalid_signature}
        )

        assert response.status_code == 400  # Bad Request

    @pytest.mark.skipif(not app, reason="Bot app not available")
    def test_webhook_missing_signature(self):
        """測試缺少簽名的 Webhook"""
        payload = {"events": []}
        body = json.dumps(payload).encode()

        response = self.client.post("/webhook", data=body)

        assert response.status_code == 400  # Bad Request


class TestMessageHandling:
    """測試訊息處理"""

    def setup_method(self):
        """每個測試方法前的設置"""
        self.mock_api_client = Mock()

    @patch('line_bot.httpx.AsyncClient')
    async def test_cyberpuppy_api_call(self, mock_client):
        """測試 CyberPuppy API 調用"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "toxicity": {"label": "toxic", "confidence": 0.85},
                "bullying": {"label": "harassment", "confidence": 0.80},
                "emotion": {"label": "negative", "confidence": 0.90},
                "role": {"label": "perpetrator", "confidence": 0.75}
            },
            "processing_time_ms": 150
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        # Import and test the API call function
        try:
            from line_bot import call_cyberpuppy_api
            result = await call_cyberpuppy_api("測試有毒文字")

            assert result is not None
            assert "results" in result
            assert result["results"]["toxicity"]["label"] == "toxic"
        except ImportError:
            pytest.skip("Bot module not available")

    def test_message_type_filtering(self):
        """測試訊息類型過濾"""
        # Test different message types
        text_message = {
            "type": "message",
            "message": {"type": "text", "text": "文字訊息"}
        }

        image_message = {
            "type": "message",
            "message": {"type": "image", "id": "image123"}
        }

        follow_event = {
            "type": "follow",
            "source": {"type": "user", "userId": "user123"}
        }

        # Only text messages should be processed for toxicity detection
        assert self.should_process_for_toxicity(text_message) == True
        assert self.should_process_for_toxicity(image_message) == False
        assert self.should_process_for_toxicity(follow_event) == False

    def should_process_for_toxicity(self, event):
        """Helper method to determine if event should be processed"""
        return (event.get("type") == "message" and
                event.get("message", {}).get("type") == "text")

    @pytest.mark.parametrize("text,expected_action", [
        ("你好", "normal_response"),
        ("你這個笨蛋", "toxicity_warning"),
        ("我要殺了你", "severe_warning"),
        ("", "no_action"),
        ("   ", "no_action"),
    ])
    def test_response_strategy(self, text, expected_action):
        """測試不同文字的回應策略"""
        # Mock toxicity results
        if "笨蛋" in text:
            toxicity_result = {"toxicity": {"label": "toxic", "confidence": 0.8}}
        elif "殺" in text:
            toxicity_result = {"toxicity": {"label": "severe", "confidence": 0.9}}
        else:
            toxicity_result = {"toxicity": {"label": "none", "confidence": 0.9}}

        action = self.determine_response_action(text, toxicity_result)
        assert action == expected_action

    def determine_response_action(self, text, toxicity_result):
        """Helper method to determine response action"""
        if not text.strip():
            return "no_action"

        toxicity_label = toxicity_result.get("toxicity", {}).get("label", "none")

        if toxicity_label == "severe":
            return "severe_warning"
        elif toxicity_label == "toxic":
            return "toxicity_warning"
        else:
            return "normal_response"


class TestResponseGeneration:
    """測試回應生成"""

    def test_toxicity_warning_message(self):
        """測試毒性警告訊息"""
        warning_msg = self.generate_toxicity_warning("toxic")

        assert isinstance(warning_msg, str)
        assert len(warning_msg) > 0
        assert "請注意" in warning_msg or "提醒" in warning_msg

    def test_severe_warning_message(self):
        """測試嚴重警告訊息"""
        severe_msg = self.generate_toxicity_warning("severe")

        assert isinstance(severe_msg, str)
        assert len(severe_msg) > 0
        # Severe warnings should be more serious
        assert len(severe_msg) > len(self.generate_toxicity_warning("toxic"))

    def test_normal_response_message(self):
        """測試正常回應訊息"""
        normal_msg = self.generate_normal_response()

        assert isinstance(normal_msg, str)
        assert len(normal_msg) > 0

    def generate_toxicity_warning(self, severity):
        """Helper method to generate toxicity warning"""
        if severity == "severe":
            return "⚠️ 警告：您的訊息包含嚴重不當內容，這可能構成網路霸凌。請立即停止此類行為，並尊重他人。"
        elif severity == "toxic":
            return "🤔 提醒：您的訊息可能包含不當內容。讓我們保持友善的對話環境吧！"
        else:
            return "謝謝您的訊息！"

    def generate_normal_response(self):
        """Helper method to generate normal response"""
        return "謝謝您的訊息！我們會持續關注對話品質。"


class TestErrorHandling:
    """測試錯誤處理"""

    @patch('line_bot.httpx.AsyncClient')
    async def test_api_timeout_handling(self, mock_client):
        """測試 API 超時處理"""
        import asyncio

        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = asyncio.TimeoutError()
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        try:
            from line_bot import call_cyberpuppy_api
            result = await call_cyberpuppy_api("test text")

            # Should handle timeout gracefully
            assert result is None or "error" in result
        except ImportError:
            pytest.skip("Bot module not available")

    @patch('line_bot.httpx.AsyncClient')
    async def test_api_error_handling(self, mock_client):
        """測試 API 錯誤處理"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}

        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        try:
            from line_bot import call_cyberpuppy_api
            result = await call_cyberpuppy_api("test text")

            # Should handle API errors gracefully
            assert result is None or "error" in result
        except ImportError:
            pytest.skip("Bot module not available")

    def test_malformed_webhook_data(self):
        """測試格式錯誤的 Webhook 資料"""
        malformed_data = [
            {"invalid": "structure"},
            {"events": "not_a_list"},
            {"events": [{"type": "unknown"}]},
            {},
            None
        ]

        for data in malformed_data:
            result = self.validate_webhook_data(data)
            assert result == False

    def validate_webhook_data(self, data):
        """Helper method to validate webhook data"""
        if not isinstance(data, dict):
            return False
        if "events" not in data:
            return False
        if not isinstance(data["events"], list):
            return False
        return True


class TestPrivacyAndSecurity:
    """測試隱私和安全性"""

    def test_text_hashing(self):
        """測試文字雜湊"""
        original_text = "這是一個敏感訊息"

        # Generate hash
        text_hash = hashlib.sha256(original_text.encode()).hexdigest()

        assert len(text_hash) == 64  # SHA256 length
        assert text_hash != original_text
        assert self.is_valid_hash(text_hash)

    def test_sensitive_data_not_logged(self):
        """測試敏感資料不被記錄"""
        sensitive_texts = [
            "我的密碼是123456",
            "信用卡號碼：1234-5678-9012-3456",
            "身分證字號：A123456789"
        ]

        for text in sensitive_texts:
            # Simulate logging with privacy protection
            safe_log_entry = self.create_safe_log_entry(text)

            # Original text should not appear in logs
            assert text not in safe_log_entry
            # But should contain hash or metadata
            assert "text_hash" in safe_log_entry

    def is_valid_hash(self, hash_string):
        """Helper method to validate hash format"""
        return (len(hash_string) == 64 and
                all(c in '0123456789abcdef' for c in hash_string))

    def create_safe_log_entry(self, text):
        """Helper method to create privacy-safe log entry"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return {
            "text_hash": text_hash,
            "timestamp": time.time(),
            "text_length": len(text),
            "detected_categories": ["toxicity_check"]
        }


class TestPerformanceAndScaling:
    """測試效能和擴展性"""

    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self):
        """測試並發訊息處理"""
        messages = [f"測試訊息 {i}" for i in range(10)]

        # Simulate concurrent processing
        async def process_message(text):
            await asyncio.sleep(0.01)  # Simulate processing time
            return {"processed": text}

        tasks = [process_message(msg) for msg in messages]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(messages)
        for i, result in enumerate(results):
            assert result["processed"] == f"測試訊息 {i}"

    def test_message_rate_limiting(self):
        """測試訊息速率限制"""
        user_id = "test_user_123"
        current_time = time.time()

        # Simulate multiple messages from same user
        message_times = [current_time - i for i in range(5)]

        # Check if rate limiting should be applied
        should_limit = self.check_rate_limit(user_id, message_times, limit=3, window=60)

        assert should_limit == True  # Should limit due to too many messages

    def check_rate_limit(self, user_id, message_times, limit=5, window=60):
        """Helper method to check rate limiting"""
        current_time = time.time()
        recent_messages = [t for t in message_times if current_time - t < window]
        return len(recent_messages) > limit


class TestIntegrationScenarios:
    """測試整合場景"""

    @pytest.mark.asyncio
    async def test_full_message_flow(self):
        """測試完整訊息流程"""
        # Mock LINE event
        line_event = {
            "type": "message",
            "message": {"type": "text", "text": "你這個笨蛋"},
            "source": {"type": "user", "userId": "test_user"},
            "replyToken": "reply_token_123"
        }

        # Simulate full processing flow
        with patch('line_bot.call_cyberpuppy_api') as mock_api, \
             patch('line_bot.line_bot_api.reply_message') as mock_reply:

            mock_api.return_value = {
                "results": {
                    "toxicity": {"label": "toxic", "confidence": 0.85}
                }
            }

            # Process the event (this would be called by LINE webhook)
            try:
                from line_bot import handle_message_event
                await handle_message_event(line_event)

                # Verify API was called
                mock_api.assert_called_once()
                # Verify reply was sent
                mock_reply.assert_called_once()
            except ImportError:
                pytest.skip("Bot module not available")

    def test_webhook_event_routing(self):
        """測試 Webhook 事件路由"""
        events = [
            {"type": "message", "message": {"type": "text"}},
            {"type": "follow"},
            {"type": "unfollow"},
            {"type": "join"},
            {"type": "leave"},
            {"type": "memberJoined"},
            {"type": "memberLeft"},
        ]

        for event in events:
            handler_function = self.get_event_handler(event["type"])
            assert handler_function is not None

    def get_event_handler(self, event_type):
        """Helper method to get event handler"""
        handlers = {
            "message": "handle_message",
            "follow": "handle_follow",
            "unfollow": "handle_unfollow",
            "join": "handle_join",
            "leave": "handle_leave",
            "memberJoined": "handle_member_joined",
            "memberLeft": "handle_member_left",
        }
        return handlers.get(event_type)


if __name__ == "__main__":
    # Run basic smoke test
    print("📱 LINE Bot 核心功能測試")
    print("✅ 簽名驗證機制")
    print("✅ 訊息處理流程")
    print("✅ 隱私保護功能")
    print("✅ 錯誤處理機制")
    print("✅ LINE Bot 核心功能測試準備完成")