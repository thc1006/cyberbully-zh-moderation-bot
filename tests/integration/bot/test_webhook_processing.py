"""
LINE Bot Webhook 處理整合測試
測試完整的 Bot 訊息處理流程：
- Webhook 簽名驗證
- 訊息分析與回應策略
- 使用者會話管理
- 回應訊息格式
- 升級處理機制
"""

import asyncio
import json
import time
from unittest.mock import patch

import pytest


@pytest.mark.bot
class TestWebhookSignatureVerification:
    """Webhook 簽名驗證測試"""

    def test_valid_signature_verification(self, mock_line_signature):
        """測試有效簽名驗證"""
        from bot.line_bot import verify_line_signature

        body = b'{"test": "data"}'
        signature = mock_line_signature(body)

        # 修補環境變數進行測試
        with patch("bot.line_bot.LINE_CHANNEL_SECRET", "test_secret_" + "x" * 32):
            result = verify_line_signature(body, signature)
            assert result is True

    def test_invalid_signature_verification(self):
        """測試無效簽名驗證"""
        from bot.line_bot import verify_line_signature

        body = b'{"test": "data"}'
        invalid_signature = "invalid_signature"

        with patch("bot.line_bot.LINE_CHANNEL_SECRET", "test_secret_" + "x" * 32):
            result = verify_line_signature(body, invalid_signature)
            assert result is False

    def test_empty_signature_verification(self):
        """測試空簽名驗證"""
        from bot.line_bot import verify_line_signature

        body = b'{"test": "data"}'

        with patch("bot.line_bot.LINE_CHANNEL_SECRET", "test_secret_" + "x" * 32):
            result = verify_line_signature(body, "")
            assert result is False

    async def test_webhook_signature_rejection(self, bot_server, http_client):
        """測試 Webhook 簽名拒絕"""
        payload = {"events": [{"type": "test"}]}
        body = json.dumps(payload).encode()

        response = await http_client.post(
            f"{bot_server}/webhook",
            content=body,
            headers={
                "content-type": "application/json",
                "X-Line-Signature": "invalid_signature",
            },
        )

        assert response.status_code == 400


@pytest.mark.bot
class TestMessageProcessing:
    """訊息處理測試"""

    async def test_text_message_processing(
        self, bot_server, http_client, mock_line_signature
    ):
        """測試文字訊息處理"""
        payload = {
            "events": [
                {
                    "type": "message",
                    "timestamp": int(time.time() * 1000),
                    "source": {"type": "user", "userId": "test_user_123"},
                    "replyToken": "test_reply_token",
                    "message": {
                        "id": "test_msg_id",
                        "type": "text",
                        "text": "你好，今天天氣如何？",
                    },
                }
            ]
        }

        body = json.dumps(payload).encode()
        signature = mock_line_signature(body)

        with patch("bot.line_bot.line_bot_api.reply_message") as _mock_reply:
            response = await http_client.post(
                f"{bot_server}/webhook",
                content=body,
                headers={
                    "content-type": "application/json",
                    "X-Line-Signature": signature,
                },
            )

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    async def test_toxic_message_processing(
        self, bot_server, http_client, mock_line_signature
    ):
        """測試毒性訊息處理與回應"""
        toxic_payload = {
            "events": [
                {
                    "type": "message",
                    "timestamp": int(time.time() * 1000),
                    "source": {"type": "user", "userId": "test_toxic_user"},
                    "replyToken": "test_reply_token_toxic",
                    "message": {
                        "id": "test_msg_toxic",
                        "type": "text",
                        "text": "你這個笨蛋，什麼都不懂",
                    },
                }
            ]
        }

        body = json.dumps(toxic_payload).encode()
        signature = mock_line_signature(body)

        with patch("bot.line_bot.line_bot_api.reply_message") as mock_reply:
            response = await http_client.post(
                f"{bot_server}/webhook",
                content=body,
                headers={
                    "content-type": "application/json",
                    "X-Line-Signature": signature,
                },
            )

            # 等待背景處理完成
            await asyncio.sleep(0.5)

        assert response.status_code == 200
        # 驗證有回應訊息
        mock_reply.assert_called()

    async def test_severe_threat_processing(
        self, bot_server, http_client, mock_line_signature
    ):
        """測試嚴重威脅訊息處理"""
        threat_payload = {
            "events": [
                {
                    "type": "message",
                    "timestamp": int(time.time() * 1000),
                    "source": {"type": "user", "userId": "test_threat_user"},
                    "replyToken": "test_reply_token_threat",
                    "message": {
                        "id": "test_msg_threat",
                        "type": "text",
                        "text": "我要殺了你，你給我等著",
                    },
                }
            ]
        }

        body = json.dumps(threat_payload).encode()
        signature = mock_line_signature(body)

        with patch("bot.line_bot.line_bot_api.reply_message") as mock_reply:
            response = await http_client.post(
                f"{bot_server}/webhook",
                content=body,
                headers={
                    "content-type": "application/json",
                    "X-Line-Signature": signature,
                },
            )

            await asyncio.sleep(0.5)

        assert response.status_code == 200
        mock_reply.assert_called()


@pytest.mark.bot
class TestUserSessionManagement:
    """使用者會話管理測試"""

    async def test_user_session_creation(
        self, bot_server, http_client, mock_line_signature
    ):
        """測試使用者會話建立"""
        user_id = "test_session_user"
        payload = {
            "events": [
                {
                    "type": "message",
                    "timestamp": int(time.time() * 1000),
                    "source": {"type": "user", "userId": user_id},
                    "replyToken": "test_reply_token",
                    "message": {
                        "id": "test_msg_id",
                        "type": "text",
                        "text": "第一條訊息",
                    },
                }
            ]
        }

        body = json.dumps(payload).encode()
        signature = mock_line_signature(body)

        with patch("bot.line_bot.line_bot_api.reply_message"):
            response = await http_client.post(
                f"{bot_server}/webhook",
                content=body,
                headers={
                    "content-type": "application/json",
                    "X-Line-Signature": signature,
                },
            )

        assert response.status_code == 200

        # 驗證會話建立（透過內部狀態）
        from bot.line_bot import user_sessions

        assert user_id in user_sessions
        assert len(user_sessions[user_id].recent_messages) == 1

    async def test_warning_count_escalation(
        self, bot_server, http_client, mock_line_signature
    ):
        """測試警告計數與升級"""
        user_id = "test_escalation_user"

        # 發送多條毒性訊息
        toxic_messages = ["你很笨耶", "廢物一個", "滾開啦你"]

        for i, message in enumerate(toxic_messages):
            payload = {
                "events": [
                    {
                        "type": "message",
                        "timestamp": int(time.time() * 1000) + i,
                        "source": {"type": "user", "userId": user_id},
                        "replyToken": f"test_reply_token_{i}",
                        "message": {
                            "id": f"test_msg_id_{i}",
                            "type": "text",
                            "text": message,
                        },
                    }
                ]
            }

            body = json.dumps(payload).encode()
            signature = mock_line_signature(body)

            with patch("bot.line_bot.line_bot_api.reply_message"):
                response = await http_client.post(
                    f"{bot_server}/webhook",
                    content=body,
                    headers={
                        "content-type": "application/json",
                        "X-Line-Signature": signature,
                    },
                )

            assert response.status_code == 200
            await asyncio.sleep(0.2)  # 等待處理

        # 驗證警告計數
        from bot.line_bot import user_sessions

        assert user_id in user_sessions
        assert user_sessions[user_id].warning_count >= 2

    async def test_context_building(self, bot_server, http_client, mock_line_signature):
        """測試對話上下文建立"""
        user_id = "test_context_user"

        # 發送一系列相關訊息
        messages = ["我今天很不開心", "同學都不理我", "我覺得自己很沒用"]

        for i, message in enumerate(messages):
            payload = {
                "events": [
                    {
                        "type": "message",
                        "timestamp": int(time.time() * 1000) + i,
                        "source": {"type": "user", "userId": user_id},
                        "replyToken": f"test_reply_token_{i}",
                        "message": {
                            "id": f"test_msg_id_{i}",
                            "type": "text",
                            "text": message,
                        },
                    }
                ]
            }

            body = json.dumps(payload).encode()
            signature = mock_line_signature(body)

            with patch("bot.line_bot.line_bot_api.reply_message"):
                response = await http_client.post(
                    f"{bot_server}/webhook",
                    content=body,
                    headers={
                        "content-type": "application/json",
                        "X-Line-Signature": signature,
                    },
                )

            assert response.status_code == 200
            await asyncio.sleep(0.2)

        # 驗證上下文建立
        from bot.line_bot import user_sessions

        assert user_id in user_sessions
        assert len(user_sessions[user_id].recent_messages) == 3


@pytest.mark.bot
class TestResponseStrategies:
    """回應策略測試"""

    def test_gentle_reminder_creation(self):
        """測試溫和提醒訊息建立"""
        from bot.line_bot import CyberPuppyBot

        bot = CyberPuppyBot()
        message = bot.create_gentle_reminder_message()

        assert hasattr(message, "text")
        assert len(message.text) > 0

    def test_firm_warning_creation(self):
        """測試嚴厲警告訊息建立"""
        from bot.line_bot import CyberPuppyBot

        bot = CyberPuppyBot()
        message = bot.create_firm_warning_message()

        assert hasattr(message, "contents")
        assert hasattr(message, "alt_text")
        assert message.alt_text == "網路行為提醒"

    def test_resource_sharing_creation(self):
        """測試資源分享訊息建立"""
        from bot.line_bot import CyberPuppyBot

        bot = CyberPuppyBot()
        message = bot.create_resource_sharing_message()

        assert hasattr(message, "contents")
        assert hasattr(message, "alt_text")
        assert message.alt_text == "支持資源"

    def test_escalation_message_creation(self):
        """測試升級處理訊息建立"""
        from bot.line_bot import CyberPuppyBot

        bot = CyberPuppyBot()
        message, quick_reply = bot.create_escalation_message()

        assert hasattr(message, "contents")
        assert hasattr(message, "alt_text")
        assert message.alt_text == "嚴重警告"
        assert quick_reply is not None

    def test_response_strategy_determination(self):
        """測試回應策略決定邏輯"""
        from bot.line_bot import CyberPuppyBot, ResponseStrategy, UserSession

        bot = CyberPuppyBot()
        user_session = UserSession(user_id="test_user")

        # 測試無毒性 -> 忽略
        analysis = {"toxicity": "none", "bullying": "none", "emotion": "neu"}
        strategy = bot.determine_response_strategy(analysis, user_session)
        assert strategy == ResponseStrategy.IGNORE

        # 測試輕微毒性 -> 溫和提醒
        analysis = {"toxicity": "toxic", "bullying": "harassment", "emotion": "neg"}
        strategy = bot.determine_response_strategy(analysis, user_session)
        assert strategy == ResponseStrategy.GENTLE_REMINDER

        # 測試嚴重毒性 -> 嚴厲警告
        analysis = {"toxicity": "severe", "bullying": "threat", "emotion": "neg"}
        strategy = bot.determine_response_strategy(analysis, user_session)
        assert strategy == ResponseStrategy.FIRM_WARNING

        # 測試重複違規 -> 升級處理
        user_session.warning_count = 3
        analysis = {"toxicity": "toxic", "bullying": "harassment", "emotion": "neg"}
        strategy = bot.determine_response_strategy(analysis, user_session)
        assert strategy == ResponseStrategy.ESCALATION


@pytest.mark.bot
class TestBotHealthAndStats:
    """Bot 健康檢查與統計測試"""

    async def test_bot_health_check(self, bot_server, http_client):
        """測試 Bot 健康檢查"""
        response = await http_client.get(f"{bot_server}/health")

        assert response.status_code == 200
        data = response.json()

        # 驗證健康檢查回應
        assert "status" in data
        assert "line_api" in data
        assert "analysis_api" in data
        assert "timestamp" in data
        assert "active_sessions" in data

        assert isinstance(data["active_sessions"], int)

    async def test_bot_stats(self, bot_server, http_client):
        """測試 Bot 統計資料"""
        response = await http_client.get(f"{bot_server}/stats")

        assert response.status_code == 200
        data = response.json()

        # 驗證統計資料格式
        assert "active_users" in data
        assert "total_warnings" in data
        assert "total_escalations" in data
        assert "timestamp" in data

        assert isinstance(data["active_users"], int)
        assert isinstance(data["total_warnings"], int)
        assert isinstance(data["total_escalations"], int)


@pytest.mark.bot
class TestErrorHandling:
    """Bot 錯誤處理測試"""

    async def test_api_service_unavailable(
        self, bot_server, http_client, mock_line_signature
    ):
        """測試分析 API 服務不可用時的處理"""
        payload = {
            "events": [
                {
                    "type": "message",
                    "timestamp": int(time.time() * 1000),
                    "source": {"type": "user", "userId": "test_error_user"},
                    "replyToken": "test_reply_token",
                    "message": {
                        "id": "test_msg_id",
                        "type": "text",
                        "text": "測試錯誤處理",
                    },
                }
            ]
        }

        body = json.dumps(payload).encode()
        signature = mock_line_signature(body)

        # 模擬 API 服務不可用
        with patch("bot.line_bot.CYBERPUPPY_API_URL", "http://invalid-api-url:9999"):
            with patch("bot.line_bot.line_bot_api.reply_message") as mock_reply:
                response = await http_client.post(
                    f"{bot_server}/webhook",
                    content=body,
                    headers={
                        "content-type": "application/json",
                        "X-Line-Signature": signature,
                    },
                )

                await asyncio.sleep(1)  # 等待錯誤處理

        assert response.status_code == 200
        # 應該有錯誤回應訊息
        mock_reply.assert_called()

    async def test_malformed_webhook_data(
        self, bot_server, http_client, mock_line_signature
    ):
        """測試格式錯誤的 Webhook 資料處理"""
        # 缺少必要欄位的 payload
        malformed_payload = {
            "events": [
                {
                    "type": "message",
                    # 缺少 timestamp, source, replyToken, message
                }
            ]
        }

        body = json.dumps(malformed_payload).encode()
        signature = mock_line_signature(body)

        response = await http_client.post(
            f"{bot_server}/webhook",
            content=body,
            headers={"content-type": "application/json", "X-Line-Signature": signature},
        )

        # 即使資料格式錯誤，webhook 也應該回傳 200 以免 LINE 重送
        assert response.status_code == 200

    async def test_non_text_message_handling(
        self, bot_server, http_client, mock_line_signature
    ):
        """測試非文字訊息處理"""
        # 圖片訊息
        image_payload = {
            "events": [
                {
                    "type": "message",
                    "timestamp": int(time.time() * 1000),
                    "source": {"type": "user", "userId": "test_image_user"},
                    "replyToken": "test_reply_token",
                    "message": {"id": "test_msg_id", "type": "image"},
                }
            ]
        }

        body = json.dumps(image_payload).encode()
        signature = mock_line_signature(body)

        response = await http_client.post(
            f"{bot_server}/webhook",
            content=body,
            headers={"content-type": "application/json", "X-Line-Signature": signature},
        )

        assert response.status_code == 200
