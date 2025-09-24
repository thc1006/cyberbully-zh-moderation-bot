"""
API 端點整合測試
測試完整的 API 功能，包括：
- 健康檢查端點
- 文本分析端點
- 錯誤處理
- 速率限制
- 回應格式驗證
"""

import asyncio
import time
from typing import Dict, Any

import httpx
import pytest

from tests.integration import MAX_RESPONSE_TIME_MS


@pytest.mark.api
class TestAPIHealthCheck:
    """API 健康檢查測試"""

    async def test_health_check_endpoint(self, api_server, http_client):
        """測試健康檢查端點"""
        response = await http_client.get(f"{api_server}/healthz")

        assert response.status_code == 200
        data = response.json()

        # 驗證回應格式
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data

        assert data["status"] == "healthy"
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    async def test_root_endpoint(self, api_server, http_client):
        """測試根端點"""
        response = await http_client.get(f"{api_server}/")

        assert response.status_code == 200
        data = response.json()

        # 驗證 API 資訊
        assert "name" in data
        assert "description" in data
        assert "version" in data
        assert data["name"] == "CyberPuppy Moderation API"


@pytest.mark.api
class TestAPIAnalyze:
    """文本分析 API 測試"""

    async def test_analyze_valid_text(self, api_server, http_client, test_data_small):
        """測試有效文本分析"""
        for test_case in test_data_small:
            payload = {"text": test_case["text"]}
            start_time = time.time()

            response = await http_client.post(f"{api_server}/analyze", json=payload)

            response_time = (time.time() - start_time) * 1000

            # 驗證回應
            assert response.status_code == 200, f"Failed for text: {test_case['text']}"

            data = response.json()
            await self._validate_analyze_response(data)

            # 驗證回應時間
            assert (
                response_time < MAX_RESPONSE_TIME_MS
            ), f"Response time {response_time}ms exceeds limit {MAX_RESPONSE_TIME_MS}ms"

            # 驗證預期結果（部分匹配，因為是模擬 API）
            expected = test_case["expected"]
            if expected["toxicity"] != "none":
                assert data["toxicity"] in ["toxic", "severe"]

    async def test_analyze_with_context(self, api_server, http_client):
        """測試帶上下文的文本分析"""
        payload = {
            "text": "那不是你的錯",
            "context": "之前有人說: 我覺得我很笨",
            "thread_id": "test_thread_123",
        }

        response = await http_client.post(f"{api_server}/analyze", json=payload)

        assert response.status_code == 200
        data = response.json()
        await self._validate_analyze_response(data)

    async def test_analyze_empty_text(self, api_server, http_client):
        """測試空文本處理"""
        payload = {"text": ""}

        response = await http_client.post(f"{api_server}/analyze", json=payload)

        assert response.status_code == 422  # Validation error

    async def test_analyze_too_long_text(self, api_server, http_client):
        """測試過長文本處理"""
        payload = {"text": "a" * 2000}  # 超過 1000 字元限制

        response = await http_client.post(f"{api_server}/analyze", json=payload)

        assert response.status_code == 422  # Validation error

    async def test_analyze_chinese_special_chars(self, api_server, http_client):
        """測試中文特殊字元處理"""
        special_texts = [
            "😊😢😡 情緒符號測試",
            "@#$%^&*() 特殊符號測試",
            "１２３４５ 全形數字測試",
            "ａｂｃｄｅ 全形英文測試",
            "繁體中文 vs 简体中文",
            "🚨⚠️💀 警告符號測試",
        ]

        for text in special_texts:
            payload = {"text": text}
            response = await http_client.post(f"{api_server}/analyze", json=payload)

            assert response.status_code == 200, f"Failed for text: {text}"
            data = response.json()
            await self._validate_analyze_response(data)

    async def _validate_analyze_response(self, data: Dict[str, Any]):
        """驗證分析回應格式"""
        # 必要欄位
        required_fields = [
            "toxicity",
            "bullying",
            "role",
            "emotion",
            "emotion_strength",
            "scores",
            "explanations",
        ]

        for field in required_fields:
            assert field in data, f"Missing field: {field}"

        # 驗證標籤值
        assert data["toxicity"] in ["none", "toxic", "severe"]
        assert data["bullying"] in ["none", "harassment", "threat"]
        assert data["role"] in ["none", "perpetrator", "victim", "bystander"]
        assert data["emotion"] in ["pos", "neu", "neg"]
        assert 0 <= data["emotion_strength"] <= 4

        # 驗證分數結構
        scores = data["scores"]
        assert "toxicity" in scores
        assert "bullying" in scores
        assert "role" in scores
        assert "emotion" in scores

        # 驗證機率分數和為 1（允許小誤差）
        for category, score_dict in scores.items():
            if isinstance(score_dict, dict):
                total = sum(score_dict.values())
                assert (
                    0.99 <= total <= 1.01
                ), f"{category} scores don't sum to 1: {total}"

        # 驗證可解釋性資料
        explanations = data["explanations"]
        assert "important_words" in explanations
        assert "method" in explanations
        assert "confidence" in explanations
        assert isinstance(explanations["important_words"], list)
        assert 0 <= explanations["confidence"] <= 1


@pytest.mark.api
class TestAPIRateLimit:
    """速率限制測試"""

    async def test_rate_limit_enforcement(self, api_server, http_client):
        """測試速率限制執行"""
        payload = {"text": "速率限制測試"}

        # 快速發送多個請求
        tasks = []
        for i in range(35):  # 超過 30/分鐘 的限制
            task = http_client.post(f"{api_server}/analyze", json=payload)
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 檢查是否有速率限制回應
        status_codes = []
        for response in responses:
            if isinstance(response, httpx.Response):
                status_codes.append(response.status_code)
            elif isinstance(response, Exception):
                # 可能是超時或其他錯誤
                continue

        # 應該有一些請求被速率限制（429 狀態碼）
        success_count = status_codes.count(200)
        rate_limited_count = status_codes.count(429)

        assert success_count <= 30, f"Too many successful requests: {success_count}"
        assert rate_limited_count > 0, "Rate limiting not enforced"


@pytest.mark.api
class TestAPIErrorHandling:
    """API 錯誤處理測試"""

    async def test_invalid_json(self, api_server, http_client):
        """測試無效 JSON 處理"""
        response = await http_client.post(
            f"{api_server}/analyze",
            content="invalid json",
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 422

    async def test_missing_required_fields(self, api_server, http_client):
        """測試缺少必要欄位"""
        payloads = [
            {},  # 空物件
            {"context": "只有上下文"},  # 缺少 text
            {"text": None},  # text 為 null
        ]

        for payload in payloads:
            response = await http_client.post(f"{api_server}/analyze", json=payload)
            assert response.status_code == 422

    async def test_unsupported_media_type(self, api_server, http_client):
        """測試不支援的媒體類型"""
        response = await http_client.post(
            f"{api_server}/analyze",
            content="text=測試",
            headers={"content-type": "application/x-www-form-urlencoded"},
        )

        assert response.status_code == 422

    async def test_method_not_allowed(self, api_server, http_client):
        """測試不允許的 HTTP 方法"""
        # 對分析端點使用 GET 方法
        response = await http_client.get(f"{api_server}/analyze")
        assert response.status_code == 405

        # 對健康檢查端點使用 POST 方法
        response = await http_client.post(f"{api_server}/healthz")
        assert response.status_code == 405


@pytest.mark.api
@pytest.mark.slow
class TestAPIStressTest:
    """API 壓力測試"""

    async def test_concurrent_requests(
        self, api_server, http_client, performance_monitor
    ):
        """測試併發請求處理"""
        payload = {"text": "併發測試訊息"}
        concurrent_requests = 10

        async with performance_monitor() as monitor:
            tasks = []
            for i in range(concurrent_requests):
                task = http_client.post(f"{api_server}/analyze", json=payload)
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

        # 驗證所有請求都成功
        for i, response in enumerate(responses):
            assert (
                response.status_code == 200
            ), f"Request {i} failed: {response.status_code}"

        # 驗證回應時間合理
        metrics = await monitor
        avg_response_time = metrics["duration"] / concurrent_requests * 1000
        assert (
            avg_response_time < MAX_RESPONSE_TIME_MS * 2
        ), f"Average response time too slow: {avg_response_time}ms"

    async def test_memory_usage_stability(self, api_server, http_client):
        """測試記憶體使用穩定性"""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # 執行多個請求
        payload = {"text": "記憶體測試訊息 " * 50}  # 較長的文本

        for i in range(20):
            response = await http_client.post(f"{api_server}/analyze", json=payload)
            assert response.status_code == 200

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 記憶體增長不應超過 50MB
        assert (
            memory_increase < 50 * 1024 * 1024
        ), f"Memory increase too large: {memory_increase / 1024 / 1024:.2f}MB"


@pytest.mark.api
class TestAPILogging:
    """API 日誌測試"""

    async def test_privacy_protection_logging(self, api_server, http_client):
        """測試隱私保護日誌記錄"""
        # 包含個人資訊的文本
        sensitive_text = (
            "我的電話是 0912345678，"
            "信用卡號碼是 1234-5678-9012-3456，"
            "身分證號碼是 A123456789"
        )

        payload = {"text": sensitive_text}
        response = await http_client.post(f"{api_server}/analyze", json=payload)

        assert response.status_code == 200
        data = response.json()

        # 驗證回應包含文本雜湊而非原文
        assert "text_hash" in data
        assert len(data["text_hash"]) == 16  # SHA-256 前 16 字元
        assert sensitive_text not in str(data)  # 原文不應出現在回應中
