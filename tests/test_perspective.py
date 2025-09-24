"""
Perspective API 整合測試
"""

import asyncio
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.cyberpuppy.arbiter.perspective import (PerspectiveAPI,
                                                PerspectiveResult,
                                                UncertaintyDetector,
                                                UncertaintyReason)


class TestUncertaintyDetector:
    """不確定性檢測器測試"""

    @pytest.fixture
    def detector(self):
        return UncertaintyDetector(
            uncertainty_threshold=0.4, confidence_threshold=0.6,
                min_confidence_gap=0.1
        )

    def test_borderline_score_detection(self, detector):
        """測試邊界分數檢測"""
        prediction_scores = {
            "sco"
                "res": {
            "toxicity": "none",
            "emotion": "neu",
        }

        should_use, analysis = detector.should_use_perspective(prediction_scores)

        assert should_use is True
        assert analysis.is_uncertain is True
        assert UncertaintyReason.BORDERLINE_SCORE in analysis.reasons

    def test_high_confidence_no_uncertainty(self, detector):
        """測試高信心度無不確定性"""
        prediction_scores = {
            "sco"
                "res": {
            "toxicity": "none",
            "emotion": "neu",
        }

        should_use, analysis = detector.should_use_perspective(prediction_scores)

        assert should_use is False
        assert analysis.is_uncertain is False
        assert len(analysis.reasons) == 0

    def test_conflicting_signals_detection(self, detector):
        """測試衝突信號檢測"""
        prediction_scores = {
            "sco"
                "res": {
            "toxicity": "none",
            "emotion": "neg",
            "emotion_strength": 4,
        }

        should_use, analysis = detector.should_use_perspective(prediction_scores)

        assert should_use is True
        assert analysis.is_uncertain is True
        assert UncertaintyReason.CONFLICTING_SIGNALS in analysis.reasons

    def test_low_confidence_gap(self, detector):
        """測試低信心度差距"""
        prediction_scores = {
            "sco"
                "res": {
            "toxicity": "none",
            "emotion": "neu",
        }

        should_use, analysis = detector.should_use_perspective(prediction_scores)

        assert should_use is True
        assert analysis.is_uncertain is True
        assert UncertaintyReason.LOW_CONFIDENCE in analysis.reasons


class TestPerspectiveAPI:
    """Perspective API 測試"""

    @pytest.fixture
    def mock_api_key(self):
        return "test_api_key_12345"

    @pytest.fixture
    def mock_response_data(self):
        return {
            "attributeScores": {
                "TOXI"
                    "CITY": {
                "SEVERE_"
                    "TOXICITY": {
                "IDENTIT"
                    "Y_ATTACK": {
                "INS"
                    "ULT": {
                "PROF"
                    "ANITY": {
                "THR"
                    "EAT": {
            },
            "detectedLanguages": ["zh"],
            "clientToken": "cyberpuppy-test",
        }

    @pytest.mark.asyncio
    async def test_successful_analysis(self, mock_api_key, mock_response_data):
        """測試成功的分析請求"""
        with patch.object(httpx.AsyncClient, "post") as mock_post:
            # 模擬成功回應
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.headers = {"X-RateLimit-Remaining": "999"}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            async with PerspectiveAPI(api_key=mock_api_key) as api:
                result = await api.analyze_comment("測試文本", lang="zh")

                assert isinstance(result, PerspectiveResult)
                assert result.toxicity_score == 0.75
                assert result.severe_toxicity_score == 0.25
                assert result.language_detected == "zh"
                assert result.api_quota_remaining == 999
                assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_api_key):
        """測試速率限制"""
        rate_limit = {"requests_per_second": 1, "requests_per_day": 10}

        with patch.object(httpx.AsyncClient, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"attributeScores": {}}
            mock_response.headers = {}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            async with PerspectiveAPI(
                api_key=mock_api_key,
                rate_limit=rate_limit
            ) as api:
                # 第一次請求應該立即執行
                start_time = asyncio.get_event_loop().time()
                await api.analyze_comment("測試文本1")

                # 第二次請求應該被速率限制延遲
                await api.analyze_comment("測試文本2")
                end_time = asyncio.get_event_loop().time()

                # 應該至少等待 1 秒
                assert end_time - start_time >= 0.9

    @pytest.mark.asyncio
    async def test_input_validation(self, mock_api_key):
        """測試輸入驗證"""
        async with PerspectiveAPI(api_key=mock_api_key) as api:
            # 空文本應該拋出 ValueError
            with pytest.raises(ValueError, match="文本不能為空"):
                await api.analyze_comment("")

            with pytest.raises(ValueError, match="文本不能為空"):
                await api.analyze_comment("   ")

    @pytest.mark.asyncio
    async def test_http_error_handling(self, mock_api_key):
        """測試 HTTP 錯誤處理"""
        with patch.object(httpx.AsyncClient, "post") as mock_post:
            # 模擬 HTTP 錯誤
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request"
            mock_post.return_value = mock_response
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "400 Bad Request", request=None, response=mock_response
            )

            async with PerspectiveAPI(api_key=mock_api_key) as api:
                with pytest.raises(httpx.HTTPStatusError):
                    await api.analyze_comment("測試文本")

    def test_missing_api_key(self):
        """測試缺少 API Key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="遺失 Perspective API Key"):
                PerspectiveAPI()

    @pytest.mark.asyncio
    async def test_quota_tracking(self, mock_api_key, mock_response_data):
        """測試配額追蹤"""
        with patch.object(httpx.AsyncClient, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.headers = {}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            async with PerspectiveAPI(api_key=mock_api_key) as api:
                # 執行幾次請求
                await api.analyze_comment("測試1")
                await api.analyze_comment("測試2")

                # 檢查配額狀態
                quota = await api.get_quota_status()
                assert quota["daily_requests_used"] == 2
                assert quota["daily_requests_limit"] == 1000

    @pytest.mark.asyncio
    async def test_long_text_truncation(
        self,
        mock_api_key,
        mock_response_data
    ):
        """測試長文本截斷"""
        with patch.object(httpx.AsyncClient, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.headers = {}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            async with PerspectiveAPI(api_key=mock_api_key) as api:
                # 創建超過 3000 字元的文本
                long_text = "測試" * 1500  # 3000 字元

                result = await api.analyze_comment(long_text)

                # 確認請求被發送（文本應被截斷但不拋出錯誤）
                assert isinstance(result, PerspectiveResult)
                mock_post.assert_called_once()


@pytest.mark.integration
class TestPerspectiveIntegration:
    """Perspective API 整合測試（需要實際 API Key）"""

    @pytest.mark.skipif(
        not os.getenv("PERSPECTI"
            "VE_API_KEY"), reason=
    )
    @pytest.mark.asyncio
    async def test_real_api_call(self):
        """測試實際 API 呼叫"""
        async with PerspectiveAPI() as api:
            result = await api.analyze_comment("This is a t"
                "est message", lang=

            assert isinstance(result, PerspectiveResult)
            assert 0 <= result.toxicity_score <= 1
            assert result.processing_time_ms > 0
            assert result.text_hash is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
