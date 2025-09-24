"""
Google Perspective API 整合模組

官方文件參考：
- Perspective API: https://developers.perspectiveapi.com/
- API Reference: https://developers.perspectiveapi.com/s/about-the-api-methods
- 申請 API Key: https://developers.perspectiveapi.com/s/docs-get-started

注意事項：
1. Perspective API 主要針對英文訓練，中文支援有限
2. 僅在本地模型不確定時（0.4 < score < 0.6）作為參考
3. 結果不直接影響最終決策，僅作為額外驗證
4. 需要申請 Google Cloud API Key 並啟用 Perspective API
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# 設定日誌
logger = logging.getLogger(__name__)


class UncertaintyReason(Enum):
    """不確定原因"""

    LOW_CONFIDENCE = "low_confidence"
    BORDERLINE_SCORE = "borderline_score"
    CONFLICTING_SIGNALS = "conflicting_signals"
    INSUFFICIENT_CONTEXT = "insufficient_context"


@dataclass
class PerspectiveResult:
    """Perspective API 結果"""

    toxicity_score: float
    severe_toxicity_score: float
    identity_attack_score: float
    insult_score: float
    profanity_score: float
    threat_score: float

    # 元資料
    text_hash: str
    language_detected: Optional[str] = None
    processing_time_ms: float = 0.0
    api_quota_remaining: Optional[int] = None

    # 內部使用
    raw_response: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class UncertaintyAnalysis:
    """不確定性分析結果"""

    is_uncertain: bool
    confidence_score: float
    reasons: List[UncertaintyReason]
    recommendation: str
    threshold_details: Dict[str, float]


class PerspectiveAPI:
    """
    Google Perspective API 客戶端

    提供毒性檢測的外部驗證服務，僅在本地模型不確定時使用
    """

    # API 端點
    BASE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

    # 支援的屬性
    ATTRIBUTES = [
        "TOXICITY",
        "SEVERE_TOXICITY",
        "IDENTITY_ATTACK",
        "INSULT",
        "PROFANITY",
        "THREAT",
    ]

    # 速率限制設定
    DEFAULT_RATE_LIMIT = {
        "requests_per_second": 1,
        "requests_per_minute": 60,
    }

    def __init__(
        self, api_key: Optional[str] = None, rate_limit: Optional[Dict[str, int]] = None
    ):
        """
        初始化 Perspective API 客戶端

        Args:
            api_key: Google Cloud API Key，如未提供則從環境變數讀取
            rate_limit: 自訂速率限制設定
        """
        self.api_key = api_key or os.getenv("PERSPECTIVE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "遺失 Perspective API Key。請設定環境變數 PERSPECTIVE_API_KEY "
                "或在初始化時提供 api_key 參數。"
            )

        # 速率限制設定
        self.rate_limit = {**self.DEFAULT_RATE_LIMIT, **(rate_limit or {})}

        # 速率限制狀態
        self._request_times: List[float] = []
        self._daily_request_count = 0
        self._daily_reset_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # HTTP 客戶端
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Content-Type": "application/json",
            },
        )

        logger.info("Perspective API 客戶端已初始化")

    async def __aenter__(self):
        """異步上下文管理器進入"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器退出"""
        await self.client.aclose()

    def _check_daily_limit(self):
        """檢查每日請求限制"""
        now = datetime.now()
        if now.date() != self._daily_reset_time.date():
            self._daily_request_count = 0
            self._daily_reset_time = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        if self._daily_request_count >= self.rate_limit["requests_per_day"]:
            raise httpx.HTTPStatusError(
                f"已達到每日請求限制 ({self.rate_limit['requests_per_day']})",
                request=None,
                response=None,
            )

    async def _rate_limit_wait(self):
        """速率限制等待"""
        current_time = time.time()

        # 清理過期的請求時間記錄
        cutoff_time = current_time - 1.0  # 1秒前
        self._request_times = [t for t in self._request_times if t > cutoff_time]

        # 檢查是否超過速率限制
        if len(self._request_times) >= self.rate_limit["requests_per_second"]:
            sleep_time = 1.0 - (current_time - self._request_times[0])
            if sleep_time > 0:
                logger.debug(f"速率限制等待 {sleep_time:.2f} 秒")
                await asyncio.sleep(sleep_time)

        # 記錄請求時間
        self._request_times.append(current_time)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    )
    async def analyze_comment(
        self,
        text: str,
        lang: str = "z" "h",
        requested_attributes: Optional[List[str]] = None,
    ) -> PerspectiveResult:
        """
        分析評論的毒性

        Args:
            text: 待分析文本
            lang: 語言代碼，預設為中文 'zh'
            requested_attributes: 要分析的屬性列表，預設分析所有支援屬性

        Returns:
            PerspectiveResult: 分析結果

        Raises:
            httpx.HTTPStatusError: API 請求錯誤
            ValueError: 輸入驗證錯誤
        """
        start_time = time.time()

        # 輸入驗證
        if not text or not text.strip():
            raise ValueError("文本不能為空")

        if len(text) > 3000:  # Perspective API 限制
            logger.warning("文本長度超過 3000 字元，將被截斷")
            text = text[:3000]

        # 檢查限制
        self._check_daily_limit()
        await self._rate_limit_wait()

        # 準備請求屬性
        attributes = requested_attributes or self.ATTRIBUTES
        requested_attributes_dict = {
            attr: {} for attr in attributes if attr in self.ATTRIBUTES
        }

        # 構建請求
        request_data = {
            "comment": {"text": text},
            "requestedAttributes": requested_attributes_dict,
            "languages": [lang],
            "doNotStore": True,  # 不儲存資料以保護隱私
            "clientToken": f"cyberpuppy-{int(time.time())}",  # 用於除錯的客戶端標識
        }

        try:
            # 發送請求
            response = await self.client.post(
                f"{self.BASE_URL}?key={self.api_key}", json=request_data
            )
            response.raise_for_status()
            result_data = response.json()

            # 更新請求計數
            self._daily_request_count += 1

            # 解析結果
            scores = result_data.get("attributeScores", {})
            processing_time = (time.time() - start_time) * 1000

            # 取得配額資訊（如果有）
            quota_remaining = None
            if "X-RateLimit-Remaining" in response.headers:
                quota_remaining = int(response.headers["X-RateLimi" "t-Remaining"])

            # 生成文本雜湊
            import hashlib

            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

            result = PerspectiveResult(
                toxicity_score=self._extract_score(scores, "TOXICITY"),
                severe_toxicity_score=self._extract_score(scores, "SEVERE_" "TOXICITY"),
                identity_attack_score=self._extract_score(scores, "IDENTIT" "Y_ATTACK"),
                insult_score=self._extract_score(scores, "INSULT"),
                profanity_score=self._extract_score(scores, "PROFANITY"),
                threat_score=self._extract_score(scores, "THREAT"),
                text_hash=text_hash,
                language_detected=result_data.get("detected" "Languages", [None])[0],
                processing_time_ms=processing_time,
                api_quota_remaining=quota_remaining,
                raw_response=result_data,
            )

            logger.info(
                f"Perspective API 分析完成 - 毒性: {result.toxicity_score:.3f}, "
                f"處理時間: {processing_time:.1f}ms"
            )

            return result

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Perspective API HTTP 錯誤: {e.response.status_code} - {e.response.text}"
            )
            raise
        except httpx.RequestError as e:
            logger.error(f"Perspective API 請求錯誤: {e}")
            raise
        except Exception as e:
            logger.error(f"Perspective API 未預期錯誤: {e}")
            raise

    def _extract_score(self, scores: Dict[str, Any], attribute: str) -> float:
        """從 API 回應中提取分數"""
        if attribute not in scores:
            return 0.0

        summary_scores = scores[attribute].get("summaryScore", {})
        return float(summary_scores.get("value", 0.0))

    async def get_quota_status(self) -> Dict[str, Any]:
        """取得 API 配額狀態"""
        return {
            "daily_requests_used": self._daily_request_count,
            "daily_requests_limit": self.rate_limit.get("requests_per_day", 1000),
            "requests_per_second_limit": self.rate_limit["requests_per_second"],
            "daily_reset_time": self._daily_reset_time.isoformat(),
            "recent_requests_count": len(self._request_times),
        }


class UncertaintyDetector:
    """
    不確定性檢測器

    判斷本地模型的預測是否存在不確定性，決定是否需要外部驗證
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.4,
        confidence_threshold: float = 0.6,
        min_confidence_gap: float = 0.1,
    ):
        """
        初始化不確定性檢測器

        Args:
            uncertainty_threshold: 不確定性下閾值
            confidence_threshold: 信心度上閾值
            min_confidence_gap: 最小信心度差距
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold
        self.min_confidence_gap = min_confidence_gap

        logger.info(
            f"不確定性檢測器已初始化 - 閾值: {uncertainty_threshold}-{confidence_threshold}"
        )

    def analyze_uncertainty(
        self, prediction_scores: Dict[str, Any]
    ) -> UncertaintyAnalysis:
        """
        分析預測結果的不確定性

        Args:
            prediction_scores: 本地模型的預測分數

        Returns:
            UncertaintyAnalysis: 不確定性分析結果
        """
        reasons = []
        is_uncertain = False
        confidence_score = 0.0
        threshold_details = {}

        # 檢查毒性分數的不確定性
        toxicity_scores = prediction_scores.get("scores", {}).get("toxicity", [])
        if toxicity_scores:
            max_score = max(toxicity_scores.values())
            second_max_score = (
                sorted(toxicity_scores.values(), reverse=True)[1]
                if len(toxicity_scores) > 1
                else 0.0
            )

            confidence_score = max_score
            threshold_details["max_toxicity_score"] = max_score
            threshold_details["confidence_gap"] = max_score - second_max_score

            # 邊界分數檢測
            if self.uncertainty_threshold < max_score < self.confidence_threshold:
                is_uncertain = True
                reasons.append(UncertaintyReason.BORDERLINE_SCORE)

            # 低信心度檢測
            if max_score - second_max_score < self.min_confidence_gap:
                is_uncertain = True
                reasons.append(UncertaintyReason.LOW_CONFIDENCE)

        # 檢查情緒與毒性的衝突信號
        emotion = prediction_scores.get("emotion", "neu")
        emotion_strength = prediction_scores.get("emotion_strength", 0)

        if emotion == "neg" and emotion_strength >= 3:
            toxicity = prediction_scores.get("toxicity", "none")
            if toxicity == "none" and max_score < 0.3:  # 強烈負面情緒但毒性很低
                is_uncertain = True
                reasons.append(UncertaintyReason.CONFLICTING_SIGNALS)

        # 檢查上下文不足
        if not prediction_scores.get("context") and confidence_score < 0.7:
            reasons.append(UncertaintyReason.INSUFFICIENT_CONTEXT)

        # 生成建議
        if is_uncertain:
            if UncertaintyReason.BORDERLINE_SCORE in reasons:
                recommendation = "建議使用外部 API 驗證邊界案例"
            elif UncertaintyReason.CONFLICTING_SIGNALS in reasons:
                recommendation = "建議驗證情緒與毒性的衝突信號"
            else:
                recommendation = "建議額外驗證以提高準確度"
        else:
            recommendation = "本地模型預測信心度足夠，無需外部驗證"

        return UncertaintyAnalysis(
            is_uncertain=is_uncertain,
            confidence_score=confidence_score,
            reasons=reasons,
            recommendation=recommendation,
            threshold_details=threshold_details,
        )

    def should_use_perspective(
        self, prediction_scores: Dict[str, Any]
    ) -> Tuple[bool, UncertaintyAnalysis]:
        """
        判斷是否應該使用 Perspective API

        Args:
            prediction_scores: 本地模型預測分數

        Returns:
            Tuple[bool, UncertaintyAnalysis]: (是否使用, 不確定性分析)
        """
        analysis = self.analyze_uncertainty(prediction_scores)

        # 只在真正不確定且有邊界分數或衝突信號時使用外部 API
        should_use = analysis.is_uncertain and (
            UncertaintyReason.BORDERLINE_SCORE in analysis.reasons
            or UncertaintyReason.CONFLICTING_SIGNALS in analysis.reasons
        )

        return should_use, analysis


async def example_usage():
    """使用範例"""

    # 檢查是否有 API Key
    api_key = os.getenv("PERSPECTIVE_API_KEY")
    if not api_key:
        logger.warning("未設定 PERSPECTIVE_API_KEY，跳過範例")
        return

    # 初始化服務
    detector = UncertaintyDetector()

    async with PerspectiveAPI(api_key) as perspective:
        # 模擬本地模型結果
        local_prediction = {
            "toxicity": "none",
            "scores": {"toxicity": {"none": 0.5, "toxic": 0.4, "severe": 0.1}},
            "emotion": "neg",
            "emotion_strength": 3,
        }

        # 檢查是否需要外部驗證
        should_use, uncertainty = detector.should_use_perspective(local_prediction)

        logger.info(f"是否需要外部驗證: {should_use}")
        logger.info(f"不確定性原因: {[r.value for r in uncertainty.reasons]}")

        if should_use:
            # 使用 Perspective API 進行驗證
            text = "這是一個測試文本"
            try:
                result = await perspective.analyze_comment(text, lang="zh")
                logger.info(f"Perspective API 毒性分數: {result.toxicity_score:.3f}")

                # 取得配額狀態
                quota = await perspective.get_quota_status()
                logger.info(f"API 配額狀態: {quota}")

            except Exception as e:
                logger.error(f"Perspective API 呼叫失敗: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
