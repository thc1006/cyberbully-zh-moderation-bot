"""
CyberPuppy API Python Client SDK
中文網路霸凌防治 API Python 客戶端

功能特色:
- 自動重試與錯誤處理
- 限流管理
- 批次處理支援
- 非同步處理
- 詳細的日誌記錄
"""

import asyncio
import hashlib
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

import httpx
from dataclasses import dataclass, field


# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """分析結果資料類別"""
    toxicity: str
    bullying: str
    role: str
    emotion: str
    emotion_strength: int
    scores: Dict[str, Any]
    explanations: Dict[str, Any]
    text_hash: str
    timestamp: str
    processing_time_ms: float
    raw_response: Dict[str, Any] = field(repr=False)


@dataclass
class ClientConfig:
    """客戶端配置"""
    api_key: str
    base_url: str = "https://api.cyberpuppy.ai"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: int = 30  # requests per minute


class CyberPuppyError(Exception):
    """CyberPuppy API 錯誤基類"""
    pass


class AuthenticationError(CyberPuppyError):
    """認證錯誤"""
    pass


class RateLimitError(CyberPuppyError):
    """限流錯誤"""
    pass


class ValidationError(CyberPuppyError):
    """輸入驗證錯誤"""
    pass


class ServiceError(CyberPuppyError):
    """服務錯誤"""
    pass


class CyberPuppyClient:
    """CyberPuppy API 客戶端"""

    def __init__(self, config: ClientConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "CyberPuppy-Python-SDK/1.0.0"
            }
        )
        self._request_times: List[float] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def _check_rate_limit(self):
        """檢查限流狀態"""
        now = time.time()
        # 移除一分鐘前的請求記錄
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.config.rate_limit:
            raise RateLimitError(
                f"Rate limit exceeded: {self.confi"
                    "g.rate_limit} requests per minute"
            )

        self._request_times.append(now)

    def _handle_error(self, response: httpx.Response) -> None:
        """處理 API 錯誤回應"""
        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        elif response.status_code == 400:
            try:
                error_data = response.json()
                raise ValidationError(error_data.get("message", "Validation failed")) 
            except json.JSONDecodeError:
                raise ValidationError("Invalid request format")
        elif response.status_code >= 500:
            raise ServiceError(f"Service error: {response.status_code}")
        else:
            raise CyberPuppyError(f"Unexpected error: {response.status_code}")

    async def _make_request(
        self, method: str, endpoint: str, data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """發送 API 請求（含重試機制）"""
        self._check_rate_limit()

        url = f"{self.config.base_url}{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.request(method, url, json=data)

                if response.is_success:
                    return response.json()

                # 如果是客戶端錯誤，不重試
                if 400 <= response.status_code < 500:
                    self._handle_error(response)

                # 服務器錯誤，進行重試
                if attempt == self.config.max_retries - 1:
                    self._handle_error(response)

            except httpx.RequestError as e:
                if attempt == self.config.max_retries - 1:
                    raise ServiceError(f"Request failed: {str(e)}")

                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")

            # 指數退避
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        raise ServiceError("Max retries exceeded")

    async def analyze_text(
        self,
        text: str,
        context: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> AnalysisResult:
        """
        分析單一文本

        Args:
            text: 待分析文本 (1-1000 字符)
            context: 對話上下文 (可選，最多 2000 字符)
            thread_id: 對話串 ID (可選，最多 50 字符)

        Returns:
            AnalysisResult: 分析結果

        Raises:
            ValidationError: 輸入驗證錯誤
            AuthenticationError: 認證失敗
            RateLimitError: 超出限流
            ServiceError: 服務錯誤
        """
        # 輸入驗證
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty")
        if len(text) > 1000:
            raise ValidationError("Text exceeds maximum le"
                "ngth of 1000 characters")
        if context and len(context) > 2000:
            raise ValidationError("Context exceeds maximum "
                "length of 2000 characters")
        if thread_id and len(thread_id) > 50:
            raise ValidationError("Thread ID exceeds maximu"
                "m length of 50 characters")

        payload = {"text": text.strip()}
        if context:
            payload["context"] = context.strip()
        if thread_id:
            payload["thread_id"] = thread_id.strip()

        logger.info(f"Analyzing text (length: {len(text)}, hash: {h"
            "ashlib.sha256(text.encode()).hexdigest()[:8]})")

        response = await self._make_request("POST", "/analyze", payload)

        return AnalysisResult(
            toxicity=response["toxicity"],
            bullying=response["bullying"],
            role=response["role"],
            emotion=response["emotion"],
            emotion_strength=response["emotion_strength"],
            scores=response["scores"],
            explanations=response["explanations"],
            text_hash=response["text_hash"],
            timestamp=response["timestamp"],
            processing_time_ms=response["processing_time_ms"],
            raw_response=response
        )

    async def analyze_batch(
        self,
        texts: List[str],
        contexts: Optional[List[str]] = None,
        thread_ids: Optional[List[str]] = None,
        max_concurrent: int = 5
    ) -> List[AnalysisResult]:
        """
        批次分析多個文本

        Args:
            texts: 文本列表
            contexts: 對應的上下文列表 (可選)
            thread_ids: 對應的對話串 ID 列表 (可選)
            max_concurrent: 最大並行請求數

        Returns:
            List[AnalysisResult]: 分析結果列表
        """
        if not texts:
            raise ValidationError("Text list cannot be empty")

        # 準備批次資料
        tasks = []
        for i, text in enumerate(texts):
            context = contexts[i] if contexts and i < len(contexts) else None
            thread_id = thread_ids[i] if thread_ids and i < len(thread_ids)
                else None
            tasks.append(self.analyze_text(text, context, thread_id))

        # 使用信號量控制並行數量
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(task):
            async with semaphore:
                return await task

        # 執行批次分析
        logger.info(f"Starting batch analysis of {len(texts)} texts")
        start_time = time.time()

        results = await asyncio.gather(
            *[analyze_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Batch analysis completed in {elapsed_time:.2f} seconds")

        # 處理結果
        successful_results = []
        failed_results = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({"index": i, "error": str(result)})
            else:
                successful_results.append(result)

        if failed_results:
            logger.warning(f"Failed to analyze {len(failed_r"
                "esults)} texts: {failed_results}")

        return successful_results

    async def health_check(self) -> Dict[str, Any]:
        """檢查 API 服務狀態"""
        response = await self._make_request("GET", "/healthz")
        return response

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """取得限流狀態"""
        now = time.time()
        recent_requests = [t for t in self._request_times if now - t < 60]

        return {
            "requests_in_last_minute": len(recent_requests),
            "rate_limit": self.config.rate_limit,
            "remaining": max(0, self.config.rate_limit - len(recent_requests)),
            "reset_time": max(recent_requests) + 60 if recent_requests else now
        }


class CyberPuppySyncClient:
    """CyberPuppy API 同步客戶端"""

    def __init__(self, config: ClientConfig):
        self.async_client = CyberPuppyClient(config)

    def analyze_text(
        self,
        text: str,
        context: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> AnalysisResult:
        """同步分析文本"""
        return asyncio.run(
            self.async_client.analyze_text(text,
            context,
            thread_id)
        )

    def analyze_batch(
        self,
        texts: List[str],
        contexts: Optional[List[str]] = None,
        thread_ids: Optional[List[str]] = None,
        max_concurrent: int = 5
    ) -> List[AnalysisResult]:
        """同步批次分析"""
        return asyncio.run(
            self.async_client.analyze_batch(
                texts,
                contexts,
                thread_ids,
                max_concurrent
            )
        )

    def health_check(self) -> Dict[str, Any]:
        """同步健康檢查"""
        return asyncio.run(self.async_client.health_check())

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """取得限流狀態"""
        return self.async_client.get_rate_limit_status()


# 使用範例
async def example_usage():
    """使用範例"""
    # 配置客戶端
    config = ClientConfig(
        api_key="cp_your_api_key_here",
        base_url="https://api.cyberpuppy.ai",
        timeout=30,
        max_retries=3,
        rate_limit=30
    )

    async with CyberPuppyClient(config) as client:
        try:
            # 健康檢查
            health = await client.health_check()
            print(f"API 狀態: {health['status']}")

            # 單一文本分析
            result = await client.analyze_text("你好，今天天氣真好！")
            print(f"毒性等級: {result.toxicity}")
            print(f"情緒: {result.emotion} (強度: {result.emotion_strength})")
            print(f"重要詞彙: {result.explanations['important_words']}")

            # 帶上下文的分析
            context_result = await client.analyze_text(
                text="我不同意你的看法",
                context="剛才討論的是關於教育政策的議題",
                thread_id="edu_discussion_001"
            )
            print(f"上下文分析結果: {context_result.toxicity}")

            # 批次分析
            texts = [
                "你好嗎？",
                "今天天氣很好",
                "你這個笨蛋！",
                "謝謝你的幫助"
            ]

            batch_results = await client.analyze_batch(texts, max_concurrent=2)
            print(f"批次分析完成，處理了 {len(batch_results)} 個文本")

            for i, result in enumerate(batch_results):
                print(f"文本 {i}: {result.toxicity}")

            # 限流狀態
            rate_status = client.get_rate_limit_status()
            print(f"剩餘請求次數: {rate_status['remaining']}")

        except ValidationError as e:
            print(f"輸入驗證錯誤: {e}")
        except AuthenticationError as e:
            print(f"認證錯誤: {e}")
        except RateLimitError as e:
            print(f"限流錯誤: {e}")
        except ServiceError as e:
            print(f"服務錯誤: {e}")
        except CyberPuppyError as e:
            print(f"API 錯誤: {e}")


def sync_example():
    """同步客戶端使用範例"""
    config = ClientConfig(api_key="cp_your_api_key_here")
    client = CyberPuppySyncClient(config)

    try:
        # 同步分析
        result = client.analyze_text("今天心情很好！")
        print(f"同步分析結果: {result.emotion}")

        # 同步批次分析
        texts = ["你好", "再見", "謝謝"]
        batch_results = client.analyze_batch(texts)
        print(f"同步批次分析完成: {len(batch_results)} 個結果")

    except CyberPuppyError as e:
        print(f"錯誤: {e}")


if __name__ == "__main__":
    print("=== 非同步範例 ===")
    asyncio.run(example_usage())

    print("\n=== 同步範例 ===")
    sync_example()
