# CyberPuppy API 錯誤代碼與故障排除指南
# Error Codes & Troubleshooting Guide

## 錯誤代碼分類 (Error Code Categories)

### HTTP 狀態碼對應表

| HTTP Status | 分類 | 描述 | 處理建議 |
|-------------|------|------|----------|
| 200 | Success | 請求成功 | 正常處理回應 |
| 400 | Client Error | 客戶端錯誤 | 檢查請求格式 |
| 401 | Authentication | 認證失敗 | 檢查 API 密鑰 |
| 403 | Authorization | 權限不足 | 檢查帳戶權限 |
| 429 | Rate Limit | 超出限流 | 減緩請求頻率 |
| 500 | Server Error | 服務器錯誤 | 稍後重試或聯絡支援 |
| 502 | Gateway Error | 網關錯誤 | 上游服務問題，稍後重試 |
| 503 | Service Unavailable | 服務不可用 | 系統維護或過載，稍後重試 |

## 詳細錯誤代碼 (Detailed Error Codes)

### 1. 認證錯誤 (Authentication Errors) - 401

#### INVALID_API_KEY
```json
{
  "error": true,
  "message": "Invalid API key",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "INVALID_API_KEY",
    "description": "提供的 API 密鑰無效或已過期"
  }
}
```

**原因**:
- API 密鑰格式錯誤
- API 密鑰已過期
- API 密鑰已被撤銷

**解決方案**:
```bash
# 檢查 API 密鑰格式
echo "你的 API 密鑰應該以 'cp_' 開頭，長度為 35 字符"

# 驗證 API 密鑰
curl -H "Authorization: Bearer cp_your_api_key" \
  "https://api.cyberpuppy.ai/healthz"
```

#### MISSING_API_KEY
```json
{
  "error": true,
  "message": "API key is required",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "MISSING_API_KEY",
    "description": "請求標頭中缺少 Authorization"
  }
}
```

**解決方案**:
```bash
# 正確的請求標頭
curl -X POST "https://api.cyberpuppy.ai/analyze" \
  -H "Authorization: Bearer cp_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"text": "測試文本"}'
```

### 2. 輸入驗證錯誤 (Validation Errors) - 400

#### EMPTY_TEXT
```json
{
  "error": true,
  "message": "輸入驗證錯誤: 文本不能為空",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "EMPTY_TEXT",
    "field": "text"
  }
}
```

**原因**:
- 文本欄位為空字符串
- 文本僅包含空白字符

**解決方案**:
```python
# Python 範例 - 正確的輸入驗證
def validate_text(text):
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    return text.strip()
```

#### TEXT_TOO_LONG
```json
{
  "error": true,
  "message": "輸入驗證錯誤: 文本長度超過限制",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "TEXT_TOO_LONG",
    "field": "text",
    "max_length": 1000,
    "actual_length": 1250
  }
}
```

**解決方案**:
```javascript
// JavaScript 範例 - 文本長度檢查
function validateTextLength(text, maxLength = 1000) {
  if (text.length > maxLength) {
    throw new Error(`Text exceeds maximum length of ${maxLength} characters`);
  }
  return text;
}
```

#### CONTEXT_TOO_LONG
```json
{
  "error": true,
  "message": "輸入驗證錯誤: 上下文長度超過限制",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "CONTEXT_TOO_LONG",
    "field": "context",
    "max_length": 2000,
    "actual_length": 2500
  }
}
```

#### INVALID_JSON_FORMAT
```json
{
  "error": true,
  "message": "JSON 格式錯誤",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "INVALID_JSON_FORMAT",
    "description": "請求體不是有效的 JSON 格式"
  }
}
```

**解決方案**:
```bash
# 檢查 JSON 格式
echo '{"text": "測試文本"}' | python -m json.tool

# 正確的 curl 請求
curl -X POST "https://api.cyberpuppy.ai/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer cp_your_api_key" \
  -d '{"text": "這是測試文本"}'
```

### 3. 限流錯誤 (Rate Limiting Errors) - 429

#### RATE_LIMIT_EXCEEDED
```json
{
  "error": true,
  "message": "Rate limit exceeded: 30 requests per minute",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "RATE_LIMIT_EXCEEDED",
    "limit": 30,
    "window": "1 minute",
    "retry_after": 45
  }
}
```

**解決方案**:
```python
# Python 範例 - 實作指數退避重試
import time
import random

def exponential_backoff_retry(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            # 指數退避 + 隨機抖動
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limited, retrying in {delay:.2f} seconds...")
            time.sleep(delay)
```

#### CONCURRENT_LIMIT_EXCEEDED
```json
{
  "error": true,
  "message": "Too many concurrent requests",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "CONCURRENT_LIMIT_EXCEEDED",
    "max_concurrent": 10,
    "current_requests": 12
  }
}
```

### 4. 權限錯誤 (Authorization Errors) - 403

#### INSUFFICIENT_PERMISSIONS
```json
{
  "error": true,
  "message": "Insufficient permissions for this operation",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "INSUFFICIENT_PERMISSIONS",
    "required_plan": "premium",
    "current_plan": "basic"
  }
}
```

#### ACCOUNT_SUSPENDED
```json
{
  "error": true,
  "message": "Account has been suspended",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "ACCOUNT_SUSPENDED",
    "reason": "policy_violation",
    "contact_support": "support@cyberpuppy.ai"
  }
}
```

### 5. 服務錯誤 (Service Errors) - 500

#### MODEL_NOT_LOADED
```json
{
  "error": true,
  "message": "模型未載入，請稍後再試",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "MODEL_NOT_LOADED",
    "description": "AI 模型正在初始化中"
  }
}
```

#### PREDICTION_FAILED
```json
{
  "error": true,
  "message": "模型預測失敗，請稍後再試",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "PREDICTION_FAILED",
    "description": "模型推理過程中發生錯誤"
  }
}
```

#### DATABASE_ERROR
```json
{
  "error": true,
  "message": "Database connection failed",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "DATABASE_ERROR",
    "description": "資料庫連接異常，正在修復中"
  }
}
```

### 6. LINE Bot 特定錯誤 (LINE Bot Specific Errors)

#### INVALID_SIGNATURE
```json
{
  "error": true,
  "message": "Invalid signature",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "code": "INVALID_SIGNATURE",
    "description": "LINE Webhook 簽名驗證失敗"
  }
}
```

**解決方案**:
```python
# Python 範例 - LINE 簽名驗證
import hmac
import hashlib
import base64

def verify_line_signature(body, signature, channel_secret):
    expected = base64.b64encode(
        hmac.new(
            channel_secret.encode('utf-8'),
            body.encode('utf-8'),
            hashlib.sha256
        ).digest()
    ).decode('utf-8')

    return hmac.compare_digest(signature, expected)
```

## 故障排除指南 (Troubleshooting Guide)

### 1. 常見問題診斷

#### 問題: API 回應緩慢
**症狀**: 請求處理時間超過 5 秒
**可能原因**:
- 模型推理負載過重
- 網路延遲
- 服務器資源不足

**診斷步驟**:
```bash
# 1. 檢查 API 健康狀態
curl "https://api.cyberpuppy.ai/healthz"

# 2. 檢查網路延遲
ping api.cyberpuppy.ai

# 3. 檢查本地時鐘同步
date
```

**解決方案**:
- 使用批次 API 處理多個文本
- 實作客戶端超時機制
- 考慮升級到更高服務等級

#### 問題: 間歇性連接失敗
**症狀**: 請求偶爾失敗但大部分時間正常
**可能原因**:
- DNS 解析問題
- SSL/TLS 握手失敗
- 負載平衡器問題

**診斷步驟**:
```bash
# 1. DNS 解析測試
nslookup api.cyberpuppy.ai

# 2. SSL 憑證檢查
openssl s_client -connect api.cyberpuppy.ai:443 -servername api.cyberpuppy.ai

# 3. 追蹤路由
traceroute api.cyberpuppy.ai
```

### 2. 客戶端實作最佳實踐

#### 錯誤處理策略
```python
import logging
from typing import Optional
import httpx

class RobustCyberPuppyClient:
    def __init__(self, api_key: str, max_retries: int = 3):
        self.api_key = api_key
        self.max_retries = max_retries
        self.session = httpx.AsyncClient()
        self.logger = logging.getLogger(__name__)

    async def analyze_with_retry(self, text: str) -> Optional[dict]:
        """帶重試機制的文本分析"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await self.session.post(
                    "https://api.cyberpuppy.ai/analyze",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"text": text},
                    timeout=30.0
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # 限流錯誤，等待後重試
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self.logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                elif response.status_code == 503:
                    # 服務不可用，短暫等待後重試
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    # 其他錯誤，記錄並停止重試
                    error_data = response.json()
                    self.logger.error(f"API error: {error_data}")
                    return None

            except httpx.TimeoutException:
                last_error = "Request timeout"
                self.logger.warning(f"Timeout on attempt {attempt + 1}")
                continue
            except httpx.RequestError as e:
                last_error = str(e)
                self.logger.warning(f"Request error on attempt {attempt + 1}: {e}")
                continue

        self.logger.error(f"All retry attempts failed. Last error: {last_error}")
        return None
```

#### 批次處理實作
```javascript
class BatchProcessor {
  constructor(client, batchSize = 10, concurrency = 3) {
    this.client = client;
    this.batchSize = batchSize;
    this.concurrency = concurrency;
  }

  async processBatch(texts) {
    const results = [];
    const errors = [];

    // 將文本分成批次
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);

      try {
        // 並行處理批次內的文本
        const batchPromises = batch.map(async (text, index) => {
          try {
            const result = await this.client.analyzeText(text);
            return { index: i + index, result };
          } catch (error) {
            return { index: i + index, error: error.message };
          }
        });

        const batchResults = await Promise.all(batchPromises);

        // 分離成功和失敗的結果
        for (const item of batchResults) {
          if (item.error) {
            errors.push(item);
          } else {
            results.push(item);
          }
        }

        // 批次間等待，避免觸發限流
        if (i + this.batchSize < texts.length) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }

      } catch (error) {
        console.error(`Batch processing failed:`, error);
        // 將整個批次標記為錯誤
        for (let j = 0; j < batch.length; j++) {
          errors.push({ index: i + j, error: error.message });
        }
      }
    }

    return { results, errors };
  }
}
```

### 3. 監控與告警設置

#### 健康檢查腳本
```bash
#!/bin/bash
# health_monitor.sh - API 健康監控腳本

API_URL="https://api.cyberpuppy.ai"
API_KEY="cp_your_api_key_here"
WEBHOOK_URL="https://your-webhook-url.com/alert"

# 檢查 API 健康狀態
check_health() {
    response=$(curl -s -w "%{http_code}" -o /tmp/health_response.json \
        -H "Authorization: Bearer $API_KEY" \
        "$API_URL/healthz")

    http_code="${response: -3}"

    if [ "$http_code" = "200" ]; then
        status=$(jq -r '.status' /tmp/health_response.json)
        if [ "$status" = "healthy" ]; then
            echo "✅ API is healthy"
            return 0
        else
            echo "⚠️ API is degraded: $status"
            return 1
        fi
    else
        echo "❌ API is down: HTTP $http_code"
        return 2
    fi
}

# 發送告警
send_alert() {
    local message="$1"
    local severity="$2"

    curl -X POST "$WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"message\": \"$message\",
            \"severity\": \"$severity\",
            \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
            \"service\": \"cyberpuppy-api\"
        }"
}

# 主邏輯
if ! check_health; then
    send_alert "CyberPuppy API health check failed" "critical"
    exit 1
fi
```

#### 效能監控
```python
# performance_monitor.py
import time
import statistics
from datetime import datetime, timedelta

class PerformanceMonitor:
    def __init__(self):
        self.response_times = []
        self.error_counts = {}
        self.success_count = 0
        self.start_time = datetime.now()

    def record_request(self, response_time: float, status_code: int):
        """記錄請求效能數據"""
        self.response_times.append(response_time)

        if status_code == 200:
            self.success_count += 1
        else:
            self.error_counts[status_code] = self.error_counts.get(status_code, 0) + 1

    def get_statistics(self) -> dict:
        """取得統計資料"""
        if not self.response_times:
            return {"message": "No data collected"}

        total_requests = len(self.response_times)
        success_rate = self.success_count / total_requests * 100

        return {
            "total_requests": total_requests,
            "success_count": self.success_count,
            "success_rate": f"{success_rate:.2f}%",
            "error_counts": self.error_counts,
            "response_times": {
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": statistics.quantiles(self.response_times, n=20)[18],
                "p99": statistics.quantiles(self.response_times, n=100)[98],
                "min": min(self.response_times),
                "max": max(self.response_times)
            },
            "uptime": str(datetime.now() - self.start_time)
        }
```

### 4. 緊急情況處理

#### API 完全不可用
**立即行動**:
1. 檢查官方狀態頁面: https://status.cyberpuppy.ai
2. 實作本地備援機制（預先訓練的簡單規則）
3. 通知用戶服務暫時中斷

**臨時解決方案**:
```python
# 簡單的本地備援檢測
def fallback_toxicity_detection(text: str) -> dict:
    """簡單的關鍵字基礎備援檢測"""
    toxic_keywords = ['笨蛋', '白痴', '去死', '滾開', '廢物']
    severe_keywords = ['殺死', '自殺', '威脅']

    text_lower = text.lower()

    has_severe = any(keyword in text_lower for keyword in severe_keywords)
    has_toxic = any(keyword in text_lower for keyword in toxic_keywords)

    if has_severe:
        toxicity = 'severe'
        confidence = 0.6
    elif has_toxic:
        toxicity = 'toxic'
        confidence = 0.7
    else:
        toxicity = 'none'
        confidence = 0.8

    return {
        'toxicity': toxicity,
        'bullying': 'harassment' if has_toxic or has_severe else 'none',
        'role': 'perpetrator' if has_toxic or has_severe else 'none',
        'emotion': 'neg' if has_toxic or has_severe else 'neu',
        'confidence': confidence,
        'fallback_mode': True
    }
```

## 支援聯絡方式 (Support Contacts)

### 技術支援
- **Email**: support@cyberpuppy.ai
- **響應時間**: 工作日 24 小時內
- **緊急問題**: 2 小時內

### 錯誤回報
- **GitHub Issues**: https://github.com/cyberpuppy/api-issues
- **錯誤追蹤**: support@cyberpuppy.ai
- **包含資訊**: 錯誤代碼、時間戳記、請求 ID

### 文檔與學習資源
- **API 文檔**: https://docs.cyberpuppy.ai
- **範例程式碼**: https://github.com/cyberpuppy/examples
- **最佳實踐**: https://docs.cyberpuppy.ai/best-practices

### 狀態監控
- **服務狀態**: https://status.cyberpuppy.ai
- **維護通知**: status-updates@cyberpuppy.ai
- **Twitter**: @CyberPuppyAI

---

**最後更新**: 2024-12-30
**版本**: 1.0.0