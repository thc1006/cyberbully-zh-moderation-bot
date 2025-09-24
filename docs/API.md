# CyberPuppy API Documentation
# 中文網路霸凌防治 API 文件

## 概述 (Overview)

CyberPuppy API 是一個專為中文文本設計的網路霸凌防治與毒性偵測服務。本 API 提供：

- **多任務分析**: 毒性偵測、霸凌行為識別、角色分析、情緒分類
- **高可解釋性**: 提供詞彙重要性分析，支援 Integrated Gradients (IG) 和 SHAP
- **隱私保護**: 不儲存原始文本，僅記錄雜湊值與分析結果
- **即時處理**: 低延遲分析，支援對話上下文
- **企業級**: 內建限流、錯誤處理、監控指標

## 基本資訊 (Basic Information)

- **Base URL**: `https://api.cyberpuppy.ai` (Production) / `http://localhost:8000` (Development)
- **版本**: v1.0.0
- **協議**: HTTPS (Production), HTTP (Development)
- **格式**: JSON
- **編碼**: UTF-8

## 認證與安全 (Authentication & Security)

### API 密鑰 (API Keys)
```http
Authorization: Bearer YOUR_API_KEY
```

### 限流政策 (Rate Limiting)
- **一般用戶**: 30 requests/minute
- **付費用戶**: 1000 requests/minute
- **企業用戶**: 無限制

### 安全特性 (Security Features)
- HTTPS 強制加密
- API 密鑰認證
- 請求簽名驗證 (Webhook)
- CORS 政策
- 輸入驗證與清理
- PII (個人識別資訊) 遮蔽

## API 端點 (Endpoints)

### 1. 文本分析 (Text Analysis)

#### POST /analyze
分析輸入文本的毒性、霸凌行為、角色與情緒

**請求格式 (Request)**:
```http
POST /analyze
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "text": "你這個笨蛋，滾開！",
  "context": "之前的對話內容...",
  "thread_id": "conversation_123"
}
```

**請求參數 (Parameters)**:
- `text` (string, required): 待分析文本 (1-1000 字符)
- `context` (string, optional): 對話上下文 (最多 2000 字符)
- `thread_id` (string, optional): 對話串 ID (最多 50 字符)

**回應格式 (Response)**:
```json
{
  "toxicity": "toxic",
  "bullying": "harassment",
  "role": "perpetrator",
  "emotion": "neg",
  "emotion_strength": 4,
  "scores": {
    "toxicity": {"none": 0.2, "toxic": 0.7, "severe": 0.1},
    "bullying": {"none": 0.2, "harassment": 0.7, "threat": 0.1},
    "role": {"none": 0.1, "perpetrator": 0.7, "victim": 0.1, "bystander": 0.1},
    "emotion": {"positive": 0.1, "neutral": 0.2, "negative": 0.7}
  },
  "explanations": {
    "important_words": [
      {"word": "笨蛋", "importance": 0.85},
      {"word": "滾開", "importance": 0.92}
    ],
    "method": "IG",
    "confidence": 0.89
  },
  "text_hash": "a1b2c3d4e5f6g7h8",
  "timestamp": "2024-12-30T10:15:30Z",
  "processing_time_ms": 145.7
}
```

**分類標籤說明**:
- **toxicity**: `none` (無毒性) | `toxic` (一般毒性) | `severe` (嚴重毒性)
- **bullying**: `none` (無霸凌) | `harassment` (騷擾) | `threat` (威脅)
- **role**: `none` (無特定角色) | `perpetrator` (施暴者) | `victim` (受害者) | `bystander` (旁觀者)
- **emotion**: `pos` (正面) | `neu` (中性) | `neg` (負面)
- **emotion_strength**: 0-4 (情緒強度等級)

### 2. 健康檢查 (Health Check)

#### GET /healthz
檢查 API 服務狀態

**請求格式**:
```http
GET /healthz
```

**回應格式**:
```json
{
  "status": "healthy",
  "timestamp": "2024-12-30T10:15:30Z",
  "version": "1.0.0",
  "uptime_seconds": 86400.5
}
```

### 3. 系統資訊 (System Information)

#### GET /
取得 API 基本資訊

**回應格式**:
```json
{
  "name": "CyberPuppy Moderation API",
  "description": "中文網路霸凌防治與毒性偵測 API",
  "version": "1.0.0",
  "docs_url": "/docs",
  "health_check": "/healthz"
}
```

## LINE Bot Webhook 端點

### POST /webhook
接收 LINE 平台的 Webhook 事件

**請求標頭**:
```http
X-Line-Signature: {signature}
```

**請求格式**:
```json
{
  "events": [
    {
      "type": "message",
      "message": {"type": "text", "text": "用戶訊息"},
      "source": {"userId": "U1234567890"},
      "replyToken": "reply_token"
    }
  ]
}
```

## 錯誤處理 (Error Handling)

### HTTP 狀態碼
- `200` OK - 請求成功
- `400` Bad Request - 請求格式錯誤
- `401` Unauthorized - 認證失敗
- `429` Too Many Requests - 超出限流
- `500` Internal Server Error - 服務器錯誤
- `502` Bad Gateway - 上游服務錯誤
- `503` Service Unavailable - 服務暫時不可用

### 錯誤回應格式
```json
{
  "error": true,
  "message": "錯誤描述",
  "timestamp": "2024-12-30T10:15:30Z",
  "details": {
    "field": "text",
    "code": "INVALID_LENGTH"
  }
}
```

### 常見錯誤碼
- `INVALID_LENGTH` - 文本長度超出限制
- `EMPTY_TEXT` - 文本為空
- `INVALID_FORMAT` - 請求格式錯誤
- `RATE_LIMIT_EXCEEDED` - 超出請求限制
- `API_KEY_INVALID` - API 密鑰無效
- `SERVICE_UNAVAILABLE` - 分析服務不可用

## 使用範例 (Usage Examples)

### cURL 範例
```bash
# 基本文本分析
curl -X POST "https://api.cyberpuppy.ai/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "你好，今天天氣真好！"
  }'

# 帶上下文的分析
curl -X POST "https://api.cyberpuppy.ai/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "我不同意你的看法",
    "context": "剛才討論的是關於教育政策的議題",
    "thread_id": "edu_discussion_001"
  }'

# 健康檢查
curl "https://api.cyberpuppy.ai/healthz"
```

### Python 範例
```python
import requests
import json

# 設定 API 配置
API_BASE_URL = "https://api.cyberpuppy.ai"
API_KEY = "your_api_key_here"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# 分析文本
def analyze_text(text, context=None, thread_id=None):
    payload = {"text": text}
    if context:
        payload["context"] = context
    if thread_id:
        payload["thread_id"] = thread_id

    response = requests.post(
        f"{API_BASE_URL}/analyze",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(f"錯誤 {response.status_code}: {response.text}")
        return None

# 使用範例
result = analyze_text("你這個笨蛋！")
if result:
    print(f"毒性等級: {result['toxicity']}")
    print(f"霸凌類型: {result['bullying']}")
    print(f"情緒: {result['emotion']}")
```

### JavaScript 範例
```javascript
// 使用 fetch API
const API_BASE_URL = 'https://api.cyberpuppy.ai';
const API_KEY = 'your_api_key_here';

async function analyzeText(text, context = null, threadId = null) {
  const payload = { text };
  if (context) payload.context = context;
  if (threadId) payload.thread_id = threadId;

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_KEY}`
      },
      body: JSON.stringify(payload)
    });

    if (response.ok) {
      const result = await response.json();
      return result;
    } else {
      console.error(`錯誤 ${response.status}: ${await response.text()}`);
      return null;
    }
  } catch (error) {
    console.error('請求失敗:', error);
    return null;
  }
}

// 使用範例
analyzeText('你今天心情如何？')
  .then(result => {
    if (result) {
      console.log('毒性等級:', result.toxicity);
      console.log('情緒:', result.emotion);
      console.log('重要詞彙:', result.explanations.important_words);
    }
  });
```

## 最佳實踐 (Best Practices)

### 1. 錯誤處理
- 總是檢查 HTTP 狀態碼
- 實施指數退避重試機制
- 記錄錯誤詳情用於除錯

### 2. 性能優化
- 批量處理多個文本
- 快取常見查詢結果
- 使用適當的超時設定

### 3. 安全考量
- 妥善保管 API 密鑰
- 在客戶端驗證輸入
- 記錄但不暴露敏感資訊

### 4. 監控與日誌
- 監控 API 響應時間
- 追蹤錯誤率與模式
- 定期檢查限流狀況

## 整合指南 (Integration Guide)

### Web 應用整合
1. 在前端表單添加即時檢查
2. 在後端 API 路由中調用分析
3. 根據分析結果執行相應動作

### 聊天機器人整合
1. 設定 Webhook 接收訊息
2. 對每條訊息進行即時分析
3. 根據風險等級發送適當回應

### 內容管理系統整合
1. 在內容發布前進行檢查
2. 自動標記可疑內容
3. 提供人工審核介面

## 支援與聯繫 (Support & Contact)

- **技術文檔**: https://docs.cyberpuppy.ai
- **API 狀態**: https://status.cyberpuppy.ai
- **技術支援**: support@cyberpuppy.ai
- **問題回報**: https://github.com/cyberpuppy/issues

## 版本歷史 (Version History)

### v1.0.0 (2024-12-30)
- 初始版本發布
- 支援中文毒性與霸凌偵測
- 多任務學習模型
- Integrated Gradients 可解釋性
- LINE Bot 整合支援

---

*本文件持續更新，最新版本請參考線上文檔*