# CyberPuppy 中文網路霸凌防治系統

## 專案概述

CyberPuppy 是一個專為中文環境設計的網路霸凌防治與毒性偵測系統，結合深度學習模型與可解釋性技術，提供即時、準確的文本分析服務。

### 主要功能

- **多任務分析**：毒性檢測、霸凌行為識別、情緒分析、角色分類
- **高可解釋性**：基於 Integrated Gradients (IG) 和 SHAP 的解釋性輸出
- **即時 API 服務**：FastAPI 構建的高效能 REST API
- **LINE Bot 整合**：完整的 LINE Messaging API 整合與 Webhook 驗證
- **隱私優先**：僅記錄雜湊摘要，不儲存原始文本內容

### 技術架構

- **模型基礎**：HuggingFace Transformers (`hfl/chinese-macbert-base`, `hfl/chinese-roberta-wwm-ext`)
- **文字處理**：OpenCC 繁簡轉換、CKIP 中文斷詞
- **可解釋性**：Captum (IG)、SHAP
- **API 框架**：FastAPI + Uvicorn
- **容器化**：Docker + Docker Compose

## 快速開始

### 1. 環境需求

- Python 3.11+
- Node.js 16+ (用於部分工具)
- Docker (可選，用於容器化部署)

### 2. 安裝依賴

```bash
# 克隆專案
git clone https://github.com/your-org/cyberpuppy-zh-moderation-bot.git
cd cyberpuppy-zh-moderation-bot

# 安裝 Python 依賴
pip install -r requirements.txt
```

### 📦 大文件下載 (必要步驟)

由於 GitHub 大小限制，部分模型檢查點和數據集文件 (>100MB) 未包含在倉庫中。首次設置時請執行：

```bash
# 自動下載所有必需的大文件
python scripts/download_datasets.py

# 或使用更全面的下載腳本
python scripts/aggressive_download.py

# 檢查所有文件是否就位
python scripts/check_datasets.py
```

**需要下載的文件包括：**
- `models/macbert_base_demo/best.ckpt` (397MB)
- `models/toxicity_only_demo/best.ckpt` (397MB)
- `data/raw/dmsc/DMSC.csv` (387MB)
- `data/raw/dmsc/dmsc_kaggle.zip` (144MB)

> 詳細說明請參閱 [`docs/LARGE_FILES_SETUP.md`](docs/LARGE_FILES_SETUP.md)

### 3. 啟動 API 服務

```bash
# 啟動分析 API (http://localhost:8000)
cd api
./start.sh  # Linux/macOS
# 或
start.bat   # Windows
```

### 4. 啟動 LINE Bot (可選)

```bash
# 設定 LINE Bot 環境變數
cp bot/.env.example bot/.env
# 編輯 .env 檔案，填入您的 LINE Bot 設定

# 啟動 LINE Bot (http://localhost:8080)
cd bot
./start.sh  # Linux/macOS
# 或
start.bat   # Windows
```

## API 使用

### 分析文本

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你這個白痴，滾開！",
    "context": "前面的對話內容（可選）",
    "thread_id": "conversation_123"
  }'
```

### 回應範例

```json
{
  "toxicity": "toxic",
  "bullying": "harassment",
  "role": "perpetrator",
  "emotion": "neg",
  "emotion_strength": 4,
  "scores": {
    "toxicity": {"none": 0.1, "toxic": 0.7, "severe": 0.2},
    "bullying": {"none": 0.15, "harassment": 0.75, "threat": 0.1},
    "role": {"none": 0.05, "perpetrator": 0.8, "victim": 0.1, "bystander": 0.05},
    "emotion": {"positive": 0.05, "neutral": 0.15, "negative": 0.8}
  },
  "explanations": {
    "important_words": [
      {"word": "白痴", "importance": 0.85},
      {"word": "滾開", "importance": 0.72}
    ],
    "method": "IG",
    "confidence": 0.89
  },
  "text_hash": "a1b2c3d4e5f6",
  "timestamp": "2024-01-15T10:30:00Z",
  "processing_time_ms": 145.2
}
```

## 外部 API 整合

### Perspective API (可選)

系統支援 Google Perspective API 作為外部驗證服務，僅在本地模型不確定時（信心度 0.4-0.6）使用。

#### 申請與設定

1. **申請 API Key**：
   - 訪問 [Perspective API 文檔](https://developers.perspectiveapi.com/)
   - 在 Google Cloud Console 中啟用 Perspective API
   - 創建 API Key

2. **環境變數設定**：
   ```bash
   # 在 .env 檔案中添加
   PERSPECTIVE_API_KEY=your_google_api_key_here

   # 可選的進階設定
   PERSPECTIVE_RATE_LIMIT_RPS=1          # 每秒請求數限制
   PERSPECTIVE_RATE_LIMIT_DAY=1000       # 每日請求數限制
   PERSPECTIVE_TIMEOUT=30.0              # 請求超時時間
   PERSPECTIVE_MAX_RETRIES=3             # 最大重試次數

   # 不確定性檢測設定
   UNCERTAINTY_THRESHOLD=0.4             # 不確定性下閾值
   CONFIDENCE_THRESHOLD=0.6              # 信心度上閾值
   MIN_CONFIDENCE_GAP=0.1                # 最小信心度差距
   ```

3. **使用說明**：
   - Perspective API 主要針對英文訓練，中文支援有限
   - 僅作為本地模型的參考驗證，不直接影響最終決策
   - 自動處理速率限制與重試機制
   - 結果僅在模型不確定時提供額外資訊

#### 整合範例

```python
from src.cyberpuppy.arbiter import validate_with_arbiter

# 模擬本地模型預測
local_prediction = {
    'toxicity': 'none',
    'scores': {'toxicity': {'none': 0.5, 'toxic': 0.4, 'severe': 0.1}}
}

# 使用仲裁服務驗證（如果需要且可用）
enhanced_prediction, metadata = await validate_with_arbiter(
    text="待分析文本",
    local_prediction=local_prediction
)

if metadata['used_external_validation']:
    print(f"Perspective 毒性分數: {metadata['perspective_result']['toxicity_score']}")
```

## LINE Bot 整合

### Webhook 設定

1. 在 [LINE Developers Console](https://developers.line.biz/) 設定 Webhook URL：
   ```
   https://your-domain.com/webhook
   ```

2. 確保正確設定環境變數：
   ```bash
   LINE_CHANNEL_ACCESS_TOKEN=your_access_token
   LINE_CHANNEL_SECRET=your_channel_secret
   ```

### 功能特色

- **嚴格簽名驗證**：HMAC-SHA256 驗證確保請求來源
- **智能回應策略**：根據毒性等級提供不同程度的提醒
- **隱私保護**：不記錄原始訊息內容，僅保存分析結果
- **錯誤恢復**：完整的重試機制與降級策略

## 開發指南

### 專案結構

```
├── api/                    # FastAPI 服務
│   ├── app.py             # 主要 API 應用
│   ├── requirements.txt   # API 依賴
│   └── Dockerfile         # API 容器配置
├── bot/                   # LINE Bot 服務
│   ├── line_bot.py       # LINE Bot 應用
│   ├── config.py         # 配置管理
│   └── requirements.txt  # Bot 依賴
├── src/cyberpuppy/       # 核心模組
│   ├── arbiter/          # 外部 API 整合
│   ├── config.py         # 全域配置
│   ├── models/           # 模型相關
│   ├── explain/          # 可解釋性模組
│   └── safety/           # 安全規則
├── data/                 # 資料集
├── tests/                # 測試檔案
├── scripts/              # 工具腳本
└── docs/                 # 文檔
```

### 執行測試

```bash
# 執行所有測試
pytest

# 執行特定模組測試
pytest tests/test_perspective.py -v

# 執行整合測試（需要 API Key）
pytest tests/test_perspective.py::TestPerspectiveIntegration -v
```

### Docker 部署

```bash
# 構建並啟動所有服務
docker-compose up --build

# 僅啟動 API 服務
docker-compose up cyberpuppy-api

# 生產環境部署（含 Nginx）
docker-compose --profile production up
```

## 配置說明

### API 配置

| 環境變數 | 預設值 | 說明 |
|---------|-------|------|
| `PORT` | 8000 | API 服務埠號 |
| `LOG_LEVEL` | info | 日誌等級 |
| `MAX_TEXT_LENGTH` | 1000 | 最大文本長度 |

### LINE Bot 配置

| 環境變數 | 必填 | 說明 |
|---------|------|------|
| `LINE_CHANNEL_ACCESS_TOKEN` | ✅ | LINE Bot 存取權杖 |
| `LINE_CHANNEL_SECRET` | ✅ | LINE Bot 頻道密鑰 |
| `CYBERPUPPY_API_URL` | | CyberPuppy API 服務網址 |

### 外部 API 配置

| 環境變數 | 必填 | 說明 |
|---------|------|------|
| `PERSPECTIVE_API_KEY` | | Google Perspective API Key |
| `PERSPECTIVE_RATE_LIMIT_RPS` | | 每秒請求數限制 |
| `PERSPECTIVE_RATE_LIMIT_DAY` | | 每日請求數限制 |

## 安全考量

- **隱私保護**：僅記錄文本雜湊值，不儲存原始內容
- **速率限制**：API 與 Bot 均實施速率限制防止濫用
- **簽名驗證**：LINE Bot 嚴格驗證 X-Line-Signature
- **輸入驗證**：完整的輸入清理與長度限制
- **錯誤處理**：詳細的錯誤記錄但不洩露敏感資訊

## 效能指標

- **API 回應時間**：< 200ms (平均)
- **準確度目標**：毒性檢測 F1 ≥ 0.78，情緒分析 F1 ≥ 0.85
- **可用性目標**：99.5% 正常運行時間
- **併發支援**：支援 100+ 併發請求

## 授權與貢獻

本專案採用 MIT 授權條款。歡迎提交問題回報和功能請求。

### 貢獻指南

1. Fork 本專案
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

## 支援

- **文檔**：[專案文檔](./docs/)
- **問題回報**：[GitHub Issues](https://github.com/your-org/cyberpuppy-zh-moderation-bot/issues)
- **討論**：[GitHub Discussions](https://github.com/your-org/cyberpuppy-zh-moderation-bot/discussions)

## 更新日誌

### v1.0.0
- 初始版本發布
- 基礎 API 服務
- LINE Bot 整合
- Perspective API 可選整合
- Docker 容器化支援