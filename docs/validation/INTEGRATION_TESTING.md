# CyberPuppy 整合測試指南

本文檔詳細說明 CyberPuppy 中文網路霸凌防治系統的整合測試架構、執行方式與最佳實務。

## 📋 目錄

1. [測試架構概覽](#測試架構概覽)
2. [測試分類](#測試分類)
3. [執行環境設定](#執行環境設定)
4. [測試執行指南](#測試執行指南)
5. [CI/CD 整合](#ci/cd-整合)
6. [效能要求](#效能要求)
7. [故障排除](#故障排除)
8. [最佳實務](#最佳實務)

## 🏗️ 測試架構概覽

整合測試架構採用多層次驗證策略：

```
┌─────────────────────────────────────────┐
│              E2E 測試層                  │
├─────────────────────────────────────────┤
│        服務整合測試層                    │
├─────────────────────────────────────────┤
│        API/Bot 整合測試層               │
├─────────────────────────────────────────┤
│        資料管道整合測試層               │
├─────────────────────────────────────────┤
│        效能基準測試層                    │
└─────────────────────────────────────────┘
```

### 測試目錄結構

```
tests/integration/
├── __init__.py                    # 整合測試套件初始化
├── conftest.py                    # 共用測試配置
├── api/                          # API 整合測試
│   └── test_api_endpoints.py     # API 端點完整測試
├── bot/                          # Bot 整合測試
│   └── test_webhook_processing.py # Webhook 處理測試
├── cli/                          # CLI 整合測試
│   └── test_cli_commands.py       # 命令列介面測試
├── pipeline/                     # 資料管道測試
│   └── test_data_pipeline.py      # 完整資料流程測試
├── performance/                  # 效能測試
│   └── test_performance_benchmarks.py
├── docker/                       # Docker 容器測試
│   ├── docker-compose.test.yml   # 測試環境配置
│   ├── Dockerfile.test           # 測試容器映像
│   └── test_docker_integration.py
├── fixtures/                     # 測試資料
│   └── chinese_toxicity_examples.py
└── .github/workflows/            # CI/CD 管道
    └── integration-tests.yml
```

## 🎯 測試分類

### 1. API 整合測試 (`@pytest.mark.api`)

**測試範圍：**
- 健康檢查端點 (`/healthz`)
- 文本分析端點 (`/analyze`)
- 錯誤處理機制
- 速率限制執行
- 回應格式驗證
- 隱私保護日誌

**關鍵測試案例：**
```python
# 基本分析功能
async def test_analyze_valid_text(api_server, http_client):
    payload = {"text": "你這個笨蛋"}
    response = await http_client.post(f"{api_server}/analyze", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["toxicity"] in ["none", "toxic", "severe"]
    assert data["processing_time_ms"] < 2000  # < 2秒要求
```

### 2. Bot 整合測試 (`@pytest.mark.bot`)

**測試範圍：**
- LINE Webhook 簽名驗證
- 訊息處理與回應策略
- 使用者會話管理
- 升級處理機制
- 多媒體訊息處理

**關鍵測試案例：**
```python
# Webhook 處理流程
async def test_toxic_message_processing(bot_server, http_client):
    payload = create_line_webhook_payload("你這個白痴")
    response = await http_client.post(f"{bot_server}/webhook", ...)

    # 驗證適當回應策略
    assert mock_reply.called  # 確認有回應
```

### 3. CLI 整合測試 (`@pytest.mark.cli`)

**測試範圍：**
- 單一文本檢測命令
- 批次檔案處理
- 不同輸出格式 (JSON, CSV)
- 錯誤處理
- 效能監控

**關鍵測試案例：**
```bash
# CLI 批次處理
python cli.py batch --input test.txt --output results.jsonl --format jsonl
```

### 4. 資料管道整合測試 (`@pytest.mark.pipeline`)

**測試範圍：**
- 下載 → 清理 → 正規化 → 訓練流程
- 標籤統一與映射
- 模型訓練與評估
- 資料一致性驗證

### 5. 效能整合測試 (`@pytest.mark.performance`)

**測試範圍：**
- API 回應時間 (< 2秒)
- 併發請求處理
- 記憶體使用監控
- 吞吐量測試
- 資源優化驗證

## ⚙️ 執行環境設定

### 本地環境設定

1. **安裝測試依賴**
```bash
pip install -r requirements-dev.txt
```

2. **設定環境變數**
```bash
export TESTING=1
export LOG_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=""  # 禁用 GPU 於測試
```

3. **啟動服務依賴**
```bash
# Redis (可選)
docker run -d -p 6379:6379 redis:7-alpine

# PostgreSQL (可選)
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=test_password_123 \
  postgres:15-alpine
```

### Docker 環境設定

```bash
# 進入測試目錄
cd tests/integration/docker

# 啟動完整測試環境
docker-compose -f docker-compose.test.yml up -d

# 執行整合測試
docker-compose -f docker-compose.test.yml run --rm integration-tests
```

## 🚀 測試執行指南

### 快速執行

```bash
# 執行所有整合測試（排除慢速測試）
pytest tests/integration/ -v -m "not slow"

# 執行特定類別測試
pytest tests/integration/ -v -m "api"
pytest tests/integration/ -v -m "bot"
pytest tests/integration/ -v -m "performance"
```

### 詳細執行

```bash
# 完整整合測試套件（包含慢速測試）
pytest tests/integration/ -v --tb=short \
  --cov=src --cov=api --cov=bot \
  --cov-report=html:reports/coverage \
  --junit-xml=reports/integration-results.xml

# 效能基準測試
pytest tests/integration/performance/ -v \
  --benchmark-json=reports/benchmark.json
```

### 並行執行

```bash
# 使用 pytest-xdist 並行執行
pytest tests/integration/ -v -n auto -m "not slow"
```

## 📊 CI/CD 整合

### GitHub Actions 工作流程

整合測試分為四個階段：

1. **基本整合測試** - 核心功能驗證
2. **效能整合測試** - 效能要求驗證
3. **Docker 整合測試** - 容器化服務測試
4. **完整端到端測試** - 僅在主分支執行

### 觸發條件

- **Push** 到 `main`, `develop` 分支
- **Pull Request** 到 `main`, `develop` 分支
- **定時執行** - 每日凌晨 2 點（UTC）

### 測試階段

```yaml
jobs:
  basic-integration:      # 基本功能測試
  performance-integration: # 效能基準測試
  docker-integration:     # 容器整合測試
  pipeline-integration:   # 資料管道測試
  end-to-end:            # 完整 E2E 測試（僅 main 分支）
```

## ⚡ 效能要求

### API 效能標準

| 指標 | 要求 | 測試方法 |
|------|------|----------|
| 單一請求回應時間 | < 2 秒 | `test_single_request_response_time` |
| 95% 回應時間 | < 3 秒 | 統計分析 |
| 併發處理能力 | ≥ 10 RPS | `test_concurrent_requests_performance` |
| 記憶體使用增長 | < 200 MB | `test_api_memory_stability` |
| 成功率 | ≥ 95% | 各種負載測試 |

### Bot 效能標準

| 指標 | 要求 | 測試方法 |
|------|------|----------|
| Webhook 處理時間 | < 1 秒 | `test_webhook_processing_time` |
| 使用者會話管理 | < 100 MB | 記憶體監控 |
| 升級處理準確性 | 100% | 邏輯驗證 |

## 🔧 故障排除

### 常見問題

**1. API 服務啟動失敗**
```bash
# 檢查埠口占用
lsof -i :8000
kill -9 <PID>

# 檢查模型檔案
ls -la models/
```

**2. Bot Webhook 簽名驗證失敗**
```bash
# 檢查環境變數
echo $LINE_CHANNEL_SECRET
echo $LINE_CHANNEL_ACCESS_TOKEN
```

**3. Docker 測試失敗**
```bash
# 檢查容器狀態
docker-compose -f docker-compose.test.yml ps

# 查看容器日誌
docker-compose -f docker-compose.test.yml logs cyberpuppy-api-test
```

**4. 記憶體不足錯誤**
```bash
# 監控記憶體使用
free -h
# 清理 Docker 資源
docker system prune -f
```

### 除錯技巧

**1. 增加日誌詳細度**
```bash
export LOG_LEVEL=DEBUG
pytest tests/integration/ -v -s
```

**2. 單一測試執行**
```bash
pytest tests/integration/api/test_api_endpoints.py::TestAPIAnalyze::test_analyze_valid_text -v -s
```

**3. 跳過慢速測試**
```bash
pytest tests/integration/ -v -m "not slow and not performance"
```

## 📝 最佳實務

### 1. 測試資料管理

- 使用 `fixtures/chinese_toxicity_examples.py` 標準測試資料
- 測試資料涵蓋各種毒性等級與邊界案例
- 避免在測試中使用真實敏感資料

### 2. 並行測試安全

- 使用獨立的資料庫/Redis 實例
- 避免全域狀態共享
- 適當的測試隔離與清理

### 3. 效能測試穩定性

- 使用統計方法處理效能變異
- 設定合理的超時與重試機制
- 監控系統資源使用

### 4. 錯誤處理驗證

- 測試各種異常情境
- 驗證適當的錯誤回應
- 確保系統能夠優雅降級

### 5. 文檔維護

- 及時更新測試文檔
- 記錄重要的測試變更
- 維護測試案例的可讀性

## 🚨 注意事項

### 安全考量

- 測試環境使用虛擬金鑰與權杖
- 不要在測試中暴露實際 API 金鑰
- 確保測試資料不含敏感資訊

### 資源管理

- 定期清理測試生成的檔案
- 監控測試環境的磁碟空間
- 適當的容器資源限制

### 測試維護

- 定期檢視失敗的測試案例
- 更新過時的測試依賴
- 保持測試案例與功能同步

---

## 📞 支援與聯絡

如需協助或回報問題：

1. 檢查 [故障排除](#故障排除) 章節
2. 查看 GitHub Issues
3. 聯絡開發團隊

**最後更新：** 2024-09-24
**版本：** 1.0.0
**維護者：** CyberPuppy 開發團隊