# CyberPuppy Docker 部署指南

## 📋 概述

本文檔提供 CyberPuppy 中文網路霸凌防治系統的 Docker 容器化部署完整指南，包含 FastAPI 後端服務和 LINE Bot 服務的容器化配置。

## 🏗️ 架構概覽

```
CyberPuppy 系統架構
├── API 服務 (Port 8000)
│   ├── FastAPI 應用
│   ├── 毒性偵測模型
│   ├── 情緒分析模型
│   └── 可解釋性分析
├── LINE Bot 服務 (Port 8080)
│   ├── LINE Webhook 處理
│   ├── 訊息分析整合
│   └── 回應策略執行
└── 共享資源
    ├── 模型檔案 (./models)
    ├── 日誌目錄 (./logs)
    └── 配置檔案 (./configs)
```

## 🚀 快速開始

### 1. 環境準備

確保系統已安裝：
- Docker >= 20.10
- Docker Compose >= 2.0
- Git

```bash
# 檢查版本
docker --version
docker-compose --version
```

### 2. 專案設置

```bash
# 克隆專案
git clone <repository-url>
cd cyberbully-zh-moderation-bot

# 準備環境變數
cp configs/docker/.env.example configs/docker/.env
```

### 3. 配置環境變數

編輯 `configs/docker/.env` 檔案，填入必要的配置：

```bash
# LINE Bot 設定 (必填)
LINE_CHANNEL_ACCESS_TOKEN=your_access_token_here
LINE_CHANNEL_SECRET=your_channel_secret_here

# API 設定
API_HOST=0.0.0.0
API_PORT=8000
CYBERPUPPY_API_URL=http://api:8000
```

### 4. 建置並啟動服務

```bash
# 建置映像並啟動服務
docker-compose up --build

# 或在背景執行
docker-compose up --build -d
```

## 📝 詳細配置

### Docker 映像說明

#### API 服務 (Dockerfile.api)
- **基礎映像**: python:3.10-slim
- **埠號**: 8000
- **功能**: FastAPI 後端 API 服務
- **健康檢查**: `GET /healthz`
- **資源限制**: 2GB RAM, 1 CPU

#### Bot 服務 (Dockerfile.bot)
- **基礎映像**: python:3.10-slim
- **埠號**: 8080
- **功能**: LINE Bot Webhook 處理
- **健康檢查**: `GET /health`
- **資源限制**: 1GB RAM, 0.5 CPU

### 環境變數詳細說明

| 變數名稱 | 說明 | 預設值 | 必填 |
|---------|------|--------|------|
| `LINE_CHANNEL_ACCESS_TOKEN` | LINE Bot 存取權杖 | - | ✅ |
| `LINE_CHANNEL_SECRET` | LINE Bot 頻道密鑰 | - | ✅ |
| `API_WORKERS` | API 工作程序數量 | 1 | ❌ |
| `LOG_LEVEL` | 日誌等級 | INFO | ❌ |
| `RATE_LIMIT_ENABLED` | 啟用速率限制 | true | ❌ |
| `DEVELOPMENT_MODE` | 開發模式 | true | ❌ |

### 持久化儲存

```yaml
volumes:
  - ./models:/app/models:ro     # 模型檔案 (唯讀)
  - ./logs:/app/logs            # 日誌檔案
  - ./data:/app/data:ro         # 訓練資料 (唯讀)
```

## 🔧 運維操作

### 基本操作

```bash
# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f
docker-compose logs -f api    # 僅 API 服務
docker-compose logs -f bot    # 僅 Bot 服務

# 重啟服務
docker-compose restart

# 停止服務
docker-compose down

# 完全清理 (包含 volumes)
docker-compose down -v
```

### 健康檢查

```bash
# 檢查 API 服務健康狀態
curl http://localhost:8000/healthz

# 檢查 Bot 服務健康狀態
curl http://localhost:8080/health

# 檢查 API 效能指標
curl http://localhost:8000/metrics
```

### 容器內部操作

```bash
# 進入 API 容器
docker-compose exec api bash

# 進入 Bot 容器
docker-compose exec bot bash

# 查看容器資源使用
docker stats cyberpuppy-api cyberpuppy-bot
```

## 🔍 故障排除

### 常見問題

#### 1. 容器啟動失敗

```bash
# 檢查日誌
docker-compose logs

# 檢查映像建置過程
docker-compose build --no-cache
```

#### 2. 模型載入失敗

```bash
# 確認模型檔案存在
ls -la models/

# 檢查檔案權限
docker-compose exec api ls -la /app/models/
```

#### 3. LINE Bot 連線問題

```bash
# 檢查環境變數
docker-compose exec bot env | grep LINE

# 測試 API 連線
docker-compose exec bot curl http://api:8000/healthz
```

#### 4. 記憶體不足

```bash
# 調整 docker-compose.yml 中的資源限制
deploy:
  resources:
    limits:
      memory: 4G  # 增加記憶體限制
```

### 效能調優

#### 1. 生產環境設定

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  api:
    environment:
      - WORKERS=4
      - LOG_LEVEL=WARNING
      - DEVELOPMENT_MODE=false
    deploy:
      replicas: 2
```

#### 2. 資源監控

```bash
# 安裝監控工具
docker run -d --name cadvisor \
  -p 8081:8080 \
  -v /:/rootfs:ro \
  -v /var/run:/var/run:ro \
  -v /sys:/sys:ro \
  -v /var/lib/docker/:/var/lib/docker:ro \
  gcr.io/cadvisor/cadvisor:latest
```

## 🔐 安全最佳實踐

### 1. 環境變數管理

```bash
# 使用 Docker Secrets (Swarm 模式)
docker secret create line_token /path/to/token
docker secret create line_secret /path/to/secret
```

### 2. 網路安全

```yaml
# 限制網路存取
networks:
  cyberpuppy-network:
    driver: bridge
    internal: true  # 內部網路
```

### 3. 容器安全

- 使用非 root 使用者執行
- 定期更新基礎映像
- 掃描安全漏洞

```bash
# 掃描映像漏洞
docker scout cves cyberpuppy-api:latest
```

## 🚀 生產環境部署

### 1. 反向代理設定 (Nginx)

```nginx
# /etc/nginx/sites-available/cyberpuppy
server {
    listen 80;
    server_name your-domain.com;
    
    location /webhook {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2. SSL 憑證設定

```bash
# 使用 Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

### 3. 監控和日誌

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## 📊 效能測試

### 負載測試

```bash
# 使用 Apache Bench
ab -n 1000 -c 10 http://localhost:8000/healthz

# 使用 curl 測試 API
for i in {1..100}; do
  curl -X POST http://localhost:8000/analyze \
    -H "Content-Type: application/json" \
    -d '{"text":"測試訊息"}'
done
```

## 📚 相關資源

- [Docker 官方文檔](https://docs.docker.com/)
- [Docker Compose 參考](https://docs.docker.com/compose/)
- [LINE Messaging API](https://developers.line.biz/en/docs/messaging-api/)
- [FastAPI 部署指南](https://fastapi.tiangolo.com/deployment/)

## 🆘 支援

如遇到問題，請：
1. 查看此文檔的故障排除章節
2. 檢查 GitHub Issues
3. 聯絡專案維護團隊

---

**注意**: 在生產環境中，請確保：
- 使用 HTTPS
- 設定適當的防火牆規則
- 定期備份重要資料
- 監控系統資源使用