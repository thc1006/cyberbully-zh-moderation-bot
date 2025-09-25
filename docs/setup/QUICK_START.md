# CyberPuppy 快速上手指南

## 🚀 5分鐘快速開始

### 1. 環境準備
```bash
# 確認 Python 版本
python --version  # 需要 3.9+

# 安裝依賴
pip install -r requirements.txt
```

### 2. 下載必要資料
```bash
# 下載預訓練模型和數據集
python scripts/download_datasets.py
```

### 3. 啟動 API 服務
```bash
# Windows
cd api && start.bat

# Linux/Mac
cd api && ./start.sh
```

服務將在 http://localhost:8000 啟動

### 4. 測試 API
```bash
# 健康檢查
curl http://localhost:8000/healthz

# 分析文本
curl -X POST http://localhost:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "你這個廢物"}'
```

## 📊 主要功能

### 毒性檢測
- **輸入**: 中文文本（繁體/簡體均可）
- **輸出**: 毒性等級 (none/toxic/severe) + 信心分數
- **可解釋性**: 標示哪些詞彙觸發毒性判定

### 情緒分析
- **分類**: positive/neutral/negative
- **強度**: 0-4 級情緒強度
- **應用**: 識別潛在受害者情緒狀態

### 霸凌角色識別
- **角色**: perpetrator(加害者)/victim(受害者)/bystander(旁觀者)
- **用途**: 協助社群管理者快速定位問題

## 🔧 進階使用

### 自訂模型訓練
```bash
# 使用自己的數據訓練
python train.py \
  --data_path data/processed/unified \
  --model_name hfl/chinese-macbert-base \
  --epochs 5 \
  --batch_size 16
```

### LINE Bot 部署
1. 申請 LINE Developers 帳號
2. 建立 Messaging API channel
3. 設定環境變數：
```bash
export LINE_CHANNEL_SECRET=your_secret
export LINE_CHANNEL_ACCESS_TOKEN=your_token
export API_ENDPOINT=http://localhost:8000
```
4. 啟動 Bot：
```bash
cd bot && python line_bot.py
```

### Docker 部署
```bash
# 建構映像
docker-compose build

# 啟動服務
docker-compose up -d
```

## 📈 效能指標

- **毒性檢測 F1**: 0.82
- **情緒分類準確率**: 0.87
- **平均回應時間**: <200ms
- **並發處理能力**: 100 req/s

## 🛠️ 故障排除

### 常見問題

**Q: 模型載入失敗**
```bash
# 重新下載模型檔案
python scripts/aggressive_download.py
```

**Q: API 無法啟動**
```bash
# 檢查 port 是否被佔用
netstat -an | grep 8000

# 更換 port
uvicorn app:app --port 8001
```

**Q: 中文顯示亂碼**
```bash
# 設定編碼
export PYTHONIOENCODING=utf-8
```

## 📚 更多資源

- [完整 API 文檔](../api/API.md)
- [文件中心](../README.md)
- [部署最佳實踐](../deployment/DEPLOYMENT.md)
- [資料標註規範](../datasets/DATA_CONTRACT.md)

## 💬 取得協助

- GitHub Issues: [報告問題](https://github.com/thc1006/cyberbully-zh-moderation-bot/issues)
- Email: cyberpuppy@example.com