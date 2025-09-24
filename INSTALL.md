# CyberPuppy 安裝指南

## 📋 系統需求

- Python 3.11+
- Git
- 約 2GB 磁碟空間 (包含模型和數據文件)

## 🚀 快速安裝

### 1. 克隆倉庫

```bash
git clone https://github.com/thc1006/cyberbully-zh-moderation-bot.git
cd cyberbully-zh-moderation-bot
```

### 2. 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

### 3. ⚠️ 下載必需的大文件

**這是關鍵步驟！** 由於 GitHub 對大文件的限制，以下文件未包含在倉庫中但對系統運行必不可少：

```bash
# 自動下載所有必需的大文件 (推薦)
python scripts/download_datasets.py

# 或使用更全面的下載腳本
python scripts/aggressive_download.py

# 驗證所有文件是否下載成功
python scripts/check_datasets.py
```

**需要下載的大文件 (>100MB)：**

| 文件路徑 | 大小 | 用途 |
|---------|------|------|
| `models/macbert_base_demo/best.ckpt` | 397MB | 多任務模型檢查點 |
| `models/toxicity_only_demo/best.ckpt` | 397MB | 毒性檢測模型檢查點 |
| `data/raw/dmsc/DMSC.csv` | 387MB | 豆瓣電影短評數據集 |
| `data/raw/dmsc/dmsc_kaggle.zip` | 144MB | DMSC 壓縮數據集 |

### 4. 驗證安裝

```bash
# 測試核心模組載入
python -c "from cyberpuppy import config, models, labeling; print('✅ 安裝成功')"

# 啟動 API 測試
cd api
python app.py
# 瀏覽器打開 http://localhost:8000/docs 查看 API 文檔
```

## 🔧 開發環境設置

如果您需要完整的開發環境：

```bash
# 安裝開發依賴
pip install -r requirements-dev.txt

# 安裝額外的可選依賴
pip install -e ".[dev,ckip,perspective]"

# 運行測試
pytest

# 代碼格式化
black src/ tests/
ruff check src/ tests/
```

## 📁 目錄結構

安裝完成後，您的目錄應該包含：

```
cyberpuppy-zh-moderation-bot/
├── src/cyberpuppy/          # 核心套件
├── api/                     # API 服務
├── bot/                     # LINE Bot
├── models/                  # 模型檢查點 (下載後)
├── data/raw/                # 原始數據集 (下載後)
├── scripts/                 # 工具腳本
├── tests/                   # 測試文件
└── docs/                    # 文檔
```

## 🐛 故障排除

### 大文件下載失敗

如果自動下載腳本失敗：

1. **檢查網路連接**
2. **手動下載** - 參閱 [`docs/LARGE_FILES_SETUP.md`](docs/LARGE_FILES_SETUP.md)
3. **重新訓練模型**：
   ```bash
   python train.py --config configs/multitask_config.yaml
   ```

### 依賴安裝問題

```bash
# 升級 pip
pip install --upgrade pip

# 清除 pip 快取
pip cache purge

# 重新安裝
pip install -r requirements.txt --force-reinstall
```

### 模組導入錯誤

確保 Python 路徑正確：

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# Windows: set PYTHONPATH=%PYTHONPATH%;%cd%\src
```

## 📞 支援

如果遇到安裝問題：

1. 查看 [Issues](https://github.com/thc1006/cyberbully-zh-moderation-bot/issues)
2. 檢查 [`docs/`](docs/) 目錄中的相關文檔
3. 提交新的 Issue 並包含錯誤信息

---

**重要提醒：** 在首次運行系統之前，請務必完成步驟 3 的大文件下載，否則 API 和模型推理功能將無法正常工作。