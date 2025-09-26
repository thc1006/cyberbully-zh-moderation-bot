# CyberPuppy 本地訓練指南

本地 Windows RTX 3050 優化的訓練啟動器，提供用戶友好的互動式界面。

## 🚀 快速開始

### 方法一：使用批次檔案 (推薦)
```cmd
# 雙擊執行或在命令列執行
train_local.bat
```

### 方法二：直接使用 Python
```cmd
python scripts/train_local.py
```

## 📋 使用前檢查

執行訓練前，建議先檢查環境：
```cmd
python scripts/check_requirements.py
```

這會檢查：
- Python 版本 (需要 3.8+)
- 必要套件安裝狀況
- GPU 支援和記憶體
- 專案目錄結構
- 磁碟空間

## 🎯 功能特色

### 🖥️ 硬體檢測
- 自動檢測 RTX 3050 並優化配置
- GPU 記憶體監控和 OOM 預防
- CPU 核心數自動配置

### ⚙️ 配置選項
1. **Conservative (記憶體保守型)**
   - RTX 3050 4GB 安全配置
   - 批次大小: 4, 梯度累積: 4
   - 混合精度訓練，梯度檢查點

2. **Aggressive (性能激進型)**
   - 需要良好散熱
   - 批次大小: 8, 梯度累積: 2
   - 更高學習率

3. **RoBERTa (更大模型)**
   - 使用 chinese-roberta-wwm-ext
   - 更好準確性但需更多記憶體
   - 自動調整批次大小

4. **Custom (自定義)**
   - 自定義所有參數

### 📊 訓練監控
- 實時進度條 (雙層：epoch + step)
- 記憶體用量監控
- 估算剩餘時間 (ETA)
- 彩色終端輸出 (Windows 相容)

### 🛡️ 錯誤處理
- OOM 自動恢復 (減小批次大小)
- 訓練中斷恢復 (Ctrl+C 安全保存)
- 檢查點自動保存
- 詳細錯誤日誌

### 📈 評估報告
訓練完成後自動生成：
```
reports/training_report_實驗名稱.json
```
包含：
- 訓練配置詳情
- 系統硬體信息
- 性能指標 (F1 分數等)
- 訓練時間統計

## 🔧 環境需求

### 必要套件
```txt
torch>=1.13.0
transformers>=4.20.0
numpy>=1.21.0
tqdm>=4.64.0
scikit-learn>=1.1.0
colorama>=0.4.5  # Windows 彩色輸出
```

### 系統需求
- Windows 10/11
- Python 3.8+
- RTX 3050 4GB 或更好的 GPU (可選)
- 至少 5GB 可用磁碟空間

## 📁 檔案結構

```
cyberbully-zh-moderation-bot/
├── train_local.bat              # Windows 啟動器
├── scripts/
│   ├── train_local.py          # 主要訓練腳本
│   ├── check_requirements.py   # 環境檢查
│   └── README_TRAINING.md      # 本文檔
├── configs/                    # 配置檔案
├── models/                     # 訓練好的模型
├── logs/                       # 訓練日誌
├── checkpoints/                # 檢查點
└── reports/                    # 評估報告
```

## 🎮 使用流程

1. **環境檢查**
   ```cmd
   python scripts/check_requirements.py
   ```

2. **啟動訓練**
   ```cmd
   train_local.bat
   ```

3. **選擇配置**
   - 根據硬體自動推薦
   - 可自定義參數

4. **設定參數**
   - 訓練輪數 (預設: 10)
   - 資料增強 (預設: 關閉)
   - 早停耐心 (預設: 3)

5. **確認開始**
   - 檢視配置摘要
   - 確認記憶體安全
   - 開始訓練

6. **監控進度**
   - 實時進度顯示
   - 記憶體用量監控
   - 錯誤自動處理

7. **查看結果**
   - 自動生成評估報告
   - 檢查點檔案保存
   - 最佳模型載入

## ⚠️ 常見問題

### Q: OOM (記憶體不足) 錯誤
**A:** 程式會自動處理：
- 自動減小批次大小
- 增加梯度累積步數
- 清理 GPU 快取
- 繼續訓練

### Q: 訓練被中斷
**A:** 支援中斷恢復：
- Ctrl+C 安全中斷
- 自動保存檢查點
- 下次啟動可選擇續訓

### Q: 沒有 GPU
**A:** 自動 CPU 後備：
- 檢測到無 GPU 時自動切換
- 調整配置以適應 CPU
- 顯示預估訓練時間

### Q: 中文顯示問題
**A:** Windows 中文支援：
- 自動設置 UTF-8 編碼
- colorama 確保顏色正常
- 批次檔案設置正確代碼頁

## 🔍 日誌和除錯

### 日誌位置
```
logs/training_local_時間戳.log
```

### 重要日誌內容
- 硬體檢測結果
- 配置參數
- 訓練進度
- 錯誤信息
- 性能指標

### 除錯步驟
1. 檢查 `check_requirements.py` 輸出
2. 查看日誌檔案
3. 確認專案目錄結構
4. 檢查可用磁碟空間

## 🛠️ 進階配置

### 自定義配置檔案
可手動編輯配置檔案：
```
configs/exp_實驗名稱.json
```

### 環境變數
```cmd
set CYBERPUPPY_USE_GPU=True
set CYBERPUPPY_BATCH_SIZE=8
set CYBERPUPPY_LOG_LEVEL=DEBUG
```

### 命令列參數 (計劃中)
```cmd
python scripts/train_local.py --config conservative --epochs 5
```

## 📞 技術支援

遇到問題時請提供：
1. `check_requirements.py` 輸出
2. 系統信息 (GPU 型號、記憶體)
3. 錯誤日誌 (`logs/` 目錄)
4. 使用的配置選項

---

**祝訓練順利！** 🐕✨