# CyberPuppy 資料集下載指南

## 目前下載狀態

### ✅ 已成功下載
1. **COLD (Chinese Offensive Language Dataset)** - 約 6MB
   - 位置: `data/raw/cold/COLDataset/`
   - 包含: train.csv (4.1MB), dev.csv (1MB), test.csv (0.8MB)
   - 共約 37,000 筆中文冒犯語言標註資料

### ⚠️ 需要手動下載的資料集

## 1. ChnSentiCorp (中文情感分析資料集)

### 方法 A: 透過 Hugging Face 網頁下載
1. 訪問: https://huggingface.co/datasets/seamew/ChnSentiCorp
2. 點擊 "Files and versions" 標籤
3. 下載所需檔案到 `data/raw/chnsenticorp/`

### 方法 B: 使用 Python 腳本（需要設定代理或 VPN）
```python
from datasets import load_dataset

# 如果需要代理
import os
os.environ['HTTP_PROXY'] = 'your_proxy_here'
os.environ['HTTPS_PROXY'] = 'your_proxy_here'

dataset = load_dataset("seamew/ChnSentiCorp")
dataset.save_to_disk("data/raw/chnsenticorp")
```

### 方法 C: 從其他來源獲取
- 原始來源: http://www.nlpir.org/?action-viewnews-itemid-77
- 備份來源: 搜尋 "ChnSentiCorp dataset download"

預期大小: ~10MB
包含: 約 12,000 筆酒店評論的情感標註（正面/負面）

## 2. DMSC v2 (豆瓣電影短評資料集)

### 下載方法
1. 訪問以下備用連結：
   - https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/dmsc_v2
   - https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments

2. 下載檔案：
   - `dmsc_v2.csv` 或類似格式的檔案
   - 放置到 `data/raw/dmsc/`

預期大小: 200-500MB
包含: 豆瓣電影短評與評分（1-5星）

## 3. NTUSD (臺大情感詞典)

### 下載方法
1. GitHub 原始庫似乎已移除，請嘗試：
   - 搜尋 "NTUSD sentiment dictionary"
   - 訪問: http://nlg.csie.ntu.edu.tw/nlpresource/NTUSD/

2. 需要的檔案：
   - `ntusd-positive.txt` - 正面詞彙
   - `ntusd-negative.txt` - 負面詞彙
   - 放置到 `data/raw/ntusd/`

預期大小: <5MB
包含: 繁體中文正負情感詞彙列表

## 4. SCCD (Session-level Chinese Cyberbullying Dataset)

### 獲取步驟
1. 閱讀論文: https://arxiv.org/abs/2506.04975
2. 聯繫論文作者獲取資料集存取權限
3. 通常需要：
   - 發送郵件給通訊作者
   - 簽署資料使用協議
   - 說明研究用途（防治網路霸凌）

4. 獲得授權後，下載並放置到 `data/external/sccd/`
   預期檔案：
   - `sccd_train.json`
   - `sccd_dev.json`
   - `sccd_test.json`

預期大小: 50-100MB
包含: 微博會話級霸凌對話標註

## 5. CHNCI (Chinese Cyberbullying Incident Dataset)

### 獲取步驟
1. 閱讀論文: https://arxiv.org/abs/2506.05380
2. 根據論文指引申請資料集
3. 可能需要：
   - 填寫線上申請表單
   - 提供機構資訊
   - 簽署倫理使用協議

4. 獲得授權後，下載並放置到 `data/external/chnci/`
   預期檔案：
   - `chnci_events.json`
   - `chnci_annotations.json`

預期大小: 50-100MB
包含: 事件級霸凌資料，含角色標註（加害者/受害者/旁觀者）

## 快速檢查資料集完整性

執行以下命令檢查所有資料集狀態：

```bash
python scripts/check_datasets.py
```

## 授權注意事項

### 學術使用
- SCCD 和 CHNCI 通常僅限學術研究使用
- 需要簽署資料使用協議
- 不可用於商業用途

### 隱私保護
- 所有包含用戶生成內容的資料集都應該：
  1. 僅用於研究和防治霸凌目的
  2. 不公開分享原始資料
  3. 遵守去識別化原則
  4. 遵守各資料集的使用條款

## 聯繫資訊模板

如需聯繫論文作者，可參考以下郵件模板：

```
Subject: Request for [Dataset Name] Access for Cyberbullying Prevention Research

Dear Professor [Name],

I am [Your Name] from [Your Institution], working on a research project focused on cyberbullying prevention in Chinese social media.

I recently read your paper "[Paper Title]" and would like to request access to the [Dataset Name] dataset for our research.

Our project aims to:
1. Develop better detection models for cyberbullying in Chinese
2. Create interpretable AI systems for content moderation
3. Protect vulnerable users in online communities

We commit to:
- Use the data solely for academic research
- Follow all ethical guidelines
- Cite your work appropriately
- Not redistribute the data

Please let me know the procedure to obtain access to the dataset.

Best regards,
[Your Name]
[Your Institution]
[Contact Information]
```

## 總結

目前狀態：
- ✅ COLD: 已下載完成
- ⚠️ ChnSentiCorp: 需要手動下載（Hugging Face）
- ⚠️ DMSC v2: 需要從備用來源下載
- ⚠️ NTUSD: 需要從備用來源下載
- 📧 SCCD: 需要聯繫作者
- 📧 CHNCI: 需要聯繫作者

總預估大小: 300-800MB（所有資料集）

---
最後更新: 2025-09-24