# 資料集下載使用指南

## 快速開始

### 使用改進的下載腳本
```bash
# 下載所有資料集
python scripts/improved_download_datasets.py

# 下載特定資料集
python scripts/improved_download_datasets.py --dataset ntusd

# 只驗證現有資料集
python scripts/improved_download_datasets.py --validate-only

# 驗證所有資料集的完整性
python scripts/validate_datasets.py
```

## 具體下載指令

### 1. ChnSentiCorp 中文情感分析資料集

#### 方法一：使用改進腳本（推薦）
```bash
python scripts/improved_download_datasets.py --dataset chnsenticorp
```

#### 方法二：手動下載 Parquet 檔案
```python
import pandas as pd

# 下載 Parquet 檔案
urls = {
    'train': 'https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/data/train-00000-of-00001.parquet',
    'validation': 'https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/data/validation-00000-of-00001.parquet',
    'test': 'https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/data/test-00000-of-00001.parquet'
}

for split, url in urls.items():
    df = pd.read_parquet(url)
    df.to_json(f'data/raw/chnsenticorp/{split}.json', orient='records', force_ascii=False)
    print(f"Downloaded {split}: {len(df)} samples")
```

#### 方法三：備用 CSV 來源
```bash
# 下載備用 CSV 檔案
curl -O https://raw.githubusercontent.com/fate233/sentiment_corpus/master/train.csv
curl -O https://raw.githubusercontent.com/fate233/sentiment_corpus/master/test.csv
```

**預期結果**：
- 檔案：`train.json`, `validation.json`, `test.json`
- 總計：約 7,767 筆記錄
- 格式：`{"label": 0/1, "text": "評論內容"}`

### 2. DMSC v2 豆瓣電影短評資料集

#### 方法一：使用現有檔案（推薦）
```bash
# 檔案已存在且完整
ls -la data/raw/dmsc/DMSC.csv
# 405.6 MB, 2,131,887 行
```

#### 方法二：重新下載
```bash
# 下載 ZIP 檔案
wget https://github.com/candlewill/dmsc-v2/releases/download/v2.0/dmsc_v2.zip
unzip dmsc_v2.zip -d data/raw/dmsc/

# 或使用改進腳本
python scripts/improved_download_datasets.py --dataset dmsc
```

**驗證命令**：
```bash
# 檢查檔案完整性
md5sum data/raw/dmsc/DMSC.csv
# 預期：4bdb50bb400d23a8eebcc11b27195fc4

# 檢查記錄數
wc -l data/raw/dmsc/DMSC.csv
# 預期：2,131,887 行
```

**預期結果**：
- 檔案：`DMSC.csv` (405.6 MB)
- 記錄數：2,131,886 筆評論（含表頭）
- 欄位：ID, Movie_Name_EN, Movie_Name_CN, Crawl_Date, Number, Username, Date, Star, Comment, Like

### 3. NTUSD 臺大情感詞典

#### 方法一：使用改進腳本（推薦）
```bash
python scripts/improved_download_datasets.py --dataset ntusd
```

#### 方法二：手動下載完整版本
```bash
# 下載完整的詞典檔案
curl -o data/raw/ntusd/positive.txt \
  https://raw.githubusercontent.com/sweslo17/chinese_sentiment/master/dict/ntusd-positive.txt

curl -o data/raw/ntusd/negative.txt \
  https://raw.githubusercontent.com/sweslo17/chinese_sentiment/master/dict/ntusd-negative.txt
```

#### 方法三：Python 下載腳本
```python
import requests
from pathlib import Path

# 創建目錄
ntusd_dir = Path("data/raw/ntusd")
ntusd_dir.mkdir(parents=True, exist_ok=True)

# 下載詞典
urls = {
    'positive.txt': 'https://raw.githubusercontent.com/sweslo17/chinese_sentiment/master/dict/ntusd-positive.txt',
    'negative.txt': 'https://raw.githubusercontent.com/sweslo17/chinese_sentiment/master/dict/ntusd-negative.txt'
}

for filename, url in urls.items():
    response = requests.get(url)
    (ntusd_dir / filename).write_text(response.text, encoding='utf-8')

    # 計算詞彙數
    words = [line.strip() for line in response.text.split('\n') if line.strip()]
    print(f"{filename}: {len(words)} 詞彙")
```

**驗證命令**：
```bash
# 檢查詞彙數量
wc -l data/raw/ntusd/positive.txt data/raw/ntusd/negative.txt

# 查看內容樣本
head -5 data/raw/ntusd/positive.txt
head -5 data/raw/ntusd/negative.txt
```

**預期結果**：
- 檔案：`positive.txt` (2,809 詞彙, ~26KB), `negative.txt` (8,277 詞彙, ~81KB)
- 總計：11,086 個中文情感詞彙
- 編碼：UTF-8 繁體中文

## Python 程式碼範例

### 載入並處理資料集

```python
import json
import pandas as pd
from pathlib import Path

# 載入 ChnSentiCorp
def load_chnsenticorp(base_dir="data/raw/chnsenticorp"):
    """載入 ChnSentiCorp 資料集"""
    datasets = {}
    for split in ['train', 'validation', 'test']:
        file_path = Path(base_dir) / f"{split}.json"
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                datasets[split] = json.load(f)
    return datasets

# 載入 DMSC
def load_dmsc(csv_path="data/raw/dmsc/DMSC.csv"):
    """載入 DMSC 資料集"""
    return pd.read_csv(csv_path, encoding='utf-8')

# 載入 NTUSD
def load_ntusd(base_dir="data/raw/ntusd"):
    """載入 NTUSD 詞典"""
    pos_file = Path(base_dir) / "positive.txt"
    neg_file = Path(base_dir) / "negative.txt"

    positive_words = pos_file.read_text(encoding='utf-8').strip().split('\n')
    negative_words = neg_file.read_text(encoding='utf-8').strip().split('\n')

    return {
        'positive': [word.strip() for word in positive_words if word.strip()],
        'negative': [word.strip() for word in negative_words if word.strip()]
    }

# 使用範例
if __name__ == "__main__":
    # 載入所有資料集
    chnsenticorp = load_chnsenticorp()
    dmsc = load_dmsc()
    ntusd = load_ntusd()

    print(f"ChnSentiCorp splits: {list(chnsenticorp.keys())}")
    print(f"DMSC records: {len(dmsc)}")
    print(f"NTUSD words: {len(ntusd['positive'])} positive, {len(ntusd['negative'])} negative")
```

### 資料集統計和品質檢查

```python
def analyze_dataset_quality():
    """分析資料集品質"""

    # ChnSentiCorp 分析
    chnsenticorp = load_chnsenticorp()
    for split, data in chnsenticorp.items():
        labels = [item['label'] for item in data]
        print(f"ChnSentiCorp {split}:")
        print(f"  Total: {len(data)}")
        print(f"  Positive: {labels.count(1)} ({labels.count(1)/len(labels)*100:.1f}%)")
        print(f"  Negative: {labels.count(0)} ({labels.count(0)/len(labels)*100:.1f}%)")

    # DMSC 分析
    dmsc = load_dmsc()
    print(f"\nDMSC Analysis:")
    print(f"  Total reviews: {len(dmsc)}")
    print(f"  Star distribution:")
    star_counts = dmsc['Star'].value_counts().sort_index()
    for star, count in star_counts.items():
        print(f"    {star} star: {count} ({count/len(dmsc)*100:.1f}%)")

    # NTUSD 分析
    ntusd = load_ntusd()
    print(f"\nNTUSD Analysis:")
    print(f"  Positive words: {len(ntusd['positive'])}")
    print(f"  Negative words: {len(ntusd['negative'])}")
    print(f"  Total words: {len(ntusd['positive']) + len(ntusd['negative'])}")
```

## 驗證方法

### 自動驗證腳本
```bash
# 驗證所有資料集
python scripts/validate_datasets.py

# 驗證特定目錄
python scripts/validate_datasets.py --data-dir /path/to/data

# 靜默模式（只顯示錯誤）
python scripts/validate_datasets.py --quiet

# 保存驗證報告
python scripts/validate_datasets.py --output validation_report.json
```

### 手動驗證檢查清單

#### ChnSentiCorp
- [ ] 存在 `train.json`, `validation.json`, `test.json`
- [ ] 每個檔案都包含 `label` 和 `text` 欄位
- [ ] 總記錄數 ≥ 7,000
- [ ] 標籤值為 0 或 1

#### DMSC v2
- [ ] 存在 `DMSC.csv` (405.6 MB)
- [ ] MD5: `4bdb50bb400d23a8eebcc11b27195fc4`
- [ ] 記錄數 ≥ 2,000,000
- [ ] 包含必要欄位：ID, Star, Comment

#### NTUSD
- [ ] 存在 `positive.txt` (≥ 2,500 詞彙)
- [ ] 存在 `negative.txt` (≥ 7,500 詞彙)
- [ ] 總詞彙數 ≥ 10,000
- [ ] UTF-8 編碼正確

## 預期檔案大小和結構

### 完整目錄結構
```
data/raw/
├── chnsenticorp/
│   ├── train.json (約 5MB, 6,000+ 記錄)
│   ├── validation.json (約 500KB, 500+ 記錄)
│   ├── test.json (約 300KB, 300+ 記錄)
│   └── download_metadata.json
├── dmsc/
│   ├── DMSC.csv (405.6MB, 2,131,886 記錄)
│   ├── dmsc_kaggle.zip (150.6MB)
│   └── download_metadata.json
└── ntusd/
    ├── positive.txt (26KB, 2,809 詞彙)
    ├── negative.txt (81KB, 8,277 詞彙)
    ├── merged_positive.txt
    ├── merged_negative.txt
    └── download_metadata.json
```

### 檔案大小總結

| 資料集 | 主要檔案 | 大小 | 記錄數/詞彙數 | 狀態 |
|--------|----------|------|---------------|------|
| ChnSentiCorp | train.json | ~5MB | 6,000+ | ✅ 可用 |
| ChnSentiCorp | validation.json | ~500KB | 500+ | ✅ 可用 |
| ChnSentiCorp | test.json | ~300KB | 300+ | ✅ 可用 |
| DMSC v2 | DMSC.csv | 405.6MB | 2,131,886 | ✅ 已驗證 |
| NTUSD | positive.txt | 26KB | 2,809 | ⚠️ 需更新 |
| NTUSD | negative.txt | 81KB | 8,277 | ⚠️ 需更新 |

## 故障排除

### 常見問題

#### 1. ChnSentiCorp 下載失敗
```bash
# 檢查網路連接
curl -I https://huggingface.co/datasets/seamew/ChnSentiCorp

# 使用備用來源
python scripts/improved_download_datasets.py --dataset chnsenticorp
```

#### 2. DMSC 檔案損壞
```bash
# 重新下載
rm data/raw/dmsc/DMSC.csv
python scripts/improved_download_datasets.py --dataset dmsc
```

#### 3. NTUSD 檔案太小
```bash
# 檢查現有檔案
wc -l data/raw/ntusd/*.txt

# 使用改進的下載方法
python scripts/improved_download_datasets.py --dataset ntusd
```

### 日誌和調試
```bash
# 啟用詳細日誌
export PYTHONUNBUFFERED=1
python scripts/improved_download_datasets.py --dataset all 2>&1 | tee download.log

# 檢查錯誤
grep -i error download.log
grep -i warning download.log
```

## 下一步

1. **執行完整下載**：`python scripts/improved_download_datasets.py`
2. **驗證資料完整性**：`python scripts/validate_datasets.py`
3. **檢查驗證報告**：確保所有資料集狀態為 PASS
4. **更新專案配置**：根據實際下載的資料集更新 `src/cyberpuppy/config.py`

完成這些步驟後，所有資料集應該準備就緒，可以開始資料預處理和模型訓練。