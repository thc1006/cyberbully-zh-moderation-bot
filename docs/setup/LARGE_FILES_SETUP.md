# Large Files Setup Guide

## 大文件設置指南

此專案包含一些大於 100MB 的文件，這些文件被 `.gitignore` 排除以減少倉庫大小。首次克隆後，您需要手動獲取這些文件。

## Required Large Files / 必需的大文件

### 1. Model Checkpoints / 模型檢查點
- `models/macbert_base_demo/best.ckpt` (397MB)
- `models/toxicity_only_demo/best.ckpt` (397MB)

### 2. Dataset Files / 數據集文件
- `data/raw/dmsc/DMSC.csv` (387MB)
- `data/raw/dmsc/dmsc_kaggle.zip` (144MB)

## How to Obtain These Files / 如何獲取這些文件

### Method 1: Using Download Scripts / 方法1：使用下載腳本

```bash
# Download all datasets including large files
python scripts/download_datasets.py

# Alternative comprehensive download
python scripts/aggressive_download.py
```

### Method 2: Training Models / 方法2：訓練模型

如果您想從頭開始訓練模型：

```bash
# Train the models to generate checkpoints
python train.py --config configs/multitask_config.yaml
```

### Method 3: Manual Download / 方法3：手動下載

如果自動腳本無法工作，請：

1. **DMSC Dataset**:
   - 從 [豆瓣電影短評數據集](https://github.com/ownthink/dmsc-v2) 下載
   - 解壓到 `data/raw/dmsc/`

2. **Model Checkpoints**:
   - 運行訓練腳本生成，或
   - 聯繫專案維護者獲取預訓練模型

## Verification / 驗證

檢查所有必需文件是否存在：

```bash
python scripts/check_datasets.py
```

## Why These Files Are Excluded / 為什麼排除這些文件

- **減少倉庫大小**: GitHub 對大文件有限制
- **提高克隆速度**: 避免下載不必要的大文件
- **靈活性**: 允許用戶選擇是否需要完整數據集

## Notes / 注意事項

- 這些文件對於完整功能是必需的
- API 和模型推理需要模型檢查點
- 數據處理腳本需要原始數據集
- 所有文件都可以通過提供的腳本重新生成或下載