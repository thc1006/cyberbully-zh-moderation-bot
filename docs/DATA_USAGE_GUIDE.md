# 訓練資料使用指南

## 概述

本文檔提供 CyberPuppy 訓練資料系統的完整使用指南，包括資料載入、預處理、特徵提取和模型訓練準備。

## 資料結構

### 資料目錄結構
```
data/processed/training_dataset/
├── train.json          # 訓練集 (25,659 樣本)
├── dev.json            # 開發集 (6,430 樣本)
├── test.json           # 測試集 (5,320 樣本)
├── detailed_statistics.json  # 詳細統計資訊
├── validation_report.json    # 資料品質報告
└── data_loader.py      # 資料載入腳本
```

### 資料格式

每個JSON檔案包含以下格式的樣本：

```json
{
  "text": "文字內容",
  "label": {
    "toxicity": "none|toxic|severe",
    "bullying": "none|harassment|threat",
    "role": "none|perpetrator|victim|bystander",
    "emotion": "pos|neu|neg",
    "emotion_strength": 0-4
  },
  "metadata": {
    "text_length": 47,
    "source": "preprocessed"
  }
}
```

## 快速開始

### 1. 基本資料載入

```python
from cyberpuppy.data.loader import create_data_loader

# 創建資料載入器
data_loader = create_data_loader(
    data_path="./data/processed/training_dataset",
    batch_size=32,
    include_features=False
)

# 獲取訓練資料載入器
train_loader = data_loader.get_dataloader('train')

# 遍歷批次資料
for batch in train_loader:
    texts = batch['texts']          # List[str]
    labels = batch['labels']        # Dict[str, torch.Tensor]

    # 各任務標籤
    toxicity_labels = labels['toxicity']
    emotion_labels = labels['emotion']
    # ... 處理邏輯
```

### 2. 包含手工特徵的載入

```python
# 載入帶特徵的資料
data_loader = create_data_loader(
    data_path="./data/processed/training_dataset",
    batch_size=32,
    include_features=True,  # 啟用特徵提取
    ntusd_path="path/to/ntusd.json"  # 可選：NTUSD詞典路徑
)

train_loader = data_loader.get_dataloader('train')

for batch in train_loader:
    texts = batch['texts']
    labels = batch['labels']
    features = batch['features']   # torch.Tensor, shape: [batch_size, feature_dim]
    feature_names = batch['feature_names']  # List[str]
```

### 3. 單一資料集使用

```python
from cyberpuppy.data.loader import TrainingDataset

# 創建單一資料集
dataset = TrainingDataset(
    data_path="./data/processed/training_dataset",
    split='train',
    include_features=True
)

# 獲取樣本
sample = dataset[0]
print(f"Text: {sample['text']}")
print(f"Labels: {sample['labels']}")
print(f"Features: {sample['features']}")

# 獲取類別權重（用於不平衡資料集）
weights = dataset.get_class_weights('toxicity')
print(f"Class weights: {weights}")
```

## 高級用法

### 1. 自定義資料預處理

```python
from cyberpuppy.data.preprocessor import DataPreprocessor

# 創建預處理器
preprocessor = DataPreprocessor(
    base_path=".",
    target_format='traditional',  # 繁體中文
    ntusd_path="path/to/ntusd.json"
)

# 執行完整預處理流水線
stats = preprocessor.process_complete_pipeline(
    output_dir="./data/processed/custom_dataset",
    balance_data=True,    # 平衡資料
    validate_data=True    # 驗證資料品質
)

print(f"Processing stats: {stats}")
```

### 2. 資料品質驗證

```python
from cyberpuppy.data.validator import validate_training_data

# 驗證資料品質
validation_report = validate_training_data(
    data_dir="./data/processed/training_dataset",
    output_dir="./data/processed/training_dataset"
)

print(f"Overall quality: {validation_report['overall_summary']['overall_quality_score']}")
```

### 3. 特徵提取

```python
from cyberpuppy.data.feature_extractor import CombinedFeatureExtractor

# 創建特徵提取器
extractor = CombinedFeatureExtractor(ntusd_path="path/to/ntusd.json")

# 提取單個文字特徵
text = "這是一個測試文字"
features = extractor.extract_features(text)

print(f"Feature names: {extractor.get_feature_names()}")
print(f"Features: {features}")

# 批量提取特徵
texts = ["文字1", "文字2", "文字3"]
batch_features = extractor.extract_batch_features(texts)
```

## 資料統計資訊

### 資料集概覽

- **總樣本數**: 37,409
- **訓練集**: 25,659 樣本 (68.6%)
- **開發集**: 6,430 樣本 (17.2%)
- **測試集**: 5,320 樣本 (14.2%)

### 標籤分佈

#### 毒性分佈
- **訓練集**: 12,673 toxic (49.4%), 12,986 none (50.6%)
- **開發集**: 3,210 toxic (49.9%), 3,220 none (50.1%)
- **測試集**: 2,106 toxic (39.6%), 3,214 none (60.4%)

#### 文字長度統計
- **平均長度**: ~48 字符
- **長度範圍**: 3-192 字符
- **中位數**: 40-41 字符

### 資料品質

- **有效樣本**: 37,393 (99.96%)
- **主要問題**: 中文字符比例不足 (16 樣本)

## 模型訓練集成

### PyTorch集成

```python
import torch
from torch.utils.data import DataLoader
from cyberpuppy.data.loader import TrainingDataset

# 創建資料集
train_dataset = TrainingDataset(
    data_path="./data/processed/training_dataset",
    split='train'
)

# 創建DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 獲取類別權重（用於損失函數）
toxicity_weights = train_dataset.get_class_weights('toxicity')
criterion = torch.nn.CrossEntropyLoss(weight=toxicity_weights)
```

### 多任務學習範例

```python
# 多任務模型訓練範例
for batch in train_loader:
    texts = batch['texts']
    labels = batch['labels']

    # 模型前向傳播（假設模型返回多個任務的輸出）
    outputs = model(texts)  # Dict[str, torch.Tensor]

    # 計算多任務損失
    total_loss = 0
    for task in ['toxicity', 'bullying', 'emotion']:
        if task in outputs and task in labels:
            task_loss = criterion(outputs[task], labels[task])
            total_loss += task_loss

    # 反向傳播
    total_loss.backward()
    optimizer.step()
```

## 常見問題

### Q: 如何處理類別不平衡？

A: 使用類別權重或重採樣：

```python
# 方法1: 使用類別權重
weights = dataset.get_class_weights('toxicity')
criterion = torch.nn.CrossEntropyLoss(weight=weights)

# 方法2: 資料平衡
preprocessor = DataPreprocessor()
datasets = preprocessor.integrate_all_datasets(balance_data=True)
```

### Q: 如何添加自定義特徵？

A: 擴展特徵提取器：

```python
from cyberpuppy.data.feature_extractor import TextFeatureExtractor

class CustomFeatureExtractor(TextFeatureExtractor):
    def extract_custom_features(self, text):
        # 添加自定義特徵邏輯
        return {'custom_feature': some_value}

    def extract_all_features(self, text):
        features = super().extract_all_features(text)
        features.update(self.extract_custom_features(text))
        return features
```

### Q: 如何處理記憶體不足？

A: 使用較小的batch_size和more_workers：

```python
data_loader = create_data_loader(
    data_path="./data/processed/training_dataset",
    batch_size=16,  # 減少batch size
    num_workers=2   # 減少工作進程數
)
```

## 性能優化建議

1. **預載入特徵**: 如果使用手工特徵，預計算並儲存特徵以避免重複計算
2. **資料快取**: 使用SSD儲存資料，或將資料載入記憶體
3. **批次大小**: 根據GPU記憶體調整批次大小
4. **工作進程**: 適當設置num_workers以平衡CPU和I/O負載

## 更新記錄

- **v1.0**: 初始版本，支援COLD資料集整合
- **v1.1**: 新增情緒分析資料支援
- **v1.2**: 新增特徵提取和資料驗證
- **v1.3**: 完善多任務學習支援

## 聯絡資訊

如有問題或建議，請聯絡開發團隊或提交Issue。