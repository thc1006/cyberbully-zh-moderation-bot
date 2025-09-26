# 中文霸凌偵測資料增強系統

## 概述

CyberPuppy 資料增強系統是專為中文霸凌偵測任務設計的綜合性資料增強解決方案。系統解決了 100% 合成標籤問題，通過多種策略提升資料多樣性和真實性，同時確保標籤一致性和品質控制。

## 🎯 核心特性

### 多策略增強
- **同義詞替換** - 基於 NTUSD 情感詞典的語義一致替換
- **回譯增強** - 中英文互譯增加自然變化
- **上下文擾動** - 使用 MacBERT [MASK] 預測的語境感知替換
- **EDA 操作** - 隨機插入、刪除、交換的輕量級增強

### 品質保證
- **標籤一致性驗證** - 確保增強後標籤與內容匹配
- **語義相似度檢查** - 維持與原文的語義關聯
- **品質門檻控制** - 過濾低品質增強樣本
- **分佈平衡維護** - 保持各類別標籤分佈

### 靈活配置
- **可配置強度** - 輕度、中度、重度增強預設
- **批次處理支援** - 高效的大規模資料處理
- **多程序加速** - 支援並行處理
- **詳細統計報告** - 增強過程監控和分析

## 🚀 快速開始

### 基本使用

```python
from cyberpuppy.data_augmentation import AugmentationPipeline

# 創建增強管道
pipeline = AugmentationPipeline()

# 準備資料
texts = ["這個很討厭", "我很開心", "今天天氣不錯"]
labels = [
    {'toxicity': 'toxic', 'emotion': 'neg'},
    {'toxicity': 'none', 'emotion': 'pos'},
    {'toxicity': 'none', 'emotion': 'neu'}
]

# 執行增強
augmented_texts, augmented_labels = pipeline.augment(texts, labels)

print(f"原始樣本數: {len(texts)}")
print(f"增強後樣本數: {len(augmented_texts)}")
```

### DataFrame 處理

```python
import pandas as pd
from cyberpuppy.data_augmentation import AugmentationPipeline

# 載入資料
df = pd.read_csv('data/processed/cold_dataset.csv')

# 創建管道
pipeline = AugmentationPipeline()

# 增強 DataFrame
augmented_df = pipeline.augment_dataframe(
    df,
    text_column='text',
    label_columns=['toxicity', 'bullying', 'emotion']
)

# 儲存結果
augmented_df.to_csv('data/processed/cold_augmented.csv', index=False)
```

### 使用執行腳本

```bash
# 基本增強
python scripts/augment_bullying_data.py \
    --input data/processed/cold_dataset.csv \
    --output data/processed/cold_augmented.csv \
    --intensity medium

# 自訂配置
python scripts/augment_bullying_data.py \
    --input data/processed/cold_dataset.csv \
    --output data/processed/cold_augmented.csv \
    --strategies synonym contextual eda \
    --augmentation-ratio 0.4 \
    --augmentations-per-text 3 \
    --save-analysis analysis_report.yaml
```

## 📋 增強策略詳解

### 1. 同義詞替換 (SynonymAugmenter)

基於 NTUSD 情感詞典的語義一致同義詞替換。

**特點:**
- 保持情感極性一致
- 支援正面、負面、中性詞彙
- 霸凌相關詞彙專門處理

**示例:**
```python
from cyberpuppy.data_augmentation import SynonymAugmenter

augmenter = SynonymAugmenter()
original = "你很笨"
augmented = augmenter.augment(original, num_augmentations=3)
# 可能結果: ["你很蠢", "你很愚笨", "你很無腦"]
```

### 2. 回譯增強 (BackTranslationAugmenter)

通過中文→英文→中文的翻譯過程引入自然變化。

**特點:**
- 保持語義核心不變
- 增加表達多樣性
- 模擬真實語言變化

**示例:**
```python
from cyberpuppy.data_augmentation import BackTranslationAugmenter

augmenter = BackTranslationAugmenter()
original = "我討厭你"
augmented = augmenter.augment(original, num_augmentations=2)
# 可能結果: ["我不喜歡你", "我憎恨你"]
```

### 3. 上下文擾動 (ContextualAugmenter)

使用 MacBERT 模型的 [MASK] 預測進行語境感知替換。

**特點:**
- 語境感知的詞彙替換
- 基於大規模預訓練模型
- 保持句法結構完整

**示例:**
```python
from cyberpuppy.data_augmentation import ContextualAugmenter

augmenter = ContextualAugmenter()
original = "這個人很討厭"
augmented = augmenter.augment(original, num_augmentations=2)
# 可能結果: ["這個人很煩人", "這個人很可惡"]
```

### 4. EDA 操作 (EDAugmenter)

輕量級的隨機操作：插入、刪除、交換。

**特點:**
- 計算成本低
- 操作簡單直接
- 適合大規模處理

**示例:**
```python
from cyberpuppy.data_augmentation import EDAugmenter

augmenter = EDAugmenter()
original = "我很討厭這個"
augmented = augmenter.augment(original, num_augmentations=3)
# 可能結果: ["我很討厭這個的", "我討厭這個", "這個我很討厭"]
```

## ⚙️ 配置選項

### AugmentationConfig

控制各個增強器的行為。

```python
from cyberpuppy.data_augmentation import AugmentationConfig

config = AugmentationConfig(
    synonym_prob=0.1,        # 同義詞替換機率
    backtrans_prob=0.3,      # 回譯機率
    contextual_prob=0.15,    # 上下文遮罩機率
    eda_prob=0.1,           # EDA 操作機率
    max_augmentations=5,     # 最大增強數量
    preserve_entities=True,  # 保留特殊符號
    quality_threshold=0.7    # 品質門檻
)
```

### PipelineConfig

控制增強管道的整體行為。

```python
from cyberpuppy.data_augmentation import PipelineConfig

config = PipelineConfig(
    # 策略選擇
    use_synonym=True,
    use_backtranslation=True,
    use_contextual=True,
    use_eda=True,

    # 增強強度
    augmentation_ratio=0.3,      # 增強資料比例
    augmentations_per_text=2,    # 每文本增強數
    max_total_augmentations=10000,

    # 品質控制
    quality_threshold=0.3,
    max_length_ratio=2.0,
    min_length_ratio=0.5,

    # 標籤平衡
    preserve_label_distribution=True,
    target_balance_ratio=1.0,

    # 處理選項
    batch_size=32,
    num_workers=4,
    use_multiprocessing=True,
    random_seed=42
)
```

### 預設強度配置

```python
from cyberpuppy.data_augmentation import create_augmentation_pipeline

# 輕度增強 - 保守策略
light_pipeline = create_augmentation_pipeline('light')
# - augmentation_ratio=0.1
# - augmentations_per_text=1
# - quality_threshold=0.5

# 中度增強 - 平衡策略
medium_pipeline = create_augmentation_pipeline('medium')
# - augmentation_ratio=0.3
# - augmentations_per_text=2
# - quality_threshold=0.3

# 重度增強 - 積極策略
heavy_pipeline = create_augmentation_pipeline('heavy')
# - augmentation_ratio=0.5
# - augmentations_per_text=3
# - quality_threshold=0.2
```

## 🔍 品質驗證系統

### 標籤一致性驗證

```python
from cyberpuppy.data_augmentation.validation import (
    LabelConsistencyValidator,
    LabelConsistencyConfig
)

# 配置驗證器
config = LabelConsistencyConfig(
    min_toxicity_confidence=0.7,
    min_bullying_confidence=0.7,
    min_emotion_confidence=0.6,
    max_length_change=0.5,
    min_semantic_similarity=0.3
)

validator = LabelConsistencyValidator(config)

# 驗證單個樣本
result = validator.validate_single_sample(
    original_text="我很討厭你",
    augmented_text="我很憎恨你",
    original_labels={'toxicity': 'toxic', 'emotion': 'neg'},
    augmented_labels={'toxicity': 'toxic', 'emotion': 'neg'}
)

print(f"驗證通過: {result.is_valid}")
print(f"信心度: {result.confidence:.3f}")
print(f"違規項目: {result.violations}")
```

### 批次驗證和報告

```python
# 驗證整個資料集
results, batch_stats = validator.validate_batch(
    original_texts, augmented_texts,
    original_labels, augmented_labels
)

# 生成報告
from cyberpuppy.data_augmentation.validation import QualityAssuranceReport

report = QualityAssuranceReport.generate_validation_report(results, batch_stats)
print(report)

# 儲存報告
QualityAssuranceReport.save_report(report, 'validation_report.txt')
```

## 📊 性能監控

### 統計資訊收集

```python
# 執行增強
pipeline = AugmentationPipeline()
augmented_texts, augmented_labels = pipeline.augment(texts, labels)

# 獲取統計資訊
stats = pipeline.get_statistics()

print(f"處理樣本數: {stats['total_processed']}")
print(f"增強樣本數: {stats['total_augmented']}")
print(f"品質過濾數: {stats['quality_filtered']}")
print(f"增強成功率: {stats['augmentation_ratio']:.2%}")
print(f"品質通過率: {stats['quality_pass_rate']:.2%}")

# 策略使用分佈
for strategy, percentage in stats['strategy_percentages'].items():
    print(f"{strategy}: {percentage:.1f}%")
```

### 詳細分析範例

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 分析文本長度分佈
original_lengths = [len(text) for text in original_texts]
augmented_lengths = [len(text) for text in augmented_texts]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(original_lengths, bins=30, alpha=0.7, label='原始')
plt.hist(augmented_lengths, bins=30, alpha=0.7, label='增強')
plt.xlabel('文本長度')
plt.ylabel('頻率')
plt.legend()
plt.title('文本長度分佈')

plt.subplot(1, 2, 2)
strategy_counts = list(stats['strategy_usage'].values())
strategy_names = list(stats['strategy_usage'].keys())
plt.pie(strategy_counts, labels=strategy_names, autopct='%1.1f%%')
plt.title('策略使用分佈')

plt.tight_layout()
plt.show()
```

## 🏗️ 進階使用案例

### 自訂增強策略

```python
from cyberpuppy.data_augmentation.augmenters import BaseAugmenter

class CustomAugmenter(BaseAugmenter):
    def augment(self, text: str, **kwargs) -> List[str]:
        # 自訂增強邏輯
        augmented = []
        # ... 實作增強邏輯
        return augmented

# 整合到管道
pipeline = AugmentationPipeline()
pipeline.augmenters['custom'] = CustomAugmenter()
```

### 特定領域適配

```python
# 為特定霸凌類型客製化
cyberbullying_config = PipelineConfig(
    # 針對霸凌偵測優化策略權重
    strategy_weights={
        'synonym': 0.4,      # 加重同義詞替換
        'backtranslation': 0.1,  # 減少回譯
        'contextual': 0.4,   # 加重語境增強
        'eda': 0.1          # 減少隨機操作
    },
    preserve_label_distribution=True,
    target_balance_ratio=1.5  # 增強少數類別
)

pipeline = AugmentationPipeline(cyberbullying_config)
```

### 多資料集整合

```python
import pandas as pd

# 處理多個資料集
datasets = ['cold', 'sccd', 'chnci']
all_augmented = []

for dataset_name in datasets:
    df = pd.read_csv(f'data/processed/{dataset_name}_dataset.csv')

    # 針對不同資料集調整配置
    if dataset_name == 'cold':
        intensity = 'heavy'  # COLD 資料較少，重度增強
    else:
        intensity = 'medium'

    pipeline = create_augmentation_pipeline(intensity)
    augmented_df = pipeline.augment_dataframe(
        df, 'text', ['toxicity', 'bullying', 'emotion']
    )

    augmented_df['source_dataset'] = dataset_name
    all_augmented.append(augmented_df)

# 合併所有增強資料
final_df = pd.concat(all_augmented, ignore_index=True)
final_df.to_csv('data/processed/all_augmented.csv', index=False)
```

## 🧪 測試和驗證

### 執行測試套件

```bash
# 執行所有測試
python -m pytest tests/test_augmentation.py -v

# 執行特定測試
python -m pytest tests/test_augmentation.py::TestSynonymAugmenter -v

# 測試覆蓋率
python -m pytest tests/test_augmentation.py --cov=cyberpuppy.data_augmentation
```

### 手動驗證範例

```python
# 測試個別增強器
from cyberpuppy.data_augmentation import SynonymAugmenter

augmenter = SynonymAugmenter()
test_cases = [
    "你很笨",
    "我討厭這個",
    "今天心情很好",
    "這個人很煩"
]

for text in test_cases:
    augmented = augmenter.augment(text, num_augmentations=3)
    print(f"原文: {text}")
    for i, aug in enumerate(augmented, 1):
        print(f"  增強{i}: {aug}")
    print()
```

## 🚀 最佳實務

### 1. 分階段增強

```python
# 第一階段：輕度增強，檢查品質
light_pipeline = create_augmentation_pipeline('light')
sample_augmented = light_pipeline.augment(sample_texts[:100], sample_labels[:100])

# 驗證品質
is_valid, report = validate_augmented_dataset(
    sample_texts[:100], sample_augmented[0],
    sample_labels[:100], sample_augmented[1]
)

if is_valid:
    # 第二階段：完整增強
    full_pipeline = create_augmentation_pipeline('medium')
    final_augmented = full_pipeline.augment(all_texts, all_labels)
```

### 2. 記憶體優化

```python
# 大資料集分批處理
def process_large_dataset(df, batch_size=1000):
    results = []

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]

        pipeline = AugmentationPipeline()
        augmented_batch = pipeline.augment_dataframe(
            batch_df, 'text', ['toxicity', 'bullying', 'emotion']
        )

        results.append(augmented_batch)

        # 清理記憶體
        del pipeline

    return pd.concat(results, ignore_index=True)
```

### 3. 品質監控

```python
# 設定品質監控閾值
def monitor_augmentation_quality(pipeline, texts, labels):
    augmented_texts, augmented_labels = pipeline.augment(texts, labels)
    stats = pipeline.get_statistics()

    # 檢查關鍵指標
    if stats['quality_pass_rate'] < 0.8:
        logger.warning("品質通過率過低，建議調整參數")

    if stats['augmentation_ratio'] < 0.2:
        logger.warning("增強比例過低，可能需要放寬限制")

    return augmented_texts, augmented_labels
```

## 🔧 故障排除

### 常見問題

1. **記憶體不足**
   ```python
   # 減少批次大小和工作程序數
   config = PipelineConfig(
       batch_size=16,  # 從 32 減少到 16
       num_workers=2,  # 從 4 減少到 2
       use_multiprocessing=False  # 關閉多程序
   )
   ```

2. **模型載入失敗**
   ```python
   # 使用較輕量的策略
   pipeline = create_augmentation_pipeline(
       'medium',
       strategies=['synonym', 'eda']  # 避免重模型
   )
   ```

3. **品質驗證過嚴**
   ```python
   # 調整驗證閾值
   validation_config = LabelConsistencyConfig(
       min_toxicity_confidence=0.5,  # 降低信心門檻
       min_semantic_similarity=0.2   # 降低相似度要求
   )
   ```

### 除錯模式

```python
import logging

# 啟用詳細日誌
logging.basicConfig(level=logging.DEBUG)

# 使用小樣本測試
test_texts = texts[:10]
test_labels = labels[:10]

pipeline = AugmentationPipeline()
augmented = pipeline.augment(test_texts, test_labels, verbose=True)
```

## 📝 參考資料

- [NTUSD 情感詞典](http://nlp.csie.ntu.edu.tw/resource/sentiment.html)
- [MacBERT 預訓練模型](https://github.com/ymcui/MacBERT)
- [Easy Data Augmentation 論文](https://arxiv.org/abs/1901.11196)
- [霸凌偵測資料集 COLD](https://github.com/hate-alert/COLD)

## 🤝 貢獻指南

歡迎提交 issue 和 pull request 來改善系統：

1. Fork 專案
2. 創建特性分支 (`git checkout -b feature/new-augmentation`)
3. 提交更改 (`git commit -am 'Add new augmentation strategy'`)
4. 推送分支 (`git push origin feature/new-augmentation`)
5. 創建 Pull Request

## 📄 授權

本專案採用 MIT 授權條款。詳見 LICENSE 文件。