# Semi-supervised Learning Framework 使用指南

## 概述

本框架提供了三種半監督學習方法來利用未標註的中文文字資料，提升中文網路霸凌偵測模型的效能：

1. **Pseudo-labeling**: 使用高信心預測作為偽標籤
2. **Self-training**: 教師-學生模型架構與知識蒸餾
3. **Consistency Regularization**: 對同一樣本的不同增強版本強制預測一致性

## 硬體需求與優化

### 最低需求
- GPU: RTX 3050 4GB 或以上
- RAM: 16GB 以上
- 儲存空間: 10GB 以上

### 優化特色
- **混合精度訓練** (FP16): 減少 GPU 記憶體使用量約 50%
- **動態批次大小調整**: 根據可用記憶體自動調整
- **梯度累積**: 模擬更大的批次大小
- **記憶體清理**: 定期清理 GPU 快取

## 快速開始

### 1. 安裝依賴

```bash
pip install torch transformers scikit-learn numpy pyyaml
pip install pytest  # 用於測試
```

### 2. 運行示例

```bash
# 運行示例腳本
python scripts/demo_semi_supervised.py

# 運行測試
python -m pytest tests/test_semi_supervised.py -v
```

### 3. 訓練模型

```bash
# Pseudo-labeling
python scripts/train_semi_supervised.py \
    --config configs/semi_supervised.yaml \
    --method pseudo_labeling \
    --output_dir outputs/pseudo_labeling

# Self-training
python scripts/train_semi_supervised.py \
    --config configs/semi_supervised.yaml \
    --method self_training \
    --output_dir outputs/self_training

# Consistency Regularization
python scripts/train_semi_supervised.py \
    --config configs/semi_supervised.yaml \
    --method consistency \
    --output_dir outputs/consistency
```

## 配置說明

### 模型配置

```yaml
model:
  name: "hfl/chinese-macbert-base"  # 或 "hfl/chinese-roberta-wwm-ext"
  num_labels_toxicity: 3  # none, toxic, severe
  num_labels_emotion: 3   # pos, neu, neg
  dropout_rate: 0.1
  max_length: 512
```

### 訓練配置

```yaml
training:
  batch_size: 4  # 基礎批次大小，會動態調整
  learning_rate: 2e-5
  weight_decay: 0.01
  gradient_accumulation_steps: 4  # 有效批次大小 = 4 * 4 = 16
  use_fp16: true  # 啟用混合精度訓練
```

### Pseudo-labeling 配置

```yaml
pseudo_labeling:
  confidence_threshold: 0.9  # 信心閾值
  min_confidence_threshold: 0.7  # 最低閾值
  max_confidence_threshold: 0.95  # 最高閾值
  threshold_decay: 0.98  # 閾值衰減率
  max_pseudo_samples: 8000  # 最大偽標籤數量
  num_iterations: 5  # 迭代次數
  epochs_per_iteration: 2  # 每次迭代的訓練輪次
```

### Self-training 配置

```yaml
self_training:
  teacher_update_frequency: 200  # 教師模型更新頻率
  student_teacher_ratio: 0.7  # 學生損失與教師損失比例
  distillation_temperature: 4.0  # 蒸餾溫度
  ema_decay: 0.999  # EMA 衰減率
  confidence_threshold: 0.8  # 信心閾值
  max_epochs: 8
```

### Consistency Regularization 配置

```yaml
consistency:
  consistency_weight: 1.0  # 一致性損失權重
  consistency_ramp_up_epochs: 3  # 權重增長輪次
  max_consistency_weight: 5.0  # 最大一致性權重
  augmentation_strength: 0.1  # 增強強度
  use_confidence_masking: true  # 使用信心遮罩
  confidence_threshold: 0.8
```

## 使用範例

### 程式化使用

```python
import torch
from transformers import AutoTokenizer
from src.cyberpuppy.semi_supervised import (
    PseudoLabelingPipeline, PseudoLabelConfig
)

# 設定模型和分詞器
model = your_model
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

# 配置 Pseudo-labeling
config = PseudoLabelConfig(
    confidence_threshold=0.9,
    max_pseudo_samples=5000
)

# 建立流水線
pipeline = PseudoLabelingPipeline(model, tokenizer, config, device='cuda')

# 生成偽標籤
pseudo_samples, stats = pipeline.generate_pseudo_labels(
    model, unlabeled_dataloader
)

print(f"生成了 {len(pseudo_samples)} 個偽標籤樣本")
```

### Self-training 範例

```python
from src.cyberpuppy.semi_supervised import (
    SelfTrainingFramework, SelfTrainingConfig
)

# 配置 Self-training
config = SelfTrainingConfig(
    distillation_temperature=4.0,
    ema_decay=0.999
)

# 建立框架
framework = SelfTrainingFramework(config, device='cuda')

# 訓練
history = framework.train(
    student_model=model,
    labeled_dataloader=labeled_loader,
    unlabeled_dataloader=unlabeled_loader,
    validation_dataloader=val_loader,
    optimizer=optimizer,
    criterion=criterion
)
```

### Consistency Regularization 範例

```python
from src.cyberpuppy.semi_supervised import (
    ConsistencyRegularizer, ConsistencyConfig
)

# 配置 Consistency Regularization
config = ConsistencyConfig(
    consistency_weight=1.0,
    augmentation_strength=0.1
)

# 建立正則化器
regularizer = ConsistencyRegularizer(config, device='cuda')

# 訓練步驟
losses = regularizer.mixed_training_step(
    model=model,
    labeled_batch=labeled_batch,
    unlabeled_batch=unlabeled_batch,
    optimizer=optimizer,
    criterion=criterion
)
```

## 文字增強方法

框架提供三種文字增強方法：

1. **Token Dropout**: 隨機將 tokens 替換為 [MASK]
2. **Token Shuffle**: 隨機交換相鄰的 tokens
3. **Synonym Replacement**: 隨機替換為其他 tokens（簡化版同義詞替換）

```python
# 使用範例
from src.cyberpuppy.semi_supervised.consistency import TextAugmenter

augmenter = TextAugmenter(augmentation_strength=0.1)

# 應用增強
augmented_batch = augmenter.augment_batch(original_batch)
```

## 記憶體優化技巧

### 1. 動態批次大小

```python
from scripts.train_semi_supervised import MemoryOptimizer

optimizer = MemoryOptimizer()
batch_size = optimizer.get_dynamic_batch_size(dataset_size, base_batch_size=8)
```

### 2. 梯度累積

```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4  # 有效批次大小 = 16
```

### 3. 混合精度訓練

```yaml
training:
  use_fp16: true
  fp16_opt_level: "O1"
```

### 4. 記憶體清理

```python
# 定期清理 GPU 快取
if batch_idx % 50 == 0:
    MemoryOptimizer.clear_cache()
```

## 評估與監控

### 1. 訓練統計

每個方法都提供詳細的訓練統計：

```python
# Pseudo-labeling 統計
stats = pipeline.get_stats()
print(f"總偽標籤數量: {stats['total_pseudo_labels']}")
print(f"閾值歷史: {stats['threshold_history']}")

# Consistency 統計
consistency_stats = regularizer.get_statistics()
print(f"平均一致性損失: {consistency_stats['avg_consistency_loss']}")
```

### 2. 模型評估

```python
# 一致性評估
metrics = regularizer.evaluate_consistency(
    model, dataloader, num_augmentations=5
)
print(f"一致性分數: {metrics['consistency_score']}")
print(f"預測穩定性: {metrics['prediction_stability']}")
```

## 最佳實踐

### 1. 超參數調整建議

- **Confidence Threshold**: 從 0.9 開始，根據驗證效能調整
- **Consistency Weight**: 從 1.0 開始，逐漸增加到 5.0
- **Distillation Temperature**: 3.0-5.0 之間效果較好
- **EMA Decay**: 0.999 通常是好的起點

### 2. 訓練策略

1. **先用有標籤資料訓練**: 獲得良好的基礎模型
2. **漸進式增加無標籤資料**: 避免低品質偽標籤干擾
3. **監控驗證效能**: 及時停止以避免過擬合
4. **組合多種方法**: 可以依序使用不同的半監督方法

### 3. 記憶體管理

- 監控 GPU 記憶體使用量
- 調整批次大小以避免 OOM
- 定期清理快取
- 使用梯度檢查點減少記憶體需求

## 常見問題

### Q: 訓練過程中出現 OOM 錯誤？

A: 嘗試以下解決方案：
1. 減少 `batch_size`
2. 增加 `gradient_accumulation_steps`
3. 啟用 `fp16` 混合精度訓練
4. 減少 `max_length`

### Q: 偽標籤品質如何評估？

A: 監控以下指標：
1. 高信心樣本比例
2. 驗證集上的效能變化
3. 信心分數分佈
4. 類別平衡性

### Q: 如何調整一致性正則化強度？

A:
1. 從較小的權重開始（1.0）
2. 觀察一致性損失和任務損失的平衡
3. 逐漸增加權重直到驗證效能不再提升

## 進階功能

### 1. 不確定性估計

```python
# 使用 Monte Carlo Dropout 估計不確定性
predictions, uncertainty = framework.predict_with_uncertainty(
    model, dataloader, num_forward_passes=10
)
```

### 2. 模型集成

可以組合多個半監督方法訓練的模型：

```python
# 載入不同方法訓練的模型
model1 = load_model("pseudo_labeling_model.pt")
model2 = load_model("self_training_model.pt")
model3 = load_model("consistency_model.pt")

# 集成預測
ensemble_predictions = (pred1 + pred2 + pred3) / 3
```

## 支援與維護

如有問題或建議，請參考：
- 測試檔案：`tests/test_semi_supervised.py`
- 示例腳本：`scripts/demo_semi_supervised.py`
- 配置檔案：`configs/semi_supervised.yaml`

定期運行測試以確保框架正常運作：

```bash
python -m pytest tests/test_semi_supervised.py -v
```