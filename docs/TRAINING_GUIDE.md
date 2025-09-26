# CyberPuppy 訓練系統使用指南

## 概述

CyberPuppy 訓練系統是一個專為中文霸凌偵測設計的完整訓練管理平台，特別針對 RTX 3050 4GB 等低記憶體 GPU 進行了優化。

## 主要功能

### 🚀 核心特性
- **記憶體優化**: 專為 RTX 3050 4GB 設計，支援自動批次大小調整
- **混合精度訓練**: FP16 大幅降低記憶體佔用
- **動態批次管理**: 自動處理 OOM 錯誤並調整批次大小
- **實驗追蹤**: 完整的訓練指標記錄和實驗管理
- **早停機制**: 防止過擬合，節省訓練時間
- **檢查點恢復**: 支援斷點續訓，避免意外中斷損失

### 📊 監控系統
- **即時指標**: 訓練損失、學習率、GPU記憶體使用
- **TensorBoard**: 視覺化訓練過程
- **記憶體監控**: 即時 GPU 記憶體使用情況
- **進度追蹤**: 彩色進度條顯示訓練狀態

## 快速開始

### 1. 環境準備

```bash
# 安裝依賴
pip install torch torchvision transformers
pip install scikit-learn tqdm tensorboard
pip install pyyaml psutil

# 檢查 GPU 環境
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. 基本訓練

```bash
# 使用預設配置開始訓練
python scripts/train_improved_model.py

# 使用 RTX 3050 優化配置
python scripts/train_improved_model.py --config configs/training/rtx3050_optimized.yaml

# 快速開發測試
python scripts/train_improved_model.py --template fast_dev --experiment-name my_test
```

### 3. 自定義參數

```bash
# 調整關鍵參數
python scripts/train_improved_model.py \
    --model-name hfl/chinese-roberta-wwm-ext \
    --batch-size 6 \
    --learning-rate 3e-5 \
    --num-epochs 15 \
    --experiment-name roberta_experiment

# 啟用 GPU 和混合精度
python scripts/train_improved_model.py \
    --gpu \
    --fp16 \
    --experiment-name gpu_fp16_test
```

## 配置系統

### 配置模板

系統提供 4 個預設模板：

1. **default**: 平衡的預設配置
2. **fast_dev**: 快速開發測試（3 epochs）
3. **production**: 生產環境配置（20 epochs）
4. **memory_efficient**: RTX 3050 專用優化

### YAML 配置檔案

每個配置檔案包含 6 個主要部分：

```yaml
model:
  name: "hfl/chinese-macbert-base"
  num_labels: 3
  dropout_rate: 0.1
  max_sequence_length: 512

data:
  train_path: "data/processed/train.json"
  val_path: "data/processed/val.json"
  batch_size: 8
  num_workers: 2

training:
  num_epochs: 10
  learning_rate: 2.0e-5
  fp16: true
  gradient_accumulation_steps: 4

optimization:
  gradient_checkpointing: true
  max_grad_norm: 1.0
  optimizer: "AdamW"

callbacks:
  early_stopping_patience: 3
  early_stopping_metric: "eval_f1_macro"
  save_top_k: 2

experiment:
  name: "my_experiment"
  seed: 42
  log_level: "INFO"
```

## RTX 3050 4GB 優化策略

### 自動記憶體優化

系統會自動檢測 GPU 記憶體並應用優化：

```python
# 自動啟用的優化項目
- 批次大小限制在 8 以下
- 梯度累積步數增加到 2+
- 啟用梯度檢查點
- 禁用 pin_memory
- 啟用 FP16 混合精度
```

### 手動優化配置

針對 4GB 記憶體的最佳實踐：

```yaml
data:
  batch_size: 4                    # 小批次
  gradient_accumulation_steps: 8   # 高累積
  num_workers: 1                   # 降低 CPU 負載
  pin_memory: false               # 釋放記憶體

model:
  max_sequence_length: 384        # 縮短序列長度
  gradient_checkpointing: true    # 必須啟用

training:
  fp16: true                      # 必須啟用

optimization:
  memory_efficient_attention: true
```

### 記憶體監控

訓練過程中實時顯示記憶體使用：

```
Step 100 | Loss: 0.4521 | LR: 1.25e-05 | GPU Mem: 3.2GB/4GB
```

當記憶體使用超過 3.5GB 時會自動發出警告。

## 實驗管理

### 實驗目錄結構

```
experiments/
├── my_experiment_20241127_143052/
│   ├── config.yaml              # 完整配置
│   ├── config.json             # JSON 格式配置
│   ├── checkpoints/            # 檢查點檔案
│   ├── tensorboard/            # TensorBoard 日誌
│   ├── metrics.json           # 訓練指標
│   ├── training_summary.json  # 訓練摘要
│   └── final_model.pt         # 最終模型
```

### 檢查點管理

系統自動保存最佳檢查點：

```python
# 檢查點包含內容
- 模型權重
- 優化器狀態
- 調度器狀態
- 訓練指標
- 完整配置
```

### 斷點續訓

```bash
# 從檢查點恢復訓練
python scripts/train_improved_model.py \
    --config experiments/my_exp/config.yaml \
    --resume-from-checkpoint experiments/my_exp/checkpoints/best_model.pt
```

## 多任務訓練

### 支援的任務

1. **毒性偵測**: 3 類別（無、有毒、嚴重）
2. **霸凌偵測**: 3 類別（無、騷擾、威脅）
3. **角色識別**: 4 類別（無、施暴者、受害者、旁觀者）
4. **情緒分析**: 3 類別（正面、中性、負面）

### 任務權重配置

```yaml
model:
  use_multitask: true
  task_weights:
    toxicity: 1.0     # 主要任務
    bullying: 1.0     # 主要任務
    emotion: 0.5      # 輔助任務
    role: 0.5         # 輔助任務
```

## 資料格式

### 訓練資料格式

```json
[
  {
    "text": "這是要分析的文本內容",
    "toxicity_labels": 0,     // 0: 無毒, 1: 有毒, 2: 嚴重
    "bullying_labels": 0,     // 0: 無霸凌, 1: 騷擾, 2: 威脅
    "emotion_labels": 1,      // 0: 負面, 1: 中性, 2: 正面
    "role_labels": 0          // 0: 無, 1: 施暴, 2: 受害, 3: 旁觀
  }
]
```

### 簡化格式（單任務）

```json
[
  {
    "text": "這是要分析的文本內容",
    "labels": 1  // 主要任務標籤
  }
]
```

## 進階功能

### 自動批次大小尋找

```yaml
training:
  auto_batch_size: true      # 啟用自動尋找
  batch_size: 8             # 起始大小
```

系統會自動測試不同批次大小，找到記憶體允許的最大值。

### 動態批次調整

訓練過程中遇到 OOM 時自動降低批次大小：

```
OOM detected, reducing batch size from 8 to 4
```

### 學習率調度

支援多種調度策略：

```yaml
training:
  lr_scheduler: "cosine"    # cosine, linear, polynomial
  warmup_ratio: 0.1        # 10% 預熱
```

### 梯度累積

模擬大批次訓練：

```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 8
  # 有效批次大小 = 4 × 8 = 32
```

## 故障排除

### 常見問題

1. **OOM 錯誤**
   ```bash
   # 解決方案：降低批次大小或啟用更多優化
   --batch-size 2 --fp16
   ```

2. **訓練緩慢**
   ```bash
   # 解決方案：增加 num_workers（但不要超過 CPU 核心數）
   --num-workers 2
   ```

3. **指標不收斂**
   ```bash
   # 解決方案：調整學習率或增加預熱
   --learning-rate 1e-5 --warmup-ratio 0.2
   ```

### 效能調優

1. **記憶體優化**
   - 啟用 gradient_checkpointing
   - 使用 FP16 混合精度
   - 降低 max_sequence_length
   - 減少 num_workers

2. **速度優化**
   - 增加 batch_size（在記憶體允許下）
   - 啟用 pin_memory（大記憶體系統）
   - 使用快速的 SSD 儲存

3. **穩定性優化**
   - 適當的學習率預熱
   - 梯度裁剪
   - 早停機制

## 監控和可視化

### TensorBoard

```bash
# 啟動 TensorBoard
tensorboard --logdir experiments/my_experiment/tensorboard

# 在瀏覽器中查看
http://localhost:6006
```

### 指標追蹤

系統自動記錄的指標：

- 訓練損失 (train_loss)
- 驗證損失 (eval_loss)
- F1 分數 (eval_f1_macro)
- 準確率 (eval_accuracy)
- 學習率 (learning_rate)
- GPU 記憶體使用 (gpu_memory)

## 最佳實踐

### 實驗設計

1. **起始實驗**: 使用 `fast_dev` 模板快速驗證
2. **參數搜索**: 使用 `hyperparameter_search` 配置
3. **正式訓練**: 使用 `production` 配置
4. **記憶體受限**: 使用 `memory_efficient` 配置

### 資料準備

1. **資料清理**: 移除過短或過長的文本
2. **標籤平衡**: 確保各類別樣本數量均衡
3. **交叉驗證**: 使用分層抽樣分割資料

### 訓練策略

1. **漸進式訓練**: 從小模型開始，逐步增加複雜度
2. **多階段訓練**: 先預訓練再微調
3. **集成學習**: 訓練多個模型並融合結果

## 擴展和自訂

### 添加新任務

1. 修改模型架構
2. 更新資料載入器
3. 調整損失函數
4. 修改評估指標

### 自訂回調

```python
from src.cyberpuppy.training.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def on_epoch_end(self, trainer, **kwargs):
        # 自訂邏輯
        pass
```

### 整合外部工具

- MLflow 實驗追蹤
- Weights & Biases 可視化
- Optuna 超參數優化

這個訓練系統為 CyberPuppy 提供了一個完整、高效、易用的訓練解決方案，特別適合資源受限的環境。