# 霸凌偵測模型訓練指南

## 🎯 目標

使用改進的架構和增強資料訓練霸凌偵測模型，達成以下目標：

- **霸凌偵測 F1 ≥ 0.75**
- **毒性偵測 F1 ≥ 0.78**
- **整體 Macro F1 ≥ 0.76**

## 🚀 快速開始

### 1. 環境準備

```bash
# 確保Python 3.8+
python --version

# 安裝依賴
pip install torch transformers scikit-learn numpy pandas PyYAML tqdm tensorboard matplotlib seaborn

# 檢查GPU (推薦RTX 3050或更好)
nvidia-smi
```

### 2. 資料準備

確保訓練資料位於以下路徑之一：
- `data/processed/training_dataset/train.json`
- `data/processed/cold/train.json`

資料格式：
```json
{
  "text": "這是一段中文文本",
  "toxicity": 0,     // 0:none, 1:toxic, 2:severe
  "bullying": 1,     // 0:none, 1:harassment, 2:threat
  "role": 2,         // 0:none, 1:perpetrator, 2:victim, 3:bystander
  "emotion": 1       // 0:negative, 1:neutral, 2:positive
}
```

### 3. 執行訓練

#### Windows:
```cmd
scripts\run_bullying_f1_training.bat
```

#### Linux/Mac:
```bash
chmod +x scripts/run_bullying_f1_training.sh
./scripts/run_bullying_f1_training.sh
```

#### Python直接執行:
```bash
python scripts/train_bullying_f1_optimizer.py \
  --config configs/training/bullying_f1_optimization.yaml \
  --experiment-name bullying_improvement_v1
```

## 📊 監控訓練

### TensorBoard視覺化

```bash
# 自動啟動TensorBoard
python scripts/launch_tensorboard.py

# 手動啟動
tensorboard --logdir experiments/bullying_f1_optimization/[實驗名稱]/tensorboard_logs
```

在瀏覽器打開 http://localhost:6006 查看：
- 訓練/驗證損失曲線
- F1分數變化
- 學習率排程
- GPU使用率

### 實時監控

```bash
# 監控訓練進度
python scripts/launch_tensorboard.py --monitor --experiment [實驗名稱]
```

## ⚙️ 配置參數

### 核心配置 (`configs/training/bullying_f1_optimization.yaml`)

```yaml
model:
  base_model: "hfl/chinese-macbert-base"
  use_focal_loss: true          # 焦點損失處理類別不平衡
  use_class_weights: true       # 自動計算類別權重
  task_weights:
    bullying: 1.5               # 重點優化霸凌任務

training:
  batch_size: 4                 # RTX 3050優化
  gradient_accumulation_steps: 8 # 等效batch_size=32
  learning_rate: 2e-5
  num_epochs: 15
  fp16: true                    # 混合精度訓練

optimization:
  early_stopping:
    patience: 3
    metric: "bullying_f1"       # 以霸凌F1為主要指標
```

### RTX 3050 記憶體優化

- `batch_size: 4` + `gradient_accumulation_steps: 8`
- `max_length: 384` (降低序列長度)
- `fp16: true` (混合精度)
- `gradient_checkpointing: true`

## 📈 結果分析

### 訓練完成後

訓練完成會生成：
```
experiments/bullying_f1_optimization/[實驗名稱]/
├── final_results.json          # 最終結果摘要
├── checkpoints/
│   ├── best.ckpt              # 最佳模型
│   └── last.ckpt              # 最後檢查點
├── tensorboard_logs/          # TensorBoard日誌
├── model_artifacts/           # 模型檔案
│   ├── model/                 # PyTorch模型
│   ├── tokenizer/             # Tokenizer
│   └── training_config.yaml   # 訓練配置
└── training.log               # 訓練日誌
```

### 目標達成檢查

```python
import json

with open('experiments/[實驗]/final_results.json', 'r') as f:
    results = json.load(f)

targets = results['target_achieved']
print(f"霸凌F1≥0.75: {'✅' if targets['bullying_f1_075'] else '❌'}")
print(f"毒性F1≥0.78: {'✅' if targets['toxicity_f1_078'] else '❌'}")
print(f"總體F1≥0.76: {'✅' if targets['overall_macro_f1_076'] else '❌'}")
```

## 🔧 調參建議

### 如果霸凌F1未達標

1. **增加霸凌任務權重**
   ```yaml
   task_weights:
     bullying: 2.0  # 增加至2.0
   ```

2. **調整焦點損失參數**
   ```yaml
   focal_loss:
     alpha: [0.2, 0.8, 1.2]  # 對霸凌類別加重
     gamma: 3.0              # 增加難樣本聚焦
   ```

3. **增加訓練輪數**
   ```yaml
   training:
     num_epochs: 20
   ```

### 如果出現過擬合

1. **增加正規化**
   ```yaml
   model:
     dropout_rate: 0.2
   training:
     weight_decay: 0.02
   ```

2. **調整學習率**
   ```yaml
   training:
     learning_rate: 1e-5
   ```

### 記憶體不足

1. **降低batch size**
   ```yaml
   training:
     batch_size: 2
     gradient_accumulation_steps: 16
   ```

2. **降低序列長度**
   ```yaml
   model:
     max_length: 256
   ```

## 🔍 進階功能

### 多實驗比較

```bash
# 比較最近的實驗
python scripts/launch_tensorboard.py --compare

# 比較特定實驗
python scripts/launch_tensorboard.py -e experiment1 -e experiment2
```

### 超參數搜尋

```bash
# 使用超參數搜尋配置
python scripts/train_bullying_f1_optimizer.py \
  --config configs/training/hyperparameter_search.yaml
```

### 模型集成

```python
# 載入多個最佳模型進行集成
from src.cyberpuppy.models.ensemble import ModelEnsemble

ensemble = ModelEnsemble([
    "experiments/exp1/model_artifacts/model",
    "experiments/exp2/model_artifacts/model",
    "experiments/exp3/model_artifacts/model"
])
```

## ⚠️ 常見問題

### 1. CUDA記憶體不足
- 降低batch_size
- 啟用gradient_checkpointing
- 使用fp16混合精度

### 2. 訓練速度太慢
- 檢查num_workers設定
- 確保使用GPU
- 減少序列長度

### 3. F1分數不收斂
- 檢查學習率是否過大
- 增加warmup步數
- 調整類別權重

### 4. 驗證損失上升
- 可能過擬合，增加正規化
- 降低學習率
- 早停機制會自動處理

## 📝 實驗記錄

建議為每次實驗記錄：
- 配置檔案變更
- 資料預處理方式
- 最終指標結果
- 觀察到的現象
- 改進方向

使用實驗名稱包含版本號：
```
bullying_f1_v1_baseline
bullying_f1_v2_focal_loss
bullying_f1_v3_class_weights
```

## 🎯 目標達成策略

1. **基線模型** - 使用預設配置建立基線
2. **焦點損失** - 啟用焦點損失處理類別不平衡
3. **類別權重** - 自動計算並應用類別權重
4. **任務權重** - 調整霸凌任務權重
5. **超參調優** - 精細調整學習率和正規化
6. **模型集成** - 組合多個最佳模型

通過系統性的方法，應該能夠達成 **霸凌F1≥0.75** 的目標！