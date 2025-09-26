# CyberPuppy 實時訓練監控器

## 概述

CyberPuppy 訓練監控器提供美觀的實時訓練進度顯示，支援 GPU 記憶體監控、最佳模型追蹤、Early Stopping 指示等功能。自動偵測 Rich 庫可用性，在不支援時優雅降級到 tqdm。

## 功能特點

✅ **實時指標顯示** - 即時顯示 Loss、F1、Accuracy 等指標
✅ **訓練進度條** - 美觀的 epoch 和 step 進度條與 ETA 估算
✅ **GPU 記憶體監控** - 實時顯示 GPU 記憶體使用情況
✅ **最佳模型追蹤** - 自動追蹤並顯示最佳指標
✅ **Early Stopping 指示** - 視覺化 early stopping 計數器
✅ **指標匯出** - 自動匯出 JSON 格式的訓練指標
✅ **Windows 相容** - 完整支援 Windows 終端環境
✅ **自動降級** - Rich 不可用時自動使用 tqdm

## 快速使用

### 基本用法

```python
from cyberpuppy.training.monitor import TrainingMonitor

# 創建監控器
monitor = TrainingMonitor(
    total_epochs=15,
    steps_per_epoch=100,
    model_name="macbert_aggressive",
    use_rich=True,  # 自動偵測
    export_dir="./training_metrics"
)

# 開始監控
monitor.start_training()

# 訓練循環
for epoch in range(15):
    monitor.start_epoch(epoch)
    monitor.set_early_stopping(patience=3, counter=early_stop_count)

    for step in range(100):
        # ... 訓練代碼 ...

        # 更新指標
        monitor.update_step(
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            val_f1=val_f1,
            val_accuracy=val_accuracy,
            learning_rate=current_lr
        )

    monitor.end_epoch(epoch)  # 自動匯出指標

monitor.stop_monitoring()
```

### 顯示效果

**Rich 版本（支援時）:**
```
═══════════════════════════════════════
  CyberPuppy Local Training Monitor
═══════════════════════════════════════
Model: macbert_aggressive
GPU: RTX 3050 (3.2GB / 4.0GB)

Epoch 5/15 [██████░░░░] 33%
├─ Train Loss: 0.3421 ↓
├─ Dev F1: 0.7234 ↑
├─ Best F1: 0.7401 (Epoch 4)
└─ ETA: 1h 23m

Early Stopping: 1/3
```

**Simple 版本（降級時）:**
```
============================================================
  macbert_aggressive Local Training Monitor
============================================================
Model: macbert_aggressive
GPU: RTX 3050 (0.0GB / 4.0GB)

Epochs: 33%|███▍      | 5/15 [01:23<02:46]
Steps (Epoch 6): 60%|██████    | 60/100 [00:45<00:30] Loss=0.3421, F1=0.7234, Best F1=0.7401, ES=1/3
```

## 主要類別

### TrainingMonitor
主要監控器類別，自動選擇最佳顯示模式。

```python
TrainingMonitor(
    total_epochs: int,
    steps_per_epoch: int,
    model_name: str = "CyberPuppy",
    use_rich: bool = True,
    export_dir: Optional[Path] = None
)
```

### GPUMonitor
GPU 記憶體監控工具。

```python
from cyberpuppy.training.monitor import GPUMonitor

gpu = GPUMonitor()
print(f"GPU: {gpu.get_device_name()}")
print(f"Memory: {gpu.get_memory_stats()}")
```

### MetricsHistory
訓練指標歷史記錄與分析。

```python
from cyberpuppy.training.monitor import MetricsHistory

history = MetricsHistory()
# ... 添加指標 ...
history.export_to_json("metrics.json")
avg_loss = history.get_recent_avg('train_loss', window=10)
```

## 指標匯出

監控器會自動匯出 JSON 格式的訓練指標：

```json
{
  "train_losses": [0.8, 0.6, 0.4],
  "val_f1_scores": [0.65, 0.75, 0.85],
  "val_accuracies": [0.7, 0.75, 0.8],
  "learning_rates": [2e-05, 2e-05, 2e-05],
  "timestamps": ["2025-09-27T03:35:14.396430", ...],
  "epochs": [0, 0, 0],
  "steps": [0, 1, 2],
  "exported_at": "2025-09-27T03:35:17.399843"
}
```

## 系統需求

- **Python**: 3.8+
- **必要依賴**: torch, numpy, tqdm
- **可選依賴**: rich (提供美觀顯示)
- **平台**: Windows, macOS, Linux

## 範例腳本

### 基本演示
```bash
cd examples
python training_monitor_demo.py full
```

### 整合範例
```bash
cd examples
python integration_example.py
```

### 特定功能測試
```bash
cd examples
python training_monitor_demo.py gpu     # GPU 監控
python training_monitor_demo.py export  # 指標匯出
python training_monitor_demo.py simple  # 簡單模式
```

## 注意事項

1. **Rich 自動偵測**: 監控器會自動偵測終端對 Rich 的支援，不支援時自動降級
2. **GPU 記憶體**: 需要 CUDA 環境才能顯示 GPU 記憶體資訊
3. **Windows 相容**: 完整支援 Windows Terminal、ConEmu 等現代終端
4. **效能影響**: 監控開銷極小（<0.1ms per update）

## 技術細節

- **Rich 偵測**: 自動檢查 `WT_SESSION`、`ConEmuANSI` 等環境變數
- **記憶體優化**: 使用 deque 限制歷史記錄大小
- **執行緒安全**: 支援多執行緒環境（未來版本）
- **錯誤處理**: 完整的異常處理與優雅降級

## 未來功能

- [ ] TensorBoard 整合
- [ ] Web 監控面板
- [ ] 分散式訓練支援
- [ ] 自訂指標追蹤
- [ ] 實時圖表顯示