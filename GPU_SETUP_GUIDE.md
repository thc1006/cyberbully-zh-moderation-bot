# GPU 加速設置指南

## 🎯 當前狀況

您的系統配備 **NVIDIA GeForce RTX 3050 (4GB VRAM)**，但目前 PyTorch 安裝的是 CPU 版本。需要重新安裝 GPU 版本的 PyTorch。

## 📦 安裝步驟

### 步驟 1: 卸載當前 CPU 版本的 PyTorch

```bash
pip uninstall torch torchvision torchaudio -y
```

### 步驟 2: 安裝 CUDA 版本的 PyTorch

根據您的 NVIDIA-SMI 顯示，您有 CUDA 13.0，但 PyTorch 目前最高支援 CUDA 12.4。請執行以下命令：

```bash
# 安裝 CUDA 12.4 版本的 PyTorch (最新穩定版)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 或者如果上面失敗，嘗試 CUDA 12.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 步驟 3: 驗證安裝

創建並運行以下測試腳本：

```python
# test_gpu.py
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory:", f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # 測試 GPU 運算
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("GPU computation test: PASSED")
else:
    print("GPU not detected!")
```

運行測試：
```bash
python test_gpu.py
```

## 🔧 如果仍然無法偵測 GPU

### 選項 A: 檢查 CUDA 工具包

1. 下載並安裝 CUDA Toolkit 12.4：
   - 前往：https://developer.nvidia.com/cuda-12-4-0-download-archive
   - 選擇：Windows → x86_64 → 11 → exe (local)
   - 下載並安裝（約 3GB）

2. 設置環境變數（安裝後自動設置，但可以檢查）：
   ```
   CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
   PATH 包含 %CUDA_PATH%\bin
   ```

3. 重新啟動命令提示字元或 PowerShell

### 選項 B: 使用 Conda 環境（推薦）

如果 pip 安裝有問題，使用 Conda 更容易管理 CUDA 依賴：

```bash
# 安裝 Miniconda (如果沒有)
# 下載：https://docs.conda.io/en/latest/miniconda.html

# 創建新環境並安裝 GPU 版 PyTorch
conda create -n cyberpuppy python=3.11
conda activate cyberpuppy
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 安裝其他依賴
pip install -r requirements.txt
```

## 📊 GPU 資源優化建議

您的 RTX 3050 有 4GB VRAM，訓練時建議：

1. **批次大小**：使用較小的 batch_size（建議 8-16）
2. **序列長度**：限制 max_length 在 128-256
3. **混合精度訓練**：使用 FP16 節省記憶體

```python
# 混合精度訓練範例
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    optimizer.zero_grad()
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## ✅ 預期結果

安裝成功後，您應該看到：
```
PyTorch version: 2.5.x+cu124
CUDA available: True
CUDA version: 12.4
GPU Name: NVIDIA GeForce RTX 3050 Laptop GPU
GPU Memory: 4.0 GB
GPU computation test: PASSED
```

## 🚀 完成後

1. 重新運行 `python train_gpu.py` 進行 GPU 加速訓練
2. API 將自動使用 GPU 進行推理（如果配置正確）
3. 訓練速度將提升 5-10 倍

## ⚠️ 注意事項

- 確保 NVIDIA 驅動程式是最新的（您的 581.29 版本是最新的）
- 如果遇到 DLL 錯誤，可能需要安裝 Visual C++ Redistributable
- GPU 訓練時會增加系統功耗和溫度，確保散熱良好