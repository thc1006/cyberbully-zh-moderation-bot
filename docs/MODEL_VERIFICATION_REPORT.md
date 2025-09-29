# 模型性能驗證報告

**日期**: 2025-09-29
**驗證工具**: `scripts/verify_model_performance.py`
**測試集**: `data/processed/training_dataset/test.json` (5,320 樣本)

---

## 執行摘要

⚠️ **關鍵發現**: 實際測試的模型性能與專案文檔中聲稱的性能存在**巨大差距**。

## 測試結果

### 1. ~~`models/gpu_trained_model`~~ 【已刪除】

**測試狀態**: ✅ 成功（有完整模型權重）

**實際性能**:
- **Macro F1**: 0.2788 (27.88%)
- **Weighted F1**: 0.4871 (48.71%)
- **Accuracy**: 0.6064 (60.64%)

**聲稱性能** (README.md):
- F1 = 0.77 (77%)

**性能差距**:
- **-63.8%** (實際 0.28 vs 聲稱 0.77)
- **遠低於目標 F1≥0.78**

#### 詳細分類結果

| 類別 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| Class_0 (none) | 0.609 | **0.973** | 0.749 | 3,214 |
| Class_1 (toxic/harassment) | 0.532 | **0.047** | **0.087** | 2,106 |
| Class_2 (severe/threat) | 0.000 | 0.000 | 0.000 | 0 |

**問題分析**:
1. **極度偏向預測 Class_0** - Recall 高達 97.3%
2. **幾乎無法檢測實際毒性內容** - Class_1 的 Recall 僅 4.7%
3. **Class_2 完全無法預測** - 測試集中沒有該類別樣本
4. **模型行為**: 將大部分樣本（包括有毒內容）錯誤分類為 "none"

**結論**: ❌ **此模型不適合生產使用**

---

### 2. `models/working_toxicity_model` ❌ 無法載入

**測試狀態**: ❌ 失敗

**錯誤原因**:
- `config.json` 缺少 `model_type` 欄位
- 使用自定義格式，不符合 HuggingFace 標準
- 無法使用 `AutoModelForSequenceClassification` 載入

**config.json 內容**:
```json
{
  "model_name": "hfl/chinese-macbert-base",
  "max_length": 256,
  "num_toxicity_classes": 3,
  "num_bullying_classes": 3,
  "num_role_classes": 4,
  "num_emotion_classes": 3,
  "use_emotion_regression": false,
  "task_weights": {
    "toxicity": 1.0,
    "bullying": 0.0,
    "role": 0.0,
    "emotion": 0.0
  }
}
```

**結論**: 需要自定義載入程式碼才能使用此模型

---

### 3. `models/bullying_a100_best` ⚠️ 無模型權重

**測試狀態**: ⚠️ 未測試（無權重檔案）

**聲稱性能** (final_results.json):
- Toxicity F1 = 0.8206
- Bullying F1 = 0.8207

**實際狀態**:
- 目錄大小: 595KB (應為 ~400MB)
- **缺少**: `.safetensors` / `.bin` 模型權重檔案
- **僅有**: tokenizer + JSON 評估結果

**結論**: ❌ **無法驗證性能聲稱，需重新訓練並正確保存**

---

### 4. `models/local_training/macbert_aggressive` ❌ 訓練失敗

**訓練記錄分析**:
- 3 個 epoch，驗證集性能下降（過擬合）
- 最佳驗證準確率: 33.5% (Epoch 3)
- 訓練集準確率: 62.9% (Epoch 2) 但驗證集僅 31%

**結論**: 訓練配置不當，未收斂

---

## 總結與建議

### 🚨 嚴重問題

1. **舊模型已移除**: 原 `gpu_trained_model` 因性能不佳已完全刪除
2. **最佳模型缺失權重**: A100 訓練的最佳模型 (F1=0.82) 無法使用
3. **無可用生產模型**: 所有測試的模型均未達到目標 F1≥0.75

### 🎯 立即行動

1. **修正文檔**:
   - 更新 README.md 中的性能數據
   - 已刪除所有不適合使用的舊模型
   - 移除對不存在模型的引用

2. **重新訓練**:
   - 使用更新後的 `notebooks/train_on_colab_a100.ipynb`
   - 確保模型權重正確保存 (~400MB)
   - 驗證 Git LFS 正確追蹤大檔案

3. **驗證流程**:
   - 訓練後立即執行 `scripts/verify_model_performance.py`
   - 確認實際 F1 分數≥目標值
   - 推送前再次驗證模型可載入

### 📊 測試環境

- **GPU**: CUDA enabled (測試時使用)
- **PyTorch**: 支援 safetensors 格式
- **測試樣本數**: 5,320
  - Class_0 (none): 3,214 (60.5%)
  - Class_1 (toxic/harassment): 2,106 (39.5%)
  - Class_2 (severe/threat): 0 (0%)

### 📎 相關檔案

- 驗證腳本: `scripts/verify_model_performance.py`
- 完整結果: `models/performance_verification_results.json`
- 訓練 Notebook: `notebooks/train_on_colab_a100.ipynb`

---

**報告生成**: 2025-09-29
**驗證工具版本**: 1.0
**測試裝置**: CUDA GPU