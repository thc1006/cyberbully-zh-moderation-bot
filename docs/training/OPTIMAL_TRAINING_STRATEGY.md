# 🎯 最佳效能訓練策略 - 霸凌偵測 F1 ≥ 0.75

## 📋 策略總覽

**目標**: 達成 F1 ≥ 0.75 的最佳霸凌偵測模型
**方法**: 多模型訓練 + 集成學習 + 智能選擇
**平台**: Google Colab (T4/V100/A100 GPU)
**Git LFS 用量**: ~1.2-1.6 GB (完全在限制內)

---

## 🚀 三階段訓練計劃

### Phase 1: 基礎模型訓練 (3 個模型並行)

訓練三個不同配置的基礎模型，確保探索最佳超參數空間：

#### Model A: 保守配置 (穩定優先)
```yaml
名稱: macbert_conservative
基礎模型: hfl/chinese-macbert-base
學習率: 1e-5 (較低，避免過擬合)
Batch size: 8
訓練輪數: 20 epochs
Early stopping: 5 epochs patience
焦點損失 alpha: 2.0, gamma: 2.5
訓練資料: 完整 77,178 樣本
預期時間: 2-3 小時 (Colab T4)
```

#### Model B: 激進配置 (效能優先)
```yaml
名稱: macbert_aggressive
基礎模型: hfl/chinese-macbert-base
學習率: 3e-5 (較高，快速收斂)
Batch size: 16 (梯度累積 x2)
訓練輪數: 15 epochs
Early stopping: 3 epochs patience
焦點損失 alpha: 2.5, gamma: 3.0
資料增強: 啟用所有策略
預期時間: 2-3 小時
```

#### Model C: RoBERTa 變體 (架構多樣性)
```yaml
名稱: roberta_balanced
基礎模型: hfl/chinese-roberta-wwm-ext
學習率: 2e-5 (中等)
Batch size: 12
訓練輪數: 18 epochs
Early stopping: 4 epochs patience
焦點損失 alpha: 2.2, gamma: 2.8
混合訓練: 50% 原始 + 50% 增強
預期時間: 2.5-3.5 小時
```

**Git LFS 用量**: 3 × 390 MB = **1.17 GB**

---

### Phase 2: 最佳模型精調 (1-2 個模型)

從 Phase 1 選出 F1 最高的 1-2 個模型，進行精調：

```yaml
配置: 基於最佳基礎模型
學習率: 降低至原來的 0.5x
訓練輪數: 10 epochs
Early stopping: 3 epochs patience
特殊技巧:
  - 不確定性加權損失
  - 對抗訓練 (FGM)
  - 學習率餘弦退火
  - 標籤平滑 (0.1)
預期時間: 1-2 小時
```

**額外 Git LFS 用量**: 1-2 × 390 MB = **390-780 MB**

---

### Phase 3: 模型集成 (可選，如單模型未達標)

如果單一模型未達 F1 ≥ 0.75，使用集成策略：

#### 策略 A: Soft Voting Ensemble
```python
# 加權平均多個模型的預測概率
final_pred = (
    0.4 * model_best1.predict_proba(text) +
    0.35 * model_best2.predict_proba(text) +
    0.25 * model_best3.predict_proba(text)
)
```

#### 策略 B: Stacking Ensemble
```python
# 使用 LightGBM 作為元學習器
meta_features = [model1_pred, model2_pred, model3_pred]
final_pred = lgbm_meta_model.predict(meta_features)
```

**優點**: 通常可提升 2-5% F1
**缺點**: 推理時間增加 2-3x
**Git LFS 用量**: 無額外用量（集成邏輯只是代碼）

---

## 📊 Git LFS 用量規劃

### 推送策略：智能分層

#### 必須推送 (高優先級)
```
models/bullying_improved/
├── best_single_model/          # 單一最佳模型
│   ├── model.safetensors      (390 MB)
│   ├── config.json            (<1 MB)
│   ├── tokenizer_config.json  (<1 MB)
│   ├── vocab.txt              (0.1 MB)
│   └── training_metrics.json  (<1 MB)
└── ensemble_models/            # 如果需要集成
    ├── model_1.safetensors    (390 MB)
    ├── model_2.safetensors    (390 MB)
    └── ensemble_config.json   (<1 MB)
```

**總用量**:
- 單模型方案: **392 MB**
- 集成方案: **1.17 GB**

#### 可選推送 (如空間充足)
```
models/experiments/             # 實驗模型
├── macbert_conservative/       (390 MB)
├── macbert_aggressive/         (390 MB)
└── roberta_balanced/           (390 MB)
```

**額外用量**: **1.17 GB**

#### 不推送 (本地保留)
```
- Optimizer states (.optimizer.pt)
- Training checkpoints (checkpoint-*.pt)
- TensorBoard logs
- 中間訓練輸出
```

---

## 🎯 推薦方案：漸進式推送

### 方案 1️⃣: 保守方案 (推薦新手)
```
第一次推送: 最佳單模型 (392 MB)
  → 如果 F1 ≥ 0.75: 完成！
  → 如果 F1 < 0.75: 進入方案 2
```

### 方案 2️⃣: 平衡方案 (推薦大多數情況)
```
第一次推送: 最佳單模型 (392 MB)
第二次推送: Top-3 模型用於集成 (1.17 GB)
  → 如果集成 F1 ≥ 0.75: 完成！
  → 如果仍未達標: 進入方案 3
```

### 方案 3️⃣: 全面方案 (追求極致效能)
```
第一次推送: 所有實驗模型 (1.17 GB)
第二次推送: 精調模型 (390-780 MB)
第三次推送: 最終集成模型 (1.17 GB)
總用量: ~2.7-3.1 GB (仍在 10 GB 限制內)
```

---

## ⚡ Colab 訓練流程優化

### 1. Clone 優化 (避免下載現有大模型)
```bash
# 方法 A: Shallow clone + LFS skip
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://github.com/yourusername/cyberbully-zh-moderation-bot.git

# 方法 B: 只下載必要的 LFS 檔案
git lfs pull --include="data/processed/*" --exclude="models/*"
```

**節省**: ~2.4 GB 下載量

### 2. 訓練資料上傳策略
```python
# 選項 A: 從 repo 讀取 (如果 data/ 較小)
data_dir = "data/processed/training_dataset/"

# 選項 B: 從 Google Drive 讀取 (如果 data/ 較大)
from google.colab import drive
drive.mount('/content/drive')
data_dir = "/content/drive/MyDrive/cyberpuppy_data/"
```

### 3. 模型上傳自動化
```python
# 訓練完成後自動推送最佳模型
def push_best_model(model_path, f1_score):
    if f1_score >= 0.75:
        print(f"✅ 達標！F1 = {f1_score:.4f}")
        # 只推送最佳模型
        os.system(f"git lfs track '{model_path}/*.safetensors'")
        os.system(f"git add {model_path}")
        os.system(f"git commit -m 'feat: Add bullying model F1={f1_score:.4f}'")
        os.system("git push origin main")
    else:
        print(f"⚠️ 未達標 F1 = {f1_score:.4f}，繼續訓練...")
```

---

## 📈 預期效能與時間

| 階段 | 模型數 | 訓練時間 (T4) | 訓練時間 (V100) | Git LFS 用量 | 預期 F1 |
|------|--------|---------------|-----------------|--------------|---------|
| Phase 1 | 3 | 6-9 小時 | 3-5 小時 | 1.17 GB | 0.65-0.75 |
| Phase 2 | 1-2 | 1-2 小時 | 0.5-1 小時 | +0.39-0.78 GB | 0.72-0.78 |
| Phase 3 | 集成 | 0.5 小時 | 0.2 小時 | 0 GB | 0.75-0.82 |
| **總計** | - | **7-12 小時** | **4-6 小時** | **1.6-2.0 GB** | **≥ 0.75** |

---

## 🎯 最終推薦：智能漸進策略

### 第一輪：快速驗證 (2-3 小時)
1. 訓練 Model B (激進配置)
2. 如果 F1 ≥ 0.75 → 推送，完成！
3. 如果 F1 < 0.75 → 進入第二輪

### 第二輪：全面探索 (6-9 小時)
1. 並行訓練 Model A + Model C
2. 選出 Top-2 模型
3. 如果任一 F1 ≥ 0.75 → 推送最佳，完成！
4. 如果都 < 0.75 → 進入第三輪

### 第三輪：精調與集成 (2-3 小時)
1. 精調 Top-2 模型
2. 建立 Soft Voting Ensemble
3. 如果集成 F1 ≥ 0.75 → 推送集成，完成！
4. 如果仍 < 0.75 → 分析問題，調整策略

---

## 💾 Git LFS 最終用量預測

| 場景 | 推送內容 | LFS 存儲 | LFS 頻寬 | 成功率 |
|------|---------|---------|---------|-------|
| 樂觀 | 單一最佳模型 | +0.39 GB | +0.39 GB | 40% |
| 標準 | Top-2 精調模型 | +0.78 GB | +0.78 GB | 80% |
| 保守 | Top-3 + 集成 | +1.17 GB | +1.17 GB | 95% |
| 極限 | 所有實驗模型 | +2.34 GB | +2.34 GB | 99% |

**你的可用空間**: 9.2 GB 存儲 + 9.8 GB 頻寬

✅ **結論**: 即使在極限情況下，也只用 25% 的空間和頻寬！

---

## 🚀 立即執行

接下來我會建立：
1. ✅ **Google Colab 訓練筆記本** (`notebooks/train_on_colab.ipynb`)
2. ✅ **智能推送腳本** (`scripts/smart_push_model.py`)
3. ✅ **訓練監控儀表板** (TensorBoard + WandB)
4. ✅ **模型集成工具** (`src/cyberpuppy/ensemble/`)

準備好了嗎？我現在開始建立完整的 Colab 訓練系統！