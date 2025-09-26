# 🚀 Google Colab 訓練完整指南

## 📋 快速開始

### 1. 準備工作

**需要準備的資訊**:
- GitHub 用戶名
- GitHub Personal Access Token (需要 `repo` 權限)
  - 取得方式: GitHub → Settings → Developer settings → Personal access tokens → Generate new token
  - 勾選權限: `repo` (完整權限)

**Colab GPU 選擇**:
- 免費版: T4 (16GB) - 訓練時間 6-9 小時
- Pro: V100 (16GB) - 訓練時間 3-5 小時
- Pro+: A100 (40GB) - 訓練時間 2-3 小時

### 2. 開啟訓練筆記本

1. 打開 `notebooks/train_on_colab.ipynb`
2. 上傳到 Google Colab:
   - 方法 A: 直接在 Colab 中開啟 GitHub 檔案
   - 方法 B: 下載後上傳到 Colab
3. 確認 GPU 啟用: Runtime → Change runtime type → GPU

### 3. 執行訓練

點擊 **Runtime → Run all** 或逐步執行每個 cell

**執行流程**:
1. ✅ 檢查 GPU
2. ✅ Clone repository (自動跳過大模型，省流量)
3. ✅ 安裝依賴
4. ✅ 訓練 3 個模型 (並行或串行)
5. ✅ 選出最佳模型
6. ✅ 自動推送 (如果 F1 ≥ 0.75)

---

## 📊 Git LFS 用量詳解

### 現有用量
```
Git LFS 存儲: 0.8 GB / 10 GB (8%)
Git LFS 頻寬: 0.2 GB / 10 GB (2%)
```

### 訓練後新增用量

#### 方案 A: 單一最佳模型 (推薦)
```
新增模型: 390 MB
配置檔案: <1 MB
-------------------
存儲總計: 1.19 GB / 10 GB (12%) ✅
頻寬總計: 0.59 GB / 10 GB (6%)  ✅
```

#### 方案 B: Top-3 模型集成
```
3 個模型: 1.17 GB
配置檔案: <1 MB
-------------------
存儲總計: 1.97 GB / 10 GB (20%) ✅
頻寬總計: 1.37 GB / 10 GB (14%) ✅
```

#### 方案 C: 所有實驗模型 (保留選項)
```
3 個基礎模型: 1.17 GB
2 個精調模型: 780 MB
集成配置: <1 MB
------------------------
存儲總計: 2.75 GB / 10 GB (28%) ✅
頻寬總計: 2.15 GB / 10 GB (22%) ✅
```

### ✅ 結論
**你的 Git LFS 空間非常充足！**
- 即使推送所有實驗模型，也只用 28% 存儲和 22% 頻寬
- 5 天後配額重置，可以進行多輪實驗
- 推薦方案 A（單一最佳模型）最經濟實用

---

## 🎯 訓練策略說明

### Phase 1: 探索階段 (6-9 小時 on T4)

訓練 3 個不同配置的模型：

**Model A - 保守配置**
```yaml
目標: 穩定性優先，避免過擬合
學習率: 1e-5 (低)
Batch size: 8
訓練輪數: 20 epochs
Early stopping: 5 epochs patience
預期 F1: 0.68-0.73
```

**Model B - 激進配置**
```yaml
目標: 快速收斂，追求高效能
學習率: 3e-5 (高)
Batch size: 16 (梯度累積 x2)
訓練輪數: 15 epochs
Early stopping: 3 epochs patience
資料增強: 全開
預期 F1: 0.70-0.76
```

**Model C - RoBERTa 變體**
```yaml
目標: 架構多樣性
基礎模型: chinese-roberta-wwm-ext
學習率: 2e-5 (中等)
Batch size: 12
訓練輪數: 18 epochs
預期 F1: 0.69-0.74
```

### Phase 2: 精調階段 (可選，1-2 小時)

如果 Phase 1 最佳模型 < 0.75:
- 選出 F1 最高的模型
- 降低學習率至原來的 0.5x
- 加入對抗訓練、標籤平滑等技巧
- 預期提升: +0.02-0.05

### Phase 3: 集成階段 (可選，0.5 小時)

如果單模型仍 < 0.75:
- 選出 Top-3 模型
- Soft Voting Ensemble
- 預期提升: +0.02-0.04

---

## 🔧 Colab 筆記本功能

### 自動化功能

1. **智能 Clone**
   ```python
   # 跳過現有大模型，只下載訓練資料
   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1
   git lfs pull --include="data/*" --exclude="models/*"
   ```
   節省下載: ~2.4 GB

2. **自動達標檢測**
   ```python
   if best_f1 >= 0.75:
       # 自動推送到 GitHub
       git push origin main
   ```

3. **TensorBoard 實時監控**
   - 自動啟動，無需手動配置
   - 實時查看訓練曲線、損失、指標

4. **錯誤恢復**
   - Colab 斷線自動保存 checkpoint
   - 可從中斷處繼續訓練

### 手動控制選項

**訓練單一模型** (快速測試):
```python
# 只執行 Model B (激進配置)
# 跳過其他兩個模型的 cell
```

**調整超參數**:
```python
# 修改 TrainingConfig
configs[1].learning_rate = 2.5e-5  # 調整學習率
configs[1].num_epochs = 20         # 增加訓練輪數
```

**啟用 WandB** (更好的實驗追蹤):
```python
import wandb
wandb.login()  # 輸入 WandB API key
# 訓練時自動上傳數據
```

---

## 📈 預期結果

### 樂觀情境 (40% 機率)
```
Model B 單次訓練即達標
F1 Score: 0.75-0.78
訓練時間: 2-3 小時 (T4)
Git LFS: +390 MB
結果: ✅ 立即推送，任務完成
```

### 標準情境 (40% 機率)
```
Phase 1 最佳 F1: 0.72-0.74
需要 Phase 2 精調
總訓練時間: 7-11 小時 (T4)
Git LFS: +780 MB
結果: ✅ 精調後達標，推送
```

### 保守情境 (15% 機率)
```
Phase 1+2 仍未達 0.75
需要 Phase 3 集成
總訓練時間: 8-12 小時 (T4)
Git LFS: +1.17 GB
結果: ✅ 集成後達標，推送
```

### 需要重新評估 (5% 機率)
```
集成後仍 < 0.75
可能原因: 資料品質、標籤問題
建議: 深度錯誤分析，調整策略
```

---

## 🛠️ 本地使用智能推送腳本

訓練完成後，也可以在本地使用智能推送腳本：

### 推送單一最佳模型
```bash
python scripts/smart_push_model.py \
  --strategy best_only \
  --target-f1 0.75
```

### 推送 Top-3 集成模型
```bash
python scripts/smart_push_model.py \
  --strategy top_3 \
  --target-f1 0.70
```

### 預演模式（不實際推送）
```bash
python scripts/smart_push_model.py \
  --strategy best_only \
  --dry-run
```

---

## 🚨 常見問題

### Q1: Colab 斷線了怎麼辦？
**A**:
- Colab 免費版會在 12 小時後斷線
- 訓練會自動保存 checkpoint
- 重新連線後可以繼續訓練
- 建議: 使用 Pro 版本 (24 小時連續運行)

### Q2: 如何節省 Colab GPU 時間？
**A**:
1. 先訓練 Model B (最有可能達標)
2. 達標後跳過其他模型
3. 使用 `--early-stopping` 提前終止
4. 啟用 FP16 混合精度訓練

### Q3: Git LFS 頻寬用完了怎麼辦？
**A**:
- 5 天後自動重置
- 可以先下載模型到本地，稍後再推送
- 使用 `--dry-run` 模式預覽推送內容

### Q4: 如何確認模型已正確推送？
**A**:
```bash
# 查看 LFS 追蹤的檔案
git lfs ls-files

# 查看 LFS 用量
git lfs status
```

### Q5: 可以同時訓練多個模型嗎？
**A**:
- Colab 單一 session 建議串行訓練
- 可以開啟多個 Colab notebook 並行訓練
- 免費版同時只能運行 1 個 GPU session

---

## 📞 技術支援

**問題排查順序**:
1. 檢查 TensorBoard 訓練曲線
2. 查看 `evaluation_results/` 錯誤分析
3. 閱讀 `docs/training/OPTIMAL_TRAINING_STRATEGY.md`
4. 調整超參數重新訓練

**相關文件**:
- `docs/training/OPTIMAL_TRAINING_STRATEGY.md` - 完整訓練策略
- `docs/datasets/DATA_CONTRACT.md` - 資料格式說明
- `docs/technical/BULLYING_DETECTION_IMPROVEMENT_GUIDE.md` - 效能改進指南

---

## ✅ 檢查清單

**訓練前**:
- [ ] GitHub Personal Access Token 已準備
- [ ] Colab Runtime 設置為 GPU
- [ ] 確認 repository 可正常 clone
- [ ] 了解訓練策略與預期時間

**訓練中**:
- [ ] TensorBoard 正常顯示
- [ ] 監控 GPU 記憶體使用
- [ ] 定期檢查訓練進度
- [ ] 保持 Colab session 活躍

**訓練後**:
- [ ] 查看最佳模型 F1 分數
- [ ] 檢查 evaluation_results/
- [ ] 確認模型已推送 (如達標)
- [ ] 更新 PROJECT_STATUS.md

---

## 🎉 預祝訓練順利！

如有任何問題，請查閱相關文件或提交 issue。