# 霸凌樣本標註工作流程

## 1. 工作流程概述

本文件描述了完整的霸凌樣本標註工作流程，從樣本選擇到最終結果驗證的所有步驟。

### 1.1 流程架構

```
樣本準備 → 主動學習選擇 → 任務分配 → 標註執行 → 品質控制 → 結果驗證 → 資料整合
```

### 1.2 參與角色

- **項目負責人**: 整體協調和品質監督
- **資料準備人員**: 樣本收集和預處理
- **標註者**: 執行標註任務
- **品質控制員**: 驗證標註結果
- **技術支援**: 系統維護和問題解決

## 2. 第一階段：樣本準備和選擇

### 2.1 樣本來源準備

**目標**: 準備足夠多樣性的候選樣本

**步驟**:
1. 收集來源資料（社群媒體、論壇、聊天記錄等）
2. 數據清理和去識別化處理
3. 格式標準化處理

**使用工具**:
- 數據清理腳本
- OpenCC（繁簡轉換）
- 隱私保護工具

### 2.2 主動學習樣本選擇

**目標**: 選擇最有價值的500個樣本進行標註

**使用腳本**: `scripts/active_learning_selector.py`

**執行命令**:
```bash
python scripts/active_learning_selector.py \
    --input data/raw/candidate_samples.json \
    --output data/annotation/selected_samples_500.json \
    --n_samples 500 \
    --uncertainty_ratio 0.7 \
    --model_name hfl/chinese-macbert-base
```

**參數說明**:
- `--input`: 候選樣本檔案
- `--output`: 選中樣本輸出檔案
- `--n_samples`: 選擇數量
- `--uncertainty_ratio`: 不確定性採樣比例
- `--model_name`: 用於嵌入的預訓練模型

**輸出**: 選中的500個高價值樣本，包含優先級標記

## 3. 第二階段：任務分配和追蹤

### 3.1 建立標註任務

**目標**: 將樣本分配給標註者並建立追蹤系統

**使用腳本**: `scripts/batch_annotation.py`

**執行命令**:
```bash
python scripts/batch_annotation.py create \
    --samples data/annotation/selected_samples_500.json \
    --task_name "cyberbully_annotation_batch1" \
    --annotators annotator1 annotator2 annotator3 \
    --overlap_ratio 0.1 \
    --difficulty_priority
```

**參數說明**:
- `--samples`: 待標註樣本檔案
- `--task_name`: 任務名稱
- `--annotators`: 標註者ID列表
- `--overlap_ratio`: 重疊樣本比例（用於計算一致性）
- `--difficulty_priority`: 按困難度優先分配

**輸出**: 為每個標註者生成獨立的任務檔案

### 3.2 建立追蹤表格

**目標**: 建立Excel追蹤表格監控進度

**使用腳本**: `scripts/annotation_tracker.py`

**執行命令**:
```bash
python scripts/annotation_tracker.py create \
    --samples data/annotation/selected_samples_500.json \
    --annotators annotator1 annotator2 annotator3 \
    --task_name "cyberbully_annotation_batch1"
```

**輸出**: 包含多個工作表的Excel追蹤檔案
- 主追蹤表: 每個樣本的標註狀態
- 標註者統計: 個人表現指標
- 進度追蹤: 時間進度圖表
- 品質控制: 品質指標監控

## 4. 第三階段：標註執行

### 4.1 標註者培訓

**培訓內容**:
1. 閱讀 `docs/ANNOTATION_GUIDE.md`
2. 理解標註維度和判斷標準
3. 練習標註範例樣本
4. 通過一致性測試（Kappa > 0.6）

### 4.2 使用標註介面

**啟動介面**:
```bash
python scripts/annotation_interface.py
```

**操作流程**:
1. 輸入標註者ID
2. 載入分配的任務檔案
3. 逐個標註樣本
4. 定期儲存進度
5. 完成後匯出結果

**標註要求**:
- 每個樣本必須完成所有5個維度的標註
- 困難案例必須添加備註說明
- 保持標註一致性
- 每天至少完成20個樣本

### 4.3 進度監控

**定期檢查**:
- 每日更新追蹤表格
- 檢查標註速度和品質
- 協助解決困難案例

**更新進度命令**:
```bash
python scripts/batch_annotation.py progress \
    --task_id cyberbully_annotation_batch1_20240115_143022_abc12345 \
    --annotator_id annotator1 \
    --progress_file data/annotation/progress/annotation_progress_annotator1.json
```

## 5. 第四階段：品質控制

### 5.1 即時品質檢查

**使用腳本**: `scripts/annotation_quality_control.py`

**執行命令**:
```bash
python scripts/annotation_quality_control.py \
    --annotation_files \
        data/annotation/results/annotator1_results.json \
        data/annotation/results/annotator2_results.json \
        data/annotation/results/annotator3_results.json \
    --output_dir data/annotation/quality_control \
    --visualize
```

**輸出**:
- 標註者間一致性報告（Kappa係數）
- 不一致案例清單
- 視覺化一致性矩陣
- 改進建議

### 5.2 異常檢測

**檢測項目**:
- 標註速度異常（過快或過慢）
- 標籤分布異常（過度偏向某個標籤）
- 邏輯一致性問題
- 困難案例比例異常

**處理措施**:
- 與標註者討論異常情況
- 重新標註問題樣本
- 加強培訓和指導

## 6. 第五階段：結果驗證

### 6.1 標註結果驗證

**使用腳本**: `scripts/annotation_validator.py`

**批次驗證命令**:
```bash
python scripts/annotation_validator.py validate_batch \
    --annotation_files \
        data/annotation/results/annotator1_results.json \
        data/annotation/results/annotator2_results.json \
        data/annotation/results/annotator3_results.json \
    --output_dir data/annotation/validation
```

**驗證項目**:
- 格式完整性檢查
- 邏輯一致性檢查
- 時間合理性檢查
- 內容合理性檢查

### 6.2 一致性檢查

**執行命令**:
```bash
python scripts/annotation_validator.py check_consistency \
    --annotation_files \
        data/annotation/results/annotator1_results.json \
        data/annotation/results/annotator2_results.json \
        data/annotation/results/annotator3_results.json \
    --output_dir data/annotation/validation
```

**目標指標**:
- 整體Kappa係數 ≥ 0.6
- 各維度Kappa係數 ≥ 0.4
- 重疊樣本一致性 ≥ 70%

## 7. 第六階段：資料整合

### 7.1 收集標註結果

**使用腳本**: `scripts/batch_annotation.py`

**執行命令**:
```bash
python scripts/batch_annotation.py collect \
    --task_id cyberbully_annotation_batch1_20240115_143022_abc12345 \
    --result_files \
        annotator1:data/annotation/results/annotator1_results.json \
        annotator2:data/annotation/results/annotator2_results.json \
        annotator3:data/annotation/results/annotator3_results.json
```

**合併策略**:
- 單一標註者樣本: 直接採用
- 多標註者樣本: 多數投票決定
- 平票情況: 標記為需要專家審核

### 7.2 最終資料準備

**輸出格式**:
```json
{
  "sample_id": "001",
  "text": "樣本內容",
  "final_annotation": {
    "toxicity": "toxic",
    "bullying": "harassment",
    "role": "perpetrator",
    "emotion": "neg",
    "emotion_strength": "3"
  },
  "annotation_metadata": {
    "annotators": ["annotator1", "annotator2"],
    "confidence": 0.85,
    "difficult": false,
    "final_review": true
  }
}
```

## 8. 品質保證檢查清單

### 8.1 樣本選擇階段
- [ ] 樣本來源多樣性充足
- [ ] 隱私保護措施到位
- [ ] 主動學習選擇合理
- [ ] 困難度分布平衡

### 8.2 任務分配階段
- [ ] 標註者負載均衡
- [ ] 重疊樣本設置合理
- [ ] 追蹤系統建立完成
- [ ] 任務說明清晰

### 8.3 標註執行階段
- [ ] 標註者培訓完成
- [ ] 標註介面測試通過
- [ ] 進度監控正常
- [ ] 困難案例及時處理

### 8.4 品質控制階段
- [ ] 一致性檢查通過
- [ ] 異常情況已處理
- [ ] 品質指標達標
- [ ] 不一致案例已討論

### 8.5 結果驗證階段
- [ ] 格式驗證通過
- [ ] 邏輯檢查無誤
- [ ] 驗證報告完成
- [ ] 改進建議實施

### 8.6 資料整合階段
- [ ] 結果收集完整
- [ ] 合併策略一致
- [ ] 最終格式正確
- [ ] 元資料完整

## 9. 故障排除指南

### 9.1 常見問題

**標註介面問題**:
- 問題: 無法載入樣本檔案
- 解決: 檢查檔案路徑和格式，確保JSON格式正確

**進度更新問題**:
- 問題: 進度檔案損壞
- 解決: 使用備份檔案，重新載入進度

**一致性過低問題**:
- 問題: Kappa係數 < 0.4
- 解決: 組織標註者討論，澄清標註標準

### 9.2 緊急聯絡

- **技術支援**: tech-support@project.com
- **項目負責人**: project-lead@project.com
- **品質控制**: quality-control@project.com

## 10. 附錄

### 10.1 檔案目錄結構
```
data/annotation/
├── tasks/           # 標註任務檔案
├── progress/        # 進度檔案
├── results/         # 標註結果
├── quality_control/ # 品質控制報告
├── validation/      # 驗證報告
├── tracking/        # 追蹤表格和圖表
└── archive/         # 歸檔檔案
```

### 10.2 重要時程

- **第1週**: 樣本準備和選擇
- **第2週**: 任務分配和標註者培訓
- **第3-4週**: 標註執行
- **第5週**: 品質控制和結果驗證
- **第6週**: 資料整合和最終檢查

### 10.3 品質指標目標

| 指標 | 目標值 | 最低要求 |
|------|--------|----------|
| 標註完成率 | 100% | 95% |
| 整體Kappa係數 | ≥ 0.7 | ≥ 0.6 |
| 各維度Kappa係數 | ≥ 0.6 | ≥ 0.4 |
| 困難案例比例 | 10-20% | 5-30% |
| 邏輯一致性 | ≥ 95% | ≥ 90% |

### 10.4 聯絡資訊

- **標註指引問題**: 參考 `docs/ANNOTATION_GUIDE.md`
- **技術文件**: 各腳本檔案內的說明文檔
- **項目Wiki**: [內部Wiki連結]

---

**版本**: v1.0
**最後更新**: 2024-01-15
**維護者**: CyberPuppy開發團隊