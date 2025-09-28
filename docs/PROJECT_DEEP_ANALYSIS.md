# 🔍 CyberPuppy 專案深度分析報告

**分析日期**: 2025-09-29
**專案路徑**: `C:\Users\thc1006\Desktop\dev\cyberbully-zh-moderation-bot`
**專案類型**: 中文網路霸凌偵測系統

---

## 📊 專案規模總覽

### 整體統計
- **總檔案數**: 9,185 個
- **專案總大小**: ~3.7 GB
- **程式碼檔案**: 254 個 Python 檔案
- **資料檔案**: 128 個 (JSON/CSV/TXT/DB)
- **測試檔案**: 70 個測試檔案
- **套件依賴**: 160 個 (64 生產 + 96 開發)

### 目錄大小分布
| 目錄 | 大小 | 說明 |
|------|------|------|
| `models/` | **2.4 GB** | 模型權重檔案 |
| `data/` | **666 MB** | 訓練與測試資料 |
| `htmlcov/` | 11 MB | 測試覆蓋率報告 |
| `tests/` | 3.8 MB | 測試程式碼 |
| `src/` | 2.8 MB | 源程式碼 |
| `scripts/` | 1.4 MB | 腳本工具 |

---

## 🗄️ 資料集詳細分析

### 資料總量
- **總資料行數**: **4,749,491 行**
- **原始資料**: 550 MB
- **處理後資料**: 101 MB
- **外部資料**: 5 KB
- **標註資料**: 28 KB

### 主要資料集

#### 1. COLD Dataset (Chinese Offensive Language Dataset)
**位置**: `data/raw/cold/COLDataset/`

| 檔案 | 樣本數 | 用途 |
|------|--------|------|
| `train.csv` | **25,727** | 訓練集 |
| `dev.csv` | **6,432** | 開發集 |
| `test.csv` | **5,324** | 測試集 |
| **總計** | **37,483** | |

#### 2. DMSC (Douban Movie Short Comments)
**位置**: `data/raw/dmsc/`
- **檔案**: `DMSC.csv`
- **大小**: **387 MB**
- **樣本數**: **2,131,887 行**
- **用途**: 情緒分析訓練

#### 3. ChnSentiCorp 中文情感語料庫
**位置**: `data/raw/chnsenticorp/`

| 檔案 | 樣本數 | 狀態 |
|------|--------|------|
| `ChnSentiCorp_htl_all.csv` | 7,767 | ✅ 可用 |
| `ChnSentiCorp_htl_all_2.csv` | 7,767 | ✅ 可用 |
| `ChnSentiCorp_from_fate233.csv` | 0 | ⚠️ 空檔案 |
| `ChnSentiCorp_gitee.csv` | 0 | ⚠️ 空檔案 |
| **總計** | **15,534** | |

#### 4. 處理後訓練資料集
**位置**: `data/processed/training_dataset/`

| 分割 | 樣本數 | 毒性標籤分布 | 文字長度 |
|------|--------|--------------|----------|
| **訓練集** | **25,659** | toxic: 12,673 (49.4%)<br>none: 12,986 (50.6%) | 平均: 47.8 字<br>最短: 3 字<br>最長: 192 字 |
| **開發集** | **6,430** | toxic: 3,210 (49.9%)<br>none: 3,220 (50.1%) | 平均: 47.3 字<br>最短: 4 字<br>最長: 190 字 |
| **測試集** | **5,320** | toxic: 2,106 (39.6%)<br>none: 3,214 (60.4%) | 平均: 48.3 字<br>最短: 4 字<br>最長: 128 字 |
| **總計** | **37,409** | | 中位數: 40-41 字 |

### 標籤體系

**現有標籤類型**:
1. **毒性 (toxicity)**: `none` | `toxic` | `severe`
2. **霸凌 (bullying)**: `none` | `harassment` | `threat`
3. **角色 (role)**: `none` | `perpetrator` | `victim` | `bystander`
4. **情緒 (emotion)**: `pos` | `neu` | `neg`
5. **情緒強度 (emotion_strength)**: 0-4 (整數)

**問題**: 當前訓練資料中，role 全部為 "none"，emotion 全部為 "neu"，emotion_strength 全部為 0

---

## 🤖 模型檔案分析

### 模型權重統計

| 模型 | 檔案大小 | 格式 | 狀態 | F1 分數 |
|------|----------|------|------|---------|
| `gpu_trained_model/model.safetensors` | **391 MB** | safetensors | ❌ 性能差 | 實測: 0.28 |
| `working_toxicity_model/pytorch_model.bin` | **397 MB** | bin | ❌ 無法載入 | - |
| `macbert_base_demo/best.ckpt` | **397 MB** | ckpt | ⚠️ 未測試 | 聲稱: 0.773 |
| `toxicity_only_demo/best.ckpt` | **397 MB** | ckpt | ⚠️ 未測試 | 聲稱: 0.783 |
| `bullying_a100_best/` | **595 KB** | - | ❌ 缺少權重 | 聲稱: 0.82 |
| `local_training/macbert_aggressive/*.pt` | **16.5 MB** | pt (7檔案) | ❌ 訓練失敗 | ~0.34 |
| **總計** | **~1.6 GB** | | | |

### 模型問題
1. **bullying_a100_best**: 聲稱最好 (F1=0.82) 但缺少模型權重檔案
2. **gpu_trained_model**: 實測 F1=0.28，遠低於聲稱 0.77
3. **working_toxicity_model**: config.json 格式錯誤，無法載入
4. **local_training**: 訓練未收斂，過擬合嚴重

---

## 🗃️ 資料庫檔案

### SQLite 資料庫
| 資料庫 | 位置 | 大小 | 用途 |
|--------|------|------|------|
| `hive.db` | `.hive-mind/` | **124 KB** | Hive-mind 協調系統 |
| `memory.db` | `.hive-mind/` | **16 KB** | Hive-mind 記憶體儲存 |
| `memory.db` | `.swarm/` | **112 KB** | Swarm 系統記憶體 |

---

## 💻 源代碼結構

### 核心模組 (`src/cyberpuppy/`)
- **檔案總數**: 79 個 Python 檔案
- **大小**: 2.8 MB

#### 模組分布
| 模組 | 功能 | 狀態 |
|------|------|------|
| `active_learning/` | 主動學習 | ✅ 實作 |
| `arbiter/` | 仲裁系統 (Perspective API) | ✅ 實作 |
| `data/` | 資料處理 | ✅ 實作 |
| `data_augmentation/` | 資料增強 | ✅ 實作 |
| `eval/` | 評估指標 | ✅ 實作 |
| `explain/` | 模型解釋 (SHAP/IG) | ✅ 實作 |
| `labeling/` | 標籤處理 | ✅ 實作 |
| `loop/` | 主動學習迴圈 | ✅ 實作 |
| `models/` | 模型定義 | ✅ 實作 |
| `safety/` | 安全規則 | ✅ 實作 |
| `training/` | 訓練管理 | ✅ 實作 |
| `web/` | Web 介面 | ✅ 實作 |

### 訓練腳本 (`scripts/`)
- **檔案數**: 45 個 Python 腳本
- **重要腳本**:
  - `train_simple_with_args.py` - 主訓練腳本
  - `download_datasets.py` - 資料集下載
  - `verify_model_performance.py` - 性能驗證
  - `optimize_thresholds.py` - 閾值優化（已刪除）

---

## 🧪 測試覆蓋率

### 測試檔案
- **測試檔案總數**: 70 個
- **測試用例總數**: 179+ 個（根據最近報告）

### 覆蓋率統計（根據最近改進）
| 模組 | 之前 | 現在 | 改進 |
|------|------|------|------|
| `detector.py` | 3.77% | **70.16%** | +66.39% |
| `metrics.py` | 0% | **86.67%** | +86.67% |
| `rules.py` | 81.69% | **90.17%** | +8.48% |

---

## 📦 Git LFS 追蹤檔案

**大檔案管理** (前 10 個):
1. `data/processed/augmentation_validation_report.json`
2. `data/processed/chnsenticorp/dev.json`
3. `data/processed/chnsenticorp/test.json`
4. `data/processed/chnsenticorp/train.json`
5. `data/processed/cold/dev_processed.csv`
6. `data/processed/cold/test_processed.csv`
7. `data/processed/cold/train_processed.csv`
8. `data/processed/cold_analysis.json`
9. `data/processed/cold_augmented.csv`
10. `data/processed/cold_augmented_stats.json`

---

## 🔧 專案配置

### 主要配置檔案
| 檔案 | 行數 | 用途 |
|------|------|------|
| `pyproject.toml` | 7,085 bytes | 專案配置與套件管理 |
| `requirements.txt` | 64 行 | 生產環境依賴 |
| `requirements-dev.txt` | 96 行 | 開發環境依賴 |
| `docker-compose.yml` | 2,437 bytes | Docker 部署配置 |

### 主要依賴套件
- **深度學習**: PyTorch, Transformers, Accelerate
- **資料處理**: NumPy, Pandas, scikit-learn
- **NLP**: jieba, OpenCC, CKIPTagger
- **解釋性**: SHAP, Captum
- **API**: FastAPI, LINE SDK
- **測試**: pytest, coverage

---

## 📈 關鍵數據總結

### 數據規模
- **總資料樣本**: ~2,200,000+ (含 DMSC)
- **訓練用樣本**: 37,409 (COLD 為主)
- **標註樣本**: 37,483 (COLD 完整標註)

### 模型規模
- **模型總大小**: ~1.6 GB
- **單個完整模型**: ~400 MB
- **可用模型數**: 0 個（無達標模型）

### 性能指標（實測）
- **最佳聲稱 F1**: 0.82 (A100，無權重)
- **最佳可用 F1**: 0.28 (gpu_trained_model，遠未達標)
- **目標 F1**: ≥0.75

### 開發規模
- **Python 程式碼**: 254 檔案
- **測試程式碼**: 70 檔案
- **測試覆蓋率**: 70-90%（核心模組）

---

## 🚨 關鍵問題

1. **無可用生產模型**: 所有實測模型 F1 < 0.30
2. **最佳模型缺失**: A100 訓練結果缺少權重檔案
3. **標籤不完整**: role/emotion 標籤未使用
4. **資料不平衡**: 測試集毒性比例 (39.6%) 與訓練集 (49.4%) 不一致
5. **模型偏差**: 現有模型過度預測 "none" 類別 (97% recall)

---

## 📋 建議優先事項

1. **立即**: 使用更新的 Colab notebook 重新訓練 A100 模型
2. **驗證**: 確保模型權重正確保存 (~400MB)
3. **測試**: 使用 `verify_model_performance.py` 驗證實際性能
4. **標籤**: 考慮整合 role/emotion 多任務學習
5. **部署**: 達到 F1≥0.75 後再考慮生產部署

---

**報告生成時間**: 2025-09-29
**分析工具版本**: 1.0
**分析者**: Claude Code Assistant