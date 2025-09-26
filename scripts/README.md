# 完整資料集下載與修復腳本

這是一個針對 CyberPuppy 專案設計的完整資料集下載與管理腳本。

## 腳本功能

- **支援多種資料集格式**：Hugging Face (.arrow)、ZIP 壓縮檔、Git 儲存庫
- **進度條顯示**：實時顯示下載進度
- **斷點續傳**：支援中斷後繼續下載
- **詳細日誌**：記錄所有操作和錯誤資訊
- **驗證功能**：檢查資料集完整性
- **最終報告**：生成詳細的驗證報告

## 支援的資料集

| 資料集 | 描述 | 大小 | 格式 | 狀態 |
|--------|------|------|------|------|
| COLD | 中文攻擊性語言檢測資料集 | ~6MB | CSV | ✅ 完整 |
| ChnSentiCorp | 中文情感分析資料集 | ~6MB | Arrow + CSV | ✅ 完整 |
| DMSC v2 | 大眾點評情感分析資料集 | ~387MB | CSV | ✅ 完整 |
| NTUSD | 台大中文情感詞典 | ~0.2MB | TXT | ✅ 完整 |

## 使用方法

### 基本用法

```bash
# 下載所有資料集
python scripts/complete_dataset_setup.py --dataset all

# 下載特定資料集
python scripts/complete_dataset_setup.py --dataset chnsenticorp

# 下載多個資料集
python scripts/complete_dataset_setup.py --dataset cold ntusd

# 只驗證現有資料集
python scripts/complete_dataset_setup.py --verify-only

# 強制重新下載
python scripts/complete_dataset_setup.py --dataset all --force

# 生成驗證報告
python scripts/complete_dataset_setup.py --report-only
```

### 命令列參數

- `--dataset`：指定要下載的資料集 (cold, chnsenticorp, dmsc, ntusd, all)
- `--output-dir`：指定輸出目錄 (預設: ./data/raw)
- `--verify-only`：只驗證，不執行下載
- `--force`：強制重新下載，覆蓋現有檔案
- `--report-only`：只生成驗證報告

## 依賴套件

```bash
pip install requests tqdm pandas datasets huggingface_hub pyarrow
```

## 輸出結構

```
data/raw/
├── cold/
│   └── COLDataset/
│       ├── train.csv
│       ├── dev.csv
│       └── test.csv
├── chnsenticorp/
│   ├── train.arrow
│   ├── validation.arrow
│   ├── test.arrow
│   └── ChnSentiCorp_htl_all.csv
├── dmsc/
│   └── DMSC.csv
├── ntusd/
│   └── data/
│       ├── 正面詞無重複_9365詞.txt
│       └── 負面詞無重複_11230詞.txt
└── dataset_report.json
```

## 驗證報告

腳本會自動生成 `dataset_report.json`，包含：

- 資料集狀態（完整、部分、缺失、錯誤）
- 檔案大小和路徑
- 下載時間和方法
- 缺失檔案列表
- 總體統計資訊

## 錯誤處理

- **網路錯誤**：自動重試，支援斷點續傳
- **檔案權限**：提供清晰的錯誤訊息
- **空間不足**：檢查並報告磁碟空間
- **格式錯誤**：驗證檔案完整性

## 日誌記錄

所有操作都會記錄到 `dataset_setup.log`，包含：
- 下載進度
- 錯誤詳情
- 執行時間
- 驗證結果

## 故障排除

### 常見問題

1. **Hugging Face 下載失敗**
   ```bash
   # 可能需要登入 Hugging Face
   huggingface-cli login
   ```

2. **Git 克隆失敗**
   ```bash
   # 檢查網路連線和 Git 配置
   git config --global http.proxy http://proxy:port
   ```

3. **權限錯誤**
   ```bash
   # Windows 上可能需要以管理員身份執行
   # 或者變更輸出目錄權限
   ```

4. **磁碟空間不足**
   ```bash
   # 檢查可用空間 (需要約 400MB)
   df -h .
   ```

### 手動修復

如果自動下載失敗，可以手動下載檔案到對應目錄：

```bash
# 手動下載範例
wget https://example.com/dataset.zip -O data/raw/dataset/dataset.zip
```

## 許可證

本腳本遵循專案許可證。各資料集有其自己的許可證條款。

## 貢獻

歡迎提交 Issue 和 Pull Request 來改進這個腳本。