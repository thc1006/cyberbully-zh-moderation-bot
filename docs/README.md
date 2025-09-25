# 📚 CyberPuppy 文件中心

## 🎯 專案概述

CyberPuppy 是專為中文環境打造的先進 AI 內容審核系統，提供即時毒性偵測、網路霸凌防治與情緒分析功能。

- **專案狀態**: v0.1.0 (85% 完成度)
- **授權**: Apache License 2.0
- **維護者**: thc1006
- **最後更新**: 2025-09-25

## 📂 文件結構

### 🚀 快速開始
如果您是第一次使用，請按照以下順序閱讀：

1. **[安裝指南](setup/INSTALL.md)** - 系統需求與環境設置
2. **[GPU 設置](setup/GPU_SETUP_GUIDE.md)** - CUDA 配置 (選用但建議)
3. **[資料集下載](datasets/DATASET_DOWNLOAD_GUIDE.md)** - 取得訓練資料
4. **[快速開始](setup/QUICK_START.md)** - 5 分鐘內啟動服務

### 📦 設置指南 (`setup/`)
- **[INSTALL.md](setup/INSTALL.md)** - 完整安裝步驟
- **[QUICK_START.md](setup/QUICK_START.md)** - 快速啟動指南
- **[GPU_SETUP_GUIDE.md](setup/GPU_SETUP_GUIDE.md)** - GPU/CUDA 配置
- **[WINDOWS_SETUP.md](setup/WINDOWS_SETUP.md)** - Windows 特定設置
- **[LARGE_FILES_SETUP.md](setup/LARGE_FILES_SETUP.md)** - 大型模型檔案管理

### 🔧 技術文件 (`technical/`)
- **[霸凌偵測改進指南](technical/BULLYING_DETECTION_IMPROVEMENT_GUIDE.md)** ⭐ - F1=0.55 改進策略
- **[TDD 實作總結](technical/TDD_IMPLEMENTATION_SUMMARY.md)** - 測試驅動開發
- **[依賴管理策略](technical/DEPENDENCY_STRATEGY.md)** - 套件版本控制
- **[依賴工作流程](technical/DEPENDENCY_WORKFLOW.md)** - 更新流程
- **[安全指南](technical/SECURITY.md)** - 安全最佳實踐
- **[內容政策](technical/POLICY.md)** - 審核政策定義
- **[測試集合分析](technical/TEST_COLLECTION_ANALYSIS.md)** - 測試覆蓋分析
- **[缺口緩解策略](technical/REMAINING_GAPS_MITIGATION.md)** - 已知問題解決

### 📊 資料集 (`datasets/`)
- **[資料合約](datasets/DATA_CONTRACT.md)** - 資料規格說明
- **[下載指南](datasets/DATASET_DOWNLOAD_GUIDE.md)** - 資料集取得方法
- **[資料集報告](datasets/FINAL_DATASET_REPORT.md)** - 資料分析結果

### 🌐 API 文件 (`api/`)
- **[API 參考](api/API.md)** - REST API 端點說明
- **[整合摘要](api/API_INTEGRATION_SUMMARY.md)** - API 整合概覽
- **[錯誤代碼](api/ERROR_CODES.md)** - 錯誤處理參考
- **[模型載入器](api/MODEL_LOADER_GUIDE.md)** - 模型管理指南

### 🚢 部署 (`deployment/`)
- **[部署指南](deployment/DEPLOYMENT.md)** - 生產環境部署
- **[部署檢查清單](deployment/DEPLOYMENT_READINESS_CHECKLIST.md)** - 上線前驗證

### ✅ 驗證報告 (`validation/`)

#### 核心驗證
- **[DoD 綜合驗證](validation/COMPREHENSIVE_DOD_VALIDATION_REPORT.md)** - 完成定義驗證
- **[執行摘要](validation/EXECUTIVE_DOD_SUMMARY.md)** - 高層次總覽
- **[整合驗證](validation/INTEGRATION_VALIDATION_REPORT.md)** - 系統整合測試
- **[整合測試](validation/INTEGRATION_TESTING.md)** - 測試程序

#### 平台相容性
- **[Windows 相容性](validation/WINDOWS_COMPATIBILITY_REPORT.md)** - Windows 支援
- **[Windows 驗證](validation/WINDOWS_VALIDATION_REPORT.md)** - Windows 測試結果

#### 測試報告
- **[覆蓋率報告](validation/COVERAGE_IMPROVEMENT_REPORT.md)** - 測試覆蓋分析
- **[診斷報告](validation/DIAGNOSTIC_REPORT.md)** - 系統診斷結果
- **[依賴測試](validation/DEPENDENCY_TESTING_REPORT.md)** - 套件相容性
- **[API 驗證修復](validation/API_VALIDATION_FIX.md)** - API 問題修正
- **[連結掃描](validation/LINK_SCAN_REPORT.md)** - 文件連結驗證
- **[手動測試](validation/test_api_manual.md)** - 手動測試程序

## 📈 專案效能指標

| 指標 | 實測值 | 目標值 | 狀態 |
|------|--------|--------|------|
| **毒性偵測 F1** | 0.77 | 0.78 | 🔄 接近達標 |
| **霸凌偵測 F1** | 0.55 | 0.75 | ⚠️ [需改進](technical/BULLYING_DETECTION_IMPROVEMENT_GUIDE.md) |
| **情緒分析 F1** | 1.00* | 0.85 | ✅ 超越目標 |
| **回應時間** | <200ms | 500ms | ✅ 達標 |
| **GPU 加速** | 5-10x | - | ✅ 實測驗證 |

*註：情緒分析在小樣本測試中表現完美，需更大規模驗證

## 🎯 開發路線圖

### 🔴 高優先級 (v0.2.0)
1. **改善霸凌偵測** - 目標 F1 ≥ 0.75
   - 參見 [改進指南](technical/BULLYING_DETECTION_IMPROVEMENT_GUIDE.md)
2. **完成資料集整合** - DMSC v2, ChnSentiCorp
3. **生產環境部署** - Docker + Kubernetes

### 🟡 中優先級 (v0.3.0)
- 建立監控系統 (Prometheus + Grafana)
- 優化 GPU 記憶體使用
- CI/CD Pipeline (GitHub Actions)

### 🟢 低優先級 (v1.0.0)
- 模型量化 (ONNX, INT8)
- 邊緣設備部署
- 擴充測試覆蓋

## 🛠️ 開發者資源

### 核心模組
```
src/cyberpuppy/
├── models/        # 模型實作
├── explain/       # 可解釋性 (IG, SHAP)
├── safety/        # 安全規則引擎
├── eval/          # 評估工具
└── labeling/      # 標籤處理
```

### API 服務
```
api/
├── app.py         # FastAPI 主程式
└── model_loader.py # 模型管理
```

### 測試指令
```bash
# 單元測試
pytest tests/ -v

# 覆蓋率報告
pytest --cov=src/cyberpuppy --cov-report=html

# API 測試
python test_api_final.py
```

## 📞 支援與聯絡

- **Email**: hctsai@linux.com
- **GitHub Issues**: [問題回報](https://github.com/thc1006/cyberbully-zh-moderation-bot/issues)
- **專案狀態**: [PROJECT_STATUS.md](../PROJECT_STATUS.md)

## 📜 授權

本專案採用 Apache License 2.0 授權 - 詳見 [LICENSE](../LICENSE)

---

**文件版本**: v1.0.0
**最後更新**: 2025-09-25
**維護團隊**: CyberPuppy Development Team