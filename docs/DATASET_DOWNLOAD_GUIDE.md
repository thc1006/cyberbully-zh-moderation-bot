# 📚 CyberPuppy 語料庫下載完整指南

所有語料庫的下載方式都已記錄在 scripts/download_datasets.py 中。

## 快速下載
```bash
python scripts/download_datasets.py --dataset all
```

## 各語料庫詳情：
1. **COLD** (~10MB) - GitHub 自動下載 ✅
2. **ChnSentiCorp** (~5MB) - Hugging Face (需修復) ⚠️
3. **DMSC v2** (387MB) - GitHub Release 自動下載 ✅
4. **NTUSD** (~1MB) - GitHub 自動下載 ✅
5. **SCCD** - 需手動申請 (arxiv.org/abs/2506.04975) ❌
6. **CHNCI** - 需手動申請 (arxiv.org/abs/2506.05380) ❌

詳見 scripts/download_datasets.py 第 46-120 行的完整配置。
