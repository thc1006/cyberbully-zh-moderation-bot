# Phase 2 Dataset Inventory

> 列出 ADR 0001 §3.2 列出的 6 個資料集，含 HF/GitHub 來源、授權、規模、用途。
> 此檔僅作盤點，**不執行下載**（GPU 暫停期間）。執行時請參考同目錄的 `phase2_download.py`。

## 已驗證可用

| 資料集 | HF dataset id | GitHub | 授權 | 規模 | 用途 | 狀態 |
|---|---|---|---|---|---|---|
| **COLD** | `thu-coai/cold` | thu-coai/COLDataset | **Apache-2.0** | 37,480 留言 | 既有 baseline；繁化後續用 | ✅ HF 可直接 `datasets.load_dataset` |
| **STATE-ToxiCN** | ❌ 無 HF | shenmeyemeifashengguo/STATE-ToxiCN | 學術用途（請查 README） | 8,029 貼文 + 9,533 quadruple + 830 詞俚語詞典 | span-level 抽取、可解釋性 | 🔧 需 git clone |
| **ToxiCN** | ❌ 無 HF | DUT-lujunyu/ToxiCN | 學術用途 | ~12K 貼文 | STATE-ToxiCN 的 base | 🔧 需 git clone |

## 待人工確認 GitHub URL（HF 無，URL 從 ADR 研究底稿引用）

| 資料集 | GitHub 推測 | 論文 | 規模 | 用途 |
|---|---|---|---|---|
| **SCCD** | （需查 arXiv 2501.15042 作者頁） | arXiv 2501.15042（COLING 2025） | 677 sessions Weibo | 對話級霸凌 |
| **CHNCI** | （需查 arXiv 2505.20654 作者頁） | arXiv 2505.20654（May 2025） | 220,676 留言 / 91 incidents | 事件級霸凌 |
| **ToxiCloakCN** | dut-lujunyu 帳號下衍生 repo | arXiv 2406.12223（EMNLP 2024） | ~12K 對抗樣本 | **僅 robustness 評測，不訓練** |
| **PANDA** | （需查 arXiv 2501.00697 作者頁） | arXiv 2501.00697（Jan 2025） | ~12K counterspeech 對 | LINE Bot 回應端，不分類 |

## 統一 schema（v2 訓練資料目標）

```jsonc
{
  "id": "<dataset>_<split>_<idx>",
  "text": "<繁中文本>",
  "context": ["<前 N 句對話，可空陣列>"],
  "label": {
    "toxicity": "none|toxic|severe",
    "bullying": "none|harassment|threat",
    "role": "none|perpetrator|victim|bystander",
    "emotion": "pos|neu|neg",
    "emotion_strength": 0
  },
  "metadata": {
    "source": "cold|sccd|chnci|state_toxicn|hk_lihkg",
    "original_label_raw": "<原始標籤>",
    "text_length": 0,
    "is_traditional": true,
    "annotation_quality": "gold|silver|weak"
  }
}
```

## 對接點：normalize 流程

每個 source 都需經過：

1. **OpenCC 繁化**（`s2twp`）— 簡 → 繁台灣腔
2. **PII 去除**（手機、Email、ID、學生姓名）
3. **長度過濾**（3-512 字）
4. **標籤對映** → unified schema
5. **去重**（SHA-256 hash on text）
6. **品質標記**（gold / silver / weak）

對應實作：`scripts/phase2_normalize.py`（Phase 2 開始時實作；目前先寫測試）

## 授權與合規備註

- COLD 授權 **Apache-2.0** → 商業可用
- STATE-ToxiCN / ToxiCN 多為**學術用途**，商業用前須與作者確認 → 影響 PadLearn 商業整合
- LIHKG / 連登 抓取 → **禁用** 個人資料；只取公開帖文，遵守 robots.txt 與速率限制；HK PDPO §64（doxxing）對齊
- 訓練前必過 PDPO impact assessment（ADR §3.5）

## 未知 / 待確認

1. SCCD / CHNCI / ToxiCloakCN 的 GitHub 實際 URL（research brief 只引用了論文 arXiv，github URL 為推論）
2. PANDA 的具體授權條款
3. STATE-ToxiCN 商業使用許可
4. SCCD/CHNCI 的單 sample 結構（需 clone 後看才能寫 normalize）

> 解除 GPU 後再執行 `python scripts/phase2_download.py --dry-run` 取得實際 manifest。
