# CLAUDE.md

> Last updated: **2026-04-18** (v5 dual-LoRA ensemble — COLD 0.8412, PCR-ToxiCN 0.6890 超越 SOTA, TC homo abs 0.8384 歷代最高)

## 專案宗旨

CyberPuppy 是以「網路霸凌防治」為核心，結合 **毒性偵測 / 霸凌分類 / 角色辨識 / 情緒分析** 的中文多任務分類系統。在私訊與群組對話中，以 **高可解釋性、低誤傷、隱私優先** 的方式提供即時提醒與教師仲裁介面。

## 當前狀態（2026-04-18）

| 維度 | 狀態 |
|---|---|
| 主模型 | **v5 dual-LoRA** = Qwen3-8B × (text LoRA-A + pinyin LoRA-B) α=0.75 ensemble |
| Backbone | `Qwen/Qwen3-8B-Base`（Apache 2.0） |
| HF artefacts | [`thc1006/cyberpuppy-v5-bilingual`](https://huggingface.co/thc1006/cyberpuppy-v5-bilingual) + [`thc1006/cyberpuppy-v5-pinyin-lora`](https://huggingface.co/thc1006/cyberpuppy-v5-pinyin-lora) |
| 訓練資料 | 179,186 樣本 = COLD + SCCD + STATE-ToxiCN + ToxiCloakCN(3×) + CNTP + 繁簡雙語 |
| GPU | RTX 5090 32 GB（CUDA 12.8、bf16 native） |

## 核心效能（v5 dual-LoRA ensemble α=0.75，2026-04-18 實測）

| Metric | v5 dual-LoRA α=0.75 | DoD | 對比 v2.2 |
|---|---|---|---|
| COLD toxicity F1_w | **0.8412** | ≥0.83 ✅ | +0.34pt |
| COLD bullying F1_w | — | ≥0.75 ✅ | — |
| ToxiCloakCN homo abs F1 | **0.8384** | — | +4.2pt |
| ToxiCloakCN homo drop | −7.41% | ≤5% ⚠️ | base 太強 (0.91) |
| **PCR-ToxiCN（真實世界）** | **0.6890** | — | **🏆 超越 SOTA 0.672** |
| CNTP homo recall drop | **−0.37%** | — | 幾乎免疫 |
| 6 繁中威脅 | **6/6** | ✅ | 維持 |

### 實驗歷程摘要（v2.2 → v5）

| 方法 | TC homo drop | PCR F1 | 結果 |
|---|---|---|---|
| v2.2 (λ=0.1) | −8.51% | — | baseline |
| v2.3 lexicon-aug | −9.23% | — | ❌ OOD 資料 |
| v2.4 JioNLP aug | −18.52% | — | ❌ clean bucket 被掏空 |
| λ=0.5 sweep | −7.26% | 0.6128 | λ 最優值 |
| v3.0 pinyin prompt | −6.70% | — | ❌ calibration 崩 |
| v4.0 dual-branch CNN | −7.49% | — | ❌ 容量失衡 400,000:1 |
| rbt4 ensemble α=0.8 | −4.92% | 0.5601 | ✅ TC DoD 但真實世界差 |
| **v5 bilingual dual-LoRA** | **−7.41%** | **0.6890** | **🏆 真實世界最佳** |

## 標籤體系（不變）

```
toxicity      : none | toxic | severe
bullying      : none | harassment | threat
role          : none | perpetrator | victim | bystander
emotion       : pos | neu | neg
emotion_strength : 0..4
```

**訓練分佈**（v2.2 train 70,870 筆）：
- `role`: none 76.7% / perpetrator 23.3% — 真信號（v1 是 100% none）
- `emotion`: neu 76.7% / neg 23.3% — 真信號（v1 是 100% neu）
- `toxicity`: none 58.8% / toxic 35.4% / severe 5.8% — severe 類有實質學習信號

## 技術棧

- **Backbone**: Qwen/Qwen3-8B-Base（Apache 2.0；ADR 拒用 Llama-Guard / ShieldGemma 因不支援中文）
- **PEFT**: peft 0.19 LoRA r=32 α=64（DoRA off — 與 gradient checkpointing 衝突）
- **Quantization**: autoawq 0.2.9（隔離 `.venv-quant` + transformers 4.51.3）
- **Trainer 自製**: bf16 + LengthBucketSampler + CloakAwareBatchSampler + focal γ=2.5 + uncertainty-weighted multi-task + adversarial consistency loss
- **解釋性**: Captum (IG)、SHAP — 已實作於 `src/cyberpuppy/explain/`
- **文字處理**: OpenCC `s2twp`（簡 → 繁台灣腔）；Qwen3 tokenizer 取代 CKIPTagger
- **API**: FastAPI + uvicorn（asyncio.Lock 序列化推論，PDPO §64 對齊）
- **Bot**: LINE Messaging API + HMAC-SHA256 webhook 驗簽
- **Arbiter**: 可選 Perspective API（`src/cyberpuppy/arbiter/perspective.py`，未必啟用）

## 雙層架構（ADR 0001）

```
使用者訊息
  │
  ├─►【Layer 1】Qwen3Guard-Gen-8B 守門員（zero-shot）
  │       └─ 高信心 unsafe → 立即觸發 LINE 警示
  │
  └─►【Layer 2】CyberPuppy v2.2（LoRA 自訓 4-head）
          ├─ toxicity / bullying / role / emotion
          └─ XAI: SHAP / IG token-level
```

驗證理由：v1/v2 漏「我打死你」(2/6)，Qwen3Guard zero-shot 抓到 6/6 但 COLD F1 只 0.7458；v2.2 則兩者都好（COLD 0.8378 + 繁中 6/6）。雙層互補。

## 三層授權（2026-04-16 修訂）

| 範疇 | 授權 | 檔案 |
|---|---|---|
| 程式碼 / 文件 / 配置 | **Apache 2.0** | `LICENSE` |
| Model weights / adapters / 量化 artefact | **CC BY-NC-SA 4.0** | `MODEL_LICENSE` |
| 訓練資料（`data/processed/v2/*`） | **不 redistribute** | `DATA_LICENSE_NOTICE.md` |

商業變體（Apache 2.0 weights）：以 COLD-only + 客戶自有資料重訓，CyberPuppy team 提供付費客製訓練服務（ADR §3.8）。

## 目錄結構（實際）

```
cyberbully-zh-moderation-bot/
├── api/
│   ├── v2_2_app.py            # ★ v2.2 主要 API（CC weights）
│   ├── app.py                 # legacy v1（保留以免 regression）
│   └── Dockerfile             # Python 3.11
├── bot/
│   ├── line_bot.py
│   └── Dockerfile
├── src/cyberpuppy/
│   ├── data/
│   │   ├── phase2.py          # ★ 6 個資料源 normalizer（COLD/SCCD/CHNCI/STATE/ToxiCloakCN）
│   │   └── phase2_download.py # dry-run-by-default downloader
│   ├── models/
│   │   ├── qwen3_multihead.py # ★ Qwen3 wrapper + 4 heads + focal + consistency loss
│   │   └── baselines.py       # legacy MacBERT
│   ├── training/
│   │   ├── bucket_sampler.py        # 變長序列分桶
│   │   └── cloak_aware_sampler.py   # 對抗 triplet 同 batch
│   ├── eval/metrics.py        # F1, calibration, multi-task aggregator
│   ├── explain/ig.py          # Integrated Gradients
│   ├── safety/rules.py        # 分級回覆策略
│   └── arbiter/perspective.py # 可選
├── scripts/
│   ├── train_qwen3_lora.py    # ★ 主訓練入口
│   ├── merge_lora.py          # fp32 merge → bf16
│   ├── quantize_awq.py        # 在 .venv-quant 跑
│   ├── verify_awq_parity.py
│   ├── benchmark_latency.py   # warmup + cuda.synchronize
│   ├── v2_2_comprehensive_eval.py
│   ├── phase2_build_v2_2.py   # 多源資料合併（含 ToxiCloakCN heldout）
│   ├── toxicloak_eval.py      # robustness 評測
│   ├── qwen3guard_baseline.py # Layer 1 守門 baseline
│   └── check_gpu.py           # 環境檢查
├── tests/                     # 273 unit tests，全綠
├── configs/training/
│   └── rtx5090_optimized.yaml # RTX 5090 config
├── docs/
│   ├── adr/0001-cyberpuppy-2026-upgrade.md   # ★ 主 ADR（10 章 + 附錄）
│   ├── adr/0001-research-brief.md             # 28 web sources
│   ├── datasets/phase2_inventory.md
│   ├── outreach/predick_reply_zh.md           # PadLearn 回信草稿
│   └── archive/                                # 舊 status / Makefile
├── reports/                   # 所有 eval JSON
├── data/
│   ├── raw/                   # COLD / NTUSD（原始）
│   ├── external/              # gitignored — git clone 上游 repo
│   └── processed/v2/          # gitignored — phase2_build_v2_2.py 重建
└── models/                    # gitignored — too big
```

## 開發環境

```bash
# 主環境（訓練 / serving / tests）
uv venv --python 3.11 .venv
.venv/bin/python -m pip install -e .          # Apache 2.0 deps
# 或 .venv/bin/pip install -e .[training]      # 加 bitsandbytes / trl

# 量化專用環境（autoawq 0.2.9 不相容主 venv 的 transformers 4.57）
uv venv --python 3.11 .venv-quant
.venv-quant/bin/pip install 'transformers==4.51.3' autoawq accelerate
```

GPU: RTX 5090, CUDA 12.8, bf16 native, 32 GB VRAM。

## 常用工作流

### 訓練
```bash
# v2.2 主訓練（~75 min, 3 epochs）
PYTHONPATH=src python scripts/train_qwen3_lora.py \
  --train data/processed/v2/v2_2_train.jsonl \
  --eval  data/processed/v2/v2_2_dev.jsonl \
  --epochs 3 --batch 6 --grad-accum 6 --max-length 192 \
  --lr 3e-5 --focal-gamma 2.5 --consistency-lambda 0.1 \
  --num-workers 8 --save-every-steps 500 \
  --output models/cyberpuppy_v2_2_qwen3_8b
```

### 部署
```bash
# 1. LoRA → merged bf16
python scripts/merge_lora.py

# 2.（可選）AWQ 量化（隔離 venv）
.venv-quant/bin/python scripts/quantize_awq.py

# 3. API serving
CP_MODEL_DIR=models/cyberpuppy_v2_2_awq \
  uvicorn api.v2_2_app:app --host 0.0.0.0 --port 8000
```

### 評測
```bash
PYTHONPATH=src python scripts/v2_2_comprehensive_eval.py
PYTHONPATH=src python scripts/toxicloak_eval.py     # robustness
PYTHONPATH=src python scripts/benchmark_latency.py --flavor bf16
```

## 風格與原則

- **明確輸出**：每次改動列出「新增 / 修改 / 刪除」清單
- **TDD 紅綠循環**：先寫 test（紅）→ 實作（綠）→ refactor。273 unit tests 全綠
- **可回溯**：所有 data 處理皆為 deterministic 腳本；訓練 random seed 固定 42
- **隱私優先**：API 層 SHA-256 hash text，原文絕不寫入日誌
- **誠實 reporting**：失敗也記錄（如 autoawq 第一次失敗 → 隔離 venv 修法）
- **不打補丁修根因**：CloakAware sampler 是因為 LengthBucketSampler 把 triplet 拆散才補上

## DoD 對照（CLAUDE.md 原始 → 2026-04-16 實際）

| 原 DoD | 狀態 |
|---|---|
| 毒性 F1 ≥ 0.78 | ✅ 0.8378 |
| 霸凌 F1 ≥ 0.75 | ✅ 0.8365 |
| 4 任務統一標籤 | ✅ |
| role / emotion 真實標籤 | ✅（v2.2 multisource 23.3% 非預設值） |
| 隱私 SHA-256 hashed | ✅ |
| FastAPI 部署可用 | ✅ `/v2/analyze` 實測 |
| LINE Bot 整合 | ⚠️ code 完備未真環境驗 |
| Docker 化 | ⚠️ Dockerfile 存在未跑 v2.2 版 |
| ToxiCloakCN robustness | ⚠️ emoji ✅ 0.37% / homo 待 v2.3 |

## ADR / Decision Log

- **0001-cyberpuppy-2026-upgrade.md** — 主 ADR：v2.0 → v2.1 → v2.2 演進、雙層架構、三層授權、Phase 5 部署
- **0001-research-brief.md** — 28 條 web sources（Qwen3、SOTA 對比、HK 政策、PDPO）
- **outreach/predick_reply_zh.md** — PadLearn 回信草稿（含三層授權對接）

## 已知限制 / 待解

1. **homophone robustness 仍未達 DoD**（v2.3.1 best −7.88% vs ≤ −5%）。**合成 homo data-aug 3 次全失敗**（v2.3 −9.23%、v2.4 −18.52%、v2.4.1 −14.40%，詳見 ADR Phase 3.4+3.5）→ **data-aug 路線已放棄**，下一步走 Phase 4 GRPO 或 pinyin-aware embedding
2. **Docker 部署未實測**（v2.2 image 未 build）
3. **LINE Bot 真實環境驗證**（需 LINE Channel credentials）
4. **CHNCI 未納入訓練**（220K 樣本中只用 0；試點時可加）
5. **autoawq 已 deprecated** → 未來遷 vLLM `llm-compressor`

## 後續路線（依 ADR 優先）

| Phase | 內容 | 狀態 |
|---|---|---|
| Phase 1 | Qwen3Guard baseline | ✅ |
| Phase 2 | 多源資料整備 | ✅ |
| Phase 3 | v2.0 → v2.1 → v2.2 訓練 | ✅ |
| Phase 4 | GRPO 安全強化 | 規劃中 |
| Phase 5 | 量化 + 部署 | ✅ MVP |
| Phase 6 | HK 試點（PadLearn 對接） | 規劃中 |

---

> **Maintainer**: thc1006 (hctsai1006@cs.nctu.edu.tw)
> **License**: Apache-2.0 (code) + CC BY-NC-SA 4.0 (weights)
> **Repo**: https://github.com/thc1006/cyberbully-zh-moderation-bot
