# ADR 0001 — CyberPuppy 2026 Upgrade Plan

- **Status**: Proposed
- **Date**: 2026-04-15
- **Authors**: thc1006 (本機分析 + Claude Code 研究代理)
- **Decision drivers**:
  - 來自 Hong Kong 教育科技 LittleP AI（PadLearn）的合作詢問，需以 **繁體中文 / 校園情境** 為主
  - 現有模型 `bullying_a100_best` 實測 F1=0.825（COLD test），但對繁中威脅句（"我打死你"、"希望你去死"）誤判為 none
  - 訓練資料 role 全為 none、emotion 全為 neu、severe 類測試集 0 樣本，「多任務」名實不符
  - 本工作站升級為 RTX 5090 (32 GB, sm_120)，硬體已不是瓶頸
- **Supersedes**: PROJECT_STATUS.md 內 RTX 3050 訓練計畫
- **Related research**: 同目錄 `0001-research-brief.md`（28 篇 web 來源、9 條已標單一來源）

---

## 1. 背景與問題陳述

CyberPuppy 目前是一個 **單一中文編碼器（hfl/chinese-macbert-base, 102 M 參數）** + **單一霸凌頭** 的分類器，COLD 測試集 weighted F1 = 0.825，符合 README 標示。但實務上有四個結構性問題：

1. **任務退化**：宣稱 4 任務（toxicity/bullying/role/emotion），實際訓練資料 role/emotion 標籤都是預設值，等同只訓練 toxicity；bullying 與 toxicity 共享同一組標籤（25,659 / 6,430 / 5,320 split 中各類別計數完全相同）。
2. **語料偏簡中、社群貼文**：COLD 是大陸社群留言；面對繁中對話、明示威脅、髒話 + 表情符號混合時泛化弱。
3. **嚴重等級空集**：test set severe = 0 樣本 → severe head 從未被學習也無法評估。
4. **基礎模型已落後**：2025-2026 開源中文 LLM（Qwen3、GLM-4.5、InternLM3）與 Qwen3Guard 守門員模型在中文 safety benchmark 上明顯超越 BERT-base 級別編碼器。

PadLearn 想把 CyberPuppy 嵌入學生對話安全模組，必要條件是：**繁中可用、能處理對話脈絡、守門時能說明根據（XAI）、p95 延遲 < 200 ms、可單機部署於學校或邊緣 GPU**。

## 2. 決策考量過的方案

| # | 方案 | 優點 | 缺點 |
|---|---|---|---|
| A | 維持 MacBERT，補資料、加強訓練 | 改動最小、推論便宜 | 上限受限於 base model；繁中泛化、long-context 對話無解 |
| B | 直接外接 OpenAI / Gemini moderation API | 立即可用 | 跨境傳輸 → PDPO/PIPL 風險；延遲不可控；商業模式套牢 |
| C | 以 Qwen3-8B/8B 微調為自家分類器 + Qwen3Guard 為守門員 | 開源 Apache-2.0、繁中與粵語涵蓋、可本機 | 訓練/推論成本高於 BERT，但 RTX 5090 可吃下 |
| D | 全 Qwen3-14B + DPO/GRPO | SOTA 表現 | 32 GB VRAM 已是上限，需 QLoRA 但 Qwen 官方不建議 |

**選擇方案 C** —— 以 Qwen3-8B 為主分類骨幹（多任務頭）+ Qwen3Guard-Gen-8B 作為前置守門員。理由：
- Qwen3 系列 Apache-2.0、明確支援繁中與粵語、bf16 LoRA 在 32 GB VRAM 內可舒適訓練（4 B ≈ 10 GB，8 B ≈ 22 GB）
- Qwen3Guard 是 2025-10 開源、explicitly 評過中文 safety 的 SOTA 守門員，可作為 (a) production 安全防線 (b) 蒸餾老師
- 不選 Llama-Guard / ShieldGemma：兩者皆**不支援中文**（Llama-Guard 3/4 只支援 8 種語言、ShieldGemma 偏英語/影像）

## 3. 決定（Decision）

採用 **「雙層架構：Qwen3Guard 即時守門 + CyberPuppy-v2 多任務分析」**。

```
使用者對話
  │
  ├─►【Layer 1 即時守門 / <50 ms】 Qwen3Guard-Gen-8B (vLLM, bf16/FP8)
  │       └─ 高信心 unsafe → 立即觸發 LINE 警示，跳過 Layer 2
  │
  └─►【Layer 2 細緻分析 / 100-150 ms】 CyberPuppy-v2 (Qwen3-8B + 4 LoRA 頭)
          ├─ toxicity head     (none / toxic / severe)
          ├─ bullying head     (none / harassment / threat)
          ├─ role head         (none / perpetrator / victim / bystander)
          ├─ emotion head      (pos / neu / neg, intensity 0-4)
          └─ XAI: SHAP / IG over span tokens
```

### 3.1 模型與訓練決定

| 項目 | 決定 | 理由 |
|---|---|---|
| 主模型 | **Qwen3.5-8B-Base（fallback Qwen3-8B）** + 4 個分類頭 | 8B bf16 LoRA ≈ 22 GB / 32 GB，留 ~10 GB 給 optimizer state + KV cache；中文 nuanced toxicity / sarcasm 比 4B 顯著強 |
| 守門模型 | **Qwen3Guard-Gen-8B-Gen** 直接部署 | Apache-2.0、現成中文 SOTA、避免重複造輪子 |
| 微調方法 | **LoRA r=32 + DoRA, target_modules="all-linear", bf16** | Unsloth 官方建議：QLoRA 4-bit 在 Qwen3 系列數值偏差大，bf16 較穩；DoRA 比純 LoRA 收斂更好 |
| 對齊方法 | **SFT → 多任務交叉熵 → 最後一階段 GRPO 安全強化** | 2025 多項研究在 14 B 規模顯示 GRPO > DPO；Qwen 團隊自己即用 curriculum GRPO 訓 Qwen3Guard |
| 量化部署 | **AWQ + Marlin kernel on vLLM 0.17+** （備案 FP8 on TensorRT-LLM） | 獨立 benchmark 顯示 AWQ-Marlin 741 tok/s，10.9× FP16；FP8 在 Blackwell 是更高上限但部署彈性較低 |

### 3.2 資料策略

新增以下中文資料集（皆 2025+ 公開，已驗證來源）：

| 資料集 | 規模 | 用途 | 來源 |
|---|---|---|---|
| **SCCD** (COLING 2025) | 677 sessions / Weibo 對話 | **對話級**霸凌；填補 CyberPuppy 缺失的脈絡能力 | arXiv 2501.15042 |
| **CHNCI** (May 2025) | 220,676 留言 / 91 incidents | **事件級**霸凌；多平台（Douyin/Weibo/Xiaohongshu/Bilibili） | arXiv 2505.20654 |
| **STATE-ToxiCN** (ACL Findings 2025) | 8,029 貼文 + 9,533 四元組 | **span-level** 標註 + 830 詞中文仇恨俚語詞典 | arXiv 2501.15451 |
| **ToxiCloakCN** (EMNLP 2024) | ~12K 對抗樣本 | **僅作 robustness 評測**；同音字、emoji 偽裝 | arXiv 2406.12223 |
| **ToxiCN-MM** (NeurIPS 2024 D&B) | 12K meme 圖文 | 未來多模態擴展（暫不訓練） | github dut-lujunyu/toxicn_mm |
| **PANDA** (Jan 2025) | ~12K counterspeech 對 | LINE Bot 回應端訓練（不參與分類） | arXiv 2501.00697 |

繁中與粵語覆蓋（**研究確認此為公開資料的真實缺口**）：
1. 用 OpenCC 將 SCCD/CHNCI/COLD 一份完整轉繁中
2. LIHKG / 連登 公開帖文有限度爬取 + 嚴格 PII 清除
3. 用 Qwen3-14B 做 prompted 合成 + **強制人工驗證 ≥ 1,500 樣本**（PANDA 論文已警告 LLM-as-Judge 在中文仇恨上有「不可忽略的標錯率」，必須抽樣驗證，不能直接 ship raw label）
4. 角色標籤（perpetrator/victim/bystander）：先用 Qwen3-14B prompt 自標，再人審 ~2K 樣本作 gold
5. emotion + intensity：以 SemEval-2025 Task 11 中文子集為 silver label，再以本地對話小批 fine-tune

### 3.3 Evaluation 體系

| 維度 | 指標 / 工具 |
|---|---|
| 主任務 F1 | macro / weighted F1 + per-class precision/recall（COLD test, SCCD test, CHNCI test） |
| 對抗 robustness | **ToxiCloakCN 必跑**（同音字、emoji 偽裝） |
| 校準 | class-wise ECE + SmoothECE (arXiv 2501.19047)；toxicity 類別不平衡時必看 |
| 對話一致性 | SCCD session-level F1（不只 message-level） |
| 延遲 | 單機 RTX 5090 / vLLM、batch=1 / batch=8 / batch=32 三點 p50/p95 |
| 安全度 | OpenCompass 司南 safety dimension |
| 人審回饋 | 每週抽 100 樣本 inter-annotator agreement (Cohen's κ) |

### 3.4 部署架構

- **推論引擎**：vLLM 0.17+（CUDA 13 wheel，已在 RTX 5090 驗證可跑）
- **API 層**：保留現有 FastAPI；`api/model_loader.py` 已於 2026-04-15 修改成自動 bf16 + cudnn benchmark
- **前置守門**：Qwen3Guard-Gen-8B 常駐記憶體（~9 GB bf16 / ~3 GB AWQ）
- **主模型**：CyberPuppy-v2-LoRA on Qwen3-8B（~10 GB bf16）
- **VRAM 預算（推論）**：Qwen3Guard-Gen-8B (~16 GB bf16) + CyberPuppy-v2-8B (~16 GB bf16) ≈ 32 GB → **必須至少一個走 AWQ**；建議守門員 AWQ (~5 GB) + 主模型 bf16 (~16 GB) + KV cache (~3-5 GB) ≈ 24-26 GB，留 6-8 GB 餘裕
- **VRAM 預算（訓練）**：Qwen3.5-8B bf16 LoRA (~22 GB) + 8-bit AdamW state (~6 GB) ≈ 28 GB → 訓練階段 Qwen3Guard 不常駐 GPU
- **快取**：Redis（已存在 docker-compose.yml）對重複文本做 SHA-256 鍵；只存 hash + score，不存原文
- **LINE Bot**：保留 HMAC-SHA256 webhook 驗簽

### 3.5 合規 / 資料保護

- **適用法規**：HK PDPO（不是 PIPL）。若任何訓練資料、託管或日誌跨境到 PRC，PIPL 第 38 條生效，須做 SCC / 安全評估 → **建議全棧託管於 HK 或本機**
- **PCPD AI Deepfake Toolkit (2025-12-17)** 已涵蓋校園場景、ImageBased Sexual Violence、cyberbullying、scams、disinformation 與 PDPO §64（doxxing），是 PadLearn 必讀
- **資料最小化**：訓練語料一律去識別（手機、Email、學生姓名、學校代碼）
- **人在迴圈（HITL）**：高信心 unsafe → 自動回應 + 標記排隊；中信心 → 推送至教師審核 dashboard
- **可解釋性**：每個 unsafe 判定附 SHAP token 重要度，老師可一鍵 override（標籤回流訓練資料池）

### 3.6 整合介面與範疇邊界

**CyberPuppy 做的（in-scope）**：訊息級毒性/霸凌/角色/情緒分類、對話脈絡感知、可解釋性輸出、LINE/HTTP 整合、教師審核回饋。

**CyberPuppy 不做的（out-of-scope，需合作方自備）**：作業批改、個性化學習路徑、試卷生成、對話式 tutoring、學科 QA、ASR/TTS、家長端 IM。

對接面（合作方視角）：

| 介面 | 形式 | 用途 |
|---|---|---|
| `POST /analyze` | OpenAI-compatible JSON | 同步單則訊息分類 |
| `POST /analyze/batch` | JSON 陣列 | 批次匯入歷史對話 |
| `POST /webhook/line` | LINE Messaging webhook | 直接掛 LINE Bot |
| WebSocket `/stream` | server-sent events | 即時對話流（PadLearn 場景） |
| Webhook callback | 客戶端 endpoint | 高風險事件回推（教師通知） |
| 教師 dashboard | React + SHAP 可視化 | override + active learning |

### 3.7 客製化分層

讓合作方在「不動 backbone」前提下調整模型行為：

| Tier | 內容 | 誰可改 | 變更時間 |
|---|---|---|---|
| **T1 閾值** | 各 head 信心閾值（嚴格 / 標準 / 寬鬆三檔） | 學校 IT | 即時，不需重訓 |
| **T2 詞庫注入** | 在地俚語/校名/敏感詞白黑名單；prompt-level rule injection | 學校 IT + 教師 | 即時，重啟服務生效 |
| **T3 私有 adapter** | 學校自有對話資料微調 LoRA adapter（資料留校） | CyberPuppy 工程支援 | 1-2 週 |
| **T4 領域擴充** | 新增頭（如自殘/物質濫用），全 backbone fine-tune | CyberPuppy team | 4-8 週 |

### 3.8 商業模式（決定）

- **核心模型權重與程式碼維持 Apache-2.0**：學校與廠商可自行下載、自部署、自修改
- **付費服務（CyberPuppy 團隊提供）**：
  - 整合與部署諮詢（按專案計費）
  - T3/T4 客製化訓練
  - SLA 維護與安全更新
  - PDPO 影響評估報告協助
- 不採雙授權、不採 enterprise-only weights — 維持社群信任、降低 PadLearn 等廠商試水溫成本

## 4. 階段性執行計畫

### Phase 0 — 基礎環境（已完成 2026-04-15）

- [x] uv venv (Python 3.11) + torch 2.11+cu128
- [x] git lfs pull → 取得既有 391 MB bullying_a100_best 權重
- [x] GPU 上重現 F1=0.825（accuracy 0.823）—— 確認舊模型可重現
- [x] 寫入 `configs/training/rtx5090_optimized.yaml`
- [x] `api/model_loader.py` 自動 bf16 + cudnn benchmark

### Phase 1 — 繁中守門 baseline（2 週）— **2026-04-15 已執行**

- [x] 6 句繁中威脅 baseline（`reports/qwen3guard_baseline.json`）：**Qwen3Guard 6/6 vs v1 2/6**
- [x] COLD test set 全跑（`reports/qwen3guard_cold_eval.json`）：
  - Qwen3Guard-Gen-8B zero-shot：**Acc 0.7430, F1 weighted 0.7458, F1 macro 0.7393**
  - v1 bullying_a100_best：Acc 0.8229, F1 weighted 0.8247, F1 macro 0.8195
  - **Qwen3Guard zero-shot 在 COLD 上比 v1 低 ~8 個 F1 點**
- [x] 原因確認（看錯誤樣本）：**標籤哲學差異**——COLD 抓「冒犯性語言」（含軟性地域/性別刻板），Qwen3Guard 抓「安全違規」（窄但嚴）
- [x] **判斷點修正**：原設「Qwen3Guard ≥ v1 0.82 即 ship」**不成立**；改為 **Layer 1 = Qwen3Guard 抓威脅 / 自殘 / 暴力（高 precision 場景）；Layer 2 = 必訓 v2 抓 nuance 冒犯**
- [x] strict mapping 對照（Controversial→safe）F1=0.5634 → 比 lenient 0.7458 更差 **18 點**；證實 COLD 的 offense 標籤涵蓋大量 Controversial 級別內容，必須由 v2 多任務頭吸收
- [x] 工作站推論優化已 land：bf16 + left-pad batched generate + max_new=12 + tf32 + cudnn benchmark → 19 samples/s（5,320 樣本 4.66 min）

### Phase 2 — 多任務資料整備（4 週）

- [ ] 下載 + 清洗 SCCD / CHNCI / STATE-ToxiCN（含 PII 移除、OpenCC 繁化）
- [ ] 設計統一 schema：`{toxicity, bullying, role, emotion, emotion_strength, context_window}`
- [ ] 對 SCCD 對話以 5-turn 滑窗切割
- [ ] 用 Qwen3-14B prompt-based 為 SCCD/CHNCI 補 role 標籤；人審 2K 樣本作 gold dev/test
- [ ] LIHKG / 連登 抓取 ~5K 繁中帖文 + 人工標註 ~1K 作為 HK 在地 evaluation set
- [ ] 寫資料卡（datasheet）與 PDPO 影響評估

### Phase 3 — CyberPuppy-v2 訓練（**2026-04-15 已執行 v2.0**）

- [x] HF 驗證：Qwen3.5-8B/-7B 在 HF **不存在**；確認採用 **Qwen3-8B-Base**（10M downloads, Apache-2.0）
- [x] 訓練腳本 `scripts/train_qwen3_lora.py` 完成 + smoke test (256 樣本 22 秒)
- [x] **v2.0 訓練實際參數**（vs ADR 草案差異標註）：
  - bf16 LoRA r=32 alpha=64, target_modules=all qkvo+gate/up/down
  - **DoRA 關閉**（DoRA 在 PEFT 內每 forward 重 materialize 4096×4096 weight + gradient checkpointing 疊加→OOM）
  - **gradient checkpointing 開啟**（節省 ~30% activation memory）
  - **batch=8 grad_accum=4 → effective 32**（batch=16 OOM；batch=8 無 GC OOM）
  - 4 個分類頭 + uncertainty-weighted Kendall multi-task loss
  - cosine LR peak 3e-5, warmup 0.1, AdamW 8-bit
  - 3 epochs on COLD train (25,659) → 2,406 optim steps in **41.7 min**
- [x] Checkpoints 已保存：
  - `models/cyberpuppy_v2_qwen3_8b/best/lora/adapter_model.safetensors` (174.6 MB)
  - `models/cyberpuppy_v2_qwen3_8b/best/heads.pt` (110 KB)
- [x] **訓練 dev F1 軌跡**：epoch0=0.9147 → epoch1=0.9227 → epoch2=**0.9239**

#### Phase E — 三方對比 (COLD test 5,320 樣本)

| Model | Acc | F1 weighted | F1 macro | 繁中 6/6 | Throughput |
|---|---|---|---|---|---|
| v1 MacBERT (391 MB) | 0.8229 | 0.8247 | 0.8195 | 2/6 (33%) | ~1925/s |
| Qwen3Guard-Gen-8B zero-shot | 0.7430 | 0.7458 | 0.7393 | 6/6 (100%) | 19/s |
| **v2 Qwen3-8B+LoRA (175 MB)** | **0.8312** | **0.8327** | **0.8271** | **5/6 (83%)** | **122/s** |

✅ v2 **每項都贏 v1**（COLD F1 +0.8 點、繁中 +50%）
✅ v2 **遠超 Qwen3Guard zero-shot**（COLD F1 +8.7 點）
⚠️ v2 漏「再嘴一句試試看，我打死你」→ **驗證 Layer 1 (Qwen3Guard) 的獨立價值**

#### DoD §9 對照

| 條件 | 目標 | 實測 | 狀態 |
|---|---|---|---|
| COLD test F1 weighted | ≥ 0.85 | 0.8327 | ⚠️ 差 1.7 點 → 待 Phase 2 多源資料補齊 |
| 繁中威脅 recall | ≥ 5/6 = 83% | 5/6 | ✅ 達標 |
| SCCD session F1 | ≥ 0.70 | N/A（未訓 SCCD） | — 等 Phase 2 |
| ToxiCloakCN emoji drop | ≤ 5% | v2.1 6.52% → **v2.2 0.37%** | ✅ 達標（v2.2） |
| ToxiCloakCN homophone drop | ≤ 5% | v2.1 10.16% → v2.2 8.51% | ⚠️ 未達，需 v2.3 |
| p95 < 200 ms | < 200 ms | **17 ms 短 / 22 ms 中 / 34 ms 長** (bf16 batch=1, RTX 5090, Phase 5 實測) | ✅✅ 大幅達標 |

#### 觀察 / 限制

- role/emotion 兩頭在 COLD 資料下退化為「永遠預測 dominant class」（acc 1.0 但 F1 macro=0），因 COLD 100% 樣本標 role=none、emotion=neu → **必須由 Phase 2 SCCD/CHNCI/SemEval 補真標**
- 訓練吞吐以 RTX 5090 / Blackwell / bf16 / no-DoRA / GC-on 為基準

### Phase 3.2 — v2.1 (multisource + focal loss, 2026-04-15)

- [x] Multisource 資料 60K（COLD 25,659 + SCCD 28,426 + STATE-ToxiCN 5,781）取代 COLD-only 25K
- [x] Focal loss γ=2.0 處理 severe (6.8%) / threat (0.2%) 類別不平衡
- [x] LengthBucketSampler 解決變長序列 OOM（三次 DoRA/batch 調試後定案）
- [x] 49 min 訓練，best=epoch 1 toxicity F1_w=0.8510
- COLD test: Acc 0.8312, F1_w 0.8327 (+0.8pt vs v1)
- 繁中威脅 5/6 (vs v1 2/6)
- role / emotion head 脫離假象（non-trivial 0.89+）

### Phase 3.3 — v2.2 (adversarial training, 2026-04-16)

- [x] 加入 ToxiCloakCN 對抗樣本訓練 (3,668 pair × 3 variants = 11,004 新樣本)
- [x] `CloakAwareBatchSampler`：強制 base/homo/emoji triplet 同 batch
- [x] `consistency_loss` (λ=0.1) 讓 clean/homo/emoji toxicity logits 對齊
- [x] 75 min 訓練，best=epoch 1 toxicity F1_w=0.8408 (multisource dev)
- **ToxiCloakCN heldout 魯棒性**（906 pairs 未參與訓練）：
  - base clean: F1_w 0.8703
  - homophone drop: v2.1 **−10.16%** → v2.2 **−8.51%** （17% 改善，仍超 DoD）
  - emoji drop: v2.1 **−6.52%** → v2.2 **−0.37%** ✅ (94% 改善，達 DoD)
- Consistency loss 軌跡：0.56 → 0.035（94% 降，證明 logits 對齊成功）
- 繁中威脅 6/6 維持，clean COLD / multisource performance 持平 v2.1

#### v2.2 觀察

- **Emoji 魯棒性完全解決**，因 emoji 保留周邊中字 token，model 只需忽略 emoji noise
- **Homophone 仍差**：中文同音字空間極大（~4K 常用字，數千音素同音），3,668 訓練 pair 不足涵蓋
- v2.3 路徑：用 STATE-ToxiCN 830 詞仇恨俚語詞典做程式化 homophone augmentation + λ=0.3

### Phase 4 — GRPO 安全強化（2 週）

- [ ] 設計 reward：F1 + 校準 + 拒答正確率 + 跨分佈一致性
- [ ] GRPO 訓練（Unsloth 或 trl 0.16+）；參考 Qwen3-8B-SafeRL pipeline
- [ ] 對抗測試（ToxiCloakCN, 自製繁中對抗集）

### Phase 5 — 量化與部署（**2026-04-16 執行 MVP**）

- [x] **LoRA merge** (`scripts/merge_lora.py`)：fp32 merge 後再降 bf16，避免小幅 LoRA delta 被 bf16 截去
  - 與 PEFT v2.2 在 106 樣本上 **100% 預測一致**；COLD first-100 F1_w 兩者皆 0.8896
  - 產出 `models/cyberpuppy_v2_2_merged/` (16 GB bf16, 4-shard safetensors)
- [ ] ~~AWQ 4-bit 量化~~ **Pivot to bf16 only**：autoawq 0.2.9 × transformers 4.57 incompatible（Catcher class 缺 `attention_type` — Qwen3 hybrid attention 新 API 未 patch）。autoawq 上游已標 deprecated，生產應遷移 `llm-compressor` (vLLM 官方)。bf16 latency 已達 DoD，不需急 AWQ。
- [x] **Latency benchmark** (`scripts/benchmark_latency.py`)：warmup 20 + cuda.synchronize()
  - RTX 5090 bf16 merged, p95 latency：
    - 短句 (~20 tok): batch=1 **17 ms** / batch=16 26 ms (619 samp/s)
    - 中句 (~100 tok): batch=1 **22 ms** / batch=16 68 ms (237 samp/s)
    - 長句 (~200 tok): batch=1 **34 ms** / batch=16 182 ms (88 samp/s)
  - **全部 p95 < 200 ms，DoD §9 達標**
- [x] **FastAPI v2.2** (`api/v2_2_app.py`)：`POST /v2/analyze` + `/healthz` 503-until-ready
  - Startup: 2.2 sec（有 3 句 warmup）
  - Inference lock 保證 single-GPU 序列化（MVP 無 dynamic batching）
  - 隱私：SHA-256 hash text，日誌不存原文（PDPO §64 對齊）
- [x] **End-to-end HTTP smoke**：6 繁中句透過 `curl` 打 `/v2/analyze`
  - 6/6 全對；server-side latency 18 ms p50（首 request 45 ms warmup）
  - "打死你" → `tox=severe bull=harassment emo=neg`

### Phase 6 — HK 在地化驗證（持續）

- [ ] **PadLearn 試點 stakeholder 對齊**：定義對接 API、SLA、資料來源、家長同意流程
- [ ] **部署選項雙軌**：
  - **On-prem / 校內 GPU**：權重與推論完全留校，CyberPuppy 提供 docker image + helm chart
  - **託管 SaaS**：HK 境內機房，僅回傳 hash + score 不存原文
- [ ] 試點：1-2 所 DSE 學校匿名對話試跑 2 週
- [ ] 教師 dashboard：誤判標記回流到 active learning queue
- [ ] PDPO 影響評估報告交付（PCPD 2025-12-17 toolkit 對齊）

**總預計工期**：12 週（Phase 1-5 連續 12 週；Phase 6 與 5 重疊開始）

## 5. 風險與緩解

| 風險 | 影響 | 緩解 |
|---|---|---|
| Qwen3Guard zero-shot 在繁中表現不如預期 | 守門層失效 | Phase 1 判斷點；備案改用 Qwen3-8B + 自訓 safety SFT |
| LIHKG / 連登 抓取觸法 | 法律 | 只抓公開帖、嚴格速率限制、上線前法律審視；資料只用於分類器訓練不公開 |
| LLM 合成資料品質差 | 模型學壞 | PANDA 論文已證實風險，**強制 ≥ 30% 抽樣人審**，rejected 樣本回到 prompt 修正 |
| 角色標籤主觀性高 | 標註不一致 | 多人標 + Cohen's κ ≥ 0.6 才 release；不達標的子集只用於 weak supervision |
| RTX 5090 為單點 | 訓練中斷 | 每 500 step checkpoint；訓練 log 同步至 GitHub Actions artifact |
| 真實校園資料含未成年 PII | PDPO 高敏感 | 試點階段絕不持久化原文，只存 SHA-256 + score；教師審核介面只顯示 token 級 SHAP，不顯示完整對話 |
| HK 稅收/資金假設錯誤 | 商業時程錯估 | **email 中 HK$5B / 智啟學教 名稱與政府公告不符**；實際確認的是 EDB 2025-12-16 公佈的 "AI for Empowering Learning and Teaching" HK$500M / 每校 50 萬 / 2025-26 至 2027-28 學年 → 在與 PadLearn 對接時須以這個事實校正期望 |

## 6. 對既有程式碼的影響

| 檔案 | 動作 |
|---|---|
| `configs/training/rtx5090_optimized.yaml` | **新增（已完成）** |
| `api/model_loader.py` | **已修改**：自動 bf16 + cudnn benchmark；下一階段改成 OpenAI client → vLLM |
| `scripts/quick_eval_test.py` / `quick_inference_test.py` | **已修改**：bf16 + GPU + 環境變數控制 batch / compile |
| `src/cyberpuppy/models/baselines.py` | 保留作為 v1 fallback；不刪除 |
| `src/cyberpuppy/models/qwen3_multitask.py` | **新增（Phase 3）** |
| `scripts/train_qwen3_lora.py` | **新增（Phase 3）** |
| `data/processed/training_dataset/*.json` | 將被 v2 schema 取代；舊版保留於 git history |
| `bot/line_bot.py` | 加入 Qwen3Guard 前置層；UI 不變 |
| `models/bullying_a100_best/` | 保留但標記為 deprecated；新模型於 `models/cyberpuppy_v2_qwen3_4b/` |
| `tests/` | 補 robustness fixture（ToxiCloakCN sample）；目前 42 fail 多為 API drift，順手修 |

## 7. 不採用的替代方案（明確記錄）

- **Llama-Guard 4 / ShieldGemma 2**：均不支援中文；對繁體中文評測缺席
- **Qwen3-14B / Qwen3.5-14B 全參 SFT**：32 GB VRAM 需 QLoRA，但 Qwen 官方不建議 4-bit 訓 Qwen3.5/3 系列；改用 Qwen3.5-8B 是性能/可訓性最佳交集
- **Qwen3-4B / Qwen3.5-4B 為主模型**：原 ADR 草案曾選 4B 為「保守」起點；但 RTX 5090 32 GB 對 8B bf16 LoRA 仍有 ~10 GB 餘裕，沒必要犧牲 quality 換 headroom，故改為 8B
- **GLM-4.5 / GLM-5**：32 B+ active 參數，超出單卡微調預算
- **DeepSeek-R1 distilled**：reasoning 蒸餾不對齊分類任務目標；附帶授權繼承複雜度
- **多 adapter 路由（MoE LoRA）**：MHR 論文證實單頭多任務在這個規模差不多；增加部署複雜度不值
- **OpenAI Moderation API 直接外接**：跨境傳輸 PDPO/PIPL 風險、商業套牢、延遲與離線可用性

## 8. Open Questions（需與 PadLearn / 其他 stakeholder 對齊）

1. PadLearn 場景是否包含 **語音轉文字**？若是，需考慮 Whisper-large-v3 或 SenseVoice 中文 ASR 串接
2. 是否要做 **多模態（迷因圖）**？若是，Phase 7 引入 ToxiCN-MM 與 Qwen3-VL
3. 部署是否在 HK 境內？若涉及大陸雲，PIPL § 38 SCC 必跑
4. 教師 override 的回饋是否可作為訓練資料？需家長同意機制
5. ~~商業授權~~ → **已決定（§3.8）**：核心 Apache-2.0，付費僅服務面（諮詢/客製/SLA）
6. 模型輸出是否需要 **PCPD-aligned 解釋**（toolkit 提到的可解釋性義務）

## 9. Acceptance Criteria（DoD）

- [ ] 繁中威脅 6 句測試集（"我打死你" 等）：toxicity recall ≥ 5/6 = 83%（v1 為 1/6 = 17%）
- [ ] COLD test weighted F1 ≥ 0.85（v1 為 0.825）
- [ ] SCCD session-level F1 ≥ 0.70
- [ ] ToxiCloakCN robustness：F1 衰退 ≤ 5% 相對 clean COLD
- [ ] p95 延遲 < 200 ms（單機 RTX 5090, vLLM, batch=1）
- [ ] 全套自動化測試（含對抗集）綠燈
- [ ] PDPO 影響評估文件交付
- [ ] 至少 1 篇技術 blog post（中英雙語）

---

## 附錄 A — 此 ADR 主要事實依據

研究來源整理見同目錄 `0001-research-brief.md`。重點：
- Qwen3-14B 與 Qwen3Guard：arXiv 2505.09388、2510.14276；Qwen 官方 blog；HuggingFace 模型卡（≥3 來源交叉）
- SCCD / CHNCI / STATE-ToxiCN / ToxiCloakCN：arXiv + ACL Anthology + GitHub repo（≥2 來源）
- LoRA / DoRA / GRPO：philschmid 2025 指南、Unsloth 官方文件、arXiv 2503.21819 / 2505.20087
- vLLM AWQ-Marlin：jarvislabs.ai benchmark + vLLM docs
- HK EDB AI 經費：info.gov.hk 2025-12-16 + news.gov.hk + chinadailyhk（**email 中 HK$5B / 智啟學教 名稱無法被任何政府來源驗證**，實際確認的是 HK$500M 的 "AI for Empowering Learning and Teaching"）
- HK PCPD：kwm.com、mayerbrown.com、pcpd.org.hk

## 附錄 B — 更新此 ADR

ADR 一旦 Accepted 不可改寫；如有新證據應建立 ADR 0002 並標 "Supersedes ADR 0001"。

## 附錄 C — 對外溝通要點（給 PadLearn / 其他合作方）

### C.1 必須校正的事實（禮貌但明確）

| 對方陳述 | 我方查證結果 | 說法建議 |
|---|---|---|
| 「2026 預算 20 億 + 5 億『智啟學教』」 | 政府公告為 **2025-09 Policy Address 2B 數位教育儲備 + 2025-12-16 EDB 撥 5 億 "AI for Empowering Learning and Teaching"**；無「智啟學教」此名 | 「我方查到的官方撥款方案是 EDB 12/16 公佈的『AI for Empowering Learning and Teaching』，每校 50 萬元，方便對齊一下嗎？」 |
| 「790 間學校已申請」 | 政府新聞稿未見此具體數字（單一來源待驗） | 「方便分享一下這個 790 的出處嗎？我們想對齊政策進度」 |
| 「F1=0.826 在繁中欺凌出色」 | F1=0.826 是真，但是在 **COLD（簡中、社群留言）測試集**；繁中威脅句、severe 等級在現行模型上有明顯弱點 | 「F1=0.826 是 COLD 簡中測試集數字；針對繁中與威脅級樣本，我們正在 Phase 1-3 重訓擴增資料集（SCCD/CHNCI/STATE-ToxiCN/LIHKG），目標 COLD F1 ≥ 0.85、繁中威脅 recall ≥ 83%」 |

### C.2 範疇邊界（讓對方知道哪邊要自備）

CyberPuppy 是 **校園安全/情緒模組**，不是學習平台主體。建議劃分：

| 模組 | 由誰負責 |
|---|---|
| 對話安全偵測、情緒分析、霸凌警示 | **CyberPuppy** |
| 教師 dashboard、家長通知、case 管理 | 雙方協作（CyberPuppy 提供 API + 樣板 React） |
| 學科 tutoring、作業批改、試卷生成、學習路徑 | **PadLearn 自家 LLM** |
| 學生資料庫、權限、SSO | **PadLearn** |

### C.3 商業模式對話模板

> CyberPuppy 核心模型與程式碼維持 Apache 2.0，貴方可立即下載、自部署、自修改，沒有授權費。
>
> 我方提供的付費服務包括：(1) 整合與部署諮詢；(2) 針對貴方資料的私有 adapter 客製化（T3）或新領域頭擴充（T4）；(3) SLA 維護與安全更新；(4) 協助 PDPO 影響評估。
>
> 適合先以 Apache-2.0 版本做 PoC 驗證，確認價值後再決定服務範圍。

### C.4 隱私 / PDPO talking points

- 我方建議 **on-prem / 校內 GPU** 部署為首選；模型權重與推論完全留校，僅 hash + 分數出校
- 若選 SaaS：HK 境內機房；不存原文，只存 SHA-256 + score
- 教師 override 與 active learning：必須先取得家長同意才能回流訓練池
- PCPD 2025-12-17 AI Deepfake Toolkit 為對齊基準，模組設計已對應其可解釋性義務

### C.5 後續邀約建議

1. 提供 ADR 0001（本檔）+ research brief（同目錄）作為附件
2. 安排 30 分鐘技術同步：(a) demo 現行 v1 (b) walkthrough Phase 1-3 (c) 對齊 PadLearn 整合面
3. 在試點前先簽 NDA + 資料處理協議（DPA）
