# ADR 0001 Supporting Research Brief — 2026 Chinese Moderation SOTA

*Compiled 2026-04-15 by Claude Code research agent. 28 web searches; ≥2 source cross-validation enforced; single-source claims explicitly flagged.*

> 完整 ADR 見同目錄 `0001-cyberpuppy-2026-upgrade.md`。本檔僅作研究底稿，便於日後追溯與補充。

## Executive Summary（10 條優先建議）

1. **以 Qwen3-4B/8B + LoRA bf16 取代 MacBERT-base 編碼器**（Apache-2.0、119 語涵蓋繁中與粵語；32 GB VRAM 充裕）。
2. **採用 Qwen3Guard (0.6/4/8B) 作為 production 守門員 baseline**；Apache-2.0、1.19 M safety 樣本訓練、明確涵蓋中文。
3. **不採用 Llama-Guard 3/4 與 ShieldGemma**：均不支援中文（Llama-Guard 系列只覆蓋 8 種語言，無中文）。
4. **資料：擴增 SCCD（對話級）+ CHNCI（事件級）+ ToxiCN/STATE-ToxiCN（span 級 + 仇恨俚語詞典）+ ToxiCloakCN（對抗 robustness eval）+ ToxiCN-MM（多模態未來擴展）**。
5. **粵語 / HK 是真實缺口**：僅有 YueTung、CantoneseLLM、Chinese ModernBERT 處理 HK 文本；無公開 HK-bullying 語料 → 必須合成 + 人工驗證。
6. **部署：vLLM 0.17+ on CUDA 13 + AWQ-Marlin（741 tok/s, 10.9× FP16）**；TensorRT-LLM 高並發快 8-13% 但部署彈性低。
7. **多任務：單一 Qwen3 backbone + 多 LoRA 頭，不分 adapter**（MHR 論文：averaged 單 adapter 已足夠）。
8. **safety 對齊用 GRPO，不只 DPO**：14 B 規模上 GRPO > DPO（多項 2025 證據 + Qwen 官方 pipeline）。
9. **HK 法規：PDPO 為主而非 PIPL；PCPD 2025-12-17 AI Deepfake Toolkit 直接適用校園場景**。
10. **HK EDB 經費實情是 HK$500 M（每校 50 萬，2025-26 至 2027-28）**，program 名稱 "AI for Empowering Learning and Teaching"；email 提到的 HK$5B / 智啟學教 在政府公告中查無實證。

---

## 1 · 中文開源 LLM ≤ 14 B（適合 RTX 5090 微調）

| 模型 | 規模 | 授權 | 中文表現 | 適合性 |
|---|---|---|---|---|
| Qwen3-4B / 8B / 14B | 4B / 8B / 14B dense | **Apache 2.0** | 119 語含繁中粵語；14B ≈ Qwen2.5-32B | ⭐⭐⭐⭐⭐（首選） |
| Qwen3Guard 0.6/4/8B | 同上 | **Apache 2.0** | 1.19M safety + Qwen3 backbone；中文 SOTA | ⭐⭐⭐⭐⭐（守門員） |
| DeepSeek-R1-Distill-Qwen-7B/14B | 7B / 14B | MIT (繼承 Qwen base) | 強，但 reasoning-oriented | ⭐⭐⭐ |
| GLM-4.5 / GLM-4.5-Air / GLM-5 | 32B+ MoE active | 開源 | 強但太大 | ❌（單卡） |
| Yi-1.5 6B/9B/34B | 6/9/34B | 非完全 OSI | 仍可用，但已老 | ⭐⭐ |
| InternLM3-8B-Instruct | 8B | Apache-style | 與 GPT-4o-mini 同級宣稱 | ⭐⭐⭐⭐ |
| Baichuan 4 | API 為主 | 不確定（單一來源宣稱） | 中文榜首 | ⭐（封閉風險） |

**MacBERT-base 已落後**：Chinese ModernBERT (arXiv 2510.12285) 為 from-scratch 中文編碼器（WWM, 8K context, 32K vocab）；ModernBERT-style 編碼器 + LLM-as-classifier 雙路徑都已超越 MacBERT 系列。

---

## 2 · 中文仇恨/霸凌資料集（2025-2026 新增）

- **COLD**（EMNLP 2022）：37,480 留言，無官方 v2。HED-COLD（EMNLP 2025）為同音字增強衍生品。
- **SCCD**（COLING 2025 / arXiv 2501.15042）：677 sessions Weibo，**首個對話級中文霸凌語料**，52.3% bullying。
- **CHNCI**（arXiv 2505.20654, May 2025）：220,676 留言 / 91 incidents，跨 Douyin/Weibo/Xiaohongshu/Bilibili，**首個事件級**。
- **STATE-ToxiCN**（ACL Findings 2025）：8,029 貼文 + 9,533 四元組（target/argument/hateful/group）+ 830 詞中文仇恨俚語詞典。
- **ToxiCloakCN**（EMNLP 2024）：~12K 對抗樣本（同音字、emoji 偽裝）；**僅作 robustness 評測**。
- **ToxiCN-MM**（NeurIPS 2024 D&B）：12K meme 圖文。
- **PANDA**（arXiv 2501.00697, Jan 2025）：~12K 中文 counterspeech 對；**警告 LLM-as-Judge 中文標錯率非小**。

**Cantonese / HK**：
- 無專屬公開仇恨語料。
- Jiang et al. arXiv 2503.03702（NAACL 2025）建 2B token 粵語語料（含 LIHKG），但通用非 bullying。
- hon9kon9ize CantoneseLLM-6B：社群模型，無正式 evaluation（單一來源）。

**角色標籤（perpetrator/victim/bystander）**：無大型監督式中文語料；2024 Tsinghua DEK 論文用無監督聚類。

---

## 3 · Fine-tuning 技術（Apr 2026 SOTA）

- **LoRA + DoRA r=16~64, target_modules="all-linear" 為 2026 默認**。
- **QLoRA 4-bit 在 Qwen3 系列數值偏差大** → Unsloth 官方建議改 bf16 LoRA。
- RTX 5090 32 GB 可吃下：0.8B→3 GB / 2B→5 GB / 4B→10 GB / 9B→22 GB。**Qwen3-8B bf16 LoRA 是 sweet spot**；14B 需 grad checkpoint 或 4-bit。
- **GRPO > DPO 在 14B 規模**（arXiv 2503.21819, 2505.20087，及 Qwen3Guard pipeline 採 curriculum GRPO）。
- KTO / IPO / ORPO 為次選；DPO 仍是好部署的 baseline。
- **CAI / RLAIF**（SparkCo 2025 報告）：Llama-3-8B 攻擊成功率降 40.8%、helpfulness 降 9.8%（單一來源、需驗證）。
- **守門員蒸餾**：MrGuard / X-Guard (EMNLP 2025) 用合成資料 + SFT + curriculum GRPO 補非英語不足。
- **SetFit + ModernBERT**：8 shot 即達近 full FT 表現；對 role/emotion bootstrap 極有用。

---

## 4 · 守門員 / 量化 / 部署（Production 2026）

**守門員模型**：

| 模型 | 中文 | 授權 | 評語 |
|---|---|---|---|
| Llama-Guard 3 (8B) / 4 (12B) | ❌（8 種語不含中文） | Llama Community | 不適用 |
| ShieldGemma 2 (4B 圖 / 9B 文) | 偏英 | Gemma | 不建議 |
| **Qwen3Guard 0.6/4/8B** | ✅ SOTA | **Apache 2.0** | **首選** |
| WildGuard / Aegis-2.0 | 英文重心，中文衰退顯著 | 各異 | 不建議 |
| NeMo Guardrails | orchestration 框架，無自家 model | Apache 2.0 | 與 Qwen3Guard 互補 |

**量化**：
- AWQ + Marlin (vLLM)：741 tok/s，10.9× FP16
- GPTQ + Marlin：712 tok/s
- GGUF Q4_K_M：92% quality retention，llama.cpp/Ollama 路徑
- FP8：Blackwell（RTX 5090）原生支援，TensorRT-LLM 強項
- NF4 bitsandbytes：唯一還支援訓練的量化（QLoRA）

**推論引擎**（單卡 RTX 5090）：
- **vLLM 0.17+**：易部署、彈性最佳
- **TensorRT-LLM**：高並發快 8-13%，FP8 利
- **SGLang**：multi-turn 共享 context 強
- **Ollama / llama.cpp**：開發便利，p95 < 200 ms 不保證

**延遲**：Qwen3-8B BF16 在 vLLM 上 ~194-197 tok/s（單一來源 HF discussion，需自行驗證）；4B-class 分類器 single forward < 100 ms p95 是合理目標。

---

## 5 · HK / 粵語 / DSE 法規與市場

**HK Education Bureau (EDB)**：
- 2025-04：Digital Policy Office 發 GenAI 技術指引
- **2025-12-16**：EDB 推出 **"AI for Empowering Learning and Teaching" Funding Programme**，HK$500 M（取自 HK$2B QEF 儲備），每校 HK$500K，2025-26 至 2027-28 學年（截止 2028-08-31）；**沒提到 bullying / safety AI** → 商機真實
- **email 中 HK$5B / 智啟學教 名稱在所有政府來源中查無實證**（這是事實校正點）

**HK PCPD（隱私專員）**：
- 2024-06-11：AI Model Personal Data Protection Framework
- 2025-03-31：GenAI 員工使用 checklist
- **2025-12-17**：**"Abuse of AI Deepfakes: Toolkit for Schools and Parents"** — 明列 image-based sexual violence、cyberbullying、scams、disinformation、PDPO §64 doxxing → **PadLearn 必讀**
- PDPO Amendment 預計 2026 全面生效；§33 跨境傳輸限制至今未強制執行

**PIPL vs PDPO**：
- HK 適用 PDPO，**不是** PIPL
- PIPL §38 將 HK 視為「跨境傳輸」，需 SCC 或安全評估
- PDPO **無資料本地化要求**，PIPL **有**
- → **建議全棧 HK 境內或本機**

**Cantonese 模型**：
- YueTung（Qwen-2.5-7B + YueData）SOTA on Yue-* benchmarks（通用，非 hate）
- Chinese ModernBERT（簡中為主）
- CantoneseLLM-6B（社群、單一來源）

**HK 學生心理健康**（市場理據）：
- HKCSS：40%+ DSE 學生顯示焦慮 / 抑鬱
- HKFP 2025-10：青少年向 AI chatbot 求心理諮商
- SCMP：HK 霸凌案件 4 年翻倍

---

## 6 · 多任務 / Context Modelling

- 對話級 detection 是 2025 active sub-field（Wiley WIREs 2025 survey 確認 context-awareness 為最大 open problem）
- Applied Sciences 2025：ALBERT + temporal/behavioral 達 88.4%；CBNet 用 10-post 滑窗
- 角色 detection：無大型監督中文語料；多用無監督聚類或心理量表
- emotion + intensity：**SemEval-2025 Task 11**（>30 語含中文，6 類 emotion + intensity，>700 隊參賽）；Chinese 主要語料：EmotionTalk、M3ED、Chinese EmoBank

---

## 7 · Evaluation 2026

- 對抗：**ToxiCloakCN 是中文同音字/emoji 偽裝的標準測試**；MMBERT (arXiv 2508.00760) 為 2025 對抗 robustness 模型
- TextFlint 中文仍可用（2021 起，無 2025 大改）
- **OpenCompass 司南**：中文 LLM eval 標準框架，含 safety dimension、CNFinBench
- 校準：ECE 對 binning 敏感；建議 **SmoothECE**（arXiv 2501.19047）+ class-wise ECE
- 人在迴圈：PANDA 經驗 — LLM-as-Judge 在中文必須抽樣人審

---

## 單一來源 / 待驗證清單

| 宣稱 | 來源 | 為何要小心 |
|---|---|---|
| Baichuan 4 是 Apache 2.0 | IntuitionLabs 概覽 | 主要 Baichuan 通訊強調 API；無 Baichuan 4 weights 授權檔 |
| 無 Yi-2 模型 | 由缺席推論 | 01.AI 可能在資料截止後發佈，需 GitHub 確認 |
| Qwen3-8B 在 RTX 5090 持續 194-197 tok/s | HF discussion thread | 單名使用者回報，非官方 benchmark |
| ShieldGemma-2B 對中文支援度 | skywork.ai | 二級彙整來源；Google 官方文件未給中文數字 |
| CantoneseLLM-6B production-ready | 單一 HF model card | 社群作品；無 model card 以外公開 evaluation |
| RTX 5090 4B 模型 sub-200 ms TTFT | Yotta Labs 部落格 | SEO 性質，雖與 arXiv 2601.09527 一致但該 arXiv 也是單一來源 |
| CAI 9.8% helpfulness drop | SparkCo 部落格 | 部落格引用未發表數字，原論文未找到 |
| Anthropic 16M / 24K accounts DeepSeek 蒸餾控訴 | CNBC / The Register / Fortune / Futurism | 4 家轉述但實質為 Anthropic 單一公司宣稱 |
| HKFYG 27.8% 焦慮率 | HKFYG 新聞稿 | 倡議組織自報，方法論未交叉檢核 |

## What I Could NOT Find（誠實標記）

1. 任何 **粵語/HK 繁中專屬公開**仇恨/霸凌語料
2. 任何 **官方 COLDv2 / COLD 更新**
3. **ShieldGemma 中文 benchmark 數字**（Google DeepMind 文件靜默）
4. ShieldGemma 2 文字版的 production-grade 中文分類
5. 大規模公開 **角色標註**（perpetrator/victim/bystander）中文語料
6. 任何證據支持 email 中 **「智啟學教」/HK$5B** 的命名與金額（實際是 "AI for Empowering Learning and Teaching"，HK$500M）
7. Qwen3Guard 在 **COLD/SCCD 上**的官方 F1 break-down（只有「中文 SOTA」泛述）
8. Qwen3-14B 在 **RTX 5090 32 GB**的官方微調 benchmark（需從 Qwen3.5 數字外推）
9. MacBERT-base vs Qwen3-4B 在 COLD 的直接公平比較
10. "CHEF" 中文對抗 robustness benchmark（命名可能與其他縮寫混淆）

---

## 主要來源 URL（依 ADR 引用順序）

完整 116 條 URL 已在研究代理人輸出內保留；本檔案僅列關鍵節點：

- Qwen3 / Qwen3Guard：https://qwenlm.github.io/blog/qwen3/、https://arxiv.org/abs/2505.09388、https://arxiv.org/abs/2510.14276
- COLD：https://aclanthology.org/2022.emnlp-main.796/
- SCCD：https://arxiv.org/abs/2501.15042
- CHNCI：https://arxiv.org/abs/2505.20654
- STATE-ToxiCN：https://arxiv.org/abs/2501.15451
- ToxiCloakCN：https://arxiv.org/abs/2406.12223
- vLLM AWQ benchmark：https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks
- HK EDB AI 經費：https://www.info.gov.hk/gia/general/202512/16/P2025121600261.htm、https://www.news.gov.hk/eng/2025/12/20251216/20251216_112101_776.html
- HK PCPD AI Deepfake Toolkit：https://www.kwm.com/hk/en/insights/latest-thinking/privacy-commissioner-issues-deepfake-toolkit-for-schools-and-parents-what-hong-kong-education-providers-need-to-know.html
- HK PDPO vs PIPL：https://hkytl.com/2025/11/05/hong-kong-data-privacy-cybersecurity-compliance/
- LoRA / Unsloth：https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Multi-Head Adapter Routing：https://arxiv.org/abs/2211.03831
- Multi-Objective GRPO：https://arxiv.org/pdf/2505.20087
- ModernBERT：https://arxiv.org/abs/2510.12285
- PANDA：https://arxiv.org/abs/2501.00697
- SemEval-2025 Task 11：https://arxiv.org/html/2503.07269v2
