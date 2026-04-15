# 回覆 Predick Ng（PadLearn / LittleP AI）— 繁中正式草稿

> 內容融合 ADR 0001 附錄 C；事實校正基於 2026-04-15 的 web 調研（28 條來源、9 條單一來源已標）。
> 寄出前請：
> 1. 換成您慣用的稱謂與簽名
> 2. 確認 ADR + research brief 兩附件版本
> 3. 視場合決定是否同步私訊先打招呼

---

**主旨**：Re: 請教 CyberPuppy 相關事宜｜派學（PadLearn）項目介紹

阿雨您好，

謝謝您詳細介紹 PadLearn 的願景與您對 CyberPuppy 的興趣，也感謝您把香港數位教育的政策背景一併分享。我用了一些時間研究貴方場景跟 CyberPuppy 目前實際的能力邊界，整理如下，希望對下一步的合作評估有幫助。

## 一、CyberPuppy 真實狀態（與 README 數字的校正）

我在過去兩天針對既有模型（`bullying_a100_best`，MacBERT-base 微調）做了完整重測，並對比 2025-10 開源的 Qwen3Guard-Gen-8B 守門員模型，誠實的數字如下：

| 模型 | COLD 測試集（5,320 樣本） | 繁中威脅句 6 例（手寫測試集） |
|---|---|---|
| 現行 v1 (MacBERT) | F1 weighted **0.825**、Accuracy 0.823 | 2/6（明示威脅句全失分） |
| Qwen3Guard-Gen-8B zero-shot | F1 weighted 0.746 | **6/6** |

兩個重點：

1. **F1=0.826 是真的**，但要強調：那是 COLD 測試集（簡體、社群留言）的數字。對應到貴方的繁中校園情境，現行模型在「我打死你」「希望你去死」這類明示威脅上有明顯盲點。
2. 我們已規劃 v2 訓練計畫對應這個缺口（見後段）。

## 二、CyberPuppy 是什麼、不是什麼

為了避免合作開始就有期望落差，先把範疇講清楚：

**CyberPuppy 做的**：
- 訊息級毒性／霸凌／角色／情緒分類
- 對話脈絡感知（v2 規劃中）
- 可解釋性輸出（SHAP / Integrated Gradients）
- LINE / HTTP API 整合
- 教師審核與回饋介面樣板

**CyberPuppy 不做**（這些 PadLearn 自家模組處理）：
- 作業批改、個性化學習路徑、模擬試卷生成
- 對話式學科 tutoring
- ASR / TTS、家長端 IM
- 學生資料庫與 SSO

## 三、回應您原信中的三個主要詢問

### 1. 訓練資料

現行 v1 用 COLD（清華 EMNLP 2022，37,480 留言）為主。**v2 規劃**將擴增四個 2024-2025 新資料集：

- **SCCD**（COLING 2025，arXiv 2501.15042）：677 個 Weibo 對話 session，**首個對話級**中文霸凌語料
- **CHNCI**（arXiv 2505.20654）：220,676 留言／91 個事件，跨 Douyin/Weibo/Xiaohongshu/Bilibili，**首個事件級**
- **STATE-ToxiCN**（ACL Findings 2025）：8,029 貼文 + 9,533 span 標註 + 830 詞中文仇恨俚語詞典
- **ToxiCloakCN**（EMNLP 2024）：~12K 對抗樣本，僅作 robustness 評測

繁中與 HK 在地語料是真實缺口；我們的計畫是 OpenCC 繁化 + 有限度抓取 LIHKG 公開帖文 + 人工驗證 ~1,500 樣本作為 HK 在地 evaluation set。

### 2. 客製化選項

我們設計成四層，PadLearn 可依需求挑：

| 層 | 內容 | 變更時間 |
|---|---|---|
| T1 | 各 head 信心閾值（嚴格／標準／寬鬆三檔） | 即時 |
| T2 | 在地俚語／校名／敏感詞詞庫注入 | 即時 |
| T3 | 學校私有對話資料微調 LoRA adapter（資料留校） | 1-2 週 |
| T4 | 新增分類頭（如自殘／物質濫用），全 backbone fine-tune | 4-8 週 |

### 3. 商業授權與技術支援（三層授權）

為尊重上游資料集（ToxiCN、STATE-ToxiCN、SCCD 等學術資料）作者意圖，我們採分層授權：

- **程式碼 / 文件 / 配置**：**Apache 2.0**（學校與廠商可自由使用、修改）
- **公開 model weights**：**CC BY-NC-SA 4.0**（學術研究、教育、非付費試點可直接使用）
- **付費服務**（CyberPuppy 團隊提供）：
  - **商業變體客製訓練**（這是貴方場景最需要的）：以 Apache-2.0 資料（COLD）+ 貴方自有學校資料（家長同意後）重訓產出**商業可用 model**，不受 CC BY-NC-SA 限制
  - 整合與部署諮詢
  - SLA 維護與安全更新
  - 協助 PDPO 影響評估

**對 PadLearn 實務意義**：
- 若貴方**內部 benchmark / 非付費試點**（如研究型學校 pilot）→ 可直接用 CC 版 weights 做 PoC，0 費用
- 若貴方進入**付費商業產品階段** → 需簽 T3/T4 客製訓練服務，我方產出乾淨授權之 commercial variant
- 這是 open-core 標準做法（類比 Red Hat / HuggingFace：公開 reference weights + 賣專業服務）

## 四、HK 政策面的兩處事實校正

我手邊查到的官方公告與您信中略有差異，不確定是版本差還是我漏看，方便對齊一下嗎？

- 您提到「2026 預算 20 億 + 5 億『智啟學教』」。我手邊查到的是：
  - 2025-09-17 施政報告預留 **2 B 數位教育儲備金**
  - 2025-12-16 EDB 公佈 **"AI for Empowering Learning and Teaching" Funding Programme**，總額 **5 億 HKD**，每校 **HK$50 萬**，使用期限至 2028-08-31
  - 計畫名稱在政府新聞稿中為英文版「AI for Empowering Learning and Teaching」，沒有「智啟學教」此中文名
- 您提到「790 間學校已申請」。這個具體數字我在 info.gov.hk / news.gov.hk / chinadailyhk 都沒查到，方便分享出處嗎？

如果您手邊有更新的官方來源，麻煩同步給我，這對我們做 HK 在地化路線圖很重要。

另外有兩個合規面相關的點先標一下：

- 香港適用 **PDPO 不是 PIPL**；若 PadLearn 任何元件部署於大陸雲，PIPL §38 跨境傳輸條款生效
- 2025-12-17 PCPD 發佈的 **"Abuse of AI Deepfakes: Toolkit for Schools and Parents"** 直接涵蓋校園 cyberbullying 與 PDPO §64 doxxing，建議納入合作前的合規對齊基準

## 五、建議下一步

如果以上方向 PadLearn 認可，我建議：

1. 我提供 ADR 0001（內部技術設計決策文件）+ 研究底稿（28 條來源），作為您內部技術評估的素材
2. 安排一場 30-45 分鐘的線上技術同步：
   - Demo 現行 v1 + 新做的 Qwen3Guard 守門 baseline
   - Walkthrough v2 訓練計畫（Phase 1-6，預計 12 週）
   - 對齊 PadLearn 整合面（API、教師 dashboard、家長同意流程）
3. 在進入試點前，雙方先簽 NDA + 資料處理協議（DPA）

謝謝您把這個機會帶來，期待後續更深入的對接。

祝 安好

[您的姓名]
CyberPuppy Maintainer
hctsai1006@cs.nctu.edu.tw

---

> 附件建議：
> 1. `docs/adr/0001-cyberpuppy-2026-upgrade.md`
> 2. `docs/adr/0001-research-brief.md`
> 3. （可選）`reports/qwen3guard_baseline.json` + `reports/qwen3guard_cold_eval.json` 作為實測佐證
