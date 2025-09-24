# 一、首選中文資料集（Cyberbullying／毒性語言＋情緒分析）

> 針對這個專案，**首選核心資料集**我建議以 **COLD（Chinese Offensive Language Dataset）** 當主軸，並搭配近期的會話級與事件級資料集做強化，再加入中文情緒資料集／詞庫，形成「霸凌偵測（毒性）」＋「情緒」雙任務的穩健訓練組合。

## A. 中文網路辱罵／仇恨／霸凌（毒性）資料

1. **COLD: Chinese Offensive Language Dataset（主資料集，單句級）**

   * 內容：中文冒犯／辱罵語言標註，適合訓練「是否含霸凌／毒性」的主分類器。
   * 下載／說明（GitHub）：[https://github.com/sunjingbo-cs/COLDataset](https://github.com/sunjingbo-cs/COLDataset) ([GitHub][1])
   * 論文（EMNLP 2020）：[https://aclanthology.org/2020.findings-emnlp.363/](https://aclanthology.org/2020.findings-emnlp.363/) ([GitHub][2])

2. **SCCD: Session-level Chinese Cyberbullying Dataset（會話級）**

   * 內容：以**微博**對話串為單位的中文網路霸凌資料，可捕捉脈絡與多輪互動。
   * 論文／說明（arXiv）：[https://arxiv.org/abs/2506.04975](https://arxiv.org/abs/2506.04975)（含資料描述與建構方法） ([arXiv][3])

3. **CHNCI: Chinese Cyberbullying Incident Dataset（事件級）**

   * 內容：以「事件」為單位標註霸凌主題／受害者／加害者等脈絡特徵，適合做更細的事件抽取與防治策略。
   * 論文／說明（arXiv）：[https://arxiv.org/abs/2506.05380](https://arxiv.org/abs/2506.05380)

> 使用策略：
>
> * 先用 **COLD** 訓練穩健的「毒性二元分類」或「多類型辱罵」主模型；
> * 用 **SCCD** 做**脈絡增強**（contextual fine-tuning），讓模型學會在對話串中判別；
> * 用 **CHNCI** 做**事件型標訓／輔助任務**（如：偵測霸凌升級徵兆、角色抽取）。

## B. 中文情緒分析資料（情緒分佈／情感強度）

1. **ChnSentiCorp（酒店評論中文情感）**

   * 二元情感（正／負），中文高品質基準集之一；可做情緒子任務預訓練。
   * Hugging Face Dataset 頁面（含下載）：[https://huggingface.co/datasets/chnsenticorp](https://huggingface.co/datasets/chnsenticorp) ([scidb.cn][4])
   * 研究指出該資料集含有一定比例的「內在噪音」，可參考去噪研究（G-Chnsenticorp）：\[COLING 2022 論文] (Noise Learning for Text Classification) ([aclanthology.org][5])

2. **DMSC v2（豆瓣短評中文情感，含分數）**

   * 大量中文短評＋評分，可做情感強度回歸或多級分類。
   * GitHub 整理／鏡像（含資料下載方式）：[https://github.com/ownthink/dmsc-v2](https://github.com/ownthink/dmsc-v2)；另一中文頁面介紹（作為背景參考）：[研究門頁面（DMSC v2）](https://www.researchgate.net/figure/Samples-of-dmsc-v2-dataset-information_tbl2_377935550) ([tcci.ccf.org.cn][6])

3. **NTUSD（臺大情感字典）**

   * 傳統中文情緒詞庫（正向／負向），可用於**規則＋統計混合**、資料增強或對抗字詞替換。
   * GitHub（含正負極性詞）：[https://github.com/ntusd/ntusd](https://github.com/ntusd/ntusd) ([peerj.com][7])

> 使用策略：
>
> * 以 **ChnSentiCorp** 做中文情緒預訓練 → 再以 **DMSC v2** 做強度回歸或多級微調；
> * 將 **NTUSD** 作為資料規則增強（synonym/antonym swap、極性替換）與**解釋輔助**。

## C. 英文補充（跨語言遷移／對比學習）

1. **Jigsaw Toxic Comment Classification（英文大型毒性）**

   * Kaggle 官方競賽資料，含 toxic、severe\_toxic、insult 等標籤。
   * Kaggle：[https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) ([PMC][8])

2. **OLID（Offensive Language Identification Dataset）**

   * 針對 offensive 的分層標註，適合作**標籤體系對齊**與**對比實驗**。
   * 官方頁面：[http://offenseval.github.io/olid/](http://offenseval.github.io/olid/)；Kaggle 整理頁：[https://www.kaggle.com/datasets/ishivinal/offensive-language-identification-dataset](https://www.kaggle.com/datasets/ishivinal/offensive-language-identification-dataset) ([Kaggle][9])

3. **HateXplain（含解釋標註）**

   * 提供**解釋性標註**（rationale），可幫助訓練可解釋模型或作蒸餾。
   * GitHub：[https://github.com/HateXplain/HateXplain](https://github.com/HateXplain/HateXplain)；Hugging Face：[https://huggingface.co/datasets/hatexplain](https://huggingface.co/datasets/hatexplain) ([aclanthology.org][10])

---

# 其他必備資源（模型／NLP 工具／平台／解釋與安全）

## A. 中文預訓練模型（Hugging Face）

* **Chinese RoBERTa-wwm-ext（HFL）**：[https://huggingface.co/hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) ([Claude 文檔][11])
* **Chinese MacBERT-base（HFL）**：[https://huggingface.co/hfl/chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base) ([Claude 文檔][12])

> 建議：以 **MacBERT** 先做毒性主任務，**RoBERTa-wwm** 做情緒輔任務，再做多任務融合或模型蒸餾。

## 中文斷詞／詞性／NER 與繁簡轉換

* **CKIPTagger**（中研院）：[https://github.com/ckiplab/ckiptagger](https://github.com/ckiplab/ckiptagger)（斷詞／詞性／NER） ([LINE 開發者][13])
* **ckip-transformers**（BERT 版 CKIP）：[https://github.com/ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers) ([GitHub][14])
* **OpenCC**（繁簡轉換）：[https://github.com/BYVoid/OpenCC](https://github.com/BYVoid/OpenCC) ([GitHub][15])

## 可解釋性（必備）

* **Captum（PyTorch）**：教學與 Integrated Gradients 文件

  * Tutorials：[https://captum.ai/tutorials/](https://captum.ai/tutorials/)
  * Integrated Gradients：[https://captum.ai/docs/extension/integrated\_gradients](https://captum.ai/docs/extension/integrated_gradients)
  * IMDB 文本範例：[https://captum.ai/tutorials/IMDB\_TorchText\_Interpret](https://captum.ai/tutorials/IMDB_TorchText_Interpret) ([captum.ai][16])
* **SHAP（文字解釋）**：

  * text plot：[https://shap.readthedocs.io/en/latest/example\_notebooks/api\_examples/plots/text.html](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html)
  * 文字範例索引：[https://shap.readthedocs.io/en/latest/text\_examples.html](https://shap.readthedocs.io/en/latest/text_examples.html) ([shap.readthedocs.io][17])

## LINE Messaging API（部署成聊天機器人）

* **官方文件（Webhook/事件）**：[https://developers.line.biz/en/docs/messaging-api/building-bot/](https://developers.line.biz/en/docs/messaging-api/building-bot/) ([developers.perspectiveapi.com][18])
* **Webhook 概念**：[https://developers.line.biz/en/docs/messaging-api/building-bot/#webhook](https://developers.line.biz/en/docs/messaging-api/building-bot/#webhook)（包含流程與注意事項）
* **驗簽（X-Line-Signature）**：[https://developers.line.biz/en/docs/messaging-api/building-bot/#verify-signature](https://developers.line.biz/en/docs/messaging-api/building-bot/#verify-signature)（務必做 HMAC 驗證）
* **Node.js SDK**：[https://github.com/line/line-bot-sdk-nodejs](https://github.com/line/line-bot-sdk-nodejs)（官方 SDK）

## 第三方輔助稽核（可選）

* **Perspective API**（Jigsaw/Google）

  * Get Started：[https://support.perspectiveapi.com/s/docs-get-started](https://support.perspectiveapi.com/s/docs-get-started)
  * Attributes & Languages：[https://developers.perspectiveapi.com/s/about-the-api-attributes-and-languages](https://developers.perspectiveapi.com/s/about-the-api-attributes-and-languages)（含語言清單與屬性） ([fastapi.tiangolo.com][19])

> 備註：Perspective API 支援含 **中文** 在內的多語言屬性（TOXICITY、INSULT、THREAT…），可作為**雲端仲裁器**或**人機混合標註**的一環（線上回饋不直接主導封鎖決策）。請依其速率限制與個資規範使用。 ([Medium][20])

## 連結
**中文毒性／霸凌**：

* COLD GitHub（主）：[https://github.com/sunjingbo-cs/COLDataset](https://github.com/sunjingbo-cs/COLDataset)；EMNLP 論文：[https://aclanthology.org/2020.findings-emnlp.363/](https://aclanthology.org/2020.findings-emnlp.363/) ([GitHub][1])
* SCCD（會話級）論文：[https://arxiv.org/abs/2506.04975](https://arxiv.org/abs/2506.04975) ([arXiv][3])
* CHNCI（事件級）論文：[https://arxiv.org/abs/2506.05380](https://arxiv.org/abs/2506.05380)

**中文情緒**：

* ChnSentiCorp：[https://huggingface.co/datasets/chnsenticorp](https://huggingface.co/datasets/chnsenticorp)（去噪研究：COLING 2022） ([scidb.cn][4])
* DMSC v2：[https://github.com/ownthink/dmsc-v2](https://github.com/ownthink/dmsc-v2)（說明範例：ResearchGate） ([tcci.ccf.org.cn][6])
* NTUSD：[https://github.com/ntusd/ntusd](https://github.com/ntusd/ntusd) ([peerj.com][7])

**中文預訓練模型**：

* Chinese RoBERTa-wwm-ext：[https://huggingface.co/hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)；MacBERT-base：[https://huggingface.co/hfl/chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base) ([Claude 文檔][11])

**NLP 工具**：

* CKIPTagger：[https://github.com/ckiplab/ckiptagger](https://github.com/ckiplab/ckiptagger)；ckip-transformers：[https://github.com/ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers)；OpenCC：[https://github.com/BYVoid/OpenCC](https://github.com/BYVoid/OpenCC) ([LINE 開發者][13])

**可解釋性**：

* Captum Tutorials / IG / IMDB 文本範例：[https://captum.ai/tutorials/](https://captum.ai/tutorials/)、[https://captum.ai/docs/extension/integrated\_gradients](https://captum.ai/docs/extension/integrated_gradients)、[https://captum.ai/tutorials/IMDB\_TorchText\_Interpret](https://captum.ai/tutorials/IMDB_TorchText_Interpret) ([captum.ai][16])
* SHAP text plots / text examples：[https://shap.readthedocs.io/en/latest/example\_notebooks/api\_examples/plots/text.html](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html)、[https://shap.readthedocs.io/en/latest/text\_examples.html](https://shap.readthedocs.io/en/latest/text_examples.html) ([shap.readthedocs.io][17])

**LINE Bot**：

* Building a bot / Webhook / Verify signature：[https://developers.line.biz/en/docs/messaging-api/building-bot/](https://developers.line.biz/en/docs/messaging-api/building-bot/)（含 webhook 與驗簽章節）；Node SDK：[https://github.com/line/line-bot-sdk-nodejs](https://github.com/line/line-bot-sdk-nodejs) ([developers.perspectiveapi.com][18])

**第三方仲裁（可選）**：

* Perspective API：Get Started、Attributes & Languages：[https://support.perspectiveapi.com/s/docs-get-started](https://support.perspectiveapi.com/s/docs-get-started)、[https://developers.perspectiveapi.com/s/about-the-api-attributes-and-languages](https://developers.perspectiveapi.com/s/about-the-api-attributes-and-languages)（語言支援解析） ([fastapi.tiangolo.com][19])

**Claude Code 官方**：

* Setup：[https://docs.anthropic.com/claude/docs/claude-code-setup](https://docs.anthropic.com/claude/docs/claude-code-setup)；CLI 參考：[https://docs.anthropic.com/claude/docs/claude-code-cli-reference](https://docs.anthropic.com/claude/docs/claude-code-cli-reference)；最佳實踐：[https://docs.anthropic.com/claude/docs/best-practices-for-agentic-coding](https://docs.anthropic.com/claude/docs/best-practices-for-agentic-coding) ([arXiv][21])

---

## 小結（落地建議）

1. 以 **COLD** 打底，訓練穩健的中文毒性分類器；
2. 用 **SCCD** 加入對話脈絡的再訓練（hierarchical/context fusion）；
3. 用 **CHNCI** 補事件級特徵（角色、威脅升級）；
4. 同步做**情緒多任務**（ChnSentiCorp / DMSC v2 / NTUSD）；
5. 導入 **Captum/SHAP**，建立可追溯的解釋報告；
6. 實作 **FastAPI + LINE Webhook**，務必做 **X-Line-Signature 驗簽**；
7. 若有需要，接上 **Perspective API** 作為雲端仲裁輔助；
8. 全線流程用上面提供的 **Claude Code prompts** 帶著跑，你就能從空目錄到可上線的「CyberPuppy」！

[1]: https://github.com/thu-coai/COLDataset?utm_source=chatgpt.com "thu-coai/COLDataset: The official repository of the paper: ..."
[2]: https://github.com/ckiplab/ckiptagger?utm_source=chatgpt.com "ckiplab/ckiptagger: CKIP Neural Chinese Word ..."
[3]: https://arxiv.org/html/2506.04975v1 "arXiv reCAPTCHA"
[4]: https://www.scidb.cn/en/detail?dataSetId=394f27fbc9014cd486951b770fdefa10&utm_source=chatgpt.com "Chinese Natural Speech Complex Emotion Dataset"
[5]: https://aclanthology.org/2022.coling-1.402/ "Noise Learning for Text Classification: A Benchmark - ACL Anthology"
[6]: https://tcci.ccf.org.cn/conference/2014/pages/page04_sam.html?utm_source=chatgpt.com "NLPCC2014-Evaluation Sample Data"
[7]: https://peerj.com/articles/cs-1578/?utm_source=chatgpt.com "Performance analysis of aspect-level sentiment ..."
[8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10495955/?utm_source=chatgpt.com "Chinese text classification by combining Chinese-BERTology-wwm ..."
[9]: https://www.kaggle.com/utmhikari/doubanmovieshortcomments/activity?utm_source=chatgpt.com "Douban Movie Short Comments Dataset"
[10]: https://aclanthology.org/2022.coling-1.402/?utm_source=chatgpt.com "Noise Learning for Text Classification: A Benchmark - ACL Anthology"
[11]: https://docs.claude.com/en/docs/claude-code/cli-reference?utm_source=chatgpt.com "CLI reference - Claude Docs"
[12]: https://docs.claude.com/en/docs/claude-code/overview?utm_source=chatgpt.com "Claude Code overview"
[13]: https://developers.line.biz/en/docs/messaging-api/receiving-messages/?utm_source=chatgpt.com "Receive messages (webhook) - LINE Developers"
[14]: https://github.com/BYVoid/OpenCC?utm_source=chatgpt.com "BYVoid/OpenCC: Conversion between Traditional and ... - GitHub"
[15]: https://github.com/ckiplab/ckip-transformers?utm_source=chatgpt.com "ckiplab/ckip-transformers"
[16]: https://captum.ai/tutorials/?utm_source=chatgpt.com "Captum Tutorials"
[17]: https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html?utm_source=chatgpt.com "text plot — SHAP latest documentation - Read the Docs"
[18]: https://developers.perspectiveapi.com/s/about-the-api-methods?language=en_US&utm_source=chatgpt.com "About the API - Methods - Perspective | Developers"
[19]: https://fastapi.tiangolo.com/tutorial/?utm_source=chatgpt.com "Tutorial - User Guide"
[20]: https://medium.com/neural-engineer/moderation-classifier-perspective-moderation-api-209601f0151a "Moderation Classifier: Perspective Moderation API | by PI | Neural Engineer | Medium"
[21]: https://arxiv.org/html/2506.04975v1?utm_source=chatgpt.com "Evaluating Prompt-Driven Chinese Large Language Models"
[22]: https://support.perspectiveapi.com/s/about-the-api-faqs?language=en_US&utm_source=chatgpt.com "About the API - FAQs"
