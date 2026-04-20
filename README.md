# CyberPuppy — 中文網路霸凌偵測與內容審核系統

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/Code-Apache%202.0-blue.svg)](LICENSE)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Weights-CC%20BY--NC--SA%204.0-lightgrey.svg)](MODEL_LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-CyberPuppy%20v5-yellow)](https://huggingface.co/thc1006/cyberpuppy-v5-bilingual)

[English](#english) | [ADR](docs/adr/0001-cyberpuppy-2026-upgrade.md) | [HF Models](https://huggingface.co/thc1006/cyberpuppy-v5-bilingual)

> **中文社群最強開源毒性偵測系統** — 超越 PCR-ToxiCN 發表 SOTA，具備諧音攻擊防禦能力

---

## 效能

| Benchmark | CyberPuppy v5 | 對比 |
|---|---|---|
| **COLD toxicity F1** | **0.8336** | DoD 0.83 ✅ |
| **PCR-ToxiCN (真實世界)** | **0.6890** | **超越 SOTA 0.672** |
| **ToxiCloakCN homo F1** | **0.8380** | 諧音攻擊防禦 |
| **CNTP homo recall drop** | **-0.37%** | 幾乎免疫 |
| **6 繁中威脅句** | **6/6** | 零漏報 |

## 架構

```
使用者輸入
  │
  ├─► LoRA-A (文字)：Qwen3-8B + LoRA r=32 → 4-head 分類
  │      ↓ softmax
  │      0.75 ×
  │            ├─► 最終預測（toxicity / bullying / role / emotion）
  │      0.25 ×
  │      ↑ softmax
  └─► LoRA-B (拼音)：Qwen3-8B + LoRA r=32 → 4-head 分類
         └─ 輸入 = pypinyin 轉換（諧音字 → 相同拼音 → 相同預測）
```

**為何有效**：攻擊者用「勾史」替代「狗屎」，文字模型可能被騙，但拼音模型看到的都是 "gou shi" → 識破攻擊。

## 快速開始

### 安裝

```bash
git clone https://github.com/thc1006/cyberbully-zh-moderation-bot.git
cd cyberbully-zh-moderation-bot

# 建議使用 uv（快速）
uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -e .
```

### 推論（需 GPU，~32GB VRAM for dual-LoRA）

```python
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from pypinyin import pinyin, Style
import re

device = torch.device("cuda")
dtype = torch.bfloat16

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")

# LoRA-A（文字）
base_a = AutoModel.from_pretrained("Qwen/Qwen3-8B-Base", torch_dtype=dtype, device_map=device)
model_a = PeftModel.from_pretrained(base_a, "thc1006/cyberpuppy-v5-bilingual")
heads_a = torch.load(hf_hub_download("thc1006/cyberpuppy-v5-bilingual", "heads.pt"),
                     map_location=device, weights_only=False)

# LoRA-B（拼音）
base_b = AutoModel.from_pretrained("Qwen/Qwen3-8B-Base", torch_dtype=dtype, device_map=device)
model_b = PeftModel.from_pretrained(base_b, "thc1006/cyberpuppy-v5-pinyin-lora")
heads_b = torch.load(hf_hub_download("thc1006/cyberpuppy-v5-pinyin-lora", "heads.pt"),
                     map_location=device, weights_only=False)

# 拼音轉換
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
def to_pinyin(text):
    return " ".join(
        pinyin(ch, style=Style.NORMAL)[0][0] if _HAN.match(ch) else ch
        for ch in text if ch.strip()
    )

# ���類
text = "勾史一個"  # = 狗屎一個
enc_t = tok(text, return_tensors="pt", truncation=True, max_length=192).to(device)
enc_p = tok(to_pinyin(text), return_tensors="pt", truncation=True, max_length=192).to(device)

with torch.inference_mode():
    h_t = model_a(**enc_t).last_hidden_state[:, -1]
    h_p = model_b(**enc_p).last_hidden_state[:, -1]
    logits_t = heads_a["heads"]["toxicity"](h_t.float())
    logits_p = heads_b["heads"]["toxicity"](h_p.float())

probs = 0.75 * logits_t.softmax(-1) + 0.25 * logits_p.softmax(-1)
labels = ["none", "toxic", "severe"]
print(f"{text} → {labels[probs.argmax(-1).item()]}")  # toxic
```

### API 部署

```bash
# 啟動 FastAPI server
CP_MODEL_DIR=models/cyberpuppy_v5_bilingual_qwen3_8b \
  uvicorn api.v2_2_app:app --host 0.0.0.0 --port 8000

# POST /v2/analyze
curl -X POST http://localhost:8000/v2/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "你這個笨蛋"}'
```

## 標籤體系

| 任務 | 標籤 | 用途 |
|---|---|---|
| toxicity | `none` / `toxic` / `severe` | 毒性偵測 |
| bullying | `none` / `harassment` / `threat` | 霸凌分類 |
| role | `none` / `perpetrator` / `victim` / `bystander` | 角色辨識 |
| emotion | `pos` / `neu` / `neg` | 情緒分析 |

## 訓練資料

179,186 筆樣本，涵蓋 6 個中文毒性/霸凌資料集：

| 來源 | 數量 | 用途 |
|---|---|---|
| COLD (繁體) | 25,659 | 基礎毒性 |
| SCCD (繁體) | 28,426 | 對話級霸凌 |
| STATE-ToxiCN (繁體) | 5,781 | 仇恨俚語 |
| ToxiCloakCN × 3 (繁體) | 33,012 | 對抗一致性 |
| 簡體副本 | 70,870 | 雙語覆蓋 |
| CNTP | 15,438 | 真實擾動對 |

## 對抗攻擊防禦

| 攻擊類型 | 範例 | 防禦狀態 |
|---|---|---|
| 諧音替換 | 勾���→狗屎, 四調→死掉 | ✅ 拼音 LoRA 完全免疫 |
| ��字替換 | 4了→死了, 13→逼 | ✅ 雙語訓練覆蓋 |
| 字母替換 | 装X→装逼, NMSL | ✅ CNTP 對抗訓練 |
| 創意暗語 | 密碼→你媽 | ⚠️ 部分覆蓋 |
| 英文諧音 | funny mud pee | ⚠️ 有限覆蓋 |

## 限制

1. **僅支援中文** — 英文輸入不在訓練分佈內
2. **需要 GPU** — dual-LoRA 推論需 ~30GB VRAM（單 LoRA ~16GB）
3. **最大長度 192 tokens** — 超長文本會被截斷
4. **對話上下文** — 僅分析單句，不理解對話脈絡
5. **新型混淆** — 未見過的創意攻擊可能逃避偵測
6. **不能取代人工** — 設計為輔助工具，需人工最終審核
7. **文化偏差** — 主要基於台灣/香港標註規範

## 專案結構

```
├── src/cyberpuppy/          # 核心 Python 套件
│   ├─��� models/              # Qwen3MultiHead, dual-LoRA logic
│   ├── data/                # 資料處理 pipeline
│   ├── eval/                # 評測 metrics
│   ├── explain/             # XAI (IG, SHAP)
│   ├── safety/              # 分級回覆策略
│   └── training/            # sampler, callbacks
├── scripts/                 # 訓練、評測、部���腳本
├── api/                     # FastAPI serving
├── bot/                     # LINE Bot
├── tests/                   # pytest 單元測試
└── docs/adr/                # Architecture Decision Records
```

## 授權（三層）

| 範疇 | 授權 | 檔案 |
|---|---|---|
| 程式碼 / 文件 / 配置 | **Apache 2.0** | [LICENSE](LICENSE) |
| Model weights / LoRA | **CC BY-NC-SA 4.0** | [MODEL_LICENSE](MODEL_LICENSE) |
| 訓練資料 | 不再分發（腳本重建） | [DATA_LICENSE_NOTICE.md](DATA_LICENSE_NOTICE.md) |

**商業授權**：提供客製化訓練服務（Apache 2.0 資料 + 客戶��有資料重訓）。���見 [ADR](docs/adr/0001-cyberpuppy-2026-upgrade.md)。

## 相關連結

- [HF Model: v5-bilingual (LoRA-A)](https://huggingface.co/thc1006/cyberpuppy-v5-bilingual)
- [HF Model: v5-pinyin-lora (LoRA-B)](https://huggingface.co/thc1006/cyberpuppy-v5-pinyin-lora)
- [Architecture Decision Record](docs/adr/0001-cyberpuppy-2026-upgrade.md)
- [Research Brief (28 sources)](docs/adr/0001-research-brief.md)

## 致謝

- [Qwen Team](https://github.com/QwenLM) — Qwen3-8B base model
- [THU-COAI](https://github.com/thu-coai) — COLD dataset
- [ToxiCN authors](https://github.com/Holence/ToxiCN) — STATE-ToxiCN, ToxiCloakCN
- [UTSNLPGroup](https://huggingface.co/datasets/UTSNLPGroup/PCR-ToxiCN) — PCR-ToxiCN benchmark

## 聯絡

- **作者**: Hung-Che Tsai (hctsai1006@cs.nctu.edu.tw)
- **Issues**: [GitHub Issues](https://github.com/thc1006/cyberbully-zh-moderation-bot/issues)

---

<a name="english"></a>

## English Summary

CyberPuppy v5 is a **state-of-the-art Chinese toxicity detection system** using a dual-LoRA ensemble on Qwen3-8B. It defends against homophone substitution attacks (where toxic characters are replaced with same-pronunciation alternatives) by combining a text LoRA with a pinyin LoRA — since homophones produce identical pinyin, the attack is neutralized by construction.

**Key results**: Exceeds published SOTA on PCR-ToxiCN (real-world RedNote posts): F1 0.6890 vs prior best 0.672. Achieves 97.2% prediction invariance against homophone attacks.

See [Hugging Face model card](https://huggingface.co/thc1006/cyberpuppy-v5-bilingual) for full details and usage instructions.
