"""Verify merged model produces same predictions as v2.2 PEFT adapter.

Runs both on 6 繁中 + 100 COLD test samples, reports agreement rate + F1.
bf16 merge should match PEFT exactly (within numerical noise).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

from cyberpuppy.data.phase2 import LABELS
from cyberpuppy.models.qwen3_multihead import HEAD_DIMS, Qwen3MultiHead

BASE = "Qwen/Qwen3-8B-Base"
ADAPTER = "models/cyberpuppy_v2_2_qwen3_8b/best"
MERGED = "models/cyberpuppy_v2_2_merged"

LABEL2ID = {task: {v: i for i, v in enumerate(vals)} for task, vals in LABELS.items()}

TC_SAMPLES = [
    "今天天氣真好，我們去公園玩吧！",
    "你這個笨蛋，滾開！",
    "我恨你，希望你去死，沒人喜歡你",
    "謝謝你的幫忙，你真的是個很棒的人",
    "再嘴一句試試看，我打死你",
    "我覺得這部電影還不錯，演員演技也很到位",
]


def build(backbone, heads_pt: str, device, dtype):
    model = Qwen3MultiHead(backbone, hidden_size=backbone.config.hidden_size).to(device=device, dtype=dtype)
    state = torch.load(heads_pt, map_location=device, weights_only=False)
    model.heads.load_state_dict(state["heads"])
    model.eval()
    return model


def predict(tokenizer, model, device, texts: list[str]) -> dict:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=192,
                     return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return {t: out.logits[t].argmax(-1).cpu().tolist() for t in HEAD_DIMS}


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # --- Sample 100 COLD test items ---
    cold = [json.loads(l) for l in open("data/processed/v2/cold_test.jsonl")][:100]
    tc_texts = TC_SAMPLES
    all_texts = [r["text"] for r in cold] + tc_texts

    # --- Model A: PEFT v2.2 ---
    print("[A] Loading v2.2 PEFT (base+adapter) ...", flush=True)
    t0 = time.time()
    peft_back = AutoModel.from_pretrained(BASE, dtype=dtype, low_cpu_mem_usage=True,
                                            attn_implementation="sdpa")
    peft_back = PeftModel.from_pretrained(peft_back, f"{ADAPTER}/lora")
    peft_model = build(peft_back, f"{ADAPTER}/heads.pt", device, dtype)
    print(f"  loaded in {time.time()-t0:.1f}s; VRAM {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)

    t0 = time.time()
    peft_preds = predict(tokenizer, peft_model, device, all_texts)
    print(f"  predicted {len(all_texts)} in {time.time()-t0:.1f}s", flush=True)

    # Free VRAM before loading merged
    del peft_model, peft_back
    torch.cuda.empty_cache()

    # --- Model B: merged ---
    print("\n[B] Loading MERGED (fp32->bf16) ...", flush=True)
    t0 = time.time()
    merged_back = AutoModel.from_pretrained(MERGED, dtype=dtype, low_cpu_mem_usage=True,
                                              attn_implementation="sdpa")
    merged_model = build(merged_back, f"{MERGED}/heads.pt", device, dtype)
    print(f"  loaded in {time.time()-t0:.1f}s; VRAM {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)

    t0 = time.time()
    merged_preds = predict(tokenizer, merged_model, device, all_texts)
    print(f"  predicted {len(all_texts)} in {time.time()-t0:.1f}s", flush=True)

    # --- Compare ---
    print("\n=== Agreement between PEFT and MERGED ===", flush=True)
    for task in HEAD_DIMS:
        agree = sum(1 for a, b in zip(peft_preds[task], merged_preds[task]) if a == b)
        print(f"  {task:<10} agreement: {agree}/{len(all_texts)} = {agree/len(all_texts):.4%}", flush=True)

    # F1 on COLD subset
    print("\n=== COLD test (first 100) F1 comparison ===", flush=True)
    y_true_tox = [LABEL2ID["toxicity"][r["label"]["toxicity"]] for r in cold]
    for label, preds in [("PEFT", peft_preds), ("MERGED", merged_preds)]:
        y_pred_tox = preds["toxicity"][:100]
        acc = accuracy_score(y_true_tox, y_pred_tox)
        f1 = f1_score(y_true_tox, y_pred_tox, average="weighted", zero_division=0)
        print(f"  {label:<8} toxicity: acc={acc:.4f}  f1_w={f1:.4f}", flush=True)

    # 6 繁中句
    print("\n=== 6 繁中威脅句 對比 ===", flush=True)
    id2tox = LABELS["toxicity"]
    for i, text in enumerate(tc_texts):
        pa = id2tox[peft_preds["toxicity"][100 + i]]
        pm = id2tox[merged_preds["toxicity"][100 + i]]
        mark = "✅" if pa == pm else "❌"
        print(f"  {mark} peft={pa:<6} merged={pm:<6} | {text}", flush=True)


if __name__ == "__main__":
    main()
