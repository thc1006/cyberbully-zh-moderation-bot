"""Real test-set evaluation against bullying_a100_best.

Optimized for RTX 5090 (Blackwell, sm_120, CUDA 12.8): bf16 + large batch +
optional torch.compile. CPU fallback works unchanged.
"""
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score)
from transformers import AutoTokenizer, BertModel

MODEL_DIR = Path("models/bullying_a100_best")
TEST_FILE = Path("data/processed/training_dataset/test.json")
LABEL2ID = {"none": 0, "toxic": 1, "severe": 2}
USE_COMPILE = os.environ.get("CP_COMPILE", "0") == "1"
BATCH = int(os.environ.get("CP_BATCH", "128"))


class MultiTaskBert(nn.Module):
    def __init__(self, base_name="hfl/chinese-macbert-base"):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_name)
        self.bullying_head = nn.Linear(768, 3)
        self.toxicity_head = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return {
            "bullying": self.bullying_head(out.pooler_output),
            "toxicity": self.toxicity_head(out.pooler_output),
        }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Device: {device}  dtype: {dtype}  batch: {BATCH}  compile: {USE_COMPILE}", flush=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = MultiTaskBert()
    state = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval().to(device=device, dtype=dtype)
    if USE_COMPILE and device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled (first batch will be slow due to graph capture)", flush=True)

    with TEST_FILE.open() as f:
        data = json.load(f)
    print(f"Test set size: {len(data)}")

    texts = [d["text"] for d in data]
    y_tox_true = [LABEL2ID[d["label"]["toxicity"]] for d in data]
    print("Truth distribution (toxicity):", Counter(y_tox_true))

    batch = BATCH
    y_tox_pred = []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tokenizer(
                chunk, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(device)
            out = model(enc["input_ids"], enc["attention_mask"])
            y_tox_pred.extend(out["toxicity"].argmax(-1).cpu().tolist())
            if (i // batch) % 10 == 0:
                done = i + len(chunk)
                rate = done / (time.time() - t0)
                eta = (len(texts) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(texts)}  {rate:.1f} samples/s  ETA {eta:.1f} s", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    print(f"\nInference done in {elapsed:.2f}s ({len(texts)/elapsed:.1f} samples/s on {device})")

    print("\n=== Toxicity head (3-way) ===")
    print(f"Accuracy: {accuracy_score(y_tox_true, y_tox_pred):.4f}")
    print(f"F1 weighted: {f1_score(y_tox_true, y_tox_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 macro:    {f1_score(y_tox_true, y_tox_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(
        y_tox_true, y_tox_pred,
        labels=[0, 1, 2],
        target_names=["none", "toxic", "severe"],
        digits=4, zero_division=0,
    ))

    # Binary view: collapse {toxic, severe} -> 1
    y_tox_true_bin = [0 if v == 0 else 1 for v in y_tox_true]
    y_tox_pred_bin = [0 if v == 0 else 1 for v in y_tox_pred]
    print("=== Binary (none vs toxic+) ===")
    print(f"Accuracy: {accuracy_score(y_tox_true_bin, y_tox_pred_bin):.4f}")
    print(f"F1 weighted: {f1_score(y_tox_true_bin, y_tox_pred_bin, average='weighted', zero_division=0):.4f}")
    print(f"F1 macro:    {f1_score(y_tox_true_bin, y_tox_pred_bin, average='macro', zero_division=0):.4f}")


if __name__ == "__main__":
    main()
