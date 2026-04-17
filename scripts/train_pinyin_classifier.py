"""Train a pinyin-only classifier using hfl/rbt4-h312 (Chinese RoBERTa-tiny).

Input: toneless pinyin text (e.g., "bai ren dou gai si")
Output: 4 classification heads (toxicity/bullying/role/emotion)

This model is INDEPENDENT of Qwen3-8B. At inference, it's ensembled with
the text model at the logit level for homophone robustness.

Usage:
  PYTHONPATH=src python scripts/train_pinyin_classifier.py \
    --train data/processed/v2/pinyin_only_v2_2_train.jsonl \
    --eval  data/processed/v2/pinyin_only_v2_2_dev.jsonl \
    --epochs 5 --batch 32 --lr 5e-5 \
    --output models/pinyin_classifier_rbt4
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup)
from sklearn.metrics import f1_score, accuracy_score

from cyberpuppy.data.phase2 import LABELS

HEAD_DIMS = {"toxicity": 3, "bullying": 3, "role": 4, "emotion": 3}
LABEL2ID = {task: {v: i for i, v in enumerate(LABELS[task])} for task in LABELS}
MODEL_ID = "hfl/rbt4-h312"


class PinyinClassifier(nn.Module):
    """Small BERT + 4 classification heads for pinyin-only input."""

    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, dim) for name, dim in HEAD_DIMS.items()
        })

    def forward(self, input_ids, attention_mask=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token pooling (BERT convention)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        logits = {name: head(pooled) for name, head in self.heads.items()}
        return logits


class PinyinDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=128, max_rows=0):
        self.records = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows: break
                self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = self.tokenizer(r["text"], truncation=True, max_length=self.max_length,
                              padding=False, return_tensors=None)
        labels = {}
        for task in HEAD_DIMS:
            v = r["label"].get(task)
            labels[task] = LABEL2ID[task].get(v, 0) if v is not None else -100
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": labels}


def collate(batch, pad_id=0):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, masks = [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * pad)
        masks.append(b["attention_mask"] + [0] * pad)
    out = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}
    label_dict = {}
    for task in HEAD_DIMS:
        label_dict[task] = torch.tensor([b["labels"][task] for b in batch])
    out["labels"] = label_dict
    return out


def evaluate(model, loader, device):
    model.eval()
    per_task = {t: {"y_true": [], "y_pred": []} for t in HEAD_DIMS}
    with torch.inference_mode():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            for task in HEAD_DIMS:
                preds = logits[task].argmax(-1).cpu().tolist()
                trues = batch["labels"][task].tolist()
                for p, t in zip(preds, trues):
                    if t < 0: continue
                    per_task[task]["y_pred"].append(p)
                    per_task[task]["y_true"].append(t)
    metrics = {}
    for task, d in per_task.items():
        if not d["y_true"]: continue
        metrics[task] = {
            "f1_weighted": float(f1_score(d["y_true"], d["y_pred"], average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(d["y_true"], d["y_pred"])),
            "n": len(d["y_true"]),
        }
    model.train()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--output", default="models/pinyin_classifier_rbt4")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    backbone = AutoModel.from_pretrained(MODEL_ID)
    model = PinyinClassifier(backbone, backbone.config.hidden_size).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,} ({params/1e6:.1f}M)  VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)

    train_ds = PinyinDataset(args.train, tokenizer, args.max_length)
    eval_ds = PinyinDataset(args.eval, tokenizer, args.max_length)
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
                               num_workers=8, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch * 2, shuffle=False,
                              collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
                              num_workers=4, pin_memory=True)

    total_steps = len(train_loader) * args.epochs
    warmup = int(total_steps * 0.1)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = {t: v.to(device) for t, v in batch["labels"].items()}

            logits = model(ids, mask)
            loss = sum(
                nn.functional.cross_entropy(logits[t], labels[t], ignore_index=-100)
                for t in HEAD_DIMS if labels[t] is not None
            ) / len(HEAD_DIMS)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        elapsed = time.time() - t_start
        metrics = evaluate(model, eval_loader, device)
        tox_f1 = metrics.get("toxicity", {}).get("f1_weighted", 0)
        print(f"ep{epoch} loss={avg_loss:.4f} tox_F1={tox_f1:.4f} "
              f"bull_F1={metrics.get('bullying', {}).get('f1_weighted', 0):.4f} "
              f"elapsed={elapsed:.0f}s", flush=True)

        if tox_f1 > best_f1:
            best_f1 = tox_f1
            torch.save({
                "model": model.state_dict(),
                "config": {"model_id": MODEL_ID, "hidden_size": backbone.config.hidden_size},
                "metrics": metrics,
            }, Path(args.output) / "best.pt")
            tokenizer.save_pretrained(args.output)
            print(f"  Saved best (tox F1={tox_f1:.4f})", flush=True)

    print(f"\nDone. Best tox F1={best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
