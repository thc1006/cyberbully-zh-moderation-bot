"""Train LoRA-B (pinyin adapter) on Qwen3-8B for pinyin-based classification.

Same base model as text LoRA-A, but trained on pinyin-converted text.
At inference, both adapters share the base model and are switched via peft.

Usage:
  PYTHONPATH=src python scripts/train_pinyin_lora.py \
    --train data/processed/v2/pinyin_only_v5_bilingual_train.jsonl \
    --eval  data/processed/v2/pinyin_only_v5_bilingual_dev.jsonl \
    --epochs 2 --batch 6 --grad-accum 6 --max-length 128 \
    --output models/cyberpuppy_v5_pinyin_lora
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score

from cyberpuppy.data.phase2 import LABELS
from cyberpuppy.models.qwen3_multihead import HEAD_DIMS, Qwen3MultiHead, build_lora_config
from cyberpuppy.training.bucket_sampler import LengthBucketSampler

DEFAULT_BASE = "Qwen/Qwen3-8B-Base"
LABEL2ID = {task: {v: i for i, v in enumerate(LABELS[task])} for task in LABELS}


class PinyinDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=128, max_rows=0):
        self.records = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lengths = [
            min(max_length, len(tokenizer.encode(r["text"], add_special_tokens=False)))
            for r in self.records
        ]

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


def collate(batch, pad_token_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    ids, masks = [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        ids.append([pad_token_id] * pad + b["input_ids"])
        masks.append([0] * pad + b["attention_mask"])
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
            attn = batch["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=attn)
            for task in HEAD_DIMS:
                preds = out.logits[task].argmax(-1).cpu().tolist()
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
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--output", default="models/cyberpuppy_v5_pinyin_lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=128,
                    help="Pinyin texts are shorter (avg ~30 tokens). 128 is plenty.")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every-steps", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--focal-gamma", type=float, default=2.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Loading {args.base}...", flush=True)
    backbone = AutoModel.from_pretrained(args.base, dtype=dtype,
                                          low_cpu_mem_usage=True, attn_implementation="sdpa")
    lora_cfg = build_lora_config(r=args.lora_r, alpha=args.lora_alpha, use_dora=False)
    backbone = get_peft_model(backbone, lora_cfg)
    backbone.enable_input_require_grads()
    backbone.gradient_checkpointing_enable()
    backbone.print_trainable_parameters()

    model = Qwen3MultiHead(backbone, hidden_size=backbone.config.hidden_size
                            ).to(device=device, dtype=dtype)
    model.focal_gamma = float(args.focal_gamma)
    print(f"Focal gamma: {model.focal_gamma}", flush=True)

    train_ds = PinyinDataset(args.train, tokenizer, args.max_length)
    eval_ds = PinyinDataset(args.eval, tokenizer, args.max_length)
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}", flush=True)

    train_sampler = LengthBucketSampler(
        train_ds.lengths, batch_size=args.batch, mega_batch_mult=50,
        shuffle=True, seed=args.seed)
    eval_sampler = LengthBucketSampler(
        eval_ds.lengths, batch_size=args.batch, mega_batch_mult=50, shuffle=False)

    nw = args.num_workers
    collate_fn = lambda b: collate(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=nw,
                               collate_fn=collate_fn, pin_memory=True,
                               persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
    eval_loader = DataLoader(eval_ds, batch_sampler=eval_sampler, num_workers=max(2, nw // 2),
                              collate_fn=collate_fn, pin_memory=True,
                              persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.1)
    print(f"Total optim steps: {total_steps}  warmup: {warmup}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f} M", flush=True)

    try:
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit(trainable, lr=args.lr, weight_decay=0.01)
        print("Optimizer: AdamW 8-bit", flush=True)
    except Exception:
        optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    global_step = 0
    t_start = time.time()

    def _save(tag, eval_metrics=None):
        ck = Path(args.output) / tag
        ck.mkdir(parents=True, exist_ok=True)
        backbone.save_pretrained(ck / "lora")
        torch.save({
            "heads": model.heads.state_dict(),
            "log_var": model.log_var.detach().cpu(),
            "head_dims": dict(model.head_dims),
            "focal_gamma": model.focal_gamma,
            "step": global_step,
            "eval_metrics": eval_metrics,
        }, ck / "heads.pt")
        tokenizer.save_pretrained(ck)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        running = n_window = 0

        for i, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = {t: v.to(device) for t, v in batch["labels"].items()}
            out = model(input_ids=ids, attention_mask=attn)
            loss = model.compute_loss(out, labels) / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            n_window += 1

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    avg = running / max(1, n_window)
                    elapsed = time.time() - t_start
                    print(f"ep{epoch} step{global_step}/{total_steps} loss={avg:.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} "
                          f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB "
                          f"elapsed={elapsed:.0f}s", flush=True)
                    running = n_window = 0
                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    _save("checkpoint_step")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        eval_metrics = evaluate(model, eval_loader, device)
        print(f"== Epoch {epoch} eval == {eval_metrics}", flush=True)
        score = eval_metrics.get("toxicity", {}).get("f1_weighted", 0.0)
        if score > best_f1:
            best_f1 = score
            _save("best", eval_metrics)
            print(f"Saved best (tox F1_w={score:.4f})", flush=True)

    _save("final")
    print(f"\nDone. Best tox F1_w={best_f1:.4f}.", flush=True)


if __name__ == "__main__":
    main()
