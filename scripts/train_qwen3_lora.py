"""Phase 3 trainer — Qwen3-8B + LoRA + 4-head multitask (ADR 0001 §3.7).

Usage:
  smoke (256 samples, 1 epoch):
    python scripts/train_qwen3_lora.py --train data/processed/v2/cold_train.jsonl \\
      --eval data/processed/v2/cold_dev.jsonl --max-train 256 --epochs 1 --batch 4

  full:
    python scripts/train_qwen3_lora.py --train data/processed/v2/cold_train.jsonl \\
      --eval data/processed/v2/cold_dev.jsonl --epochs 3 --batch 16 \\
      --output models/cyberpuppy_v2_qwen3_8b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from peft import get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup)

from cyberpuppy.data.phase2 import LABELS
from cyberpuppy.models.qwen3_multihead import (HEAD_DIMS, Qwen3MultiHead,
                                                build_lora_config)

DEFAULT_BASE = "Qwen/Qwen3-8B-Base"
LABEL2ID = {task: {v: i for i, v in enumerate(LABELS[task])} for task in LABELS}


@dataclass
class TrainConfig:
    base_model: str = DEFAULT_BASE
    train_path: str = "data/processed/v2/cold_train.jsonl"
    eval_path: str = "data/processed/v2/cold_dev.jsonl"
    output_dir: str = "models/cyberpuppy_v2_qwen3_8b"
    max_train: int = 0  # 0 = all
    max_eval: int = 0
    epochs: int = 3
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 3e-5
    warmup_ratio: float = 0.1
    max_length: int = 256
    weight_decay: float = 0.01
    lora_r: int = 32
    lora_alpha: int = 64
    log_every: int = 20
    save_every_epochs: int = 1
    seed: int = 42


class JsonlClassificationDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 256, max_rows: int = 0) -> None:
        self.records = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        text = r["text"]
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length, padding=False, return_tensors=None
        )
        labels = {}
        for task in HEAD_DIMS:
            v = r["label"].get(task)
            labels[task] = LABEL2ID[task].get(v, 0) if v is not None else -1
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


def collate(batch, pad_token_id: int):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attn = [], []
    for b in batch:
        ids = b["input_ids"]
        m = b["attention_mask"]
        pad = max_len - len(ids)
        # left pad (decoder LM convention so last-token pool is real)
        input_ids.append([pad_token_id] * pad + ids)
        attn.append([0] * pad + m)
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }
    label_dict = {}
    for task in HEAD_DIMS:
        vals = [b["labels"][task] for b in batch]
        label_dict[task] = torch.tensor(vals, dtype=torch.long)
    out["labels"] = label_dict
    return out


def evaluate(model, loader, device) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    per_task: dict = {t: {"y_true": [], "y_pred": []} for t in HEAD_DIMS}
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            for task in HEAD_DIMS:
                preds = out.logits[task].argmax(-1).cpu().tolist()
                trues = batch["labels"][task].tolist()
                # mask -1 (missing)
                for p, t in zip(preds, trues):
                    if t < 0:
                        continue
                    per_task[task]["y_pred"].append(p)
                    per_task[task]["y_true"].append(t)
    metrics = {}
    for task, d in per_task.items():
        if not d["y_true"]:
            metrics[task] = {"f1_weighted": 0.0, "accuracy": 0.0, "n": 0}
            continue
        metrics[task] = {
            "f1_weighted": float(f1_score(d["y_true"], d["y_pred"], average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(d["y_true"], d["y_pred"])),
            "n": len(d["y_true"]),
        }
    model.train()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--train", default="data/processed/v2/cold_train.jsonl")
    ap.add_argument("--eval", default="data/processed/v2/cold_dev.jsonl")
    ap.add_argument("--output", default="models/cyberpuppy_v2_qwen3_8b")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True,
                    help="Enable gradient checkpointing (saves ~30-40%% activation memory, ~30%% slower)")
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    ap.add_argument("--dora", action="store_true", default=False,
                    help="Use DoRA (magnitude-decomposed LoRA). Costs ~2-3 GB extra VRAM due "
                         "to weight rematerialization every forward. OFF by default on 32 GB GPUs.")
    args = ap.parse_args()

    cfg = TrainConfig(
        base_model=args.base, train_path=args.train, eval_path=args.eval,
        output_dir=args.output, max_train=args.max_train, max_eval=args.max_eval,
        epochs=args.epochs, batch_size=args.batch, grad_accum=args.grad_accum,
        lr=args.lr, max_length=args.max_length, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, log_every=args.log_every, seed=args.seed,
    )

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB", flush=True)
    print(f"Device: {device}  dtype: {dtype}", flush=True)
    print(f"Config: {cfg}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading backbone {cfg.base_model} (bf16) ...", flush=True)
    t0 = time.time()
    backbone = AutoModel.from_pretrained(
        cfg.base_model, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
    )
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    lora_cfg = build_lora_config(r=cfg.lora_r, alpha=cfg.lora_alpha, use_dora=args.dora)
    print(f"LoRA: r={cfg.lora_r} alpha={cfg.lora_alpha} dora={args.dora}", flush=True)
    backbone = get_peft_model(backbone, lora_cfg)
    backbone.print_trainable_parameters()

    if args.gradient_checkpointing:
        # Required for PEFT + checkpointing: inputs must require grad for
        # checkpointed sub-modules to chain gradients through LoRA.
        backbone.enable_input_require_grads()
        backbone.gradient_checkpointing_enable()
        print("Gradient checkpointing: ON", flush=True)
    else:
        print("Gradient checkpointing: OFF", flush=True)

    model = Qwen3MultiHead(backbone, hidden_size=backbone.config.hidden_size).to(device=device, dtype=dtype)
    # heads must stay in bf16 too (already from .to)

    train_ds = JsonlClassificationDataset(cfg.train_path, tokenizer, cfg.max_length, cfg.max_train)
    eval_ds = JsonlClassificationDataset(cfg.eval_path, tokenizer, cfg.max_length, cfg.max_eval)
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id), pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id), pin_memory=True,
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(total_steps * cfg.warmup_ratio)
    print(f"Total optim steps: {total_steps}  warmup: {warmup}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f} M", flush=True)
    try:
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        print("Optimizer: AdamW 8-bit (bitsandbytes)", flush=True)
    except Exception:
        optim = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        print("Optimizer: AdamW (fp32 fallback)", flush=True)
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    log_lines: list = []
    best_f1 = -1.0
    global_step = 0
    t_start = time.time()

    for epoch in range(cfg.epochs):
        running = 0.0
        n_in_window = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = {t: v.to(device) for t, v in batch["labels"].items()}
            out = model(input_ids=input_ids, attention_mask=attn)
            # mask -1 (missing label) by replacing with valid ignored id via cross_entropy ignore_index
            for t in HEAD_DIMS:
                if (labels[t] < 0).any():
                    # any sample with -1 label for this task -> set to None (skip task)
                    if (labels[t] >= 0).all():
                        pass
                    else:
                        labels[t] = None  # type: ignore
            loss = model.compute_loss(out, labels) / cfg.grad_accum
            loss.backward()
            running += loss.item() * cfg.grad_accum
            n_in_window += 1
            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % cfg.log_every == 0:
                    avg = running / max(1, n_in_window)
                    elapsed = time.time() - t_start
                    msg = (f"ep{epoch} step{global_step}/{total_steps} "
                           f"loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} "
                           f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB "
                           f"elapsed={elapsed:.0f}s")
                    print(msg, flush=True)
                    log_lines.append(msg)
                    running = 0.0
                    n_in_window = 0

        # eval at end of epoch
        eval_metrics = evaluate(model, eval_loader, device)
        print(f"== Epoch {epoch} eval == {eval_metrics}", flush=True)
        log_lines.append(f"epoch{epoch} eval: {eval_metrics}")

        # save adapter + heads if best
        tox_f1 = eval_metrics.get("toxicity", {}).get("f1_weighted", 0.0)
        if tox_f1 > best_f1:
            best_f1 = tox_f1
            ck_dir = Path(cfg.output_dir) / "best"
            ck_dir.mkdir(parents=True, exist_ok=True)
            backbone.save_pretrained(ck_dir / "lora")
            torch.save({
                "heads": model.heads.state_dict(),
                "log_var": model.log_var.detach().cpu(),
                "head_dims": dict(model.head_dims),
                "epoch": epoch,
                "eval_metrics": eval_metrics,
            }, ck_dir / "heads.pt")
            tokenizer.save_pretrained(ck_dir)
            print(f"Saved best (toxicity F1={tox_f1:.4f}) -> {ck_dir}", flush=True)

    # always save final
    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(final_dir / "lora")
    torch.save({
        "heads": model.heads.state_dict(),
        "log_var": model.log_var.detach().cpu(),
        "head_dims": dict(model.head_dims),
    }, final_dir / "heads.pt")
    tokenizer.save_pretrained(final_dir)
    Path(cfg.output_dir, "train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nDone. Best toxicity F1={best_f1:.4f}. Final adapter -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()
