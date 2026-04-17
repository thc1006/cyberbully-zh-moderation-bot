"""v4.0 dual-branch trainer: Qwen3-8B text + DBIT pinyin CNN + contrastive alignment.

Based on train_qwen3_lora.py but adds:
  - PinyinBranch (38K params) as second encoder
  - PinyinCharTokenizer for char-level pinyin input
  - Contrastive loss (bidirectional InfoNCE) for text↔pinyin alignment
  - DualBranchMultiHead replacing Qwen3MultiHead

Usage:
  PYTHONPATH=src python scripts/train_v4_dual_branch.py \
    --train data/processed/v2/v2_2_train.jsonl \
    --eval  data/processed/v2/v2_2_dev.jsonl \
    --epochs 2 --batch 6 --grad-accum 6 --max-length 192 \
    --lr 3e-5 --focal-gamma 2.5 --consistency-lambda 0.5 \
    --contrastive-lambda 0.1 --pinyin-max-length 128 \
    --output models/cyberpuppy_v4_dual_qwen3_8b
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from peft import get_peft_model
from pypinyin import Style, pinyin as get_pinyin
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from cyberpuppy.data.phase2 import LABELS
from cyberpuppy.models.dual_branch_multihead import DualBranchMultiHead
from cyberpuppy.models.pinyin_cnn_encoder import PinyinBranch, PinyinCharTokenizer
from cyberpuppy.models.qwen3_multihead import (
    HEAD_DIMS,
    build_lora_config,
    consistency_loss,
)
from cyberpuppy.training.bucket_sampler import LengthBucketSampler
from cyberpuppy.training.cloak_aware_sampler import CloakAwareBatchSampler

DEFAULT_BASE = "Qwen/Qwen3-8B-Base"
LABEL2ID = {task: {v: i for i, v in enumerate(LABELS[task])} for task in LABELS}

import re
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _text_to_toneless_pinyin(text: str) -> str:
    """Convert Chinese chars in text to tone-stripped pinyin, space-separated."""
    syllables = []
    for ch in text:
        if _HAN_RE.match(ch):
            try:
                py = get_pinyin(ch, style=Style.NORMAL, errors="ignore")
                if py and py[0] and py[0][0]:
                    syllables.append(py[0][0])
            except Exception:
                pass
    return " ".join(syllables)


@dataclass
class TrainConfig:
    base_model: str = DEFAULT_BASE
    train_path: str = ""
    eval_path: str = ""
    output_dir: str = "models/cyberpuppy_v4_dual_qwen3_8b"
    max_train: int = 0
    max_eval: int = 0
    epochs: int = 2
    batch_size: int = 6
    grad_accum: int = 6
    lr: float = 3e-5
    warmup_ratio: float = 0.1
    max_length: int = 192
    weight_decay: float = 0.01
    lora_r: int = 32
    lora_alpha: int = 64
    log_every: int = 50
    seed: int = 42


class DualBranchDataset(Dataset):
    """JSONL dataset that produces BOTH text tokens AND pinyin char IDs."""

    def __init__(self, path, tokenizer, pinyin_tok, max_length=192,
                 pinyin_max_length=128, max_rows=0):
        self.records = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.pinyin_tok = pinyin_tok
        self.max_length = max_length
        self.pinyin_max_length = pinyin_max_length
        self.lengths = [
            min(max_length, len(tokenizer.encode(r["text"], add_special_tokens=False)))
            for r in self.records
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = r["text"]
        # Text tokens (Qwen3 BPE)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding=False, return_tensors=None)
        # Pinyin chars
        py_str = _text_to_toneless_pinyin(text)
        py_ids = self.pinyin_tok.encode(py_str)[:self.pinyin_max_length]

        labels = {}
        for task in HEAD_DIMS:
            v = r["label"].get(task)
            labels[task] = LABEL2ID[task].get(v, 0) if v is not None else -100
        pair_id = int(r.get("metadata", {}).get("cloak_pair_id", -1))

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pinyin_ids": py_ids,
            "labels": labels,
            "pair_id": pair_id,
        }


def collate(batch, pad_token_id, pinyin_pad_id=0):
    # Text: left-pad (decoder LM convention)
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attn = [], []
    for b in batch:
        ids, m = b["input_ids"], b["attention_mask"]
        pad = max_len - len(ids)
        input_ids.append([pad_token_id] * pad + ids)
        attn.append([0] * pad + m)

    # Pinyin: right-pad (CNN convention)
    max_py = max(len(b["pinyin_ids"]) for b in batch)
    pinyin_ids = []
    for b in batch:
        py = b["pinyin_ids"]
        pinyin_ids.append(py + [pinyin_pad_id] * (max_py - len(py)))

    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "pinyin_ids": torch.tensor(pinyin_ids, dtype=torch.long),
    }
    label_dict = {}
    for task in HEAD_DIMS:
        label_dict[task] = torch.tensor([b["labels"][task] for b in batch], dtype=torch.long)
    out["labels"] = label_dict
    out["pair_ids"] = torch.tensor([b.get("pair_id", -1) for b in batch], dtype=torch.long)
    return out


def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    per_task = {t: {"y_true": [], "y_pred": []} for t in HEAD_DIMS}
    with torch.inference_mode():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            py = batch["pinyin_ids"].to(device)
            out = model(input_ids=ids, attention_mask=attn, pinyin_ids=py)
            for task in HEAD_DIMS:
                preds = out.logits[task].argmax(-1).cpu().tolist()
                trues = batch["labels"][task].tolist()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--output", default="models/cyberpuppy_v4_dual_qwen3_8b")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--focal-gamma", type=float, default=2.5)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--consistency-lambda", type=float, default=0.5)
    ap.add_argument("--contrastive-lambda", type=float, default=0.1,
                    help="Weight for bidirectional InfoNCE text↔pinyin alignment loss")
    ap.add_argument("--pinyin-max-length", type=int, default=128)
    ap.add_argument("--pinyin-embed-dim", type=int, default=64)
    ap.add_argument("--pinyin-hidden-dim", type=int, default=256)
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

    # Tokenizers
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    pinyin_tok = PinyinCharTokenizer()
    print(f"Pinyin tokenizer: vocab={pinyin_tok.vocab_size}", flush=True)

    # Backbone
    print(f"Loading backbone {cfg.base_model} ...", flush=True)
    backbone = AutoModel.from_pretrained(cfg.base_model, dtype=dtype,
                                          low_cpu_mem_usage=True, attn_implementation="sdpa")
    lora_cfg = build_lora_config(r=cfg.lora_r, alpha=cfg.lora_alpha, use_dora=False)
    backbone = get_peft_model(backbone, lora_cfg)
    backbone.enable_input_require_grads()
    backbone.gradient_checkpointing_enable()
    backbone.print_trainable_parameters()

    # Pinyin branch
    pinyin_branch = PinyinBranch(
        vocab_size=pinyin_tok.vocab_size,
        embed_dim=args.pinyin_embed_dim,
        hidden_dim=args.pinyin_hidden_dim,
    )
    pinyin_params = sum(p.numel() for p in pinyin_branch.parameters())
    print(f"PinyinBranch: {pinyin_params:,} params ({pinyin_params/1e3:.1f}K)", flush=True)

    # Dual-branch model
    model = DualBranchMultiHead(
        backbone, pinyin_branch, hidden_size=backbone.config.hidden_size,
    ).to(device=device, dtype=dtype)
    model.focal_gamma = float(args.focal_gamma)
    print(f"Focal gamma: {model.focal_gamma}  Consistency λ: {args.consistency_lambda}  "
          f"Contrastive λ: {args.contrastive_lambda}", flush=True)

    # Datasets
    train_ds = DualBranchDataset(cfg.train_path, tokenizer, pinyin_tok,
                                  cfg.max_length, args.pinyin_max_length, cfg.max_train)
    eval_ds = DualBranchDataset(cfg.eval_path, tokenizer, pinyin_tok,
                                 cfg.max_length, args.pinyin_max_length, cfg.max_eval)
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}", flush=True)

    # Samplers
    if args.consistency_lambda > 0:
        train_sampler = CloakAwareBatchSampler(
            train_ds.records, batch_size=cfg.batch_size, mega_batch_mult=50,
            shuffle=True, seed=cfg.seed)
        print(f"Sampler: CloakAware (λ_cons={args.consistency_lambda})", flush=True)
    else:
        train_sampler = LengthBucketSampler(
            train_ds.lengths, batch_size=cfg.batch_size, mega_batch_mult=50,
            shuffle=True, seed=cfg.seed)
    eval_sampler = LengthBucketSampler(
        eval_ds.lengths, batch_size=cfg.batch_size, mega_batch_mult=50, shuffle=False)

    nw = args.num_workers
    collate_fn = lambda b: collate(b, tokenizer.pad_token_id, pinyin_tok.pad_id)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=nw,
                               collate_fn=collate_fn, pin_memory=True,
                               persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
    eval_loader = DataLoader(eval_ds, batch_sampler=eval_sampler, num_workers=max(2, nw // 2),
                              collate_fn=collate_fn, pin_memory=True,
                              persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(total_steps * cfg.warmup_ratio)
    print(f"Total optim steps: {total_steps}  warmup: {warmup}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f} M "
          f"(LoRA + heads + pinyin_branch + pinyin_proj)", flush=True)

    try:
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        print("Optimizer: AdamW 8-bit", flush=True)
    except Exception:
        optim = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        print("Optimizer: AdamW fp32 fallback", flush=True)
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)

    if device.type == "cuda":
        vram = torch.cuda.memory_allocated() / 1024**3
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Pre-train VRAM: {vram:.1f}/{total_vram:.1f} GB ({(1-vram/total_vram)*100:.0f}% headroom)", flush=True)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    log_lines = []
    best_f1 = -1.0
    global_step = 0
    t_start = time.time()

    def _save(tag, eval_metrics=None):
        ck = Path(cfg.output_dir) / tag
        ck.mkdir(parents=True, exist_ok=True)
        backbone.save_pretrained(ck / "lora")
        torch.save({
            "heads": model.heads.state_dict(),
            "pinyin_branch": model.pinyin_branch.state_dict(),
            "pinyin_proj": model.pinyin_proj.state_dict(),
            "log_var": model.log_var.detach().cpu(),
            "head_dims": dict(model.head_dims),
            "focal_gamma": model.focal_gamma,
            "step": global_step,
            "eval_metrics": eval_metrics,
        }, ck / "heads.pt")
        tokenizer.save_pretrained(ck)

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)
        running = running_cons = running_cont = 0.0
        cons_fire = n_window = 0

        for i, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            py = batch["pinyin_ids"].to(device)
            labels = {t: v.to(device) for t, v in batch["labels"].items()}

            out = model(input_ids=ids, attention_mask=attn, pinyin_ids=py)
            task_loss = model.compute_loss(out, labels)

            # Consistency loss (ToxiCloakCN triplets)
            cons_loss = torch.tensor(0.0, device=device, dtype=task_loss.dtype)
            if args.consistency_lambda > 0:
                pair_ids = batch["pair_ids"].to(device)
                cons_loss = consistency_loss(out.logits["toxicity"], pair_ids)
                if cons_loss.item() > 1e-6:
                    running_cons += cons_loss.item()
                    cons_fire += 1

            # Contrastive loss (text↔pinyin alignment)
            cont_loss = torch.tensor(0.0, device=device, dtype=task_loss.dtype)
            if args.contrastive_lambda > 0:
                cont_loss = model.compute_contrastive_loss(out)
                running_cont += cont_loss.item()

            loss = (task_loss
                    + args.consistency_lambda * cons_loss
                    + args.contrastive_lambda * cont_loss) / cfg.grad_accum
            loss.backward()
            running += loss.item() * cfg.grad_accum
            n_window += 1

            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg = running / max(1, n_window)
                    avg_cons = running_cons / max(1, cons_fire) if cons_fire else 0
                    avg_cont = running_cont / max(1, n_window)
                    elapsed = time.time() - t_start
                    msg = (f"ep{epoch} step{global_step}/{total_steps} "
                           f"loss={avg:.4f} cons={avg_cons:.3f}|{cons_fire} "
                           f"cont={avg_cont:.3f} lr={sched.get_last_lr()[0]:.2e} "
                           f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB "
                           f"elapsed={elapsed:.0f}s")
                    print(msg, flush=True)
                    log_lines.append(msg)
                    running = running_cons = running_cont = 0.0
                    cons_fire = n_window = 0

                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    _save("checkpoint_step")
                    print(f"  [save] checkpoint_step at step {global_step}", flush=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        eval_metrics = evaluate(model, eval_loader, device)
        print(f"== Epoch {epoch} eval == {eval_metrics}", flush=True)
        log_lines.append(f"epoch{epoch} eval: {eval_metrics}")

        score = eval_metrics.get("toxicity", {}).get("f1_weighted", 0.0)
        if score > best_f1:
            best_f1 = score
            _save("best", eval_metrics)
            print(f"Saved best (toxicity F1_w={score:.4f}) -> {Path(cfg.output_dir) / 'best'}", flush=True)

    _save("final")
    Path(cfg.output_dir, "train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nDone. Best toxicity F1_w={best_f1:.4f}. Final -> {Path(cfg.output_dir) / 'final'}", flush=True)


if __name__ == "__main__":
    main()
