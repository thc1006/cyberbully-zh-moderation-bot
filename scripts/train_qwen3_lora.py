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
                                                build_lora_config,
                                                consistency_loss)
from cyberpuppy.training.bucket_sampler import LengthBucketSampler
from cyberpuppy.training.cloak_aware_sampler import CloakAwareBatchSampler

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
        # Pre-compute token lengths once for bucket sampler. Truncation cap
        # keeps outliers from inflating activation budget calculations.
        self.lengths: list[int] = [
            min(self.max_length, len(tokenizer.encode(r["text"], add_special_tokens=False)))
            for r in self.records
        ]

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
            # -100 = PyTorch CE ignore_index (handled per-sample by focal_ce)
            labels[task] = LABEL2ID[task].get(v, 0) if v is not None else -100
        # ToxiCloakCN samples carry cloak_pair_id; others get -1 (ignored by consistency loss)
        pair_id = int(r.get("metadata", {}).get("cloak_pair_id", -1))
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "pair_id": pair_id,
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
    out["pair_ids"] = torch.tensor(
        [b.get("pair_id", -1) for b in batch], dtype=torch.long
    )
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
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="LoRA peak LR; v2.0 used 3e-5, v2.1 default 5e-5 for larger data")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="Focal-loss gamma per Lin et al.; 0 -> plain CE, "
                         "2.0 default for severe class imbalance (severe 6.8%%, threat 0.2%%)")
    ap.add_argument("--save-every-steps", type=int, default=500,
                    help="Save mid-train checkpoint every N optim steps (0=disable)")
    ap.add_argument("--eval-every-steps", type=int, default=0,
                    help="Run dev eval every N optim steps (0=epoch-end only)")
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
    ap.add_argument("--consistency-lambda", type=float, default=0.0,
                    help="Weight for ToxiCloakCN adversarial consistency loss (v2.2). "
                         "Tries to equalize toxicity logits across base/homo/emoji of same pair_id. "
                         "0 = off (v2.1 behavior); 0.1 recommended for v2.2.")
    ap.add_argument("--num-workers", type=int, default=12,
                    help="DataLoader workers (RTX 5090 24 cores + NVMe; default 12).")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="Path to checkpoint dir to resume from (e.g., "
                         "models/cyberpuppy_v6_text_lora/checkpoint_step). "
                         "Loads: LoRA adapter, heads, optimizer, scheduler, "
                         "global_step, best_f1, RNG state.")
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

    if args.resume_from:
        # Load LoRA adapter weights from checkpoint dir
        from peft import PeftModel
        lora_path = Path(args.resume_from) / "lora"
        print(f"Resuming LoRA from {lora_path}", flush=True)
        backbone = PeftModel.from_pretrained(backbone, str(lora_path), is_trainable=True)
    else:
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
    model.focal_gamma = float(args.focal_gamma)
    print(f"Focal gamma: {model.focal_gamma}", flush=True)
    print(f"Consistency λ: {args.consistency_lambda}", flush=True)

    if args.resume_from:
        heads_path = Path(args.resume_from) / "heads.pt"
        if heads_path.exists():
            st = torch.load(heads_path, map_location=device, weights_only=False)
            model.heads.load_state_dict(st["heads"])
            if "log_var" in st:
                model.log_var.data = st["log_var"].to(device=device, dtype=dtype)
            print(f"Resumed heads from {heads_path}", flush=True)
    # heads must stay in bf16 too (already from .to)

    train_ds = JsonlClassificationDataset(cfg.train_path, tokenizer, cfg.max_length, cfg.max_train)
    eval_ds = JsonlClassificationDataset(cfg.eval_path, tokenizer, cfg.max_length, cfg.max_eval)
    print(f"Train: {len(train_ds)}  Eval: {len(eval_ds)}", flush=True)

    # Use CloakAwareBatchSampler when adversarial consistency is active —
    # guarantees ToxiCloakCN triplets land together so consistency_loss fires.
    # Otherwise regular length-bucket sampling.
    if args.consistency_lambda > 0:
        train_sampler = CloakAwareBatchSampler(
            train_ds.records, batch_size=cfg.batch_size, mega_batch_mult=50,
            shuffle=True, seed=cfg.seed,
        )
        print(f"Sampler: CloakAware (λ={args.consistency_lambda} active) — triplets kept together", flush=True)
    else:
        train_sampler = LengthBucketSampler(
            train_ds.lengths, batch_size=cfg.batch_size, mega_batch_mult=50,
            shuffle=True, seed=cfg.seed,
        )
        print("Sampler: LengthBucket (consistency off)", flush=True)
    eval_sampler = LengthBucketSampler(
        eval_ds.lengths, batch_size=cfg.batch_size, mega_batch_mult=50,
        shuffle=False,
    )
    # Workstation-optimized (RTX 5090 32 GB + NVMe + 24 cores):
    #   - persistent_workers: avoid per-epoch respawn overhead
    #   - prefetch_factor=4: pre-load 4 batches/worker (NVMe is fast enough)
    #   - pin_memory: async H2D transfer
    nw = args.num_workers
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=nw,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        pin_memory=True, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    eval_loader = DataLoader(
        eval_ds, batch_sampler=eval_sampler, num_workers=max(2, nw // 2),
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        pin_memory=True, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    print(f"Length bucketing: ON (mega={cfg.batch_size * 50}); peak per-batch len bounded to longest in batch.", flush=True)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(total_steps * cfg.warmup_ratio)
    print(f"Total optim steps: {total_steps}  warmup: {warmup}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f} M", flush=True)

    # torch.compile: disabled for max_length≥384 (inductor attention kernels
    # allocate L²-scaled workspace that OOMs on 32 GB at L=384). Safe at L≤192.
    # suppress_errors: if compile crashes (e.g., DoRA dynamic-shape guards),
    # fall back to eager mode instead of crashing the entire training run.
    torch._dynamo.config.suppress_errors = True
    if cfg.max_length <= 256:
        try:
            model = torch.compile(model, mode="default")
            print("torch.compile: enabled (mode=default, suppress_errors=True)", flush=True)
        except Exception as e:
            print(f"torch.compile: skipped ({e})", flush=True)
    else:
        print(f"torch.compile: disabled (max_length={cfg.max_length} > 256 → inductor OOM risk)", flush=True)

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
    epoch = 0  # current epoch, made accessible to _save_resume
    t_start = time.time()

    def _save(tag: str, eval_metrics: dict | None = None) -> None:
        ck_dir = Path(cfg.output_dir) / tag
        ck_dir.mkdir(parents=True, exist_ok=True)
        backbone.save_pretrained(ck_dir / "lora")
        torch.save({
            "heads": model.heads.state_dict(),
            "log_var": model.log_var.detach().cpu(),
            "head_dims": dict(model.head_dims),
            "focal_gamma": model.focal_gamma,
            "step": global_step,
            "eval_metrics": eval_metrics,
        }, ck_dir / "heads.pt")
        tokenizer.save_pretrained(ck_dir)

    def _save_resume(tag: str) -> None:
        """Save full training state for resume (optimizer, scheduler, RNG)."""
        ck_dir = Path(cfg.output_dir) / tag
        ck_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "best_f1": best_f1,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "cfg": cfg.__dict__,
        }, ck_dir / "resume_state.pt")

    def _load_resume(resume_dir: str) -> int:
        """Load training state from checkpoint. Returns starting epoch.

        If resume_state.pt exists (new checkpoint): full resume with optimizer,
        scheduler, RNG, global_step.
        If only heads.pt exists (old checkpoint): load LoRA+heads only, infer
        epoch from heads.pt step count, fresh optimizer.
        """
        nonlocal global_step, best_f1
        resume_path = Path(resume_dir) / "resume_state.pt"
        heads_path = Path(resume_dir) / "heads.pt"

        if resume_path.exists():
            state = torch.load(resume_path, map_location=device, weights_only=False)
            optim.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["torch_rng"].cpu() if state["torch_rng"].is_cuda else state["torch_rng"])
            if state.get("cuda_rng") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda_rng"]])
            global_step = state["global_step"]
            best_f1 = state["best_f1"]
            start_epoch = state["epoch"]
            print(f"  [resume] FULL resume from {resume_path}: step={global_step}, "
                  f"epoch={start_epoch}, best_f1={best_f1:.4f}", flush=True)
            return start_epoch

        if heads_path.exists():
            # Old checkpoint: LoRA+heads loaded in main flow, infer epoch from step
            st = torch.load(heads_path, map_location="cpu", weights_only=False)
            saved_step = st.get("step", 0)
            # Scheduler needs fast-forward to match step (cosine decay)
            for _ in range(saved_step):
                sched.step()
            global_step = saved_step
            eval_metrics = st.get("eval_metrics")
            if eval_metrics and "toxicity" in eval_metrics:
                best_f1 = eval_metrics["toxicity"].get("f1_weighted", -1.0)
            # Infer epoch from step / steps_per_epoch
            start_epoch = saved_step // max(1, steps_per_epoch)
            print(f"  [resume] PARTIAL resume (old checkpoint, no optimizer): "
                  f"step={global_step}, inferred epoch={start_epoch}, "
                  f"best_f1={best_f1:.4f}. Optimizer restarted fresh.", flush=True)
            return start_epoch

        print(f"  [resume] Nothing to resume at {resume_dir}; starting fresh.", flush=True)
        return 0

    # Apply resume state (optimizer, scheduler, step, best_f1) now that they exist
    start_epoch = 0
    if args.resume_from:
        start_epoch = _load_resume(args.resume_from)

    if device.type == "cuda":
        vram_gb = torch.cuda.memory_allocated() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        headroom = (1 - vram_gb / vram_total) * 100
        print(f"Pre-train VRAM: {vram_gb:.1f}/{vram_total:.1f} GB ({headroom:.0f}% headroom)", flush=True)

    for epoch in range(start_epoch, cfg.epochs):
        train_sampler.set_epoch(epoch)
        running = 0.0
        running_cons = 0.0
        cons_fire_count = 0
        n_in_window = 0
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = {t: v.to(device) for t, v in batch["labels"].items()}
            out = model(input_ids=input_ids, attention_mask=attn)
            # Per-sample masking: -100 is PyTorch CE ignore_index — focal_ce
            # already handles it, no batch-level skipping needed.
            task_loss = model.compute_loss(out, labels)
            # v2.2: adversarial consistency — force base/homo/emoji of same
            # pair_id to have similar toxicity logits.
            cons_loss = torch.tensor(0.0, device=task_loss.device, dtype=task_loss.dtype)
            if args.consistency_lambda > 0:
                pair_ids = batch["pair_ids"].to(device)
                cons_loss = consistency_loss(out.logits["toxicity"], pair_ids)
                loss = (task_loss + args.consistency_lambda * cons_loss) / cfg.grad_accum
                if cons_loss.item() > 1e-6:
                    running_cons += cons_loss.item()
                    cons_fire_count += 1
            else:
                loss = task_loss / cfg.grad_accum
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
                    cons_tag = ""
                    if args.consistency_lambda > 0:
                        avg_cons = running_cons / max(1, cons_fire_count)
                        cons_tag = f" cons(avg|fired)={avg_cons:.3f}|{cons_fire_count}"
                    msg = (f"ep{epoch} step{global_step}/{total_steps} "
                           f"loss={avg:.4f}{cons_tag} lr={sched.get_last_lr()[0]:.2e} "
                           f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB "
                           f"elapsed={elapsed:.0f}s")
                    print(msg, flush=True)
                    log_lines.append(msg)
                    running = 0.0
                    running_cons = 0.0
                    cons_fire_count = 0
                    n_in_window = 0
                # mid-train safety checkpoint (regenerable but saves time on crash)
                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    _save("checkpoint_step")
                    _save_resume("checkpoint_step")
                    print(f"  [save] checkpoint_step at step {global_step} (+resume state)", flush=True)
                # mid-train eval (optional)
                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    mid = evaluate(model, eval_loader, device)
                    print(f"  [mid-eval step {global_step}] {mid}", flush=True)
                    log_lines.append(f"mid-eval step {global_step}: {mid}")

        # Free fragmented CUDA memory before eval (prevents OOM on long sequences)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # eval at end of epoch
        # Reset torch.compile cache before eval to avoid DoRA dynamic-shape
        # assertion crash when switching train→eval→train modes (symbol guard
        # invalidation on s52). Costs ~1-2 min recompile but prevents crash.
        torch._dynamo.reset()
        eval_metrics = evaluate(model, eval_loader, device)
        torch._dynamo.reset()  # also reset after eval before next epoch's train
        print(f"== Epoch {epoch} eval == {eval_metrics}", flush=True)
        log_lines.append(f"epoch{epoch} eval: {eval_metrics}")

        # save adapter + heads if best (track macro F1 of toxicity for severe-class fairness)
        tox = eval_metrics.get("toxicity", {})
        score = tox.get("f1_weighted", 0.0)
        if score > best_f1:
            best_f1 = score
            _save("best", eval_metrics)
            _save_resume("best")
            print(f"Saved best (toxicity F1_w={score:.4f}) -> {Path(cfg.output_dir) / 'best'}", flush=True)

        # Always save end-of-epoch resume point (separate from best)
        _save("epoch_end")
        _save_resume("epoch_end")

    # always save final
    _save("final")
    Path(cfg.output_dir, "train_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nDone. Best toxicity F1_w={best_f1:.4f}. Final adapter -> {Path(cfg.output_dir) / 'final'}", flush=True)


if __name__ == "__main__":
    main()
