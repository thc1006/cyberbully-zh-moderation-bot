"""Merge v2.2 LoRA adapter into Qwen3-8B-Base for inference deploy.

Safety: merge in fp32 (avoid bf16 rounding of small LoRA deltas), then cast
back to bf16 for the saved artefact. The 4 custom classification heads are
NOT part of the LoRA, so they stay in `heads.pt` next to the merged model.

Output layout:
  models/cyberpuppy_v2_2_merged/
    config.json, *.safetensors, tokenizer files   (HF-standard)
    heads.pt                                       (copied from adapter)

Usage:
  python scripts/merge_lora.py \\
      --adapter models/cyberpuppy_v2_2_qwen3_8b/best \\
      --base Qwen/Qwen3-8B-Base \\
      --out models/cyberpuppy_v2_2_merged
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="models/cyberpuppy_v2_2_qwen3_8b/best")
    ap.add_argument("--base", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--out", default="models/cyberpuppy_v2_2_merged")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading base {args.base} in fp32 (preserves LoRA delta precision)...", flush=True)
    t0 = time.time()
    # fp32 merge: tiny LoRA updates (r=32 delta) survive; bf16 would round some away.
    base = AutoModel.from_pretrained(args.base, dtype=torch.float32, low_cpu_mem_usage=True)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    print(f"[2/4] Attaching LoRA adapter {args.adapter}/lora ...", flush=True)
    model = PeftModel.from_pretrained(base, f"{args.adapter}/lora")
    print(f"  merged {sum(1 for _ in model.named_modules())} modules", flush=True)

    print("[3/4] merge_and_unload()...", flush=True)
    t1 = time.time()
    merged = model.merge_and_unload()
    print(f"  done in {time.time()-t1:.1f}s", flush=True)

    print(f"[4/4] Casting to bf16 and saving to {out} ...", flush=True)
    merged = merged.to(torch.bfloat16)
    merged.save_pretrained(out)
    # Tokenizer from BASE (our training-saved tokenizer is transformers 5.x format
    # and incompatible with 4.57; base tokenizer is functionally identical).
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.save_pretrained(out)
    # Copy classification heads alongside.
    src_heads = Path(args.adapter) / "heads.pt"
    shutil.copy2(src_heads, out / "heads.pt")
    print(f"  wrote: {sorted(p.name for p in out.iterdir())}", flush=True)

    print("\nDone. Verify with scripts/verify_merged_parity.py.", flush=True)


if __name__ == "__main__":
    main()
