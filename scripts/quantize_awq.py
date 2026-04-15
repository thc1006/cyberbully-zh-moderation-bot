"""AWQ 4-bit quantization of the merged v2.2 model.

Notes:
- autoawq 0.2.9 needs a monkey-patch for transformers 4.57+ GELU rename.
- AWQ quantizes backbone Q/K/V/O + MLP matrices only. Our 4 classification
  heads are NOT touched; they stay bf16 in heads.pt.
- AWQ expects CausalLM; our merged model is AutoModel (no lm_head). We wrap
  it as Qwen3ForCausalLM by adding a dummy lm_head at load time — the dummy
  lm_head is never used by our inference (we use hidden_states -> custom heads).

Calibration: 128 samples drawn from multisource train.

Output: models/cyberpuppy_v2_2_awq/
"""
from __future__ import annotations

# Designed to run under .venv-quant (transformers 4.51.3 + autoawq 0.2.9),
# NOT under main .venv (transformers 4.57) — autoawq 0.2.9's Catcher class
# incompatible with Qwen3 hybrid attention API introduced in 4.52+.
# Keep a monkey-patch fallback for the legacy GELU rename just in case.
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh  # type: ignore[attr-defined]

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def load_calibration(path: str, n: int, seed: int = 42) -> list[str]:
    rows = [json.loads(l) for l in open(path)]
    random.Random(seed).shuffle(rows)
    texts = [r["text"] for r in rows if len(r.get("text", "")) >= 10][:n]
    print(f"  calibration: {len(texts)} samples from {path}", flush=True)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="models/cyberpuppy_v2_2_merged")
    ap.add_argument("--out", default="models/cyberpuppy_v2_2_awq")
    ap.add_argument("--calib", default="data/processed/v2/v2_2_train.jsonl")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--w-bit", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }
    print(f"[1/3] Loading merged model as AWQ CausalLM wrapper (safetensors, ~16 GB bf16)...", flush=True)
    # AutoAWQForCausalLM loads as Qwen3ForCausalLM; our merged dir has AutoModel
    # weights but same arch key 'Qwen3Model' → autoawq wraps it.
    model = AutoAWQForCausalLM.from_pretrained(
        args.merged, safetensors=True, device_map={"": "cuda"}, trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.merged, trust_remote_code=False)

    print(f"[2/3] Calibration + quantization (w_bit={args.w_bit}, group={args.q_group_size})...", flush=True)
    calib = load_calibration(args.calib, args.n_calib)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib, max_calib_seq_len=192)

    print(f"[3/3] Saving quantized model to {out} ...", flush=True)
    model.save_quantized(str(out), safetensors=True)
    tokenizer.save_pretrained(str(out))
    # Copy classification heads + any aux files.
    shutil.copy2(Path(args.merged) / "heads.pt", out / "heads.pt")

    # Report size
    total_bytes = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"\nDone. AWQ artefact size: {total_bytes/1024**3:.2f} GB", flush=True)
    print(f"Files: {sorted(p.name for p in out.iterdir())}", flush=True)


if __name__ == "__main__":
    main()
