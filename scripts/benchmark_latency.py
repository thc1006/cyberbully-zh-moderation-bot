"""Latency benchmark: bf16 merged vs AWQ quantized v2.2.

Careful methodology (RTX 5090 / Blackwell specific):
- 20 warmup iterations (kernel compile + CUDA graph cache warm)
- torch.cuda.synchronize() around each measurement
- Realistic input mix: short 繁中 (~20 chars) + medium (~100) + long (~200)
- Metrics: p50 / p95 / p99 per batch size
- Compares both flavors so we can decide if AWQ's F1 tradeoff buys worthwhile speedup

Usage:
  python scripts/benchmark_latency.py --flavor bf16
  python scripts/benchmark_latency.py --flavor awq
"""
from __future__ import annotations

# Monkey-patch for AWQ import compatibility under transformers ≥4.52
# (PytorchGELUTanh renamed to GELUTanh; autoawq 0.2.9 still uses the old name).
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh  # type: ignore[attr-defined]

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from cyberpuppy.models.qwen3_multihead import HEAD_DIMS, Qwen3MultiHead

SAMPLES_MIX = {
    "short": "你這個笨蛋，滾開！",  # ~8 tokens
    "medium": "我最近心情真的很差，昨天跟同事吵架了，今天還被老闆罵。感覺整個世界都在和我作對，連回家的路上公車都遲到半小時。",  # ~50 tokens
    "long": "這家餐廳的服務態度真的很差，從一開始排隊就讓人等了四十分鐘，點完餐又過了半小時才上菜，最氣的是湯根本是冷的，服務生還一副不耐煩的樣子。我跟朋友一致認為以後絕對不會再來，也會在網路上提醒大家避雷。" * 2,  # ~200 tokens
}


def load_bf16(merged_dir: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(merged_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    backbone = AutoModel.from_pretrained(merged_dir, dtype=torch.bfloat16,
                                           low_cpu_mem_usage=True, attn_implementation="sdpa")
    model = Qwen3MultiHead(backbone, hidden_size=backbone.config.hidden_size).to(device=device, dtype=torch.bfloat16)
    state = torch.load(f"{merged_dir}/heads.pt", map_location=device, weights_only=False)
    model.heads.load_state_dict(state["heads"])
    model.eval()
    return tokenizer, model


def load_awq(awq_dir: str, device: torch.device):
    # AWQ quantized weights are loaded via standard transformers w/ quant config
    tokenizer = AutoTokenizer.from_pretrained(awq_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    # AWQ models come back via AutoModel if config has quantization_config; require
    # awq kernels registered. transformers 4.57 supports awq via autoawq install.
    backbone = AutoModel.from_pretrained(awq_dir, device_map={"": "cuda"},
                                           low_cpu_mem_usage=True)
    # backbone is already on cuda; heads need explicit device + dtype
    model = Qwen3MultiHead(backbone, hidden_size=backbone.config.hidden_size).to(device=device, dtype=torch.float16)
    state = torch.load(f"{awq_dir}/heads.pt", map_location=device, weights_only=False)
    # heads were saved as bf16; load and cast to match AWQ compute path (fp16)
    heads_state = {k: v.to(torch.float16) for k, v in state["heads"].items()}
    model.heads.load_state_dict(heads_state)
    model.eval()
    return tokenizer, model


def benchmark(
    tokenizer,
    model,
    device,
    text: str,
    batch_size: int,
    n_iter: int = 100,
    warmup: int = 20,
) -> dict:
    """Return p50 / p95 / p99 latency in ms over n_iter for given batch_size."""
    inputs = [text] * batch_size
    enc = tokenizer(inputs, padding=True, truncation=True, max_length=192,
                     return_tensors="pt").to(device)

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "p50_ms": statistics.median(times),
        "p95_ms": statistics.quantiles(times, n=20)[-1],  # 95th
        "p99_ms": statistics.quantiles(times, n=100)[-1],  # 99th
        "mean_ms": statistics.mean(times),
        "throughput_sps": batch_size * 1000 / statistics.mean(times),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flavor", choices=["bf16", "awq"], required=True)
    ap.add_argument("--merged-dir", default="models/cyberpuppy_v2_2_merged")
    ap.add_argument("--awq-dir", default="models/cyberpuppy_v2_2_awq")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--out", default="reports/v2_2_latency.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"=== Flavor: {args.flavor} ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.time()
    if args.flavor == "bf16":
        tokenizer, model = load_bf16(args.merged_dir, device)
    else:
        tokenizer, model = load_awq(args.awq_dir, device)
    print(f"Load time: {time.time()-t0:.1f}s; VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)

    results: dict = {"flavor": args.flavor, "gpu": torch.cuda.get_device_name(0)}
    for length_label, text in SAMPLES_MIX.items():
        for bs in args.batch_sizes:
            stats = benchmark(tokenizer, model, device, text, bs, n_iter=args.n_iter)
            key = f"{length_label}_batch{bs}"
            results[key] = stats
            print(f"  [{length_label:<6} batch={bs:<3}] p50={stats['p50_ms']:.1f}ms "
                  f"p95={stats['p95_ms']:.1f}ms p99={stats['p99_ms']:.1f}ms  "
                  f"throughput={stats['throughput_sps']:.1f} samp/s", flush=True)

    # Merge with any previous flavor's results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
        except Exception:
            pass
    prev[args.flavor] = results
    out_path.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {args.flavor} results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
