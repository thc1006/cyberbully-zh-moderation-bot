"""ToxiCloakCN robustness eval (ADR 0001 DoD §9).

Runs v2.1 on:
  - base_data.tsv       (clean, upper bound)
  - homo_full.tsv       (homophone-substituted, e.g., 白人 -> 拜仁)
  - emoji_full.tsv      (emoji-substituted, e.g., 白 -> 👌)

Reports F1 drop from clean to each adversarial. DoD target: drop ≤ 5%.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModel, AutoTokenizer

from cyberpuppy.data.phase2 import LABELS, Phase2Normalizer
from cyberpuppy.models.qwen3_multihead import HEAD_DIMS, Qwen3MultiHead

DEFAULT_BASE = "Qwen/Qwen3-8B-Base"
DEFAULT_ADAPTER = "models/cyberpuppy_v2_1_qwen3_8b/best"
DATA_DIR = Path("data/external/ToxiCloakCN/Datasets")

ID2LABEL = {task: list(vals) for task, vals in LABELS.items()}


def load_v2_1(adapter_dir: str, base_id: str = DEFAULT_BASE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  dtype: {dtype}", flush=True)

    # Tokenizer saved during training was transformers 5.x format (extra_special_tokens
    # as list); current stack is 4.57.6 which expects dict. Load from base repo instead
    # — identical tokenizer, compatible format.
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    backbone = AutoModel.from_pretrained(base_id, dtype=dtype, low_cpu_mem_usage=True,
                                          attn_implementation="sdpa")
    backbone = PeftModel.from_pretrained(backbone, f"{adapter_dir}/lora")
    model = Qwen3MultiHead(backbone, hidden_size=backbone.config.hidden_size).to(device=device, dtype=dtype)
    head_state = torch.load(f"{adapter_dir}/heads.pt", map_location=device, weights_only=False)
    model.heads.load_state_dict(head_state["heads"])
    model.eval()
    print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)
    return tokenizer, model, device


def predict_toxicity(tokenizer, model, device, texts: list[str], batch: int = 32) -> list[int]:
    preds = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=192,
                         return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        preds.extend(out.logits["toxicity"].argmax(-1).cpu().tolist())
    return preds


def eval_variant(tokenizer, model, device, tsv_path: Path, label: str) -> dict:
    """ToxiCloak TSV: columns = content, toxic (0 = non-toxic, 1 = toxic).

    Normalize text via same OpenCC繁化 the model was trained on.
    Convert binary toxic -> 3-class toxicity: 0 -> none(0), 1 -> toxic(1).
    Collapse v2.1 predictions the same way: severe(2) -> treat as toxic for binary compare.
    """
    print(f"\n=== {label}: {tsv_path.name} ===", flush=True)
    df = pd.read_csv(tsv_path, sep="\t")
    n = Phase2Normalizer(min_chars=3, max_chars=512)
    texts = [n.opencc_convert(str(t)) for t in df["content"].tolist()]
    y_true = [int(v) for v in df["toxic"].tolist()]

    t0 = time.time()
    y_pred_3way = predict_toxicity(tokenizer, model, device, texts)
    elapsed = time.time() - t0

    # Binary collapse: {none} vs {toxic, severe}
    y_pred_bin = [0 if p == 0 else 1 for p in y_pred_3way]

    acc = accuracy_score(y_true, y_pred_bin)
    f1_w = f1_score(y_true, y_pred_bin, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    toxic_recall = f1_score(y_true, y_pred_bin, average="binary", pos_label=1, zero_division=0)
    print(f"  n={len(texts)}  elapsed={elapsed:.1f}s  {len(texts)/elapsed:.1f} samp/s", flush=True)
    print(f"  Acc={acc:.4f}  F1_w={f1_w:.4f}  F1_m={f1_m:.4f}  Toxic-F1={toxic_recall:.4f}", flush=True)
    print(classification_report(y_true, y_pred_bin, target_names=["non-toxic", "toxic"],
                                 digits=4, zero_division=0))
    return {
        "variant": label,
        "n": len(texts),
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "f1_toxic": toxic_recall,
        "elapsed_s": elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="reports/v2_1_toxicloak_eval.json")
    args = ap.parse_args()

    tokenizer, model, device = load_v2_1(args.adapter, args.base)

    variants = [
        ("base_clean", DATA_DIR / "base_data.tsv"),
        ("homophone_full", DATA_DIR / "Full_Perturbation" / "homo_full.tsv"),
        ("emoji_full", DATA_DIR / "Full_Perturbation" / "emoji_full.tsv"),
    ]
    results = []
    for name, path in variants:
        results.append(eval_variant(tokenizer, model, device, path, name))

    # Robustness drop vs clean
    base_f1 = results[0]["f1_weighted"]
    print("\n=== Robustness drop vs clean ===", flush=True)
    print(f"{'variant':<20} {'F1_w':>8}  {'drop':>8}  {'drop %':>8}")
    for r in results:
        drop = base_f1 - r["f1_weighted"]
        pct = drop / base_f1 * 100 if base_f1 > 0 else 0
        marker = "✅" if pct <= 5.0 or r['variant'] == 'base_clean' else "⚠️"
        print(f"  {marker} {r['variant']:<18} {r['f1_weighted']:>8.4f}  {drop:>+8.4f}  {pct:>+7.2f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        json.dump({
            "model": "cyberpuppy_v2_1_qwen3_8b",
            "adapter": args.adapter,
            "dod_target": "F1 drop ≤ 5% per ADR 0001 §9",
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
