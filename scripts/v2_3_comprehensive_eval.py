"""v2.2 comprehensive eval — DoD §9 showdown.

Runs v2.2 on:
1. COLD test       (apples-to-apples vs v1 / v2 / v2.1 / Qwen3Guard)
2. Multisource test (same as v2.1 eval)
3. ToxiCloakCN held-out base   (unseen clean adversarial set)
4. ToxiCloakCN held-out homo   (unseen homophone)
5. ToxiCloakCN held-out emoji  (unseen emoji)
6. 6 繁中威脅句 (Traditional Chinese threats)

Workstation-optimized: batch 64 (inference-only, ~10 GB VRAM headroom to
grow; v2.1 used 32).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModel, AutoTokenizer

from cyberpuppy.data.phase2 import LABELS
from cyberpuppy.models.qwen3_multihead import HEAD_DIMS, Qwen3MultiHead

DEFAULT_BASE = "Qwen/Qwen3-8B-Base"
DEFAULT_ADAPTER = "models/cyberpuppy_v2_3_qwen3_8b/best"

ID2LABEL = {task: list(vals) for task, vals in LABELS.items()}
LABEL2ID = {task: {v: i for i, v in enumerate(vals)} for task, vals in LABELS.items()}

SAMPLES_TC = [
    ("今天天氣真好，我們去公園玩吧！", "none"),
    ("你這個笨蛋，滾開！", "toxic"),
    ("我恨你，希望你去死，沒人喜歡你", "toxic"),
    ("謝謝你的幫忙，你真的是個很棒的人", "none"),
    ("再嘴一句試試看，我打死你", "toxic"),
    ("我覺得這部電影還不錯，演員演技也很到位", "none"),
]


def load_model(adapter_dir: str, base_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  dtype: {dtype}", flush=True)

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
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)
    return tokenizer, model, device


def predict_batch(tokenizer, model, device, texts: list[str], max_length: int = 192) -> dict:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return {task: out.logits[task].argmax(-1).cpu().tolist() for task in HEAD_DIMS}


def eval_jsonl(tokenizer, model, device, path: Path, name: str, batch: int) -> dict:
    print(f"\n=== {name}: {path.name} ===", flush=True)
    with path.open() as f:
        records = [json.loads(line) for line in f]
    texts = [r["text"] for r in records]
    y_true = {t: [] for t in HEAD_DIMS}
    y_pred = {t: [] for t in HEAD_DIMS}
    # Aligned per-sample: we only compute classification_report on samples that have each task's label
    t0 = time.time()
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        chunk_records = records[i:i + batch]
        preds = predict_batch(tokenizer, model, device, chunk)
        for j, r in enumerate(chunk_records):
            for task in HEAD_DIMS:
                gold = r["label"].get(task)
                if gold is None:
                    continue
                y_true[task].append(LABEL2ID[task][gold])
                y_pred[task].append(preds[task][j])
    elapsed = time.time() - t0

    metrics = {}
    for task in HEAD_DIMS:
        if not y_true[task]:
            metrics[task] = {"n": 0}
            continue
        metrics[task] = {
            "n": len(y_true[task]),
            "accuracy": float(accuracy_score(y_true[task], y_pred[task])),
            "f1_weighted": float(f1_score(y_true[task], y_pred[task], average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true[task], y_pred[task], average="macro", zero_division=0)),
        }
    print(f"  n={len(texts)}  elapsed={elapsed:.1f}s  {len(texts)/elapsed:.1f} samp/s", flush=True)
    for task, m in metrics.items():
        if m["n"] == 0:
            continue
        print(f"  {task:<10} n={m['n']:<6} acc={m['accuracy']:.4f} f1_w={m['f1_weighted']:.4f} f1_m={m['f1_macro']:.4f}", flush=True)
    return {"name": name, "elapsed_s": elapsed, "throughput_sps": len(texts) / elapsed, "metrics": metrics}


def eval_heldout_by_variant(tokenizer, model, device, path: Path, batch: int) -> dict:
    """Group toxicloak_heldout by cloak_variant -> separate F1 per base/homo/emoji."""
    print(f"\n=== ToxiCloakCN heldout (by variant): {path.name} ===", flush=True)
    with path.open() as f:
        records = [json.loads(line) for line in f]
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        v = r["metadata"].get("cloak_variant", "unknown")
        by_variant[v].append(r)

    # Run each variant through model
    results = {}
    for variant in ("base", "homo", "emoji"):
        if variant not in by_variant:
            continue
        subset = by_variant[variant]
        texts = [r["text"] for r in subset]
        y_true = [LABEL2ID["toxicity"][r["label"]["toxicity"]] for r in subset]
        # Collapse to binary (none vs any-toxic) for direct compare with v2.1 toxicloak_eval.py
        y_true_bin = [0 if y == 0 else 1 for y in y_true]
        y_pred_3way: list[int] = []
        t0 = time.time()
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            preds = predict_batch(tokenizer, model, device, chunk)
            y_pred_3way.extend(preds["toxicity"])
        elapsed = time.time() - t0
        y_pred_bin = [0 if p == 0 else 1 for p in y_pred_3way]
        m = {
            "n": len(texts),
            "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
            "f1_weighted": float(f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0)),
            "f1_toxic": float(f1_score(y_true_bin, y_pred_bin, average="binary", pos_label=1, zero_division=0)),
            "elapsed_s": elapsed,
        }
        results[variant] = m
        print(f"  {variant:<8} n={m['n']:<4} acc={m['accuracy']:.4f} f1_w={m['f1_weighted']:.4f} f1_toxic={m['f1_toxic']:.4f} ({elapsed:.1f}s)", flush=True)

    # Robustness drop from base
    if "base" in results:
        base_f1 = results["base"]["f1_weighted"]
        print(f"\n  === Robustness drop (vs held-out base) ===", flush=True)
        for v, m in results.items():
            drop = base_f1 - m["f1_weighted"]
            pct = drop / base_f1 * 100 if base_f1 > 0 else 0
            mark = "✅" if v == "base" or pct <= 5.0 else "⚠️"
            print(f"  {mark} {v:<8} F1_w={m['f1_weighted']:.4f}  drop={drop:+.4f} ({pct:+.2f}%)", flush=True)
    return results


def eval_traditional_threats(tokenizer, model, device) -> list[dict]:
    texts = [t for t, _ in SAMPLES_TC]
    golds = [g for _, g in SAMPLES_TC]
    preds = predict_batch(tokenizer, model, device, texts)
    out = []
    correct = 0
    for text, gold, tox_id, bull_id in zip(texts, golds, preds["toxicity"], preds["bullying"]):
        tox = ID2LABEL["toxicity"][tox_id]
        bull = ID2LABEL["bullying"][bull_id]
        gold_unsafe = gold != "none"
        pred_unsafe = tox != "none" or bull != "none"
        ok = gold_unsafe == pred_unsafe
        correct += int(ok)
        out.append({"text": text, "gold": gold, "pred_toxicity": tox, "pred_bullying": bull, "ok": ok})
    print(f"\n=== 繁中威脅 6 句結果: {correct}/6 = {correct/6:.0%} ===", flush=True)
    for r in out:
        mark = "✅" if r["ok"] else "❌"
        print(f"  {mark} gold={r['gold']:<5} tox={r['pred_toxicity']:<5} bull={r['pred_bullying']:<10} | {r['text']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="reports/v2_3_comprehensive_eval.json")
    args = ap.parse_args()

    tokenizer, model, device = load_model(args.adapter, args.base)

    all_results = {}
    all_results["cold_test"] = eval_jsonl(
        tokenizer, model, device,
        Path("data/processed/v2/cold_test.jsonl"), "COLD test", args.batch,
    )
    all_results["multisource_test"] = eval_jsonl(
        tokenizer, model, device,
        Path("data/processed/v2/v2_3_test.jsonl"), "multisource test", args.batch,
    )
    all_results["toxicloak_heldout"] = eval_heldout_by_variant(
        tokenizer, model, device,
        Path("data/processed/v2/toxicloak_heldout.jsonl"), args.batch,
    )
    all_results["traditional_threats"] = eval_traditional_threats(tokenizer, model, device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({
            "model": "cyberpuppy_v2_3_qwen3_8b",
            "adapter": args.adapter,
            "results": all_results,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSaved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
