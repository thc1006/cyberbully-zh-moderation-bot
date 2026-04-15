"""CyberPuppy v2 evaluation: COLD test + 6 繁中威脅句 + 對比 v1/Qwen3Guard.

Reuses the same Qwen3MultiHead wrapper + LoRA adapter saved by train_qwen3_lora.py.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import (accuracy_score, classification_report, f1_score)
from transformers import AutoModel, AutoTokenizer

from cyberpuppy.data.phase2 import LABELS
from cyberpuppy.models.qwen3_multihead import HEAD_DIMS, Qwen3MultiHead

DEFAULT_BASE = "Qwen/Qwen3-8B-Base"
DEFAULT_ADAPTER = "models/cyberpuppy_v2_qwen3_8b/best"
TEST_FILE = Path("data/processed/v2/cold_test.jsonl")

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


def load_v2(adapter_dir: str, base_id: str = DEFAULT_BASE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}  dtype: {dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Loading base {base_id} ...", flush=True)
    backbone = AutoModel.from_pretrained(base_id, dtype=dtype, low_cpu_mem_usage=True,
                                          attn_implementation="sdpa")
    print(f"Loading LoRA adapter {adapter_dir}/lora ...", flush=True)
    backbone = PeftModel.from_pretrained(backbone, f"{adapter_dir}/lora")

    model = Qwen3MultiHead(backbone, hidden_size=backbone.config.hidden_size).to(device=device, dtype=dtype)
    head_state = torch.load(f"{adapter_dir}/heads.pt", map_location=device, weights_only=False)
    model.heads.load_state_dict(head_state["heads"])
    if "log_var" in head_state and isinstance(model.log_var, torch.nn.Parameter):
        with torch.no_grad():
            model.log_var.copy_(head_state["log_var"].to(device).to(model.log_var.dtype))
    model.eval()
    print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)
    return tokenizer, model, device


def predict_batch(tokenizer, model, device, texts: list[str], max_length: int = 256) -> dict[str, list[int]]:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return {task: out.logits[task].argmax(-1).cpu().tolist() for task in HEAD_DIMS}


def eval_on_jsonl(tokenizer, model, device, path: Path, batch: int = 32) -> dict:
    with path.open() as f:
        records = [json.loads(line) for line in f]
    texts = [r["text"] for r in records]
    y_true = {task: [] for task in HEAD_DIMS}
    y_pred = {task: [] for task in HEAD_DIMS}

    n = len(texts)
    t0 = time.time()
    for i in range(0, n, batch):
        chunk = texts[i:i + batch]
        chunk_records = records[i:i + batch]
        preds = predict_batch(tokenizer, model, device, chunk)
        for j, r in enumerate(chunk_records):
            for task in HEAD_DIMS:
                gold_str = r["label"].get(task)
                if gold_str is None:
                    continue
                y_true[task].append(LABEL2ID[task][gold_str])
                y_pred[task].append(preds[task][j])
        if (i // batch) % 10 == 0:
            done = i + len(chunk)
            rate = done / (time.time() - t0)
            print(f"  {done:>5}/{n}  {rate:5.1f} samp/s", flush=True)
    elapsed = time.time() - t0

    metrics = {}
    for task in HEAD_DIMS:
        if not y_true[task]:
            metrics[task] = {"n": 0}
            continue
        report = classification_report(
            y_true[task], y_pred[task], target_names=ID2LABEL[task],
            labels=list(range(len(ID2LABEL[task]))),
            digits=4, zero_division=0, output_dict=True,
        )
        metrics[task] = {
            "n": len(y_true[task]),
            "accuracy": float(accuracy_score(y_true[task], y_pred[task])),
            "f1_weighted": float(f1_score(y_true[task], y_pred[task], average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true[task], y_pred[task], average="macro", zero_division=0)),
            "per_class": {cls: report[cls] for cls in ID2LABEL[task] if cls in report},
        }
    return {"metrics": metrics, "elapsed_s": elapsed, "throughput_sps": n / elapsed}


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
        out.append({"text": text, "gold": gold, "pred_toxicity": tox,
                     "pred_bullying": bull, "ok": ok})
    print(f"\n=== 繁中威脅 6 句結果: {correct}/6 = {correct/6:.0%} ===", flush=True)
    for r in out:
        mark = "✅" if r["ok"] else "❌"
        print(f"  {mark} gold={r['gold']:<5} tox={r['pred_toxicity']:<5} bull={r['pred_bullying']:<10} | {r['text']}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--test", default=str(TEST_FILE))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="reports/v2_eval.json")
    args = ap.parse_args()

    tokenizer, model, device = load_v2(args.adapter, args.base)

    print(f"\n=== Eval on {args.test} ===", flush=True)
    test_result = eval_on_jsonl(tokenizer, model, device, Path(args.test), batch=args.batch)
    print(f"\nElapsed: {test_result['elapsed_s']:.1f}s, throughput {test_result['throughput_sps']:.1f} samp/s", flush=True)
    for task, m in test_result["metrics"].items():
        print(f"\n--- {task} (n={m['n']}) ---", flush=True)
        if m["n"] == 0:
            continue
        print(f"  acc={m['accuracy']:.4f}  f1_w={m['f1_weighted']:.4f}  f1_m={m['f1_macro']:.4f}", flush=True)
        for cls, c in m["per_class"].items():
            print(f"    {cls:<12} P={c['precision']:.4f} R={c['recall']:.4f} F1={c['f1-score']:.4f} n={int(c['support'])}", flush=True)

    threat_results = eval_traditional_threats(tokenizer, model, device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        json.dump({
            "model": "cyberpuppy_v2_qwen3_8b",
            "adapter": args.adapter,
            "test_set": str(args.test),
            "test_result": test_result,
            "traditional_threats": threat_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
