"""Compare AWQ 4-bit vs merged bf16 on COLD test — F1 drop ≤ 2% gate.

Runs main .venv (transformers 4.57) — loads AWQ via the standard HF
`quantization_config` in config.json (no autoawq runtime needed).
"""
from __future__ import annotations

# Monkey-patch needed whenever autoawq 0.2.9 is imported under transformers ≥4.52
# (PytorchGELUTanh renamed to GELUTanh). transformers triggers this import when
# loading a model with awq quantization_config.
import transformers.activations as _act
if not hasattr(_act, "PytorchGELUTanh"):
    _act.PytorchGELUTanh = _act.GELUTanh  # type: ignore[attr-defined]

import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

from cyberpuppy.data.phase2 import LABELS
from cyberpuppy.models.qwen3_multihead import HEAD_DIMS, Qwen3MultiHead

MERGED = "models/cyberpuppy_v2_2_merged"
AWQ = "models/cyberpuppy_v2_2_awq"
COLD_TEST = "data/processed/v2/cold_test.jsonl"

LABEL2ID = {task: {v: i for i, v in enumerate(vals)} for task, vals in LABELS.items()}
ID2TOX = LABELS["toxicity"]


def load(model_dir: str, is_awq: bool, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if is_awq:
        # AWQ models carry quantization_config in config.json; transformers
        # >=4.36 can load them directly. Note: heads must be fp16 to match
        # AWQ activation dtype.
        backbone = AutoModel.from_pretrained(model_dir, device_map={"": "cuda"},
                                               low_cpu_mem_usage=True)
        head_dtype = torch.float16
    else:
        backbone = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True,
                                               attn_implementation="sdpa")
        head_dtype = torch.bfloat16

    model = Qwen3MultiHead(backbone, hidden_size=backbone.config.hidden_size).to(
        device=device, dtype=head_dtype,
    )
    state = torch.load(f"{model_dir}/heads.pt", map_location=device, weights_only=False)
    # Cast heads to match AWQ fp16 compute path if needed
    heads = {k: v.to(head_dtype) for k, v in state["heads"].items()}
    model.heads.load_state_dict(heads)
    model.eval()
    return tokenizer, model


def predict_batch(tokenizer, model, device, texts, batch: int = 32):
    preds = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=192,
                         return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        preds.extend(out.logits["toxicity"].argmax(-1).cpu().tolist())
    return preds


def main() -> None:
    device = torch.device("cuda")
    data = [json.loads(l) for l in open(COLD_TEST)]
    texts = [r["text"] for r in data]
    y_true = [LABEL2ID["toxicity"][r["label"]["toxicity"]] for r in data]
    print(f"COLD test: {len(texts)} samples", flush=True)

    results = {}
    for name, path, is_awq in [("bf16 merged", MERGED, False), ("AWQ 4-bit", AWQ, True)]:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        tokenizer, model = load(path, is_awq, device)
        print(f"  Loaded in {time.time()-t0:.1f}s; VRAM {torch.cuda.memory_allocated()/1024**3:.2f} GB", flush=True)
        t0 = time.time()
        y_pred = predict_batch(tokenizer, model, device, texts)
        elapsed = time.time() - t0
        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results[name] = {"accuracy": acc, "f1_w": f1w, "f1_m": f1m, "elapsed_s": elapsed}
        print(f"  acc={acc:.4f}  f1_w={f1w:.4f}  f1_m={f1m:.4f}", flush=True)
        print(f"  inference: {elapsed:.1f}s ({len(texts)/elapsed:.1f} samp/s)", flush=True)
        del model, tokenizer
        torch.cuda.empty_cache()

    # Delta
    bf = results["bf16 merged"]
    awq = results["AWQ 4-bit"]
    drop_acc = bf["accuracy"] - awq["accuracy"]
    drop_f1 = bf["f1_w"] - awq["f1_w"]
    speedup = bf["elapsed_s"] / max(awq["elapsed_s"], 1e-6)
    print("\n=== bf16 vs AWQ ===", flush=True)
    print(f"  Acc drop:  {drop_acc:+.4f} ({drop_acc/bf['accuracy']*100:+.2f}%)")
    print(f"  F1_w drop: {drop_f1:+.4f} ({drop_f1/bf['f1_w']*100:+.2f}%)")
    print(f"  AWQ speedup: {speedup:.2f}×")
    mark = "✅" if drop_f1 / bf['f1_w'] <= 0.02 else "⚠️"
    print(f"  {mark} DoD ≤ 2% drop target")

    Path("reports/v2_2_awq_parity.json").write_text(
        json.dumps({"results": results, "drop_acc": drop_acc, "drop_f1": drop_f1,
                     "speedup": speedup}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nSaved: reports/v2_2_awq_parity.json")


if __name__ == "__main__":
    main()
