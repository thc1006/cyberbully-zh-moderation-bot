"""Train the Learned Router on dev set using frozen dual-LoRA hidden states.

Pipeline:
  1. Load both LoRA models (frozen)
  2. Forward all dev samples → collect (text_hidden, text_probs, pinyin_probs, label)
  3. Train router MLP to minimize NLLLoss on routed ensemble
  4. Evaluate on test sets (COLD, PCR, TC homo) and compare vs fixed α=0.75

Usage:
  PYTHONPATH=src python scripts/train_router.py
"""
from __future__ import annotations

import json, time, torch, torch.nn as nn
from pathlib import Path
from pypinyin import Style, pinyin as get_pinyin
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from cyberpuppy.models.qwen3_multihead import Qwen3MultiHead
from cyberpuppy.models.learned_router import LearnedRouter, router_ensemble
from cyberpuppy.data.phase2 import LABELS
from datasets import load_dataset
import re

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ALPHA_FIXED = 0.75


def text_to_pinyin(text):
    syls = []
    for ch in text:
        if _HAN_RE.match(ch):
            try:
                py = get_pinyin(ch, style=Style.NORMAL, errors="ignore")
                if py and py[0] and py[0][0]:
                    syls.append(py[0][0])
            except:
                pass
        elif ch not in ' \t\n':
            syls.append(ch)
    return " ".join(syls)


@torch.inference_mode()
def extract_features(model_a, model_b, tok, texts, device, batch_size=16):
    """Extract text hidden states, text probs, pinyin probs for all samples."""
    all_hidden = []
    all_text_probs = []
    all_pinyin_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        pinyins = [text_to_pinyin(t) for t in batch]

        t_enc = tok(batch, return_tensors='pt', padding=True,
                    truncation=True, max_length=192).to(device)
        p_enc = tok(pinyins, return_tensors='pt', padding=True,
                    truncation=True, max_length=192).to(device)

        t_out = model_a(input_ids=t_enc['input_ids'],
                        attention_mask=t_enc['attention_mask'])
        p_out = model_b(input_ids=p_enc['input_ids'],
                        attention_mask=p_enc['attention_mask'])

        # Pooled hidden state from text model (last token, after dropout)
        all_hidden.append(t_out.pooled.cpu())

        # Toxicity probs
        t_probs = torch.softmax(t_out.logits['toxicity'].float(), -1)
        p_probs = torch.softmax(p_out.logits['toxicity'].float(), -1)
        all_text_probs.append(t_probs.cpu())
        all_pinyin_probs.append(p_probs.cpu())

        if (i // batch_size) % 20 == 0:
            print(f"  {i}/{len(texts)}", flush=True)

    return (torch.cat(all_hidden),
            torch.cat(all_text_probs),
            torch.cat(all_pinyin_probs))


def main():
    device = torch.device('cuda')
    dtype = torch.bfloat16

    # ================================================================
    # Stage 1: Load models and extract features
    # ================================================================
    print("=" * 60)
    print("Loading dual-LoRA models...")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B-Base')
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # LoRA-A (text)
    base_a = AutoModel.from_pretrained('Qwen/Qwen3-8B-Base', dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation='sdpa')
    lora_a = PeftModel.from_pretrained(base_a,
        'models/cyberpuppy_v5_bilingual_qwen3_8b/best/lora')
    model_a = Qwen3MultiHead(lora_a, hidden_size=lora_a.config.hidden_size).to(device, dtype)
    st_a = torch.load('models/cyberpuppy_v5_bilingual_qwen3_8b/best/heads.pt',
        map_location=device, weights_only=False)
    model_a.heads.load_state_dict(st_a['heads'])
    model_a.eval()
    del base_a

    # LoRA-B (pinyin)
    base_b = AutoModel.from_pretrained('Qwen/Qwen3-8B-Base', dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation='sdpa')
    lora_b = PeftModel.from_pretrained(base_b,
        'models/cyberpuppy_v5_pinyin_lora/best/lora')
    model_b = Qwen3MultiHead(lora_b, hidden_size=lora_b.config.hidden_size).to(device, dtype)
    st_b = torch.load('models/cyberpuppy_v5_pinyin_lora/best/heads.pt',
        map_location=device, weights_only=False)
    model_b.heads.load_state_dict(st_b['heads'])
    model_b.eval()
    del base_b

    # ================================================================
    # Stage 2: Extract features from dev set (train router) + test sets
    # ================================================================
    print("\nLoading datasets...")

    # Dev set (for training router)
    dev_data = [json.loads(l) for l in
                Path('data/processed/v2/v2_2_dev.jsonl').open()]
    dev_texts = [r['text'] for r in dev_data]
    dev_labels = [1 if r['label']['toxicity'] in ('toxic', 'severe') else 0
                  for r in dev_data]

    # Test sets
    cold = [json.loads(l) for l in
            Path('data/processed/v2/cold_test.jsonl').open()]
    cold_texts = [r['text'] for r in cold]
    cold_labels = [1 if r['label']['toxicity'] in ('toxic', 'severe') else 0
                   for r in cold]

    ds = load_dataset("UTSNLPGroup/PCR-ToxiCN", cache_dir="data/external/PCR-ToxiCN")
    pcr_texts = list(ds['train']['text'])
    pcr_labels = list(ds['train']['offensive_label'])

    heldout = [json.loads(l) for l in
               Path('data/processed/v2/toxicloak_heldout.jsonl').open()]
    homo = [r for r in heldout if r['metadata']['cloak_variant'] == 'homo']
    homo_texts = [r['text'] for r in homo]
    homo_labels = [1 if r['label']['toxicity'] in ('toxic', 'severe') else 0
                   for r in homo]

    print(f"Dev: {len(dev_texts)}, COLD: {len(cold_texts)}, "
          f"PCR: {len(pcr_texts)}, TC homo: {len(homo_texts)}")

    print("\nExtracting features (dev)...")
    dev_h, dev_tp, dev_pp = extract_features(model_a, model_b, tok,
                                             dev_texts, device)
    print("Extracting features (COLD)...")
    cold_h, cold_tp, cold_pp = extract_features(model_a, model_b, tok,
                                                cold_texts, device)
    print("Extracting features (PCR)...")
    pcr_h, pcr_tp, pcr_pp = extract_features(model_a, model_b, tok,
                                             pcr_texts, device)
    print("Extracting features (TC homo)...")
    homo_h, homo_tp, homo_pp = extract_features(model_a, model_b, tok,
                                                homo_texts, device)

    # Free GPU
    del model_a, model_b, lora_a, lora_b
    torch.cuda.empty_cache()

    # ================================================================
    # Stage 3: Train router on dev set
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Training Learned Router on dev set")
    print(f"{'=' * 60}")

    hidden_size = dev_h.shape[1]
    router = LearnedRouter(hidden_size=hidden_size).cuda()
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    dev_labels_t = torch.tensor(dev_labels, dtype=torch.long).cuda()
    dev_h_cuda = dev_h.float().cuda()
    dev_tp_cuda = dev_tp.float().cuda()
    dev_pp_cuda = dev_pp.float().cuda()

    best_loss = float('inf')
    best_state = None

    for epoch in range(200):
        router.train()
        # Mini-batch training
        perm = torch.randperm(len(dev_labels_t))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(perm), 128):
            idx = perm[i:i+128]
            h = dev_h_cuda[idx]
            tp = dev_tp_cuda[idx]
            pp = dev_pp_cuda[idx]
            labels = dev_labels_t[idx]

            alpha = router(h)
            ensemble = router_ensemble(tp, pp, alpha)
            loss = nn.NLLLoss()(ensemble.clamp(min=1e-7).log(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in router.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} (best={best_loss:.4f})")

    router.load_state_dict(best_state)
    router.eval()
    print(f"\nBest router loss: {best_loss:.4f}")

    # ================================================================
    # Stage 4: Evaluate on all benchmarks
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Evaluation: Router vs Fixed α=0.75")
    print(f"{'=' * 60}")

    def evaluate(name, hidden, text_probs, pinyin_probs, labels):
        h = hidden.float().cuda()
        tp = text_probs.float().cuda()
        pp = pinyin_probs.float().cuda()

        with torch.no_grad():
            alpha = router(h)

        # Router ensemble
        ens_router = router_ensemble(tp, pp, alpha).cpu()
        preds_router = (ens_router.argmax(-1) > 0).long().tolist()

        # Fixed α=0.75
        ens_fixed = (ALPHA_FIXED * tp + (1 - ALPHA_FIXED) * pp).cpu()
        preds_fixed = (ens_fixed.argmax(-1) > 0).long().tolist()

        f1_router = f1_score(labels, preds_router, average='weighted', zero_division=0)
        f1_fixed = f1_score(labels, preds_fixed, average='weighted', zero_division=0)
        acc_router = accuracy_score(labels, preds_router)
        acc_fixed = accuracy_score(labels, preds_fixed)

        # α statistics
        alpha_cpu = alpha.cpu()
        print(f"\n  {name}:")
        print(f"    Fixed α=0.75: F1={f1_fixed:.4f}  Acc={acc_fixed:.4f}")
        print(f"    Learned α:    F1={f1_router:.4f}  Acc={acc_router:.4f}  "
              f"(Δ={f1_router-f1_fixed:+.4f})")
        print(f"    α stats: mean={alpha_cpu.mean():.3f} "
              f"std={alpha_cpu.std():.3f} "
              f"min={alpha_cpu.min():.3f} max={alpha_cpu.max():.3f}")
        return f1_fixed, f1_router

    evaluate("Dev (train)", dev_h, dev_tp, dev_pp, dev_labels)
    results = {}
    results['COLD'] = evaluate("COLD (test)", cold_h, cold_tp, cold_pp, cold_labels)
    results['PCR'] = evaluate("PCR (test)", pcr_h, pcr_tp, pcr_pp, pcr_labels)
    results['TC_homo'] = evaluate("TC homo (test)", homo_h, homo_tp, homo_pp, homo_labels)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, (fixed, routed) in results.items():
        delta = routed - fixed
        tag = "✓" if delta > 0 else "✗"
        print(f"  {name:10s}  fixed={fixed:.4f}  router={routed:.4f}  Δ={delta:+.4f} {tag}")

    # Save router
    out_path = Path('models/learned_router/router.pt')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'hidden_size': hidden_size}, out_path)
    print(f"\nRouter saved to {out_path}")


if __name__ == "__main__":
    main()
