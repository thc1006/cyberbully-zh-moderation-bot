"""Disagreement-based LLM cascade evaluation.

Pipeline:
  1. Run dual-LoRA ensemble (text + pinyin) on ALL samples
  2. Identify text-pinyin disagreement samples
  3. Free GPU → load Qwen3-8B Instruct (chat template)
  4. Few-shot classify ONLY disagreement samples
  5. Merge: agree → ensemble result, disagree → LLM result

Usage:
  PYTHONPATH=src python scripts/eval_cascade.py
  PYTHONPATH=src python scripts/eval_cascade.py --llm Qwen/Qwen3-8B-Base  # fallback
"""
from __future__ import annotations

import argparse, json, re, time, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
from pypinyin import Style, pinyin as get_pinyin
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from cyberpuppy.models.qwen3_multihead import HEAD_DIMS, Qwen3MultiHead
from cyberpuppy.data.phase2 import LABELS
from datasets import load_dataset

_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LABEL2ID = {task: {v: i for i, v in enumerate(LABELS[task])} for task in LABELS}
ID2LABEL = {task: list(vals) for task, vals in LABELS.items()}
ALPHA = 0.75

def text_to_pinyin(text):
    syls = []
    for ch in text:
        if _HAN_RE.match(ch):
            try:
                py = get_pinyin(ch, style=Style.NORMAL, errors="ignore")
                if py and py[0] and py[0][0]: syls.append(py[0][0])
            except: pass
        elif ch not in ' \t\n': syls.append(ch)
    return " ".join(syls)

SYSTEM_PROMPT = """你是一位中文網路安全專家，專門辨識各種隱藏式有毒內容。用戶會使用以下手法：
- 諧音字：「勾史」=「狗屎」，「四調」=「死掉」
- 數字代替：「4了」=「死了」，「13」=「B（逼）」
- 字母代替：「装X」=「装逼」，「G8」=「雞巴」，「NMSL」=「你媽死了」
- 詞級暗語：「密碼」=「你媽」，「春竹」=「蠢豬」
- 英文諧音：「funny mud pee」=「放你媽屁」

你必須只回答「有毒」或「安全」兩個字，不要解釋。"""

FEW_SHOT_EXAMPLES = [
    ("密碼4了", "有毒"),
    ("装X", "有毒"),
    ("今天天氣真好，我們去公園玩吧", "安全"),
    ("勾史", "有毒"),
    ("你真太6了", "安全"),
    ("你是亞里士多德的妹妹", "有毒"),
    ("四調的黑人是好人", "有毒"),
    ("我對你一見粽情", "安全"),
]

# Legacy: plain text prompt for Base model fallback
FEW_SHOT_PROMPT = """你是一位中文網路安全專家。網路用戶經常使用以下手法隱藏有毒內容：
- 諧音字：「勾史」=「狗屎」，「四調」=「死掉」
- 數字代替：「4了」=「死了」，「13」=「B（逼）」
- 字母代替：「装X」=「装逼」，「G8」=「雞巴」，「NMSL」=「你媽死了」
- 詞級暗語：「密碼」=「你媽」，「春竹」=「蠢豬」
- 英文諧音：「funny mud pee」=「放你媽屁」

以下是一些例子：
例1：「密碼4了」→ 有毒（密碼≈你媽, 4=死, 意思「你媽死了」）
例2：「装X」→ 有毒（X=逼, 意思「装逼」）
例3：「今天天氣真好，我們去公園玩吧」→ 安全
例4：「勾史」→ 有毒（諧音「狗屎」）
例5：「你真太6了」→ 安全（6=厲害，是讚美）
例6：「你是亞里士多德的妹妹」→ 有毒（暗示「你是妓女」的委婉說法）
例7：「四調的黑人是好人」→ 有毒（四調=死掉，意思「死掉的黑人是好人」）
例8：「我對你一見粽情」→ 安全（諧音「一見鍾情」，是甜話）

請判斷以下文字是否有毒，只回答「有毒」或「安全」：
「{text}」
判斷："""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', default='Qwen/Qwen3-8B',
                        help='LLM for cascade tier 2 (default: Qwen/Qwen3-8B Instruct)')
    args = parser.parse_args()
    llm_name = args.llm
    use_chat_template = 'Base' not in llm_name  # Instruct models use chat template

    device = torch.device('cuda')
    dtype = torch.bfloat16

    # ================================================================
    # Stage 1: Run dual-LoRA ensemble on all data
    # ================================================================
    print("=" * 60)
    print("STAGE 1: Dual-LoRA ensemble on all samples")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B-Base')
    tok.padding_side = 'left'
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id

    # Load text LoRA-A
    print("Loading LoRA-A (text)...", flush=True)
    base_a = AutoModel.from_pretrained('Qwen/Qwen3-8B-Base', dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation='sdpa')
    lora_a = PeftModel.from_pretrained(base_a, 'models/cyberpuppy_v5_bilingual_qwen3_8b/best/lora')
    model_a = Qwen3MultiHead(lora_a, hidden_size=lora_a.config.hidden_size).to(device, dtype)
    st_a = torch.load('models/cyberpuppy_v5_bilingual_qwen3_8b/best/heads.pt',
        map_location=device, weights_only=False)
    model_a.heads.load_state_dict(st_a['heads']); model_a.eval()
    del base_a

    # Load pinyin LoRA-B
    print("Loading LoRA-B (pinyin)...", flush=True)
    base_b = AutoModel.from_pretrained('Qwen/Qwen3-8B-Base', dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation='sdpa')
    lora_b = PeftModel.from_pretrained(base_b, 'models/cyberpuppy_v5_pinyin_lora/best/lora')
    model_b = Qwen3MultiHead(lora_b, hidden_size=lora_b.config.hidden_size).to(device, dtype)
    st_b = torch.load('models/cyberpuppy_v5_pinyin_lora/best/heads.pt',
        map_location=device, weights_only=False)
    model_b.heads.load_state_dict(st_b['heads']); model_b.eval()
    del base_b

    def tier1_predict(texts, batch_size=16):
        """Returns text_binary, pinyin_binary, ensemble_binary for each sample."""
        all_t, all_p, all_e = [], [], []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            pinyins = [text_to_pinyin(t) for t in batch]
            t_enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=192).to(device)
            p_enc = tok(pinyins, return_tensors='pt', padding=True, truncation=True, max_length=192).to(device)
            with torch.inference_mode():
                t_out = model_a(input_ids=t_enc['input_ids'], attention_mask=t_enc['attention_mask'])
                p_out = model_b(input_ids=p_enc['input_ids'], attention_mask=p_enc['attention_mask'])
            t_probs = torch.softmax(t_out.logits['toxicity'].float(), -1)
            p_probs = torch.softmax(p_out.logits['toxicity'].float(), -1)
            ens = ALPHA * t_probs + (1 - ALPHA) * p_probs

            all_t.extend((t_probs.argmax(-1) > 0).cpu().tolist())
            all_p.extend((p_probs.argmax(-1) > 0).cpu().tolist())
            all_e.extend((ens.argmax(-1) > 0).cpu().tolist())
        return all_t, all_p, all_e

    # Load datasets
    print("\nLoading datasets...", flush=True)
    cold = [json.loads(l) for l in Path('data/processed/v2/cold_test.jsonl').open()]
    cold_texts = [r['text'] for r in cold]
    cold_labels = [1 if r['label']['toxicity'] in ('toxic','severe') else 0 for r in cold]

    ds = load_dataset("UTSNLPGroup/PCR-ToxiCN", cache_dir="data/external/PCR-ToxiCN")
    pcr_texts = list(ds['train']['text'])
    pcr_labels = list(ds['train']['offensive_label'])

    heldout = [json.loads(l) for l in Path('data/processed/v2/toxicloak_heldout.jsonl').open()]
    homo = [r for r in heldout if r['metadata']['cloak_variant'] == 'homo']
    homo_texts = [r['text'] for r in homo]
    homo_labels = [1 if r['label']['toxicity'] in ('toxic','severe') else 0 for r in homo]

    # Run Tier 1 on all
    print("Running Tier 1 on COLD...", flush=True)
    cold_t, cold_p, cold_e = tier1_predict(cold_texts)
    print("Running Tier 1 on PCR-ToxiCN...", flush=True)
    pcr_t, pcr_p, pcr_e = tier1_predict(pcr_texts)
    print("Running Tier 1 on TC homo...", flush=True)
    homo_t, homo_p, homo_e = tier1_predict(homo_texts)

    # Collect disagreement indices
    cold_disagree = [i for i in range(len(cold_t)) if cold_t[i] != cold_p[i]]
    pcr_disagree = [i for i in range(len(pcr_t)) if pcr_t[i] != pcr_p[i]]
    homo_disagree = [i for i in range(len(homo_t)) if homo_t[i] != homo_p[i]]

    print(f"\nDisagreement: COLD {len(cold_disagree)}/{len(cold_t)} "
          f"PCR {len(pcr_disagree)}/{len(pcr_t)} "
          f"TC_homo {len(homo_disagree)}/{len(homo_t)}")

    # Free GPU
    print("\nFreeing Tier 1 models...", flush=True)
    del model_a, model_b, lora_a, lora_b
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # ================================================================
    # Stage 2: LLM few-shot on disagreement samples
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"STAGE 2: {llm_name} few-shot on disagreement samples")
    print(f"  Chat template: {use_chat_template}")
    print(f"{'=' * 60}")

    all_disagree_texts = (
        [cold_texts[i] for i in cold_disagree] +
        [pcr_texts[i] for i in pcr_disagree] +
        [homo_texts[i] for i in homo_disagree]
    )
    print(f"Total disagreement samples: {len(all_disagree_texts)}", flush=True)

    # Load LLM — use separate tokenizer for Instruct (has chat template)
    print(f"Loading {llm_name} CausalLM...", flush=True)
    gen_tok = AutoTokenizer.from_pretrained(llm_name)
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id
    gen_model = AutoModelForCausalLM.from_pretrained(llm_name,
        dtype=dtype, device_map=device, low_cpu_mem_usage=True)
    gen_model.eval()
    print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB", flush=True)

    def build_chat_prompt(text):
        """Build chat-template prompt with few-shot examples as multi-turn."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex_text, ex_label in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": f"「{ex_text}」"})
            messages.append({"role": "assistant", "content": ex_label})
        messages.append({"role": "user", "content": f"「{text}」"})
        return gen_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    def llm_classify(text):
        if use_chat_template:
            prompt = build_chat_prompt(text)
        else:
            prompt = FEW_SHOT_PROMPT.format(text=text)
        inputs = gen_tok(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = gen_model.generate(**inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=gen_tok.eos_token_id)
        gen = gen_tok.decode(out[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        if "有毒" in gen:
            return 1
        elif "安全" in gen:
            return 0
        else:
            return -1  # unparseable

    # Classify all disagreement samples
    t0 = time.time()
    llm_preds = {}
    unparseable = 0
    for name, indices, texts in [
        ("COLD", cold_disagree, cold_texts),
        ("PCR", pcr_disagree, pcr_texts),
        ("TC_homo", homo_disagree, homo_texts),
    ]:
        for idx in indices:
            pred = llm_classify(texts[idx])
            llm_preds[(name, idx)] = pred
            if pred == -1:
                unparseable += 1
        print(f"  {name}: {len(indices)} samples classified", flush=True)

    elapsed = time.time() - t0
    print(f"LLM classification: {len(llm_preds)} samples in {elapsed:.0f}s "
          f"({elapsed/max(1,len(llm_preds)):.2f}s/sample)")
    print(f"  Unparseable: {unparseable}/{len(llm_preds)} "
          f"({100*unparseable/max(1,len(llm_preds)):.1f}%)")

    # Free LLM
    del gen_model, gen_tok
    torch.cuda.empty_cache()

    # ================================================================
    # Stage 3: Merge results
    # ================================================================
    print(f"\n{'=' * 60}")
    print("STAGE 3: Merge ensemble + LLM results")
    print(f"{'=' * 60}")

    def merge_and_eval(name, texts, labels, t_preds, p_preds, e_preds, disagree_idx):
        # Final predictions: agree → ensemble, disagree → LLM (fallback to ensemble)
        final_ens_only = list(e_preds)  # baseline: ensemble only
        final_full = list(e_preds)      # full cascade: all disagree → LLM
        final_asym = list(e_preds)      # asymmetric: only override when LLM says toxic

        for idx in disagree_idx:
            llm_pred = llm_preds.get((name, idx), -1)
            if llm_pred >= 0:
                final_full[idx] = llm_pred
                if llm_pred == 1:  # toxic-only override
                    final_asym[idx] = 1

        f1_ens = f1_score(labels, final_ens_only, average='weighted', zero_division=0)
        f1_full = f1_score(labels, final_full, average='weighted', zero_division=0)
        f1_asym = f1_score(labels, final_asym, average='weighted', zero_division=0)
        acc_ens = accuracy_score(labels, final_ens_only)
        acc_full = accuracy_score(labels, final_full)
        acc_asym = accuracy_score(labels, final_asym)

        print(f"\n  {name} ({len(disagree_idx)} disagree / {len(labels)} total):")
        print(f"    Ensemble only:   F1={f1_ens:.4f}  Acc={acc_ens:.4f}")
        print(f"    + Full cascade:  F1={f1_full:.4f}  Acc={acc_full:.4f}  (Δ={f1_full-f1_ens:+.4f})")
        print(f"    + Asym (toxic):  F1={f1_asym:.4f}  Acc={acc_asym:.4f}  (Δ={f1_asym-f1_ens:+.4f})")
        return f1_ens, f1_full, f1_asym

    results = {}
    results['COLD'] = merge_and_eval("COLD", cold_texts, cold_labels,
        cold_t, cold_p, cold_e, cold_disagree)
    results['PCR'] = merge_and_eval("PCR", pcr_texts, pcr_labels,
        pcr_t, pcr_p, pcr_e, pcr_disagree)
    results['TC_homo'] = merge_and_eval("TC_homo", homo_texts, homo_labels,
        homo_t, homo_p, homo_e, homo_disagree)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY — LLM: {llm_name}")
    print(f"{'=' * 60}")
    for name in ['COLD', 'PCR', 'TC_homo']:
        ens, full, asym = results[name]
        best = max(ens, full, asym)
        tag = '★' if best > ens else ''
        print(f"  {name:10s}  ens={ens:.4f}  full={full:.4f}  asym={asym:.4f}  {tag}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
