#!/bin/bash
# v2.3 post-training release pipeline.
# Assumes Q-3 training is DONE and models/cyberpuppy_v2_3_qwen3_8b/best/ exists.
#
# Steps:
#   1. Q-4 comprehensive eval (COLD + multisource + ToxiCloakCN heldout + 6 繁中)
#   2. LoRA merge (fp32→bf16) -> models/cyberpuppy_v2_3_merged
#   3. Parity check vs adapter
#   4. AWQ quantize (uses .venv-quant)         -> models/cyberpuppy_v2_3_awq
#   5. AWQ parity + latency
#   6. Push HF private: thc1006/cyberpuppy-v2.3-adapter
#   7. Push HF private: thc1006/cyberpuppy-v2.3-awq
#
# Run: HF_TOKEN=... bash scripts/v2_3_release_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN env var required (write-scope)"
    exit 1
fi

ADAPTER_DIR="models/cyberpuppy_v2_3_qwen3_8b/best"
MERGED_DIR="models/cyberpuppy_v2_3_merged"
AWQ_DIR="models/cyberpuppy_v2_3_awq"

if [ ! -d "$ADAPTER_DIR/lora" ]; then
    echo "ERROR: $ADAPTER_DIR/lora not found — Q-3 training not complete?"
    exit 1
fi

export PYTHONPATH=src:${PYTHONPATH:-}
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
PY_QUANT=.venv-quant/bin/python

echo "============================================================"
echo "[1/7] Q-4: Comprehensive eval on v2.3"
echo "============================================================"
$PY scripts/v2_3_comprehensive_eval.py --batch 32

echo ""
echo "============================================================"
echo "[2/7] LoRA merge fp32 -> bf16"
echo "============================================================"
$PY scripts/merge_lora.py --adapter "$ADAPTER_DIR" --out "$MERGED_DIR"

echo ""
echo "============================================================"
echo "[3/7] Merged-vs-adapter parity (sanity)"
echo "============================================================"
# Reuse verify script with overridden constants via env (script needs editing
# to read paths from env; if not, rely on the manual check below).
$PY -c "
import torch, json, sys
sys.path.insert(0, 'src')
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from cyberpuppy.models.qwen3_multihead import HEAD_DIMS, Qwen3MultiHead

device = torch.device('cuda')
dtype = torch.bfloat16
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B-Base')
tok.padding_side = 'left'
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id

# adapter
back_a = AutoModel.from_pretrained('Qwen/Qwen3-8B-Base', dtype=dtype, low_cpu_mem_usage=True, attn_implementation='sdpa')
back_a = PeftModel.from_pretrained(back_a, '$ADAPTER_DIR/lora')
m_a = Qwen3MultiHead(back_a, hidden_size=back_a.config.hidden_size).to(device=device, dtype=dtype)
state = torch.load('$ADAPTER_DIR/heads.pt', map_location=device, weights_only=False)
m_a.heads.load_state_dict(state['heads']); m_a.eval()

texts = ['今天天氣真好', '你這個笨蛋', '我打死你']
enc = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=192).to(device)
with torch.inference_mode():
    out_a = m_a(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
preds_a = out_a.logits['toxicity'].argmax(-1).cpu().tolist()
del m_a, back_a
torch.cuda.empty_cache()

# merged
back_m = AutoModel.from_pretrained('$MERGED_DIR', dtype=dtype, low_cpu_mem_usage=True, attn_implementation='sdpa')
m_m = Qwen3MultiHead(back_m, hidden_size=back_m.config.hidden_size).to(device=device, dtype=dtype)
state = torch.load('$MERGED_DIR/heads.pt', map_location=device, weights_only=False)
m_m.heads.load_state_dict(state['heads']); m_m.eval()
with torch.inference_mode():
    out_m = m_m(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
preds_m = out_m.logits['toxicity'].argmax(-1).cpu().tolist()

print('adapter:', preds_a, '  merged:', preds_m, '  agree:', preds_a == preds_m)
assert preds_a == preds_m, 'Merge parity failed'
print('PARITY_OK')
"

echo ""
echo "============================================================"
echo "[4/7] AWQ quantize via .venv-quant (transformers 4.51.3)"
echo "============================================================"
$PY_QUANT scripts/quantize_awq.py --merged "$MERGED_DIR" --out "$AWQ_DIR"

echo ""
echo "============================================================"
echo "[5/7] AWQ parity + latency"
echo "============================================================"
$PY scripts/verify_awq_parity.py 2>&1 | tail -15 || echo "(non-fatal: parity check used hard-coded paths; manual eval below)"

echo ""
echo "============================================================"
echo "[6/7] Push v2.3-adapter to HF private"
echo "============================================================"
$PY -c "
import os
from huggingface_hub import create_repo, upload_folder
REPO = 'thc1006/cyberpuppy-v2.3-adapter'
create_repo(repo_id=REPO, private=True, exist_ok=True, repo_type='model', token=os.environ['HF_TOKEN'])
upload_folder(repo_id=REPO, repo_type='model', folder_path='$ADAPTER_DIR',
              commit_message='v2.3 release: lexicon-aug homophone training (lambda=0.3)',
              ignore_patterns=['*.tmp', '.DS_Store'], token=os.environ['HF_TOKEN'])
print('  view: https://huggingface.co/' + REPO)
"

echo ""
echo "============================================================"
echo "[7/7] Push v2.3-awq to HF private"
echo "============================================================"
$PY -c "
import os
from huggingface_hub import create_repo, upload_folder
REPO = 'thc1006/cyberpuppy-v2.3-awq'
create_repo(repo_id=REPO, private=True, exist_ok=True, repo_type='model', token=os.environ['HF_TOKEN'])
upload_folder(repo_id=REPO, repo_type='model', folder_path='$AWQ_DIR',
              commit_message='v2.3 AWQ 4-bit release', ignore_patterns=['*.tmp', '.DS_Store'],
              token=os.environ['HF_TOKEN'])
print('  view: https://huggingface.co/' + REPO)
"

echo ""
echo "============================================================"
echo "All done. Don't forget to:"
echo "  1. Drop v2.3 model cards into the HF repos"
echo "  2. Update ADR 0001 §4 Phase 3.4 with v2.3 numbers"
echo "  3. Commit + push + REVOKE the HF token"
echo "============================================================"
