#!/bin/bash
# λ sweep: 0.5 / 0.7 / 1.0 on v2.2 data (same as v2.3.1 but higher λ)
# Each run: 2 epochs (v2.3.1 best was ep0/ep1, 3rd epoch doesn't help)
# Total: ~53 min × 3 = ~2.7 hours
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH=src:${PYTHONPATH:-}
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

TRAIN=data/processed/v2/v2_2_train.jsonl
EVAL=data/processed/v2/v2_2_dev.jsonl

for LAMBDA in 0.5 0.7 1.0; do
    TAG=$(echo $LAMBDA | tr '.' '_')
    OUT="models/cyberpuppy_v2_5_lambda${TAG}_qwen3_8b"
    REPORT="reports/v2_5_lambda${TAG}_eval.json"

    echo ""
    echo "============================================================"
    echo "  λ=${LAMBDA}  output=${OUT}"
    echo "============================================================"

    # Train (2 epochs, same base config as v2.3.1)
    $PY scripts/train_qwen3_lora.py \
        --train "$TRAIN" --eval "$EVAL" \
        --epochs 2 --batch 6 --grad-accum 6 --max-length 192 \
        --lr 3e-5 --focal-gamma 2.5 --consistency-lambda "$LAMBDA" \
        --num-workers 8 --log-every 100 --save-every-steps 1000 \
        --output "$OUT"

    echo ""
    echo "--- Eval λ=${LAMBDA} ---"

    # Comprehensive eval
    $PY scripts/v2_3_comprehensive_eval.py \
        --adapter "${OUT}/best" --batch 64 --out "$REPORT"

    echo ""
    echo "--- λ=${LAMBDA} DONE ---"
    echo ""
done

echo "============================================================"
echo "ALL λ SWEEP DONE. Reports:"
echo "  reports/v2_5_lambda0_5_eval.json"
echo "  reports/v2_5_lambda0_7_eval.json"
echo "  reports/v2_5_lambda1_0_eval.json"
echo "============================================================"
