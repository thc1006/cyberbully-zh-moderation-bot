# ADR 0002 — Migrate Quantization from autoawq to llm-compressor

- **Status**: Proposed (deferred; not yet executed)
- **Date**: 2026-04-16
- **Supersedes**: ADR 0001 §4 Phase 5 quantization choice (autoawq 0.2.9)
- **Triggers re-execution**: when (a) llm-compressor 0.9.0+ proves stable
  with our v2.2/v2.3 weights, or (b) autoawq breaks on a future
  transformers upgrade we cannot avoid.

## Context

ADR 0001 Phase 5 shipped v2.2 with **autoawq 0.2.9** for AWQ 4-bit
quantization, achieved via an isolated `.venv-quant` (transformers
4.51.3) because autoawq's `Catcher` class is incompatible with
transformers ≥ 4.52 hybrid attention API.

autoawq is **officially deprecated**; the maintainer points users to
vLLM's [llm-compressor](https://github.com/vllm-project/llm-compressor),
which since 0.8.0 (2025-10) supports Qwen3 and since 0.9.0 (2026-01)
supports generalized AWQ + MXFP4 + attention quantization.

This ADR records the migration plan when we choose to execute.

## Decision

When migrating, replace `scripts/quantize_awq.py` (autoawq path) with a
new `scripts/quantize_llm_compressor.py` driven by llm-compressor's
oneshot API. Keep autoawq script in repo (under archive/) for
reproducibility of v2.2.

### Key differences

| Aspect | autoawq 0.2.9 | llm-compressor 0.9+ |
|---|---|---|
| Maintenance | deprecated, no updates | active (vLLM team) |
| transformers compat | 4.51.3 only | tracks current |
| Quant schemes | W4A16 GEMM | W4A16, W8A8, FP8, MXFP4, attention quant |
| API style | imperative `model.quantize()` | declarative `oneshot(model, recipe=...)` |
| vLLM serving | requires post-load wrapping | native — `compressed_tensors` format |
| Calibration | manual list of texts | dataset abstraction (HF datasets compatible) |
| Multi-modifier | one quant scheme per run | combines AWQ + GPTQ in single oneshot |

### Sample recipe (W4A16 AWQ)

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.awq import AWQModifier
from transformers import AutoModelForCausalLM
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(
    "models/cyberpuppy_v2_2_merged", torch_dtype="auto"
)
calib = load_dataset("json",
                      data_files="data/processed/v2/v2_2_train.jsonl",
                      split="train").shuffle(seed=42).select(range(128))

recipe = AWQModifier(
    bits=4,
    group_size=128,
    targets=["Linear"],
    ignore=["lm_head", "*classifier*"],   # heads stay fp16
)

oneshot(
    model=model,
    dataset=calib,
    recipe=recipe,
    output_dir="models/cyberpuppy_v2_2_awq_lc",
    max_seq_length=192,
    num_calibration_samples=128,
)
```

## Migration plan

Estimated total: ~1.5 hr.

1. **L0 — Preflight** (10 min)
   - Create `.venv-quant-lc/` (Python 3.11 + torch cu128 + llm-compressor 0.9+)
   - `pip install llmcompressor==0.9.* compressed-tensors`
   - Verify AWQModifier supports Qwen3 architecture
2. **L1 — Recipe + script** (30 min)
   - Write `scripts/quantize_llm_compressor.py` (TDD: tests/test_lc_recipe.py)
   - Recipe ignores classification heads (preserve fp16 precision)
3. **L2 — Quantize** (15 min)
   - Run on `models/cyberpuppy_v2_2_merged`
   - Output: `models/cyberpuppy_v2_2_awq_lc/`
4. **L3 — Parity** (15 min)
   - Reuse `scripts/verify_awq_parity.py` with new path
   - Gate: COLD F1 drop ≤ 2% (same as ADR 0001)
5. **L4 — Latency bench** (10 min)
   - `scripts/benchmark_latency.py --flavor awq-lc`
   - Compare against autoawq numbers (expect within 5%)
6. **L5 — Wire into API** (10 min)
   - `api/v2_2_app.py` already auto-detects via `quantization_config`
   - llm-compressor outputs `compressed-tensors` format → may need
     transformers ≥ 4.55 + `compressed-tensors` runtime install
7. **L6 — HF re-release** (10 min)
   - New repo: `thc1006/cyberpuppy-v2.2-awq-lc` (or replace AWQ with re-quantized version after parity verified)

## Risk / mitigation

| Risk | Mitigation |
|---|---|
| llm-compressor's AWQ output format differs from autoawq → loading code change | Test load via `transformers.AutoModel.from_pretrained` first; if `device_map={"":"cuda"}` works without code change, no API change needed |
| Quality regression vs autoawq | Parity gate + held-out ToxiCloakCN eval before replacing public artefact |
| compressed-tensors runtime not installable in air-gapped sites | Pin SBOM; provide tarball install option |

## When to revisit this ADR

- llm-compressor 1.0 release (signals API stability)
- autoawq breaks on a transformers upgrade we need (e.g. for Qwen3.5)
- a customer requires vLLM-native serving
- moving v2.3 to AWQ (good opportunity to switch tooling once)

## Decision: defer

Current AWQ artefact (autoawq) works, F1 drop only 0.22%, latency met
DoD. There is no production blocker. Migrate when v2.3 ships, since
re-quantizing weights is the natural breakpoint anyway.
