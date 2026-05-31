# bullying_a100_best (legacy MacBERT bullying model, F1≈0.826)

> **The large weight files were removed from git history on 2026-06-01.**
> They bloated the repo via Git LFS (`best_model.pt` + `pytorch_model.bin`, ~409 MB each =
> 818 MB, i.e. ~93% of every clone's LFS download). The current production model is the
> **v6.0 Qwen3-8B dual-LoRA** hosted on HuggingFace, not this one — see the project `CLAUDE.md`.

## What is still tracked here

Only the lightweight metadata needed to understand / reconstruct the model:

- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`
- `convert_model.py` — converts `best_model.pt` (training checkpoint) → `pytorch_model.bin` (HF format)
- `*_results.json`, `deployment_info.json` — eval / deployment records

## How to get the weights

These two files are intentionally **git-ignored** now:

- `best_model.pt` — full training checkpoint
- `pytorch_model.bin` — HuggingFace inference weights (produced by `convert_model.py`)

Obtain them from one of:

1. **HuggingFace** (recommended) — upload to / download from a model repo, e.g.
   `huggingface-cli download <org>/bullying-a100-best pytorch_model.bin --local-dir .`
2. **GitHub Release asset** — attach the file to a tagged release and `wget` the asset URL.
3. **Local backup** — a copy was made during the cleanup at
   `Desktop/MAY/_cyberpuppy_legacy_model_backup/bullying_a100_best/`.

Place the downloaded file back into this directory; it will be ignored by git automatically.
