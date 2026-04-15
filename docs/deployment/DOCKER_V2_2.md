# Docker — v2.2 Quickstart

> v2.2 (Qwen3-8B + LoRA + AWQ) deployment via docker-compose.
> See `on_prem_deploy.md` (HK pilot) for the school-side install runbook.
> Last updated: 2026-04-16

## Prerequisites

- Docker ≥ 24.0
- NVIDIA Container Toolkit (`nvidia-smi` works inside Docker)
- ≥ 8 GB free VRAM (AWQ uses 4.5 GB, plus headroom)
- ≥ 10 GB free disk

## Modes

| Profile | Command | Image | Port | Use case |
|---|---|---|---|---|
| `v1` | `docker compose --profile v1 up -d` | api/Dockerfile (MacBERT) | 8000 | Legacy / regression |
| `v2_2` | `docker compose --profile v2_2 up -d` | api/Dockerfile.v2_2 (Qwen3 + AWQ) | 8002 | **Production** |
| `all` | both | both | 8000 + 8002 | Side-by-side comparison |

## Quick start (v2.2 with HF download)

```bash
# 1. Get a HF read-token from https://huggingface.co/settings/tokens
export HF_TOKEN=hf_...your_read_token...

# 2. Configure
cat > configs/docker/.env <<EOF
CP_HF_REPO=thc1006/cyberpuppy-v2.2-awq
HF_TOKEN=${HF_TOKEN}
PERSPECTIVE_API_KEY=          # optional; leave empty to disable arbiter
EOF

# 3. Boot
docker compose --profile v2_2 up -d

# 4. Wait until healthy (first run downloads ~5.7 GB to ./.cache/huggingface)
until curl -fsS http://localhost:8002/healthz; do sleep 5; done

# 5. Smoke test
curl -X POST http://localhost:8002/v2/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"你這個笨蛋，滾開！"}'
```

## Quick start (v2.2 with locally pre-downloaded weights)

```bash
# Pre-download weights (once)
huggingface-cli download thc1006/cyberpuppy-v2.2-awq \
  --local-dir models/cyberpuppy_v2_2_awq

# Configure to use local path (no HF auth needed at runtime)
cat > configs/docker/.env <<EOF
CP_MODEL_DIR=/models/cyberpuppy_v2_2_awq
CP_HF_REPO=
EOF

docker compose --profile v2_2 up -d
```

## Configuration reference

All env vars are optional; sensible defaults shipped.

| Var | Default | Purpose |
|---|---|---|
| `CP_MODEL_DIR` | `/models/cyberpuppy_v2_2_awq` | Path inside container to model dir |
| `CP_HF_REPO` | (empty) | If set, download from HF instead of local mount |
| `HF_TOKEN` | (empty) | Required only when `CP_HF_REPO` is private |
| `HF_CACHE_DIR` | `./.cache/huggingface` | Host path to mount as HF cache (persists) |
| `PERSPECTIVE_API_KEY` | (empty) | If set, enables the optional second-opinion arbiter |
| `CP_PERSPECTIVE_UNCERTAIN_BELOW` | `0.7` | Local confidence threshold below which Perspective is consulted |
| `PORT` | `8000` | Inside container |

## GPU verification inside container

```bash
docker compose --profile v2_2 exec api_v2_2 \
  python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Logs

```bash
docker compose --profile v2_2 logs -f api_v2_2
```

PDPO §64 alignment: container only logs SHA-256 hashes + scores, never raw text.

## Updating the model

```bash
# Stop
docker compose --profile v2_2 down

# Pull latest version (e.g. v2.3)
huggingface-cli download thc1006/cyberpuppy-v2.3-awq \
  --local-dir models/cyberpuppy_v2_3_awq

# Edit configs/docker/.env
sed -i 's|cyberpuppy_v2_2_awq|cyberpuppy_v2_3_awq|' configs/docker/.env

# Restart
docker compose --profile v2_2 up -d
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/healthz` returns 503 for >3 min | Model still loading | Tail logs; first run downloads ~5.7 GB |
| `CUDA out of memory` | Another process holding GPU | `nvidia-smi` to find culprit |
| `403 Forbidden` on HF download | HF token missing or wrong scope | Check `HF_TOKEN` is read-or-write |
| `quantization_config not found` | Pointed at adapter dir, not AWQ dir | Set `CP_MODEL_DIR` to AWQ artefact, not LoRA |
| `'Catcher' has no attribute 'attention_type'` | Trying to RE-quantize, not load | Use `.venv-quant` for quantization; the served container only loads |

See `docs/hk_pilot/on_prem_deploy.md` for the school-side guide and SLA details.
