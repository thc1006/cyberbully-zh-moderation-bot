# On-premise Deployment Runbook (HK Schools)

> Step-by-step guide for deploying CyberPuppy v2.2 inside a school's own
> network. Designed for **PDPO §33 zero-cross-border** scenarios.
>
> **Audience**: school IT / system integrator
> **Last updated**: 2026-04-16

## 0. Why on-premise?

For HK schools handling minor data, the cleanest privacy posture is to
**never let raw text leave the school's network**. On-premise deployment:

- ✅ No PDPO §33 cross-border transfer concerns
- ✅ No third-party sub-processor disclosures needed
- ✅ Schools own all logs, hashes, model artefacts
- ✅ Air-gapped option for ultra-sensitive use
- ⚠️ Requires capable on-site GPU (one-time CapEx)
- ⚠️ School IT handles updates / patches

## 1. Hardware recommendations

### Minimum (single-school, ≤ 1,000 students)

| Component | Spec | Why |
|---|---|---|
| GPU | NVIDIA RTX 4090 / 5070 Ti / Ada A2000 (16-24 GB) | AWQ 4-bit fits in ~5 GB, headroom for OS / batch |
| CPU | 8-core / 16-thread | Concurrent FastAPI workers |
| RAM | 32 GB | Model load buffer + Linux page cache |
| Storage | 512 GB NVMe SSD | Model weights ~6 GB + logs (90-day retention) |
| Network | Internal LAN; outbound HTTPS optional for model updates | — |

### Recommended (multi-school cluster, ≤ 10,000 students)

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5090 (32 GB) or L40 (48 GB) |
| CPU | 16-core |
| RAM | 64 GB |
| Storage | 1 TB NVMe in RAID-1 |
| Network | 10 GbE for shared model storage |

### NOT supported

- CPU-only inference (10× slower; will miss SLA)
- Macs (no CUDA support)
- Non-NVIDIA GPUs (AWQ kernels are CUDA-only at this version)

## 2. Software prerequisites

| Software | Version | Note |
|---|---|---|
| Linux kernel | ≥ 5.15 | Ubuntu 22.04 LTS recommended |
| NVIDIA driver | ≥ 535 (550 for RTX 5090) | `nvidia-smi` should run |
| Docker | ≥ 24.0 | with NVIDIA Container Toolkit |
| Disk free | ≥ 50 GB | Image + model + logs + headroom |

```bash
# Verify GPU visible to Docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 3. Deployment options

### Option A — Docker Compose (simplest)

```bash
# 1. Clone code (read-only)
git clone https://github.com/thc1006/cyberbully-zh-moderation-bot
cd cyberbully-zh-moderation-bot

# 2. Pull pre-built image (or build locally — see Option B)
docker pull ghcr.io/thc1006/cyberpuppy-api:v2.2-awq    # or v2.2-bf16

# 3. Pull model weights (one-time, 5.7 GB for AWQ)
mkdir -p models
huggingface-cli login   # provide read-token; required for private repo access
huggingface-cli download thc1006/cyberpuppy-v2.2-awq \
    --local-dir models/cyberpuppy_v2_2_awq

# 4. Edit configs/docker/.env (copy from .env.example)
cp .env.example configs/docker/.env
# Set:  CP_MODEL_DIR=/models/cyberpuppy_v2_2_awq
#       PORT=8000

# 5. Start
docker compose --profile v2_2 up -d
```

### Option B — Build image locally (air-gapped sites)

```bash
# Pre-fetch model weights on internet-connected machine
huggingface-cli download thc1006/cyberpuppy-v2.2-awq \
    --local-dir ./models/cyberpuppy_v2_2_awq

# Build image with model baked in (heavier but offline-installable)
docker build -t cyberpuppy-api:v2.2-awq-baked \
  --build-arg INCLUDE_MODEL=true \
  -f api/Dockerfile.v2_2 .

# Save / transfer / load on air-gapped host
docker save cyberpuppy-api:v2.2-awq-baked | gzip > cyberpuppy-v2.2-awq.tar.gz
# Transfer via approved channel
gunzip -c cyberpuppy-v2.2-awq.tar.gz | docker load
```

### Option C — Kubernetes / Helm (enterprise)

```bash
helm repo add cyberpuppy https://thc1006.github.io/cyberbully-zh-moderation-bot/charts
helm install cyberpuppy cyberpuppy/v2_2 \
  --set image.tag=v2.2-awq \
  --set model.source=huggingface \
  --set model.repo=thc1006/cyberpuppy-v2.2-awq \
  --set replicas=1 \
  --set resources.limits."nvidia\.com/gpu"=1
```

(Helm chart is a roadmap item; current state ships docker-compose only.)

## 4. Smoke test

```bash
# Wait until /healthz returns 200 (typically 30-60s including model warmup)
until curl -fsS http://localhost:8000/healthz; do sleep 5; done

# 6 Traditional Chinese sentinel tests
for text in "今天天氣真好" "你這個笨蛋滾開" "我打死你"; do
  curl -s -X POST http://localhost:8000/v2/analyze \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"$text\"}" | jq '{tox: .toxicity.label, lat: .latency_ms}'
done
```

Expected: ~20 ms p50 latency on RTX 5090, ~50 ms on RTX 4090.

## 5. Monitoring

### Prometheus scrape

```yaml
scrape_configs:
  - job_name: cyberpuppy
    static_configs:
      - targets: ['cyberpuppy:8000']
    metrics_path: /v2/dashboard/metrics
```

### Key alerts

| Metric | Warn | Page |
|---|---|---|
| `cyberpuppy_p95_latency_ms` | > 100 | > 300 |
| `cyberpuppy_5xx_rate` | > 1% | > 5% |
| `cyberpuppy_vram_gb` | > 90% | > 95% |
| `cyberpuppy_queue_depth` | > 50 | > 200 |

## 6. Updates & versioning

- Provider publishes a new model version on a quarterly cadence
- Update procedure:
  1. `docker pull ghcr.io/thc1006/cyberpuppy-api:v2.3-awq`
  2. `huggingface-cli download thc1006/cyberpuppy-v2.3-awq --local-dir ./models/cyberpuppy_v2_3_awq`
  3. `docker compose --profile v2_3 up -d` (blue-green)
  4. Verify smoke test on new endpoint
  5. Cut over reverse proxy
  6. `docker compose --profile v2_2 down` after 24h soak

## 7. Backup & disaster recovery

| Asset | Backup method | RPO | RTO |
|---|---|---|---|
| Model weights | Local NVMe + offsite mirror | 90 days | 1 hour |
| Hash + score logs | Daily snapshot to school-managed storage | 24 hours | 4 hours |
| Override / consent records | Daily snapshot, encrypted at rest | 24 hours | 4 hours |

## 8. Security hardening checklist

- [ ] HTTPS only (TLS 1.3, HSTS preload)
- [ ] API key rotation every 90 days
- [ ] Container runs as non-root (`cyberpuppy` user, already in image)
- [ ] Network policy: only school's LMS / messaging system can hit `/v2/analyze`
- [ ] Admin endpoints (`/v2/admin/*`) restricted to school admin VLAN
- [ ] Logs egress only to school-controlled SIEM
- [ ] Quarterly dependency vulnerability scan (provider supplies SBOM)
- [ ] Annual penetration test (school's choice of vendor)

## 9. Cost estimate (single-school, monthly)

| Item | Cost (HKD) |
|---|---|
| GPU server amortized over 3 years (RTX 4090 build, ~80,000 HKD) | ~2,200 |
| Power (300W avg × 24 × 30 / 1000 × $1.5/kWh) | ~325 |
| Internet (existing school uplink) | included |
| Provider on-prem support (optional, T3 tier) | ~5,000 |
| Total | **~7,500 / month** (or 2,500 without provider support) |

Compare with EDB "AI for Empowering Learning and Teaching" grant of
HK$500,000 over 3 years per school: **easily covered**.

## 10. Support

- **Documentation**: this file + ADR 0001
- **Issues**: https://github.com/thc1006/cyberbully-zh-moderation-bot/issues
- **Email**: hctsai1006@cs.nctu.edu.tw
- **Response SLA**: per service agreement (typical: 1 business day for P3,
  4 hours for P2, 1 hour for P1)

---

*This runbook will be revised after the first PadLearn pilot. Suggestions
welcome via GitHub issues.*
