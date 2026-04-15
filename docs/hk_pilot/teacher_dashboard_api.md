# Teacher Dashboard — API Schema

> Backend API contract for the school-side dashboard that consumes
> CyberPuppy v2.2 predictions, manages teacher overrides, and feeds an
> active-learning queue.
>
> **Status**: design / contract draft (no backend implementation in CyberPuppy core).
> **Last updated**: 2026-04-16

## Design principles

1. **Privacy first**: all texts identified by SHA-256 hash; raw text only
   exists in memory of the inference step and the teacher-review UI.
2. **Human-in-the-loop**: high-confidence "unsafe" predictions enter a
   review queue, never trigger automatic action.
3. **Dual-layer surfacing**: both Layer 1 (Qwen3Guard sentinel) and
   Layer 2 (CyberPuppy multi-head) verdicts shown side-by-side so
   teachers can see model disagreement.
4. **Family consent gate**: any data point flagged for active-learning
   collection must reference an active consent record.
5. **Audit log immutable**: all teacher actions append to a write-once log.

---

## 1. Inference (called by the LMS / messaging layer, not by teacher)

### `POST /v2/analyze`

Already implemented in `api/v2_2_app.py`. School integrates this directly.

**Request**:
```json
{
  "text": "你這個笨蛋，滾開！",
  "context": "optional prior turn for session-level analysis",
  "thread_id": "session-abc-123"
}
```

**Response**:
```json
{
  "toxicity":   { "label": "toxic",       "scores": { "none": 0.02, "toxic": 0.94, "severe": 0.04 } },
  "bullying":   { "label": "harassment",  "scores": { "none": 0.05, "harassment": 0.91, "threat": 0.04 } },
  "role":       { "label": "perpetrator", "scores": { "none": 0.10, "perpetrator": 0.85, "victim": 0.03, "bystander": 0.02 } },
  "emotion":    { "label": "neg",         "scores": { "pos": 0.01, "neu": 0.05, "neg": 0.94 } },
  "text_hash":  "f33b6d9dbf276189",
  "model_version": "v2.2",
  "latency_ms": 18.3
}
```

---

## 2. Review queue (consumed by teacher dashboard)

### `GET /v2/dashboard/queue`

Returns pending review items, ordered by severity then recency.

**Query params**:
| Param | Default | Description |
|---|---|---|
| `school_id` | (from API key) | Restrict to caller's school |
| `severity_min` | `toxic` | one of `toxic`, `severe` |
| `limit` | 50 | max items |
| `since` | (none) | ISO8601 timestamp |

**Response** (200):
```json
{
  "items": [
    {
      "queue_id": "q-2026-04-16-0001",
      "text_hash": "f33b6d9dbf276189",
      "redacted_preview": "你這個[REDACTED]，滾開！",
      "channel": "messenger | classroom | private",
      "thread_id": "session-abc-123",
      "verdicts": {
        "layer1_qwen3guard": { "label": "controversial", "categories": ["Violent"] },
        "layer2_cyberpuppy": {
          "toxicity": "toxic",
          "bullying": "harassment",
          "role": "perpetrator",
          "emotion": "neg"
        },
        "agreement": "agree"
      },
      "captured_at": "2026-04-16T08:33:12+08:00",
      "consent_record_id": "consent-2026-fall-001",
      "explainability_url": "/v2/dashboard/explain/q-2026-04-16-0001"
    }
  ],
  "next_cursor": "q-2026-04-16-0050"
}
```

### `GET /v2/dashboard/explain/{queue_id}`

Returns SHAP / Integrated Gradients token-level importance for a queue item.

**Response**:
```json
{
  "queue_id": "q-2026-04-16-0001",
  "method": "integrated_gradients",
  "task": "toxicity",
  "tokens": [
    { "token": "你",   "importance":  0.12 },
    { "token": "這個", "importance":  0.08 },
    { "token": "笨蛋", "importance":  0.71 },
    { "token": "，",   "importance":  0.01 },
    { "token": "滾開", "importance":  0.58 }
  ],
  "confidence": 0.94
}
```

---

## 3. Teacher override (writes to audit log)

### `POST /v2/dashboard/queue/{queue_id}/decide`

**Request**:
```json
{
  "decision": "confirmed | dismissed | escalate-counselor | escalate-parent",
  "override_label": { "toxicity": "toxic", "bullying": "harassment" },
  "notes": "Discussed in homeroom. No further action needed.",
  "feedback_for_training": true,
  "reviewer_id": "teacher-042"
}
```

- `decision` is required.
- `override_label` optional — used when teacher disagrees with model.
- `feedback_for_training`: if `true`, the (text_hash + override_label)
  enters the active-learning pool (see §4). Requires active consent.
- `reviewer_id` extracted from auth token; included for audit redundancy.

**Response** (200):
```json
{ "queue_id": "q-2026-04-16-0001", "decision": "dismissed", "audit_log_id": "log-2026-04-16-9127" }
```

---

## 4. Active learning (re-training feedback loop)

### `GET /v2/admin/active-learning/queue`

Admin-only. Returns aggregate counts of teacher overrides eligible for
re-training. Raw text NOT included unless `--include-text` flag (which
requires both school admin + provider engineer co-sign).

```json
{
  "total_overrides": 1247,
  "by_disagreement_type": {
    "model_too_strict": 312,
    "model_too_lenient": 521,
    "task_label_disagreement": 414
  },
  "ready_for_training": 821,
  "consent_blocked": 426
}
```

### `POST /v2/admin/active-learning/export`

Generates a training increment for the next CyberPuppy version, gated by
consent and admin co-sign. Output is a JSONL file delivered via signed S3
URL (24h expiry).

---

## 5. Family consent

### `POST /v2/dashboard/consent`

Records a guardian's consent for using the student's data in the
active-learning loop. **Without an active consent record, override
feedback cannot enter training.**

```json
{
  "student_pseudonym_id": "student-2026-001",
  "guardian_signature_hash": "sha256:...",
  "scope": ["bullying-screening", "active-learning"],
  "expires_at": "2027-08-31",
  "language": "zh-HK"
}
```

### `DELETE /v2/dashboard/consent/{consent_id}`

Right-to-be-forgotten. Triggers cascade:
1. Marks consent record `revoked`
2. Removes student's hashes from active-learning pool within 24h
3. Future inference still works (predictions are not stored), but
   predictions about this student no longer enter the queue

---

## 6. Health and metrics

### `GET /v2/dashboard/health`

```json
{
  "ready": true,
  "model_version": "v2.2",
  "uptime_s": 86400,
  "queue_depth": 12,
  "p95_latency_ms_24h": 22.1,
  "vram_gb": 4.5
}
```

### `GET /v2/dashboard/metrics`

Prometheus-format scrape endpoint (text/plain).

---

## 7. Authentication

- API key per school (`Authorization: Bearer cp_<school_id>_<random>`)
- Teacher / counselor / admin role within a school: short-lived JWT
  signed by the school's identity provider, validated by middleware.
- Admin endpoints (`/v2/admin/*`) require school-admin role + audit logging.

## 8. Rate limits

- 100 req/sec per school (burst 200) — typical school comfortably under
- Override / consent endpoints: 10 req/sec per teacher

## 9. Error format

```json
{
  "error": "consent_required",
  "message": "feedback_for_training=true requires an active consent record for this thread",
  "request_id": "req-2026-04-16-77f3"
}
```

| Code | HTTP | Meaning |
|---|---|---|
| `model_not_ready` | 503 | Model still loading after restart |
| `consent_required` | 403 | Active-learning feedback without consent |
| `quota_exceeded` | 429 | Per-school rate limit hit |
| `text_too_long` | 400 | > 1000 chars |
| `unsupported_language` | 400 | Detected non-Chinese input |

## 10. SLA (default; tunable per school)

| Metric | Target |
|---|---|
| Availability | 99.5% / month |
| `/v2/analyze` p95 latency | < 200 ms |
| `/v2/dashboard/queue` p95 latency | < 500 ms |
| Incident response (P1) | < 1 hour |
| Severity-1 model bug fix | < 7 days |

---

*All endpoints return `application/json` unless noted. All timestamps
are ISO 8601 with timezone. No personal data flows to logs — only
hashes and IDs.*
