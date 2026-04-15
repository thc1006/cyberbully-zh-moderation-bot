"""CyberPuppy v2.2 FastAPI server (ADR 0001 Phase 5).

New `/v2/analyze` endpoint backed by merged Qwen3-8B + 4 multi-task heads.
Kept separate from api/app.py (v1 MacBERT loader) to avoid regression risk.

Design:
- Single worker holds model in VRAM (~14 GB bf16 on RTX 5090)
- `torch.inference_mode()` + `asyncio.Lock` for GIL-safe inference
- `/healthz` returns 503 until model ready (ADR §3.4)
- No PII logging — hash text with SHA-256 for correlation only

Run:
  uvicorn api.v2_2_app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

# Allow 'from cyberpuppy...' when run via uvicorn (cwd = repo root).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberpuppy.data.phase2 import LABELS  # noqa: E402
from cyberpuppy.models.qwen3_multihead import HEAD_DIMS, Qwen3MultiHead  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("cyberpuppy.v2_2")

MODEL_DIR = os.environ.get("CP_MODEL_DIR", "models/cyberpuppy_v2_2_merged")
MAX_TEXT_LEN = 1000
MAX_CONTEXT_LEN = 2000
MAX_TOKEN_LEN = 192  # matches training max_length

ID2LABEL = {task: list(vals) for task, vals in LABELS.items()}

_state: Dict[str, Any] = {
    "tokenizer": None,
    "model": None,
    "device": None,
    "ready": False,
    "startup_time": None,
    "lock": None,
    "metrics": {"requests": 0, "errors": 0, "total_latency_ms": 0.0},
}


# ---- Pydantic schemas ---------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    context: Optional[str] = Field(None, max_length=MAX_CONTEXT_LEN)
    thread_id: Optional[str] = Field(None, max_length=50)


class HeadResult(BaseModel):
    label: str
    scores: Dict[str, float]


class AnalyzeResponse(BaseModel):
    toxicity: HeadResult
    bullying: HeadResult
    role: HeadResult
    emotion: HeadResult
    text_hash: str
    model_version: str = "v2.2"
    latency_ms: float


# ---- Lifespan: load model, mark ready -----------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    log.info(f"Loading v2.2 merged model from {MODEL_DIR} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    backbone = AutoModel.from_pretrained(
        MODEL_DIR, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
    )
    model = Qwen3MultiHead(backbone, hidden_size=backbone.config.hidden_size).to(
        device=device, dtype=dtype
    )
    heads_state = torch.load(f"{MODEL_DIR}/heads.pt", map_location=device, weights_only=False)
    model.heads.load_state_dict(heads_state["heads"])
    model.eval()

    # Warmup: 3 forward passes to compile kernels
    with torch.inference_mode():
        for txt in ("你好", "這個世界很複雜", "你這個笨蛋滾開"):
            enc = tokenizer(txt, return_tensors="pt", padding=True, truncation=True,
                              max_length=MAX_TOKEN_LEN).to(device)
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    torch.cuda.synchronize() if device.type == "cuda" else None

    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["device"] = device
    _state["lock"] = asyncio.Lock()
    _state["ready"] = True
    _state["startup_time"] = time.time() - t0
    log.info(f"Model ready. Startup: {_state['startup_time']:.1f}s; VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    yield

    log.info("Shutting down.")
    _state["ready"] = False


app = FastAPI(
    title="CyberPuppy v2.2 API",
    description="Chinese cyberbullying detection — Qwen3-8B + LoRA multi-task (ADR 0001 Phase 5)",
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["GET", "POST"], allow_headers=["*"],
)


# ---- Core inference -----------------------------------------------------

def _predict_one(text: str) -> Dict[str, HeadResult]:
    tokenizer = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                     max_length=MAX_TOKEN_LEN).to(device)
    with torch.inference_mode():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    result = {}
    for task in HEAD_DIMS:
        logits = out.logits[task][0].float()
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        idx = int(logits.argmax().item())
        vocab = ID2LABEL[task]
        result[task] = HeadResult(
            label=vocab[idx],
            scores={vocab[i]: round(float(probs[i]), 4) for i in range(len(vocab))},
        )
    return result


# ---- Endpoints -----------------------------------------------------------

@app.get("/healthz")
async def health():
    if not _state["ready"]:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                              detail="Model not ready")
    return {
        "ready": True,
        "model_version": "v2.2",
        "startup_time_s": _state["startup_time"],
        "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2) if _state["device"].type == "cuda" else None,
        "requests_served": _state["metrics"]["requests"],
    }


@app.post("/v2/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty after stripping")

    t0 = time.perf_counter()
    async with _state["lock"]:
        # Single-GPU serial — no batching yet (MVP)
        try:
            heads = _predict_one(text)
        except Exception as e:
            _state["metrics"]["errors"] += 1
            log.exception("inference_error")
            raise HTTPException(status_code=500, detail=f"inference_error: {e.__class__.__name__}")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _state["metrics"]["requests"] += 1
    _state["metrics"]["total_latency_ms"] += elapsed_ms

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    # Intentionally NO text logging — PDPO §64 / ADR §3.5 privacy-first.
    log.info(f"analyze hash={text_hash} tox={heads['toxicity'].label} "
             f"bull={heads['bullying'].label} latency_ms={elapsed_ms:.1f}")

    return AnalyzeResponse(
        toxicity=heads["toxicity"],
        bullying=heads["bullying"],
        role=heads["role"],
        emotion=heads["emotion"],
        text_hash=text_hash,
        latency_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    uvicorn.run("api.v2_2_app:app", host="0.0.0.0", port=8000, reload=False)
