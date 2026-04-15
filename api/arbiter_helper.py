"""Optional Perspective API arbiter for v2.2 API.

Designed to be safe-by-default: returns None unless explicitly enabled,
never blocks the main response on Perspective errors.

ADR 0001 §3.3 — Perspective is "輔助，不直接決策"; this helper preserves
that intent. The local model's verdict is always authoritative; this
function only adds a `perspective_score` field to the response when the
local model is uncertain.

Env vars:
- PERSPECTIVE_API_KEY            — Google API key (off if unset)
- CP_PERSPECTIVE_UNCERTAIN_BELOW — confidence threshold (default 0.7)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Allow relative import from src/ when this module is loaded via uvicorn.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberpuppy.arbiter.perspective import PerspectiveAPI  # noqa: E402

log = logging.getLogger("cyberpuppy.arbiter")


async def maybe_perspective_score(
    text: str,
    local_confidence: float,
) -> Optional[dict]:
    """Return Perspective scores iff arbiter is enabled AND local is uncertain.

    Args:
        text: Original input text (≤ 3000 chars; Perspective's hard limit).
        local_confidence: Max softmax over toxicity head from the local model.

    Returns:
        dict of Perspective category scores, or None if disabled / confident /
        on any error (best-effort augmentation).
    """
    api_key = os.environ.get("PERSPECTIVE_API_KEY", "").strip()
    if not api_key:
        return None

    threshold = float(os.environ.get("CP_PERSPECTIVE_UNCERTAIN_BELOW", "0.7"))
    if local_confidence >= threshold:
        return None  # local model confident enough; save the API call

    try:
        async with PerspectiveAPI(api_key=api_key) as api:
            result = await api.analyze_comment(text=text)
    except Exception as exc:  # network / quota / parse / anything
        log.warning("perspective_arbiter_error", extra={"err": str(exc)[:100]})
        return None

    return {
        "toxicity": float(result.toxicity_score),
        "severe_toxicity": float(result.severe_toxicity_score),
        "threat": float(result.threat_score),
        "insult": float(result.insult_score),
        "identity_attack": float(result.identity_attack_score),
        "profanity": float(result.profanity_score),
    }
