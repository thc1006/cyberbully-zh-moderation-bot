"""Phase 2 dataset normalization (ADR 0001 §3.2).

Source-specific normalizers + unified schema for COLD / SCCD / CHNCI / STATE-ToxiCN.
Currently implements COLD; other sources land in subsequent commits.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from opencc import OpenCC

logger = logging.getLogger(__name__)

LABELS: dict[str, tuple[str, ...]] = {
    "toxicity": ("none", "toxic", "severe"),
    "bullying": ("none", "harassment", "threat"),
    "role": ("none", "perpetrator", "victim", "bystander"),
    "emotion": ("pos", "neu", "neg"),
}

# Reused patterns; identical to api/app.py PII_PATTERNS but precompiled.
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[A-Z][0-9]{9}\b"), "[ID]"),
    (re.compile(r"\b\w+@\w+\.\w+\b"), "[EMAIL]"),
    (re.compile(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"), "[CARD]"),
    (re.compile(r"\b\d{10,11}\b"), "[PHONE]"),
]


@dataclass
class UnifiedRecord:
    text: str
    label: dict[str, Any]
    metadata: dict[str, Any]
    context: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Phase2Normalizer:
    """Pipeline: opencc → pii scrub → length filter → label map → dedup."""

    def __init__(
        self,
        target: str = "traditional",
        min_chars: int = 3,
        max_chars: int = 512,
    ) -> None:
        self.target = target
        self.min_chars = min_chars
        self.max_chars = max_chars
        # s2twp: Simplified -> Traditional Taiwan with phrase conversion
        config = "s2twp" if target == "traditional" else "tw2sp"
        self._cc = OpenCC(config)

    # ---- Atomic ops -------------------------------------------------------

    def opencc_convert(self, text: str) -> str:
        return self._cc.convert(text)

    def scrub_pii(self, text: str) -> str:
        out = text
        for pat, repl in _PII_PATTERNS:
            out = pat.sub(repl, out)
        return out

    def is_valid_length(self, text: str) -> bool:
        return self.min_chars <= len(text) <= self.max_chars

    def text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # ---- Source: COLD -----------------------------------------------------

    def normalize_cold_row(self, row: dict[str, Any]) -> UnifiedRecord | None:
        raw_text = row.get("TEXT", "") or ""
        raw_label = int(row.get("label", 0))
        raw_fine = int(row.get("fine-grained-label", 0))

        text = self.opencc_convert(raw_text).strip()
        text = self.scrub_pii(text)
        if not self.is_valid_length(text):
            return None

        toxicity = "toxic" if raw_label == 1 else "none"
        bullying = "harassment" if raw_label == 1 else "none"

        return UnifiedRecord(
            text=text,
            label={
                "toxicity": toxicity,
                "bullying": bullying,
                "role": "none",
                "emotion": "neu",
                "emotion_strength": 0,
            },
            metadata={
                "source": "cold",
                "original_label_raw": f"{raw_label}/{raw_fine}",
                "text_length": len(text),
                "is_traditional": self.target == "traditional",
                "annotation_quality": "gold",
            },
        )

    def process_cold_dataframe(
        self, df: pd.DataFrame, dedup: bool = False
    ) -> list[UnifiedRecord]:
        records: list[UnifiedRecord] = []
        seen: set[str] = set()
        for row in df.to_dict(orient="records"):
            rec = self.normalize_cold_row(row)
            if rec is None:
                continue
            if dedup:
                h = self.text_hash(rec.text)
                if h in seen:
                    continue
                seen.add(h)
            records.append(rec)
        return records
