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

    # ---- Source: STATE-ToxiCN (ACL Findings 2025) ------------------------

    def normalize_state_toxicn_row(self, row: dict[str, Any]) -> UnifiedRecord | None:
        """STATE-ToxiCN sentence-level + span-level → unified.

        Maps:
        - sen_hate "1" → toxicity=toxic, bullying=harassment, emotion=neg
        - 3+ hate quadruples → toxicity=severe, bullying=threat
        - sen_hate "0" → all none / neu
        """
        raw_text = row.get("content", "") or ""
        text = self.opencc_convert(raw_text).strip()
        text = self.scrub_pii(text)
        if not self.is_valid_length(text):
            return None

        sen_hate = str(row.get("sen_hate", "0")).strip() == "1"
        # count hate quadruples (Q1..Q5 hateful == "hate")
        hate_groups: list[str] = []
        for i in range(1, 6):
            ht = str(row.get(f"Q{i} hateful", "")).strip().lower()
            grp = str(row.get(f"Q{i} Group", "")).strip()
            if ht == "hate" and grp and grp.lower() != "non-hate":
                hate_groups.append(grp)

        if sen_hate and len(hate_groups) >= 3:
            toxicity, bullying = "severe", "threat"
        elif sen_hate:
            toxicity, bullying = "toxic", "harassment"
        else:
            toxicity, bullying = "none", "none"
        emotion = "neg" if sen_hate else "neu"

        return UnifiedRecord(
            text=text,
            label={
                "toxicity": toxicity,
                "bullying": bullying,
                "role": "perpetrator" if sen_hate else "none",
                "emotion": emotion,
                "emotion_strength": min(len(hate_groups), 4) if sen_hate else 0,
            },
            metadata={
                "source": "state_toxicn",
                "original_id": row.get("id"),
                "platform": row.get("platform", ""),
                "topic": row.get("topic", ""),
                "hate_groups": ",".join(hate_groups),
                "n_hate_quadruples": len(hate_groups),
                "text_length": len(text),
                "is_traditional": self.target == "traditional",
                "annotation_quality": "gold",
            },
        )

    def process_state_toxicn_records(
        self, rows: list[dict[str, Any]], dedup: bool = False
    ) -> list[UnifiedRecord]:
        records: list[UnifiedRecord] = []
        seen: set[str] = set()
        for row in rows:
            rec = self.normalize_state_toxicn_row(row)
            if rec is None:
                continue
            if dedup:
                h = self.text_hash(rec.text)
                if h in seen:
                    continue
                seen.add(h)
            records.append(rec)
        return records

    # ---- Source: SCCD (Yang et al., COLING 2025) -------------------------

    def normalize_sccd_comment(
        self,
        comment_row: dict[str, Any],
        post_severity: str = "low",
    ) -> UnifiedRecord | None:
        """SCCD comment-level → unified.

        Maps:
        - label "Non-CB" → toxicity=none
        - label starts with "CB" → toxicity by post_severity (low/med→toxic, high→severe)
        - label "CB-Threat" → bullying=threat (else harassment)
        """
        raw_text = str(comment_row.get("comment_content", "") or "")
        text = self.opencc_convert(raw_text).strip()
        text = self.scrub_pii(text)
        if not self.is_valid_length(text):
            return None

        label = str(comment_row.get("label", "Non-CB")).strip()
        is_cb = label.lower().startswith("cb")
        if not is_cb:
            toxicity, bullying, emotion = "none", "none", "neu"
        else:
            sev = post_severity.lower()
            toxicity = "severe" if sev == "high" else "toxic"
            bullying = "threat" if "threat" in label.lower() else "harassment"
            emotion = "neg"

        return UnifiedRecord(
            text=text,
            label={
                "toxicity": toxicity,
                "bullying": bullying,
                "role": "perpetrator" if is_cb else "none",
                "emotion": emotion,
                "emotion_strength": {"low": 2, "medium": 3, "high": 4}.get(post_severity.lower(), 0) if is_cb else 0,
            },
            metadata={
                "source": "sccd",
                "comment_id": comment_row.get("comment_id"),
                "post_id": comment_row.get("post_id"),
                "to_id": comment_row.get("to_id", ""),
                "raw_label": label,
                "post_severity": post_severity,
                "text_length": len(text),
                "is_traditional": self.target == "traditional",
                "annotation_quality": "gold",
            },
        )

    def process_sccd_comments(
        self,
        comments: list[dict[str, Any]],
        post_severity_map: dict[str, str],
        dedup: bool = False,
    ) -> list[UnifiedRecord]:
        records: list[UnifiedRecord] = []
        seen: set[str] = set()
        for c in comments:
            sev = post_severity_map.get(str(c.get("post_id", "")), "low")
            rec = self.normalize_sccd_comment(c, post_severity=sev)
            if rec is None:
                continue
            if dedup:
                h = self.text_hash(rec.text)
                if h in seen:
                    continue
                seen.add(h)
            records.append(rec)
        return records

    # ---- Source: CHNCI (Zhu/Zou/Wu, May 2025) ----------------------------

    def normalize_chnci_row(
        self,
        row: dict[str, Any],
        incident_label: str = "non-cyberbullying",
        incident_name: str = "",
    ) -> UnifiedRecord | None:
        """CHNCI per-comment → unified.

        Maps (3-annotator majority vote on label1/2/3):
        - majority 1 → toxicity=toxic, bullying=harassment, emotion=neg
        - majority 0 → toxicity=none
        - diff > 0 (annotators disagreed) → annotation_quality=silver
        """
        raw_text = str(row.get("content", "") or "")
        text = self.opencc_convert(raw_text).strip()
        text = self.scrub_pii(text)
        if not self.is_valid_length(text):
            return None

        labels: list[int] = []
        for k in ("label1", "label2", "label3"):
            v = row.get(k)
            if v is None or v == "":
                continue
            try:
                labels.append(int(float(v)))
            except (ValueError, TypeError):
                continue
        if labels:
            majority = 1 if sum(1 for x in labels if x >= 1) > len(labels) / 2 else 0
        else:
            # fall back to incident label
            majority = 1 if incident_label.lower().startswith("cyber") else 0

        diff = float(row.get("diff", 0) or 0)
        quality = "silver" if diff > 0 else "gold"

        if majority == 1:
            toxicity, bullying, emotion = "toxic", "harassment", "neg"
        else:
            toxicity, bullying, emotion = "none", "none", "neu"

        return UnifiedRecord(
            text=text,
            label={
                "toxicity": toxicity,
                "bullying": bullying,
                "role": "perpetrator" if majority == 1 else "none",
                "emotion": emotion,
                "emotion_strength": 3 if majority == 1 else 0,
            },
            metadata={
                "source": "chnci",
                "incident_name": incident_name,
                "incident_label": incident_label,
                "platform": row.get("platform", ""),
                "annotator_labels": labels,
                "annotator_diff": diff,
                "text_length": len(text),
                "is_traditional": self.target == "traditional",
                "annotation_quality": quality,
            },
        )

    def process_chnci_dir(
        self,
        root: Any,
        dedup: bool = False,
    ) -> list[UnifiedRecord]:
        """Walk a CHNCI dataset/{cyberbullying,non-cyberbullying}/*.csv tree."""
        from pathlib import Path
        root = Path(root)
        records: list[UnifiedRecord] = []
        seen: set[str] = set()
        for incident_label in ("cyberbullying", "non-cyberbullying"):
            sub = root / incident_label
            if not sub.exists():
                continue
            for csv_path in sub.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_path, encoding="utf-8", errors="replace")
                except Exception:
                    continue
                for row in df.to_dict(orient="records"):
                    rec = self.normalize_chnci_row(
                        row, incident_label=incident_label, incident_name=csv_path.stem,
                    )
                    if rec is None:
                        continue
                    if dedup:
                        h = self.text_hash(rec.text)
                        if h in seen:
                            continue
                        seen.add(h)
                    records.append(rec)
        return records
