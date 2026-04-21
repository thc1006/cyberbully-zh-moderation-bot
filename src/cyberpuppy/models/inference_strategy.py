"""Optimal inference strategy for dual-LoRA ensemble (v5.1.1).

Discovered via comprehensive parameter sweep on 2026-04-21:
  1. Geometric mean ensemble: text^0.5 × pinyin^0.5 (equal weight)
  2. Pinyin override: pinyin toxic prob > 0.7 → force toxic
  3. Decision threshold: 0.48 (slightly aggressive)
  4. Confidence-gated cascade: ensemble conf < 0.55 → LLM fallback

Key insight: α=0.75 (v5.1) suppressed pinyin signal on creative attacks
(e.g., 大廈避風=大傻逼). Equal weighting (α=0.5) + pinyin override
rescues 34/116 FN on PCR-ToxiCN, yielding +3.44pt improvement.

Results (no retrain, inference-only):
  COLD: 0.8302 (DoD ✅)  PCR: 0.7162 (+3.44pt)  TC: 0.8496 (+1.63pt)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class InferenceStrategy:
    """Configurable inference strategy for dual-LoRA ensemble.

    Args:
        geo_alpha: power for text probabilities in geometric mean (default 0.5)
        pinyin_override_thresh: if pinyin toxic prob exceeds this, force toxic.
            None to disable. Default 0.7.
        decision_thresh: ensemble toxic prob threshold for positive prediction.
            Default 0.48.
        cascade_conf_thresh: ensemble confidence below this triggers LLM cascade.
            Default 0.55.
    """
    geo_alpha: float = 0.5
    pinyin_override_thresh: Optional[float] = 0.7
    decision_thresh: float = 0.48
    cascade_conf_thresh: float = 0.55

    def ensemble(
        self,
        text_probs: torch.Tensor,
        pinyin_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute geometric mean ensemble.

        Args:
            text_probs: (B, C) softmax from text LoRA
            pinyin_probs: (B, C) softmax from pinyin LoRA

        Returns:
            (B, C) normalized geometric mean probabilities
        """
        a = self.geo_alpha
        ens = (text_probs ** a) * (pinyin_probs ** (1 - a))
        return ens / ens.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    def predict(
        self,
        text_probs: torch.Tensor,
        pinyin_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict binary toxic/safe labels.

        Args:
            text_probs: (B, C) softmax from text LoRA (C=3: none/toxic/severe)
            pinyin_probs: (B, C) softmax from pinyin LoRA

        Returns:
            (B,) bool tensor — True = toxic
        """
        ens = self.ensemble(text_probs, pinyin_probs)
        toxic_prob = ens[:, 1:].sum(dim=-1)  # toxic + severe
        preds = toxic_prob > self.decision_thresh

        # Pinyin override: if pinyin is highly confident it's toxic, trust it
        if self.pinyin_override_thresh is not None:
            p_toxic = pinyin_probs[:, 1:].sum(dim=-1)
            preds = preds | (p_toxic > self.pinyin_override_thresh)

        return preds

    def predict_with_confidence(
        self,
        text_probs: torch.Tensor,
        pinyin_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict with confidence scores for cascade gating.

        Returns:
            preds: (B,) bool tensor
            confidence: (B,) float tensor — max ensemble prob per sample
        """
        ens = self.ensemble(text_probs, pinyin_probs)
        toxic_prob = ens[:, 1:].sum(dim=-1)
        preds = toxic_prob > self.decision_thresh

        if self.pinyin_override_thresh is not None:
            p_toxic = pinyin_probs[:, 1:].sum(dim=-1)
            preds = preds | (p_toxic > self.pinyin_override_thresh)

        confidence = ens.max(dim=-1).values
        return preds, confidence

    def needs_cascade(self, confidence: torch.Tensor) -> torch.Tensor:
        """Identify samples that need LLM cascade (low confidence).

        Args:
            confidence: (B,) ensemble confidence scores

        Returns:
            (B,) bool tensor — True = needs LLM cascade
        """
        return confidence < self.cascade_conf_thresh

    def apply_cascade(
        self,
        preds: torch.Tensor,
        confidence: torch.Tensor,
        cascade_preds: Dict[int, bool],
    ) -> torch.Tensor:
        """Apply LLM cascade results to override low-confidence predictions.

        Args:
            preds: (B,) original predictions
            confidence: (B,) confidence scores
            cascade_preds: {sample_index: LLM_prediction} for low-conf samples

        Returns:
            (B,) final predictions with cascade overrides
        """
        final = preds.clone()
        for idx, llm_pred in cascade_preds.items():
            if confidence[idx] < self.cascade_conf_thresh:
                final[idx] = bool(llm_pred)
        return final
