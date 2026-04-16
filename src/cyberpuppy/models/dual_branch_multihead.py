"""Dual-branch multi-task classifier: Qwen3 text + DBIT-style pinyin CNN.

Based on 5-source verified SOTA approach:
  - Semantic-Phonetic Collaboration (Expert Systems 2025)
  - MMBERT (AAAI 2026)
  - CCTAgent (Expert Systems 2025)
  - CHOLD (Springer 2025)
  - ChineseBERT (ACL 2021)

Architecture:
  Branch 1: Qwen3-8B + LoRA → text_pooled (B, 4096)
  Branch 2: PinyinBranch (~38K params) → pinyin_pooled (B, 256)
  Fusion: text_pooled + proj(pinyin_pooled) → fused (B, 4096)
  Heads: 4 × Linear (toxicity, bullying, role, emotion)
  Losses: task_loss + λ_cons × consistency + λ_cont × contrastive(text, pinyin)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .pinyin_cnn_encoder import PinyinBranch, contrastive_loss
from .qwen3_multihead import MultiTaskOutput, Qwen3MultiHead


class DualBranchMultiHead(Qwen3MultiHead):
    """Qwen3MultiHead + DBIT-style pinyin branch with contrastive alignment.

    Inherits: compute_loss, heads, log_var, focal_gamma, _pool, dropout.
    Overrides: forward (adds pinyin branch + additive fusion).
    Adds: contrastive_loss method, pinyin_branch, pinyin_proj.
    """

    def __init__(
        self,
        backbone: nn.Module,
        pinyin_branch: PinyinBranch,
        hidden_size: Optional[int] = None,
        contrastive_temp: float = 0.07,
        **kwargs: Any,
    ) -> None:
        super().__init__(backbone, hidden_size=hidden_size, **kwargs)
        self.pinyin_branch = pinyin_branch
        self.contrastive_temp = contrastive_temp

        # Project pinyin hidden_dim → text hidden_size for additive fusion.
        # Small random init (std=0.02): starts as near-zero residual.
        self.pinyin_proj = nn.Linear(pinyin_branch.hidden_dim, self.hidden_size, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.pinyin_proj.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pinyin_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> MultiTaskOutput:
        """Forward with dual-branch fusion.

        Args:
            input_ids: (B, L) token IDs for Qwen3
            attention_mask: (B, L)
            pinyin_ids: (B, L_py) char-level pinyin IDs from PinyinCharTokenizer.
                        If None, falls back to text-only (parent behavior).
        """
        # Branch 1: Qwen3 text encoder (existing path)
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs,
        )
        text_pooled = self._pool(out.last_hidden_state, attention_mask)
        text_pooled = self.dropout(text_pooled)  # (B, 4096)

        # Branch 2: Pinyin CNN encoder (new)
        if pinyin_ids is not None:
            pinyin_pooled = self.pinyin_branch(pinyin_ids)  # (B, 256)
            # Match dtype (bf16 during training)
            pinyin_pooled = pinyin_pooled.to(dtype=text_pooled.dtype)
            pinyin_projected = self.pinyin_proj(pinyin_pooled)  # (B, 4096)

            # Additive fusion (LoRA-like residual)
            fused = text_pooled + pinyin_projected
        else:
            fused = text_pooled
            pinyin_pooled = None

        # 4 classification heads on fused representation
        logits = {name: head(fused) for name, head in self.heads.items()}

        return MultiTaskOutput(
            logits=logits,
            pooled=fused,
            extra={
                "text_pooled": text_pooled,
                "pinyin_pooled": pinyin_pooled,
            },
        )

    def compute_contrastive_loss(
        self,
        outputs: MultiTaskOutput,
    ) -> torch.Tensor:
        """Bidirectional InfoNCE between text and pinyin representations.

        Aligns same-sample text↔pinyin pairs. Homophones share pinyin →
        their text representations get pulled closer in embedding space.
        """
        text_pooled = outputs.extra.get("text_pooled")
        pinyin_pooled = outputs.extra.get("pinyin_pooled")

        if text_pooled is None or pinyin_pooled is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Project pinyin to text space for contrastive comparison
        pinyin_projected = self.pinyin_proj(pinyin_pooled)

        return contrastive_loss(
            text_pooled.float(),
            pinyin_projected.float(),
            temperature=self.contrastive_temp,
        )
