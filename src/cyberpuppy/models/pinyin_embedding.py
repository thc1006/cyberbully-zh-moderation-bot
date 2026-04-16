"""Pinyin auxiliary embedding for homophone-robust Chinese classification.

Injects phonetic (pinyin) features into Qwen3 token embeddings so that
homophones like 智商/之上 (both "zhishang") share identical pinyin
representations. This lets the model "hear" toxicity even when toxic
characters are replaced with same-sound alternatives.

Architecture:
  input_ids → embed_tokens → char_embed (B, L, D)
  input_ids → PinyinLookupTable → pinyin_ids → PinyinProjection → fused (B, L, D)

Design invariant: homophones MUST map to the same pinyin_id sequence.
"""
from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn as nn
from pypinyin import Style, pinyin


# CJK Unified Ideographs (basic + Extension A + Compatibility Ideographs)
_HAN_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$")


class PinyinLookupTable:
    """Maps Qwen3 token IDs to pinyin vocabulary IDs.

    Built once at init from a HuggingFace tokenizer. For each token:
      - Chinese text → tone-stripped pinyin string → hashed to pinyin_id
      - Non-Chinese text → sentinel_id (0)

    The table is a 1-D LongTensor: table[token_id] = pinyin_id.
    """

    sentinel_id: int = 0

    def __init__(self, tokenizer, *, debug: bool = False) -> None:
        vocab_size = tokenizer.vocab_size
        # Extend table to cover pad_token_id even if it's beyond vocab_size
        pad_id = getattr(tokenizer, "pad_token_id", None) or 0
        table_size = max(vocab_size, pad_id + 1)

        # Phase 1: decode each token → compute pinyin string
        token_pinyin_strings: list[str] = []
        for tid in range(vocab_size):
            text = tokenizer.decode([tid]).strip()
            py_str = self._text_to_pinyin(text)
            token_pinyin_strings.append(py_str)

        # Phase 2: build pinyin vocabulary (unique strings → IDs)
        # ID 0 is reserved for sentinel (non-Chinese)
        unique_pinyins = sorted(set(s for s in token_pinyin_strings if s))
        self._pinyin_to_id = {py: i + 1 for i, py in enumerate(unique_pinyins)}
        self._id_to_pinyin = {i + 1: py for i, py in enumerate(unique_pinyins)}
        self._id_to_pinyin[0] = ""

        # Phase 3: build the lookup tensor (extended to cover pad_token_id)
        table = torch.zeros(table_size, dtype=torch.long)
        for tid, py_str in enumerate(token_pinyin_strings):
            if py_str:
                table[tid] = self._pinyin_to_id[py_str]
            # else: stays 0 (sentinel)
        # IDs beyond vocab_size (e.g., pad_token_id) default to sentinel (0)
        self.table = table

        # Store per-token pinyin strings for debugging / string-level comparison
        # Omit in production (saves ~750 KB) with debug=False
        if debug:
            self._token_pinyin_strings = token_pinyin_strings
        else:
            self._token_pinyin_strings = token_pinyin_strings  # needed for pinyin_string_for_tokens
            # TODO: could trim to only keep non-empty entries if memory is tight

    @property
    def pinyin_vocab_size(self) -> int:
        """Number of unique pinyin IDs (including sentinel 0)."""
        return len(self._pinyin_to_id) + 1  # +1 for sentinel

    @staticmethod
    def _text_to_pinyin(text: str) -> str:
        """Convert text to tone-stripped pinyin string. Empty if not Chinese."""
        if not text or not _HAN_RE.match(text):
            return ""
        try:
            pys = pinyin(text, style=Style.NORMAL, errors="ignore")
            return "".join(p[0] for p in pys if p and p[0])
        except Exception:
            return ""

    def pinyin_string_for_tokens(self, token_ids: list[int]) -> str:
        """Concatenated pinyin string for a sequence of token IDs.

        Used for homophone verification: homophones should produce
        identical concatenated strings regardless of BPE segmentation.
        """
        parts = []
        for tid in token_ids:
            s = self._token_pinyin_strings[tid]
            if s:
                parts.append(s)
        return "".join(parts)

    def lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Batch lookup: input_ids (B, L) → pinyin_ids (B, L).

        Handles device transfer automatically.
        """
        device = input_ids.device
        if self.table.device != device:
            self.table = self.table.to(device)
        return self.table[input_ids]


class PinyinProjection(nn.Module):
    """Fuses character embeddings with pinyin embeddings via additive residual.

    Design: fused = base_embed + pinyin_proj(pinyin_embed)

    This is analogous to LoRA's philosophy: a small low-rank additive
    modification to the pretrained representation. The base_embed passes
    through UNCHANGED; pinyin contributes a residual that carries phonetic
    information.

    Given:
      base_embed: (B, L, base_dim)    — from Qwen3 embed_tokens
      pinyin_ids: (B, L)              — from PinyinLookupTable

    Returns:
      fused: (B, L, base_dim)         — same shape as input

    Parameter count: pinyin_embed (vocab × pinyin_dim) + pinyin_proj (pinyin_dim × base_dim)
    With vocab=15K, pinyin_dim=64, base_dim=4096: ~1.2M params total (vs 17M for concat+linear).
    """

    def __init__(
        self,
        base_dim: int,
        pinyin_vocab_size: int,
        pinyin_dim: int = 64,
    ) -> None:
        super().__init__()
        self.pinyin_embed = nn.Embedding(pinyin_vocab_size, pinyin_dim)
        self.pinyin_proj = nn.Linear(pinyin_dim, base_dim, bias=False)

        # Small random init (std=0.02) for pinyin_proj so:
        # 1. At training start, residual ≈ 0.02 × sqrt(64) ≈ 0.16 per dim
        #    (tiny vs base_embed range [-3, 3]) → near-identity behavior
        # 2. Gradient flows to pinyin_embed from step 1 (no zero-weight barrier)
        with torch.no_grad():
            nn.init.normal_(self.pinyin_proj.weight, std=0.02)

    def forward(
        self,
        base_embed: torch.Tensor,
        pinyin_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse character and pinyin embeddings.

        Args:
            base_embed: (B, L, base_dim) character embeddings
            pinyin_ids: (B, L) pinyin vocabulary IDs

        Returns:
            (B, L, base_dim) fused embeddings
        """
        pinyin_emb = self.pinyin_embed(pinyin_ids)  # (B, L, pinyin_dim)
        # Match dtype of base_embed (bf16 during training)
        pinyin_emb = pinyin_emb.to(dtype=base_embed.dtype)
        residual = self.pinyin_proj(pinyin_emb)  # (B, L, base_dim)
        return base_embed + residual
