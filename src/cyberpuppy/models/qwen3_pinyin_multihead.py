"""Qwen3MultiHead with pinyin auxiliary embedding (Path B2).

Subclasses Qwen3MultiHead to inject phonetic features via additive
residual. Homophones (智商/之上) get identical pinyin embeddings →
model can "hear" toxicity even when toxic characters are replaced.

Usage:
  lookup = PinyinLookupTable(tokenizer)
  model = PinyinAwareQwen3MultiHead(backbone, hidden_size=4096, pinyin_lookup=lookup)
  out = model(input_ids=ids, attention_mask=mask)  # same API as parent
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .pinyin_embedding import PinyinLookupTable, PinyinProjection
from .qwen3_multihead import MultiTaskOutput, Qwen3MultiHead


class PinyinAwareQwen3MultiHead(Qwen3MultiHead):
    """Qwen3MultiHead extended with pinyin auxiliary embedding.

    Forward path:
      input_ids → embed_tokens → base_embed
      input_ids → pinyin_lookup → pinyin_ids → PinyinProjection → residual
      fused = base_embed + residual
      fused → transformer layers (LoRA) → pool → 4 heads → logits

    Everything else (compute_loss, heads, log_var, focal_gamma) inherited.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: Optional[int] = None,
        pinyin_lookup: Optional[PinyinLookupTable] = None,
        pinyin_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(backbone, hidden_size=hidden_size, **kwargs)
        if pinyin_lookup is None:
            raise ValueError("pinyin_lookup is required for PinyinAwareQwen3MultiHead")
        self.pinyin_lookup = pinyin_lookup
        self.pinyin_proj = PinyinProjection(
            base_dim=self.hidden_size,
            pinyin_vocab_size=pinyin_lookup.pinyin_vocab_size,
            pinyin_dim=pinyin_dim,
        )

    def _find_embed_tokens(self) -> nn.Embedding:
        """Navigate PeftModel / raw backbone to find the token embedding layer.

        Supports:
          - Raw Qwen3Model: backbone.embed_tokens
          - AutoModel wrapper: backbone.model.embed_tokens
          - PeftModel: backbone.base_model.model.embed_tokens
          - PeftModel (alt): backbone.model.model.embed_tokens
        """
        backbone = self.backbone
        candidates = [
            "embed_tokens",
            "model.embed_tokens",
            "base_model.model.embed_tokens",
            "model.model.embed_tokens",
        ]
        for path in candidates:
            obj = backbone
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.Embedding):
                    return obj
            except AttributeError:
                continue
        raise AttributeError(
            f"Cannot find embed_tokens in backbone ({type(backbone).__name__}). "
            f"Tried: {candidates}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> MultiTaskOutput:
        # 1. Get character embeddings from backbone's embed_tokens
        embed_layer = self._find_embed_tokens()
        base_embed = embed_layer(input_ids)

        # 2. Compute pinyin IDs and apply additive fusion
        pinyin_ids = self.pinyin_lookup.lookup(input_ids)
        fused = self.pinyin_proj(base_embed, pinyin_ids)

        # 3. Forward through backbone with fused embeddings
        #    (bypasses backbone's internal embed_tokens — we supply inputs_embeds)
        out = self.backbone(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            **kwargs,
        )

        # 4. Pool + classify (identical to parent Qwen3MultiHead)
        hidden = out.last_hidden_state
        pooled = self._pool(hidden, attention_mask)
        pooled = self.dropout(pooled)
        logits = {name: head(pooled) for name, head in self.heads.items()}
        return MultiTaskOutput(logits=logits, pooled=pooled)
