"""Qwen3 multi-head classifier wrapper for CyberPuppy v2 (ADR 0001 §3.7).

Structure:
  Qwen3 backbone (LoRA-adapted) -> last-token pool -> 4 linear heads
  (toxicity / bullying / role / emotion). Loss = uncertainty-weighted multi-task.

Designed so unit tests can pass a tiny fake backbone (no real model load).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical head sizes — must match labels.label_map / phase2 LABELS.
HEAD_DIMS: Dict[str, int] = {
    "toxicity": 3,
    "bullying": 3,
    "role": 4,
    "emotion": 3,
}


@dataclass
class MultiTaskOutput:
    logits: Dict[str, torch.Tensor]
    pooled: Optional[torch.Tensor] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Qwen3MultiHead(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: Optional[int] = None,
        head_dims: Optional[Dict[str, int]] = None,
        pool: str = "last",
        dropout: float = 0.1,
        use_uncertainty_weighting: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head_dims = head_dims or dict(HEAD_DIMS)
        self.pool = pool
        self.dropout = nn.Dropout(dropout)

        if hidden_size is None:
            hidden_size = getattr(backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("hidden_size must be provided or inferable from backbone.config")

        self.hidden_size = hidden_size
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, dim) for name, dim in self.head_dims.items()
        })

        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            # log_var per task; precision = exp(-log_var); init to 0 -> precision=1
            self.log_var = nn.Parameter(torch.zeros(len(self.head_dims)))
        else:
            self.register_buffer("log_var", torch.zeros(len(self.head_dims)))

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pool == "last":
            # For left-padded decoder LM, the last token is the most recent real token.
            return hidden_states[:, -1, :]
        if self.pool == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        raise ValueError(f"Unknown pool: {self.pool}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> MultiTaskOutput:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = out.last_hidden_state
        pooled = self._pool(hidden, attention_mask)
        pooled = self.dropout(pooled)
        logits = {name: head(pooled) for name, head in self.heads.items()}
        return MultiTaskOutput(logits=logits, pooled=pooled)

    def compute_loss(
        self,
        outputs: MultiTaskOutput,
        labels: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        return uncertainty_weighted_loss(
            outputs.logits, labels, self.log_var, task_order=list(self.head_dims.keys())
        )


def uncertainty_weighted_loss(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, Optional[torch.Tensor]],
    log_var: torch.Tensor,
    task_order: Iterable[str],
) -> torch.Tensor:
    """Kendall et al. multi-task uncertainty weighting.

    L = sum_t  precision_t * CE_t + log(var_t) / 2
    where precision_t = exp(-log_var_t).

    Tasks whose label is None are skipped (silver/missing labels).
    """
    total: Optional[torch.Tensor] = None
    for idx, task in enumerate(task_order):
        target = labels.get(task)
        if target is None:
            continue
        ce = F.cross_entropy(logits[task], target)
        precision = torch.exp(-log_var[idx])
        term = precision * ce + 0.5 * log_var[idx]
        total = term if total is None else total + term
    if total is None:
        # No labels supplied; return a zero tensor that still tracks log_var if needed.
        total = log_var.sum() * 0.0
    return total


def build_lora_config(
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    use_dora: bool = True,
    target_modules: Optional[Iterable[str]] = None,
):
    """Return a peft.LoraConfig pre-tuned for Qwen3 (ADR 0001 §3.1).

    Imported lazily so unit tests don't require peft for non-LoRA tests.
    """
    from peft import LoraConfig

    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    kwargs: Dict[str, Any] = dict(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
        task_type="FEATURE_EXTRACTION",
    )
    # peft >=0.10 supports DoRA via use_dora flag; fail soft on older versions.
    try:
        return LoraConfig(use_dora=use_dora, **kwargs)
    except TypeError:
        return LoraConfig(**kwargs)
