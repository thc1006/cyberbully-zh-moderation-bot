"""Learned Router for dynamic α text/pinyin ensemble blending.

Instead of a fixed α=0.75, the router predicts per-sample optimal blending
weight based on the text LoRA's hidden state. This allows:
- Clean text → α≈1.0 (trust text, pinyin may add noise)
- Suspected homophone attack → α≈0.5 (lean on pinyin)
- Ambiguous cases → learned optimal trade-off
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class LearnedRouter(nn.Module):
    """Lightweight MLP that predicts blending α from text hidden states.

    Architecture: Linear(H→64) → GELU → Linear(64→1) → Sigmoid
    Parameters: ~260K for H=4096, well under 50K for H≤512
    """

    def __init__(self, hidden_size: int, init_alpha: float = 0.75):
        super().__init__()
        mid = min(8, max(4, hidden_size // 512))
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.Linear(mid, 1),
        )
        # Initialize: small random weights + bias so sigmoid(output) ≈ init_alpha
        # sigmoid(x) = init_alpha → x = log(α / (1-α))
        bias_init = math.log(init_alpha / (1 - init_alpha))
        nn.init.normal_(self.net[0].weight, std=0.01)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.constant_(self.net[2].bias, bias_init)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict α ∈ [0,1] per sample.

        Args:
            hidden_states: (B, H) text LoRA last hidden state

        Returns:
            alpha: (B, 1) blending weight for text probabilities
        """
        return torch.sigmoid(self.net(hidden_states))


def router_ensemble(
    text_probs: torch.Tensor,
    pinyin_probs: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Blend text and pinyin probabilities using per-sample α.

    Args:
        text_probs: (B, C) softmax probabilities from text LoRA
        pinyin_probs: (B, C) softmax probabilities from pinyin LoRA
        alpha: (B, 1) per-sample blending weight for text

    Returns:
        (B, C) blended probability distribution
    """
    return alpha * text_probs + (1 - alpha) * pinyin_probs
