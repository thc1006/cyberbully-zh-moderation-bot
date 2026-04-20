"""Gradient-Constrained Loss (GCLoss) for attention concentration.

Inspired by ToxiTrace (ACL 2026): forces model gradients to concentrate on
toxic evidence tokens rather than dispersing across the input. This improves
robustness to token-level perturbation attacks (homophones, typos) because
the model learns to rely on specific evidence rather than distributed patterns.

Usage in training:
    1. Forward pass → get logits
    2. Compute CE loss
    3. Compute input gradient magnitudes (via autograd)
    4. Generate evidence mask (top-k or threshold on gradient)
    5. GCLoss = mean gradient magnitude on NON-evidence positions
    6. Total loss = CE + λ_gc * GCLoss
"""
from __future__ import annotations

import torch


def gradient_constrained_loss(
    token_grads: torch.Tensor,
    evidence_mask: torch.BoolTensor,
) -> torch.Tensor:
    """Penalize gradient magnitude on non-evidence token positions.

    Args:
        token_grads: (B, L) gradient magnitude per token position
        evidence_mask: (B, L) True for evidence tokens (should have gradient)

    Returns:
        Scalar loss — mean gradient on non-evidence positions.
        Returns 0 if no non-evidence positions exist.
    """
    # Non-evidence mask
    non_evidence = ~evidence_mask  # (B, L)

    # If no evidence specified → can't constrain (don't know where grad should be)
    # If all positions are evidence → nothing to penalize
    if not evidence_mask.any() or not non_evidence.any():
        return token_grads.sum() * 0.0

    # Mean gradient magnitude on non-evidence positions
    non_ev_grads = token_grads[non_evidence]
    return non_ev_grads.mean()


def topk_evidence_mask(
    token_grads: torch.Tensor,
    k: int = 3,
) -> torch.BoolTensor:
    """Generate evidence mask from top-k gradient positions per sample.

    The top-k positions with highest gradient are considered "evidence" —
    the model is allowed to attend to them. All other positions should
    have low gradient (enforced by GCLoss).

    Args:
        token_grads: (B, L) gradient magnitude per token
        k: number of top positions to mark as evidence

    Returns:
        (B, L) boolean mask with True for top-k positions per sample
    """
    B, L = token_grads.shape
    k = min(k, L)
    _, topk_idx = token_grads.topk(k, dim=-1)  # (B, k)
    mask = torch.zeros(B, L, dtype=torch.bool, device=token_grads.device)
    mask.scatter_(1, topk_idx, True)
    return mask


def threshold_evidence_mask(
    token_grads: torch.Tensor,
    threshold: float = 0.5,
) -> torch.BoolTensor:
    """Generate evidence mask from gradient threshold.

    Positions with gradient magnitude above threshold are evidence.

    Args:
        token_grads: (B, L) gradient magnitude per token
        threshold: relative threshold (fraction of max gradient per sample)

    Returns:
        (B, L) boolean mask
    """
    # Per-sample threshold relative to max
    max_grad = token_grads.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    relative = token_grads / max_grad
    return relative >= threshold
