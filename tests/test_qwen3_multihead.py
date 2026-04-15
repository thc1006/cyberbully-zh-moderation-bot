"""TDD tests for Qwen3MultiHead classifier wrapper (ADR 0001 §3.7).

Uses a tiny fake backbone — no real model loading, no GPU.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit

from cyberpuppy.models.qwen3_multihead import (HEAD_DIMS,
                                                MultiTaskOutput,
                                                Qwen3MultiHead,
                                                build_lora_config,
                                                uncertainty_weighted_loss)


class _FakeBackbone(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed = nn.Embedding(1000, hidden)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        h = self.embed(input_ids)
        return SimpleNamespace(last_hidden_state=h, hidden_states=(h,))


# ---- 1. HEAD_DIMS spec ---------------------------------------------------

def test_head_dims_match_adr() -> None:
    assert HEAD_DIMS == {
        "toxicity": 3,
        "bullying": 3,
        "role": 4,
        "emotion": 3,
    }


# ---- 2. Construction -----------------------------------------------------

def test_qwen3_multihead_builds_4_heads() -> None:
    backbone = _FakeBackbone(hidden=64)
    model = Qwen3MultiHead(backbone, hidden_size=64)
    assert set(model.heads.keys()) == {"toxicity", "bullying", "role", "emotion"}
    assert model.heads["toxicity"].out_features == 3
    assert model.heads["role"].out_features == 4


def test_qwen3_multihead_log_var_per_task() -> None:
    backbone = _FakeBackbone(hidden=64)
    model = Qwen3MultiHead(backbone, hidden_size=64, use_uncertainty_weighting=True)
    assert model.log_var.shape == (4,)
    # initialized to zero -> precision = 1
    assert torch.allclose(model.log_var, torch.zeros(4))


# ---- 3. Forward pass -----------------------------------------------------

def test_forward_returns_logits_for_each_head() -> None:
    backbone = _FakeBackbone(hidden=64)
    model = Qwen3MultiHead(backbone, hidden_size=64)
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert isinstance(out, MultiTaskOutput)
    assert out.logits["toxicity"].shape == (2, 3)
    assert out.logits["bullying"].shape == (2, 3)
    assert out.logits["role"].shape == (2, 4)
    assert out.logits["emotion"].shape == (2, 3)


def test_forward_uses_last_token_pool() -> None:
    """For left-padded decoder-only LM, pool = hidden_state[:, -1]."""
    backbone = _FakeBackbone(hidden=64)
    model = Qwen3MultiHead(backbone, hidden_size=64, pool="last")
    input_ids = torch.randint(0, 1000, (3, 8))
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    # 3 batch, 3 classes for toxicity
    assert out.logits["toxicity"].shape == (3, 3)


# ---- 4. Loss -------------------------------------------------------------

def test_uncertainty_weighted_loss_basic() -> None:
    logits = {
        "toxicity": torch.randn(4, 3),
        "bullying": torch.randn(4, 3),
        "role": torch.randn(4, 4),
        "emotion": torch.randn(4, 3),
    }
    labels = {
        "toxicity": torch.tensor([0, 1, 2, 0]),
        "bullying": torch.tensor([0, 1, 2, 0]),
        "role": torch.tensor([0, 1, 2, 3]),
        "emotion": torch.tensor([0, 1, 2, 0]),
    }
    log_var = torch.zeros(4, requires_grad=True)
    loss = uncertainty_weighted_loss(logits, labels, log_var, task_order=list(HEAD_DIMS.keys()))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    # gradient flows
    loss.backward()
    assert log_var.grad is not None


def test_uncertainty_weighted_loss_skips_missing_task() -> None:
    """If a task's label is None, that task is excluded from the loss."""
    logits = {
        "toxicity": torch.randn(2, 3),
        "bullying": torch.randn(2, 3),
        "role": torch.randn(2, 4),
        "emotion": torch.randn(2, 3),
    }
    labels = {
        "toxicity": torch.tensor([0, 1]),
        "bullying": None,
        "role": None,
        "emotion": None,
    }
    log_var = torch.zeros(4)
    loss = uncertainty_weighted_loss(logits, labels, log_var, task_order=list(HEAD_DIMS.keys()))
    assert torch.isfinite(loss)


def test_model_compute_loss_integration() -> None:
    backbone = _FakeBackbone(hidden=64)
    model = Qwen3MultiHead(backbone, hidden_size=64, use_uncertainty_weighting=True)
    input_ids = torch.randint(0, 1000, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    labels = {
        "toxicity": torch.tensor([0, 1]),
        "bullying": torch.tensor([0, 1]),
        "role": torch.tensor([0, 1]),
        "emotion": torch.tensor([0, 1]),
    }
    loss = model.compute_loss(out, labels)
    assert torch.isfinite(loss)
    loss.backward()


# ---- 5. LoRA config ------------------------------------------------------

def test_build_lora_config_default_r_alpha() -> None:
    cfg = build_lora_config()
    assert cfg.r == 32
    assert cfg.lora_alpha == 64
    # use_dora flag should be settable
    assert hasattr(cfg, "use_dora")


def test_build_lora_config_targets_qwen_modules() -> None:
    cfg = build_lora_config()
    # Should target attention + MLP projections (Qwen3 module names)
    assert set(cfg.target_modules) >= {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_build_lora_config_custom_r() -> None:
    cfg = build_lora_config(r=8, alpha=16)
    assert cfg.r == 8
    assert cfg.lora_alpha == 16


# ---- 6. Focal loss + per-sample masking (G-2 TDD) ------------------------

def test_focal_loss_reduces_easy_example_weight() -> None:
    """Focal (gamma>0) must give lower loss than plain CE for confident-correct samples."""
    # Confident-correct logits
    logits = {"toxicity": torch.tensor([[5.0, 0.0, 0.0]])}
    labels = {"toxicity": torch.tensor([0])}
    log_var = torch.zeros(4)

    ce = uncertainty_weighted_loss(logits, labels, log_var, task_order=list(HEAD_DIMS.keys()),
                                    focal_gamma=0.0)
    fl = uncertainty_weighted_loss(logits, labels, log_var, task_order=list(HEAD_DIMS.keys()),
                                    focal_gamma=2.0)
    assert fl.item() < ce.item(), f"focal ({fl.item():.4f}) should be < CE ({ce.item():.4f})"


def test_focal_loss_keeps_hard_example_weight() -> None:
    """For confident-WRONG samples, focal preserves (or slightly inflates) the loss vs CE."""
    # Confident WRONG: high logit on class 0, but true label is class 1
    logits = {"toxicity": torch.tensor([[5.0, 0.0, 0.0]])}
    labels = {"toxicity": torch.tensor([1])}
    log_var = torch.zeros(4)
    ce = uncertainty_weighted_loss(logits, labels, log_var, task_order=list(HEAD_DIMS.keys()),
                                    focal_gamma=0.0)
    fl = uncertainty_weighted_loss(logits, labels, log_var, task_order=list(HEAD_DIMS.keys()),
                                    focal_gamma=2.0)
    # focal should be approx equal or larger since p_true is tiny
    assert fl.item() >= 0.95 * ce.item()


def test_loss_ignore_index_per_sample() -> None:
    """Samples with label == -100 are skipped per-task per-sample."""
    logits = {
        "toxicity": torch.randn(4, 3),
        "bullying": torch.randn(4, 3),
        "role": torch.randn(4, 4),
        "emotion": torch.randn(4, 3),
    }
    # mark sample 0 as missing for toxicity only
    labels = {
        "toxicity": torch.tensor([-100, 1, 2, 0]),
        "bullying": torch.tensor([0, 1, 2, 0]),
        "role": torch.tensor([0, 1, 2, 3]),
        "emotion": torch.tensor([0, 1, 2, 0]),
    }
    log_var = torch.zeros(4, requires_grad=True)
    loss = uncertainty_weighted_loss(logits, labels, log_var,
                                      task_order=list(HEAD_DIMS.keys()))
    assert torch.isfinite(loss)
    loss.backward()
    assert log_var.grad is not None


def test_loss_all_samples_masked_returns_zero_for_task() -> None:
    """If every sample for a task is -100, that task contributes nothing (no NaN)."""
    logits = {
        "toxicity": torch.randn(2, 3),
        "bullying": torch.randn(2, 3),
        "role": torch.randn(2, 4),
        "emotion": torch.randn(2, 3),
    }
    labels = {
        "toxicity": torch.tensor([-100, -100]),
        "bullying": torch.tensor([0, 1]),
        "role": torch.tensor([0, 1]),
        "emotion": torch.tensor([0, 1]),
    }
    log_var = torch.zeros(4)
    loss = uncertainty_weighted_loss(logits, labels, log_var,
                                      task_order=list(HEAD_DIMS.keys()))
    assert torch.isfinite(loss)
    assert loss.item() > 0  # other 3 tasks still contribute


# ---- 7. Consistency loss for adversarial training (v2.2) --------------

def test_consistency_loss_zero_for_identical_logits() -> None:
    """Clean vs cloaked versions of SAME pair_id must have low loss when
    model is already robust (identical logits -> zero MSE)."""
    from cyberpuppy.models.qwen3_multihead import consistency_loss

    # 3 variants (base/homo/emoji) of same pair_id=0, all identical logits
    toxicity_logits = torch.tensor([
        [2.0, 1.0, 0.0],  # pair 0 base
        [2.0, 1.0, 0.0],  # pair 0 homo
        [2.0, 1.0, 0.0],  # pair 0 emoji
    ])
    pair_ids = torch.tensor([0, 0, 0])
    loss = consistency_loss(toxicity_logits, pair_ids)
    assert loss.item() < 1e-6


def test_consistency_loss_nonzero_for_divergent_variants() -> None:
    """Model predicts different logits for clean vs cloaked -> positive loss."""
    from cyberpuppy.models.qwen3_multihead import consistency_loss

    toxicity_logits = torch.tensor([
        [5.0, 0.0, 0.0],  # pair 0 base: confident non-toxic
        [0.0, 5.0, 0.0],  # pair 0 homo: confident toxic (bad! should be consistent)
        [5.0, 0.0, 0.0],  # pair 0 emoji: non-toxic
    ])
    pair_ids = torch.tensor([0, 0, 0])
    loss = consistency_loss(toxicity_logits, pair_ids)
    assert loss.item() > 1.0  # significant mismatch


def test_consistency_loss_ignores_unpaired_samples() -> None:
    """pair_id=-1 means "no pair" (from non-ToxiCloakCN sources); must skip."""
    from cyberpuppy.models.qwen3_multihead import consistency_loss

    toxicity_logits = torch.tensor([
        [5.0, 0.0, 0.0],  # lone sample (pair_id=-1)
        [0.0, 5.0, 0.0],  # another lone (pair_id=-1)
        [1.0, 1.0, 1.0],  # pair 7 base
        [1.0, 1.0, 1.0],  # pair 7 homo
    ])
    pair_ids = torch.tensor([-1, -1, 7, 7])
    loss = consistency_loss(toxicity_logits, pair_ids)
    assert loss.item() < 1e-6  # only pair 7 is checked, and they match


def test_consistency_loss_requires_at_least_two_variants_per_pair() -> None:
    """Pair with only one sample in the batch contributes nothing."""
    from cyberpuppy.models.qwen3_multihead import consistency_loss

    toxicity_logits = torch.tensor([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
    ])
    pair_ids = torch.tensor([3, 4])  # two different pairs, one sample each
    loss = consistency_loss(toxicity_logits, pair_ids)
    assert loss.item() < 1e-6


def test_consistency_loss_backprop() -> None:
    """Loss must be differentiable."""
    from cyberpuppy.models.qwen3_multihead import consistency_loss

    logits = torch.tensor([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
    ], requires_grad=True)
    pair_ids = torch.tensor([1, 1])
    loss = consistency_loss(logits, pair_ids)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
