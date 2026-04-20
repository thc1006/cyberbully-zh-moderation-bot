"""TDD tests for Gradient-Constrained Loss (GCLoss).

GCLoss concentrates model attention on toxic evidence tokens by penalizing
high gradient magnitude on non-evidence positions. This forces the model to
rely on actual toxic tokens rather than spurious correlations, improving
robustness to token-level perturbation attacks.

Reference: ToxiTrace (ACL 2026) — gradient-aligned training.
"""
import pytest
import torch
import torch.nn as nn


class TestGCLossComputation:
    """Test GCLoss function correctness."""

    def test_gcloss_zero_when_attention_focused(self):
        """GCLoss should be 0 when gradients are already focused on evidence."""
        from cyberpuppy.models.gcloss import gradient_constrained_loss

        # Simulate: gradient only on position 2 (the evidence token)
        # token_grads shape: (B, L) — gradient magnitude per token
        token_grads = torch.zeros(1, 5)
        token_grads[0, 2] = 1.0  # only position 2 has gradient
        evidence_mask = torch.zeros(1, 5, dtype=torch.bool)
        evidence_mask[0, 2] = True  # position 2 is evidence

        loss = gradient_constrained_loss(token_grads, evidence_mask)
        assert loss.item() == 0.0, f"Expected 0, got {loss.item()}"

    def test_gcloss_positive_when_gradient_dispersed(self):
        """GCLoss should be positive when gradients leak to non-evidence tokens."""
        from cyberpuppy.models.gcloss import gradient_constrained_loss

        # Gradient dispersed across all tokens
        token_grads = torch.ones(1, 5)
        evidence_mask = torch.zeros(1, 5, dtype=torch.bool)
        evidence_mask[0, 2] = True  # only position 2 is evidence

        loss = gradient_constrained_loss(token_grads, evidence_mask)
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

    def test_gcloss_increases_with_more_leakage(self):
        """More gradient on non-evidence tokens → higher GCLoss."""
        from cyberpuppy.models.gcloss import gradient_constrained_loss

        evidence_mask = torch.zeros(1, 5, dtype=torch.bool)
        evidence_mask[0, 2] = True

        # Low leakage
        grads_low = torch.zeros(1, 5)
        grads_low[0, 2] = 1.0
        grads_low[0, 0] = 0.1  # small leak

        # High leakage
        grads_high = torch.zeros(1, 5)
        grads_high[0, 2] = 1.0
        grads_high[0, 0] = 0.9  # large leak

        loss_low = gradient_constrained_loss(grads_low, evidence_mask)
        loss_high = gradient_constrained_loss(grads_high, evidence_mask)
        assert loss_high > loss_low

    def test_gcloss_batch(self):
        """GCLoss works on batches."""
        from cyberpuppy.models.gcloss import gradient_constrained_loss

        B, L = 4, 10
        token_grads = torch.rand(B, L)
        evidence_mask = torch.zeros(B, L, dtype=torch.bool)
        evidence_mask[:, 3:5] = True  # positions 3-4 are evidence

        loss = gradient_constrained_loss(token_grads, evidence_mask)
        assert loss.shape == ()  # scalar
        assert loss.item() >= 0

    def test_gcloss_no_evidence_returns_zero(self):
        """If no evidence mask is set (all False), loss should be 0."""
        from cyberpuppy.models.gcloss import gradient_constrained_loss

        token_grads = torch.ones(2, 5)
        evidence_mask = torch.zeros(2, 5, dtype=torch.bool)  # no evidence

        loss = gradient_constrained_loss(token_grads, evidence_mask)
        assert loss.item() == 0.0


class TestEvidenceMaskGeneration:
    """Test automatic evidence mask generation from attention/gradient."""

    def test_topk_evidence_mask(self):
        """Generate evidence mask from top-k gradient positions."""
        from cyberpuppy.models.gcloss import topk_evidence_mask

        # Gradient magnitudes
        grads = torch.tensor([[0.1, 0.5, 0.9, 0.2, 0.8]])
        mask = topk_evidence_mask(grads, k=2)

        # Top 2 positions: 2 (0.9) and 4 (0.8)
        assert mask[0, 2] == True
        assert mask[0, 4] == True
        assert mask.sum() == 2

    def test_topk_evidence_mask_batch(self):
        """Top-k mask works per sample in batch."""
        from cyberpuppy.models.gcloss import topk_evidence_mask

        grads = torch.tensor([
            [0.1, 0.9, 0.5, 0.2],
            [0.8, 0.1, 0.3, 0.7],
        ])
        mask = topk_evidence_mask(grads, k=2)

        # Sample 0: top-2 = positions 1, 2
        assert mask[0, 1] == True
        # Sample 1: top-2 = positions 0, 3
        assert mask[1, 0] == True
        assert mask[1, 3] == True

    def test_threshold_evidence_mask(self):
        """Generate evidence mask from threshold on gradient magnitude."""
        from cyberpuppy.models.gcloss import threshold_evidence_mask

        grads = torch.tensor([[0.1, 0.5, 0.9, 0.2, 0.8]])
        mask = threshold_evidence_mask(grads, threshold=0.6)

        assert mask[0, 2] == True  # 0.9 > 0.6
        assert mask[0, 4] == True  # 0.8 > 0.6
        assert mask[0, 0] == False  # 0.1 < 0.6
        assert mask.sum() == 2


class TestGCLossIntegration:
    """Test GCLoss integration with model training."""

    def test_gcloss_gradient_flows(self):
        """GCLoss produces gradients for backprop."""
        from cyberpuppy.models.gcloss import gradient_constrained_loss

        token_grads = torch.rand(2, 8, requires_grad=True)
        evidence_mask = torch.zeros(2, 8, dtype=torch.bool)
        evidence_mask[:, 3:5] = True

        loss = gradient_constrained_loss(token_grads, evidence_mask)
        loss.backward()
        assert token_grads.grad is not None

    def test_gcloss_combined_with_ce(self):
        """GCLoss can be added to CE loss without breaking training."""
        from cyberpuppy.models.gcloss import gradient_constrained_loss

        # Simulate a simple model
        model = nn.Linear(10, 3)
        x = torch.randn(4, 10, requires_grad=True)
        labels = torch.randint(0, 3, (4,))

        logits = model(x)
        ce_loss = nn.CrossEntropyLoss()(logits, labels)

        # Simulate token gradients (from input)
        token_grads = torch.rand(4, 10)
        evidence_mask = torch.zeros(4, 10, dtype=torch.bool)
        evidence_mask[:, 2:5] = True

        gc_loss = gradient_constrained_loss(token_grads, evidence_mask)
        total_loss = ce_loss + 0.1 * gc_loss

        total_loss.backward()
        assert model.weight.grad is not None
