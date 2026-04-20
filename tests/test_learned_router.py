"""TDD tests for Learned Router (dynamic α per-sample blending).

The router is a small MLP that takes text LoRA hidden states and predicts
the optimal α for text/pinyin ensemble blending per sample.
"""
import pytest
import torch
import torch.nn as nn


class TestLearnedRouterArchitecture:
    """Test the router network architecture and forward pass."""

    def test_router_output_shape(self):
        """Router outputs a scalar α ∈ [0,1] per sample."""
        from cyberpuppy.models.learned_router import LearnedRouter

        router = LearnedRouter(hidden_size=4096)
        x = torch.randn(8, 4096)  # batch=8, hidden_size=4096
        alpha = router(x)
        assert alpha.shape == (8, 1), f"Expected (8,1), got {alpha.shape}"

    def test_router_output_range(self):
        """α must be in [0, 1] (sigmoid output)."""
        from cyberpuppy.models.learned_router import LearnedRouter

        router = LearnedRouter(hidden_size=4096)
        x = torch.randn(32, 4096)
        alpha = router(x)
        assert (alpha >= 0).all() and (alpha <= 1).all()

    def test_router_initial_bias_near_075(self):
        """Router should initialize near α=0.75 (our known optimal fixed value)."""
        from cyberpuppy.models.learned_router import LearnedRouter

        router = LearnedRouter(hidden_size=4096, init_alpha=0.75)
        x = torch.zeros(16, 4096)  # zero input → only bias matters
        alpha = router(x)
        # Should be near 0.75 at initialization
        assert (alpha.mean() - 0.75).abs() < 0.1, f"Initial α={alpha.mean():.3f}, expected ~0.75"

    def test_router_parameter_count(self):
        """Router should be lightweight (<50K parameters)."""
        from cyberpuppy.models.learned_router import LearnedRouter

        router = LearnedRouter(hidden_size=4096)
        n_params = sum(p.numel() for p in router.parameters())
        assert n_params < 50_000, f"Router has {n_params} params, expected <50K"

    def test_router_gradient_flow(self):
        """Gradients flow through the router for training."""
        from cyberpuppy.models.learned_router import LearnedRouter

        router = LearnedRouter(hidden_size=4096)
        x = torch.randn(4, 4096, requires_grad=True)
        alpha = router(x)
        loss = alpha.sum()
        loss.backward()
        assert x.grad is not None


class TestRouterEnsemble:
    """Test the router-based ensemble logic."""

    def test_ensemble_with_router(self):
        """Router-based ensemble should blend text/pinyin probabilities."""
        from cyberpuppy.models.learned_router import router_ensemble

        text_probs = torch.tensor([[0.8, 0.15, 0.05]])  # confident "none"
        pinyin_probs = torch.tensor([[0.3, 0.5, 0.2]])  # says "toxic"
        alpha = torch.tensor([[0.9]])  # trust text heavily

        result = router_ensemble(text_probs, pinyin_probs, alpha)
        assert result.shape == (1, 3)
        # With α=0.9: 0.9*0.8 + 0.1*0.3 = 0.75 for "none"
        expected_none = 0.9 * 0.8 + 0.1 * 0.3
        assert abs(result[0, 0].item() - expected_none) < 1e-5

    def test_ensemble_alpha_one_equals_text_only(self):
        """α=1.0 should give pure text predictions."""
        from cyberpuppy.models.learned_router import router_ensemble

        text_probs = torch.tensor([[0.1, 0.7, 0.2]])
        pinyin_probs = torch.tensor([[0.9, 0.05, 0.05]])
        alpha = torch.tensor([[1.0]])

        result = router_ensemble(text_probs, pinyin_probs, alpha)
        assert torch.allclose(result, text_probs, atol=1e-6)

    def test_ensemble_alpha_zero_equals_pinyin_only(self):
        """α=0.0 should give pure pinyin predictions."""
        from cyberpuppy.models.learned_router import router_ensemble

        text_probs = torch.tensor([[0.1, 0.7, 0.2]])
        pinyin_probs = torch.tensor([[0.9, 0.05, 0.05]])
        alpha = torch.tensor([[0.0]])

        result = router_ensemble(text_probs, pinyin_probs, alpha)
        assert torch.allclose(result, pinyin_probs, atol=1e-6)

    def test_ensemble_batch(self):
        """Router ensemble works on batches with different α per sample."""
        from cyberpuppy.models.learned_router import router_ensemble

        B = 4
        text_probs = torch.softmax(torch.randn(B, 3), -1)
        pinyin_probs = torch.softmax(torch.randn(B, 3), -1)
        alpha = torch.rand(B, 1)

        result = router_ensemble(text_probs, pinyin_probs, alpha)
        assert result.shape == (B, 3)
        # Results should still be valid probability distributions
        assert torch.allclose(result.sum(-1), torch.ones(B), atol=1e-5)


class TestRouterTraining:
    """Test the router training procedure."""

    def test_router_loss_decreases(self):
        """Router training should decrease loss over steps."""
        from cyberpuppy.models.learned_router import LearnedRouter, router_ensemble

        torch.manual_seed(42)
        router = LearnedRouter(hidden_size=128)
        optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)

        # Synthetic data: text is better for some samples, pinyin for others
        N = 64
        hidden_states = torch.randn(N, 128)
        text_probs = torch.softmax(torch.randn(N, 3), -1)
        pinyin_probs = torch.softmax(torch.randn(N, 3), -1)
        labels = torch.randint(0, 3, (N,))

        losses = []
        for _ in range(50):
            alpha = router(hidden_states)
            ensemble = router_ensemble(text_probs, pinyin_probs, alpha)
            loss = nn.CrossEntropyLoss()(ensemble.log(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_router_learns_correct_preference(self):
        """Router should learn to prefer text when text is more accurate."""
        from cyberpuppy.models.learned_router import LearnedRouter, router_ensemble

        torch.manual_seed(42)
        router = LearnedRouter(hidden_size=32, init_alpha=0.5)
        optimizer = torch.optim.Adam(router.parameters(), lr=5e-2)

        # Create scenario: first half → text is correct, second half → pinyin is correct
        N = 32
        hidden_states = torch.zeros(N, 32)
        hidden_states[:N//2, 0] = 5.0   # strong "text-friendly" signal
        hidden_states[N//2:, 1] = 5.0   # strong "pinyin-friendly" signal

        labels = torch.zeros(N, dtype=torch.long)  # all label=0

        # text correct for first half, pinyin correct for second half
        text_probs = torch.zeros(N, 3)
        text_probs[:N//2, 0] = 0.9; text_probs[:N//2, 1] = 0.05; text_probs[:N//2, 2] = 0.05
        text_probs[N//2:, 1] = 0.9; text_probs[N//2:, 0] = 0.05; text_probs[N//2:, 2] = 0.05

        pinyin_probs = torch.zeros(N, 3)
        pinyin_probs[:N//2, 1] = 0.9; pinyin_probs[:N//2, 0] = 0.05; pinyin_probs[:N//2, 2] = 0.05
        pinyin_probs[N//2:, 0] = 0.9; pinyin_probs[N//2:, 1] = 0.05; pinyin_probs[N//2:, 2] = 0.05

        for _ in range(500):
            alpha = router(hidden_states)
            ensemble = router_ensemble(text_probs, pinyin_probs, alpha)
            # NLLLoss on log-probs
            loss = nn.NLLLoss()(ensemble.clamp(min=1e-7).log(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check: first half should have higher α (prefer text)
        with torch.no_grad():
            alpha = router(hidden_states)
        assert alpha[:N//2].mean() > alpha[N//2:].mean(), \
            f"Router didn't learn preference: text-friendly α={alpha[:N//2].mean():.3f}, pinyin-friendly α={alpha[N//2:].mean():.3f}"
