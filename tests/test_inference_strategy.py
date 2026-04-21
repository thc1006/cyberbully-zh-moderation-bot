"""TDD tests for optimal inference strategy.

Strategy v5.1.1:
  1. Geometric mean ensemble: text^α × pinyin^(1-α), α=0.5
  2. Pinyin override: if pinyin toxic prob > 0.7 → force toxic
  3. Decision threshold: ensemble toxic prob > 0.48 → toxic
  4. Confidence-gated cascade: if ensemble confidence < 0.55 → LLM

Discovered via comprehensive parameter sweep on 2026-04-21.
"""
import pytest
import torch
import numpy as np


class TestEnsembleStrategy:
    """Test the core ensemble + override strategy."""

    def test_geometric_mean_default_alpha(self):
        """Default alpha should be 0.5 (equal weight)."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy()
        assert s.geo_alpha == 0.5

    def test_geometric_mean_computation(self):
        """Geometric mean: normalize(text^α × pinyin^(1-α))."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy(geo_alpha=0.5)

        text_probs = torch.tensor([[0.8, 0.15, 0.05]])
        pinyin_probs = torch.tensor([[0.2, 0.6, 0.2]])

        result = s.ensemble(text_probs, pinyin_probs)
        assert result.shape == (1, 3)
        # Should sum to 1
        assert abs(result.sum().item() - 1.0) < 1e-5

        # With α=0.5 (equal weight), result should be geometric mean
        expected = (text_probs ** 0.5) * (pinyin_probs ** 0.5)
        expected = expected / expected.sum(-1, keepdim=True)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_pinyin_override_triggers(self):
        """When pinyin toxic > threshold, prediction should be toxic."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy(pinyin_override_thresh=0.7)

        # Text says safe, pinyin says very toxic
        text_probs = torch.tensor([[0.9, 0.08, 0.02]])
        pinyin_probs = torch.tensor([[0.1, 0.75, 0.15]])  # toxic=0.75+0.15=0.90 > 0.7

        pred = s.predict(text_probs, pinyin_probs)
        assert pred[0].item() == True, "Should override to toxic when pinyin > 0.7"

    def test_pinyin_override_does_not_trigger_below_thresh(self):
        """When pinyin toxic <= threshold, no override."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy(pinyin_override_thresh=0.7)

        text_probs = torch.tensor([[0.9, 0.08, 0.02]])
        pinyin_probs = torch.tensor([[0.4, 0.5, 0.1]])  # toxic=0.6 < 0.7

        pred = s.predict(text_probs, pinyin_probs)
        # Ensemble should dominate — text says safe with 0.9
        assert pred[0].item() == False

    def test_decision_threshold(self):
        """Predictions use decision_thresh, not fixed 0.5."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy

        text_probs = torch.tensor([[0.55, 0.30, 0.15]])
        pinyin_probs = torch.tensor([[0.55, 0.30, 0.15]])

        # With thresh=0.50: toxic_prob=0.45 < 0.50 → safe
        s50 = InferenceStrategy(decision_thresh=0.50)
        assert s50.predict(text_probs, pinyin_probs)[0].item() == False

        # With thresh=0.42: toxic_prob=0.45 > 0.42 → toxic
        s42 = InferenceStrategy(decision_thresh=0.42)
        assert s42.predict(text_probs, pinyin_probs)[0].item() == True

    def test_batch_prediction(self):
        """Strategy works on batches."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy()

        B = 8
        text_probs = torch.softmax(torch.randn(B, 3), -1)
        pinyin_probs = torch.softmax(torch.randn(B, 3), -1)

        preds = s.predict(text_probs, pinyin_probs)
        assert preds.shape == (B,)
        assert preds.dtype == torch.bool

    def test_confidence_output(self):
        """Strategy returns confidence for cascade gating."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy()

        text_probs = torch.tensor([[0.9, 0.08, 0.02], [0.34, 0.33, 0.33]])
        pinyin_probs = torch.tensor([[0.85, 0.10, 0.05], [0.34, 0.33, 0.33]])

        preds, conf = s.predict_with_confidence(text_probs, pinyin_probs)
        assert conf.shape == (2,)
        # First sample should be high confidence, second low
        assert conf[0] > conf[1]

    def test_backward_compat_v51(self):
        """Can instantiate with v5.1 settings (α=0.75, no override, thresh=0.5)."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy(geo_alpha=0.75, pinyin_override_thresh=None,
                              decision_thresh=0.5)
        text_probs = torch.tensor([[0.8, 0.15, 0.05]])
        pinyin_probs = torch.tensor([[0.2, 0.6, 0.2]])

        result = s.ensemble(text_probs, pinyin_probs)
        expected = (text_probs ** 0.75) * (pinyin_probs ** 0.25)
        expected = expected / expected.sum(-1, keepdim=True)
        assert torch.allclose(result, expected, atol=1e-5)


class TestCascadeGating:
    """Test confidence-gated cascade logic."""

    def test_identify_low_confidence(self):
        """Correctly identify samples below confidence threshold."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy(cascade_conf_thresh=0.55)

        text_probs = torch.tensor([
            [0.9, 0.08, 0.02],  # high conf
            [0.34, 0.33, 0.33], # low conf
            [0.51, 0.30, 0.19], # borderline
        ])
        pinyin_probs = torch.tensor([
            [0.85, 0.10, 0.05],
            [0.40, 0.35, 0.25],
            [0.50, 0.30, 0.20],
        ])

        preds, conf = s.predict_with_confidence(text_probs, pinyin_probs)
        needs_cascade = s.needs_cascade(conf)

        # Sample 0: high conf → no cascade
        assert needs_cascade[0].item() == False
        # Sample 1: low conf → cascade
        assert needs_cascade[1].item() == True

    def test_apply_cascade_results(self):
        """Cascade results override low-confidence predictions."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy(cascade_conf_thresh=0.55)

        preds = torch.tensor([True, False, True])
        conf = torch.tensor([0.9, 0.4, 0.8])
        cascade_preds = {1: True}  # LLM says sample 1 is toxic

        final = s.apply_cascade(preds, conf, cascade_preds)
        assert final[0] == True   # kept (high conf)
        assert final[1] == True   # overridden by cascade
        assert final[2] == True   # kept (high conf)

    def test_cascade_skip_when_no_low_conf(self):
        """No cascade needed when all samples are high confidence."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy(cascade_conf_thresh=0.55)

        conf = torch.tensor([0.9, 0.85, 0.92])
        needs = s.needs_cascade(conf)
        assert needs.sum() == 0


class TestEndToEnd:
    """Integration tests for the full strategy."""

    def test_pinyin_override_rescues_suppressed_signal(self):
        """The key insight: pinyin detects 大廈避風=大傻逼 but text suppresses it."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy()  # α=0.5, override=0.7, thresh=0.48

        # Simulated: 大廈避風了 — text safe, pinyin very toxic
        text_probs = torch.tensor([[0.85, 0.10, 0.05]])   # text: safe
        pinyin_probs = torch.tensor([[0.15, 0.65, 0.20]])  # pinyin: toxic (0.85)

        pred = s.predict(text_probs, pinyin_probs)
        assert pred[0].item() == True, "Pinyin override should rescue this case"

    def test_safe_joke_not_overridden(self):
        """Safe homophone jokes (嫦娥=change) should NOT be flagged."""
        from cyberpuppy.models.inference_strategy import InferenceStrategy
        s = InferenceStrategy()

        # Both models say safe — no override
        text_probs = torch.tensor([[0.85, 0.10, 0.05]])
        pinyin_probs = torch.tensor([[0.70, 0.20, 0.10]])  # pinyin toxic=0.30 < 0.7

        pred = s.predict(text_probs, pinyin_probs)
        assert pred[0].item() == False, "Safe jokes should stay safe"
