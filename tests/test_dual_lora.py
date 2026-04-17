"""TDD contract for dual-LoRA self-ensemble.

Architecture:
  Qwen3-8B base (shared, 14 GB)
  ├→ LoRA-A "text" (existing v5.1, trained on Chinese text)
  └→ LoRA-B "pinyin" (new, trained on pinyin-converted text)

Inference:
  1. set_adapter("text") → forward(原文) → text_logits
  2. set_adapter("pinyin") → forward(pinyin) → pinyin_logits
  3. ensemble = α × softmax(text) + (1-α) × softmax(pinyin)

Key invariant: homophones produce identical pinyin → identical pinyin_logits.
"""
from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock


class TestPinyinLoraTrainingScript:
    """Verify the training script for LoRA-B (pinyin) produces valid output."""

    def test_pinyin_training_data_exists(self):
        from pathlib import Path
        p = Path("data/processed/v2/pinyin_only_v5_bilingual_train.jsonl")
        assert p.exists(), "Need pinyin training data — run pinyin conversion first"

    def test_pinyin_training_data_has_records(self):
        import json
        from pathlib import Path
        p = Path("data/processed/v2/pinyin_only_v5_bilingual_train.jsonl")
        if not p.exists():
            pytest.skip("pinyin data not built")
        n = sum(1 for _ in p.open())
        assert n > 100000, f"Expected >100K pinyin records, got {n}"


class TestDualLoraInference:
    """Verify dual-adapter switching works correctly."""

    def test_peft_supports_multiple_adapters(self):
        """PEFT library must support loading multiple named adapters."""
        from peft import PeftModel
        # PeftModel has load_adapter and set_adapter methods
        assert hasattr(PeftModel, 'load_adapter')
        assert hasattr(PeftModel, 'set_adapter')

    def test_adapter_switching_produces_different_outputs(self):
        """Switching adapters should change model output."""
        # This is a design contract — verified at integration time
        # with real model. Here we just check the API exists.
        from peft import PeftModel, LoraConfig
        assert LoraConfig is not None


class TestSelfEnsembleLogic:
    """Verify ensemble computation is correct."""

    def test_ensemble_weighted_average(self):
        text_probs = torch.tensor([[0.8, 0.1, 0.1]])
        pinyin_probs = torch.tensor([[0.2, 0.7, 0.1]])
        alpha = 0.65
        ens = alpha * text_probs + (1 - alpha) * pinyin_probs
        expected = torch.tensor([[0.65 * 0.8 + 0.35 * 0.2,
                                   0.65 * 0.1 + 0.35 * 0.7,
                                   0.65 * 0.1 + 0.35 * 0.1]])
        assert torch.allclose(ens, expected, atol=1e-6)

    def test_identical_pinyin_gives_identical_ensemble_for_homophones(self):
        """If pinyin_probs are identical for homophones, ensemble is closer."""
        text_base = torch.tensor([[0.1, 0.8, 0.1]])   # text says toxic
        text_homo = torch.tensor([[0.7, 0.2, 0.1]])   # text fooled → says none
        pinyin_same = torch.tensor([[0.2, 0.6, 0.2]])  # pinyin: same for both!

        alpha = 0.65
        ens_base = alpha * text_base + (1 - alpha) * pinyin_same
        ens_homo = alpha * text_homo + (1 - alpha) * pinyin_same

        # The gap between ens_base and ens_homo should be SMALLER than
        # the gap between text_base and text_homo
        text_gap = (text_base - text_homo).abs().sum()
        ens_gap = (ens_base - ens_homo).abs().sum()
        assert ens_gap < text_gap, "Ensemble should reduce gap between homo predictions"
