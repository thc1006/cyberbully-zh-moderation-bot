"""TDD contract for dual-branch architecture (DBIT-style).

Based on: "Homophone-aware offensive language detection via semantic-phonetic
collaboration" (Expert Systems 2025) + DBIT GitHub (hjhhlc/DBIT).

Architecture:
  Branch 1: Qwen3-8B text encoder (existing) → text_pooled (B, 4096)
  Branch 2: Depthwise CNN pinyin encoder (~38K params) → pinyin_pooled (B, 256)
  Fusion: additive residual → fused (B, 4096) → 4 classification heads
  Loss: task_loss + λ_cons × consistency + λ_cont × contrastive(text, pinyin)
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cyberpuppy.models.pinyin_cnn_encoder import (
    PinyinCharTokenizer,
    PinyinCNNEncoder,
    PinyinBranch,
)


# ===========================================================================
# PinyinCharTokenizer
# ===========================================================================

class TestPinyinCharTokenizer:
    @pytest.fixture
    def tok(self):
        return PinyinCharTokenizer()

    def test_creates_instance(self, tok):
        assert tok is not None

    def test_vocab_size_small(self, tok):
        """Char-level: 26 letters + space + pad + unk = ~29-35 tokens."""
        assert 25 <= tok.vocab_size <= 50

    def test_encode_pinyin_string(self, tok):
        ids = tok.encode("bai ren")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) == len("bai ren")  # char-level

    def test_homophones_produce_identical_ids(self, tok):
        """Core invariant from pinyin_prompt: homophones → same pinyin string → same IDs."""
        from cyberpuppy.data.pinyin_prompt import _chinese_chars_to_pinyin
        py_a = _chinese_chars_to_pinyin("白人都該死")
        py_b = _chinese_chars_to_pinyin("拜仁都該死")
        assert py_a == py_b  # same pinyin string
        assert tok.encode(py_a) == tok.encode(py_b)  # same IDs

    def test_batch_encode(self, tok):
        result = tok.batch_encode(["bai ren", "zhi shang"], max_length=20)
        assert result.shape[0] == 2
        assert result.shape[1] == 20  # padded to max_length
        assert result.dtype == torch.long

    def test_padding(self, tok):
        result = tok.batch_encode(["bai", "bai ren dou"], max_length=15)
        # Shorter one should be padded (right-pad with pad_id)
        assert result[0, 3].item() == tok.pad_id  # "bai" is 3 chars, pos 3 should be pad

    def test_unknown_chars_handled(self, tok):
        """Non-Latin chars should map to unk, not crash."""
        ids = tok.encode("bai人")  # mixed Latin + Chinese
        assert len(ids) == 4  # b, a, i, 人(→unk)


# ===========================================================================
# PinyinCNNEncoder (DBIT Depthwise CNN)
# ===========================================================================

class TestPinyinCNNEncoder:
    @pytest.fixture
    def encoder(self):
        return PinyinCNNEncoder(input_dim=64, hidden_dim=256)

    def test_creates_instance(self, encoder):
        assert encoder is not None

    def test_output_shape(self, encoder):
        x = torch.randn(4, 20, 64)  # (B=4, L=20, input_dim=64)
        out = encoder(x)
        assert out.shape == (4, 256)  # (B, hidden_dim)

    def test_parameter_count_tiny(self, encoder):
        total = sum(p.numel() for p in encoder.parameters())
        assert total < 100_000, f"Too many params: {total} (should be ~20K)"

    def test_variable_length_input(self, encoder):
        """AdaptiveMaxPool1d should handle different sequence lengths."""
        x1 = torch.randn(2, 10, 64)
        x2 = torch.randn(2, 50, 64)
        out1 = encoder(x1)
        out2 = encoder(x2)
        assert out1.shape == out2.shape == (2, 256)

    def test_gradient_flows(self, encoder):
        x = torch.randn(2, 15, 64)
        out = encoder(x)
        out.sum().backward()
        assert encoder.depthwise.weight.grad is not None


# ===========================================================================
# PinyinBranch (Embedding + CNN)
# ===========================================================================

class TestPinyinBranch:
    @pytest.fixture
    def branch(self):
        return PinyinBranch(vocab_size=35, embed_dim=64, hidden_dim=256)

    def test_creates_instance(self, branch):
        assert branch is not None

    def test_forward_from_ids(self, branch):
        ids = torch.randint(0, 35, (4, 20))  # (B=4, L=20)
        out = branch(ids)
        assert out.shape == (4, 256)

    def test_total_params_under_100k(self, branch):
        total = sum(p.numel() for p in branch.parameters())
        assert total < 100_000, f"PinyinBranch should be <100K params, got {total}"

    def test_gradient_flows_to_embedding(self, branch):
        ids = torch.randint(1, 35, (2, 10))
        out = branch(ids)
        out.sum().backward()
        assert branch.embed.weight.grad is not None
        assert branch.embed.weight.grad.abs().sum() > 0


# ===========================================================================
# Contrastive Loss (bidirectional InfoNCE, DBIT style)
# ===========================================================================

class TestContrastiveLoss:
    def test_contrastive_loss_scalar(self):
        from cyberpuppy.models.pinyin_cnn_encoder import contrastive_loss
        text_feat = torch.randn(8, 256)
        pinyin_feat = torch.randn(8, 256)
        loss = contrastive_loss(text_feat, pinyin_feat, temperature=0.07)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_identical_features_low_loss(self):
        from cyberpuppy.models.pinyin_cnn_encoder import contrastive_loss
        feat = torch.randn(8, 256)
        loss_identical = contrastive_loss(feat, feat, temperature=0.07)
        loss_random = contrastive_loss(feat, torch.randn_like(feat), temperature=0.07)
        assert loss_identical < loss_random

    def test_gradient_flows(self):
        from cyberpuppy.models.pinyin_cnn_encoder import contrastive_loss
        text_feat = torch.randn(4, 128, requires_grad=True)
        pinyin_feat = torch.randn(4, 128, requires_grad=True)
        loss = contrastive_loss(text_feat, pinyin_feat)
        loss.backward()
        assert text_feat.grad is not None
        assert pinyin_feat.grad is not None
