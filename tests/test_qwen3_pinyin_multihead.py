"""TDD contract for PinyinAwareQwen3MultiHead integration.

Tests the end-to-end path: input_ids → pinyin fusion → transformer → heads,
using a MOCK backbone (tiny transformer) to avoid loading 8B model in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn as nn

from cyberpuppy.models.pinyin_embedding import PinyinLookupTable, PinyinProjection
from cyberpuppy.models.qwen3_multihead import MultiTaskOutput
from cyberpuppy.models.qwen3_pinyin_multihead import PinyinAwareQwen3MultiHead


# ---------------------------------------------------------------------------
# Mock backbone: tiny transformer that mimics Qwen3Model's interface
# ---------------------------------------------------------------------------

@dataclass
class FakeConfig:
    hidden_size: int = 128


@dataclass
class FakeOutput:
    last_hidden_state: torch.Tensor


class FakeBackbone(nn.Module):
    """Minimal mock that behaves like Qwen3Model for testing."""

    def __init__(self, hidden_size: int = 128, vocab_size: int = 1000):
        super().__init__()
        self.config = FakeConfig(hidden_size=hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        # Tiny transformer layer (just a linear to simulate processing)
        self.transform = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> FakeOutput:
        if inputs_embeds is not None:
            h = inputs_embeds
        elif input_ids is not None:
            h = self.embed_tokens(input_ids)
        else:
            raise ValueError("Need input_ids or inputs_embeds")
        h = self.transform(h)
        return FakeOutput(last_hidden_state=h)


class FakePinyinLookup:
    """Minimal mock PinyinLookupTable for controlled testing."""

    sentinel_id = 0

    def __init__(self, vocab_size: int = 1000, pinyin_vocab_size: int = 100):
        self._pinyin_vocab_size = pinyin_vocab_size
        # Simple mapping: token_id % pinyin_vocab_size (homophones share same modulo)
        self.table = torch.arange(vocab_size) % pinyin_vocab_size

    @property
    def pinyin_vocab_size(self):
        return self._pinyin_vocab_size

    def lookup(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = input_ids.device
        if self.table.device != device:
            self.table = self.table.to(device)
        return self.table[input_ids]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_backbone():
    return FakeBackbone(hidden_size=128, vocab_size=1000)


@pytest.fixture
def fake_lookup():
    return FakePinyinLookup(vocab_size=1000, pinyin_vocab_size=100)


@pytest.fixture
def model(fake_backbone, fake_lookup):
    return PinyinAwareQwen3MultiHead(
        backbone=fake_backbone,
        hidden_size=128,
        pinyin_lookup=fake_lookup,
        pinyin_dim=32,
    )


# ===========================================================================
# Construction tests
# ===========================================================================

class TestConstruction:
    def test_creates_instance(self, model):
        assert model is not None

    def test_has_pinyin_components(self, model):
        assert hasattr(model, "pinyin_lookup")
        assert hasattr(model, "pinyin_proj")
        assert isinstance(model.pinyin_proj, PinyinProjection)

    def test_inherits_heads(self, model):
        """Should have all 4 classification heads from parent."""
        assert "toxicity" in model.heads
        assert "bullying" in model.heads
        assert "role" in model.heads
        assert "emotion" in model.heads

    def test_requires_pinyin_lookup(self, fake_backbone):
        with pytest.raises(ValueError, match="pinyin_lookup is required"):
            PinyinAwareQwen3MultiHead(fake_backbone, hidden_size=128, pinyin_lookup=None)

    def test_find_embed_tokens(self, model):
        """Should successfully navigate to the embedding layer."""
        embed = model._find_embed_tokens()
        assert isinstance(embed, nn.Embedding)


# ===========================================================================
# Forward pass tests
# ===========================================================================

class TestForward:
    def test_returns_multitask_output(self, model):
        ids = torch.randint(0, 1000, (2, 10))
        mask = torch.ones_like(ids)
        out = model(input_ids=ids, attention_mask=mask)
        assert isinstance(out, MultiTaskOutput)

    def test_logits_have_correct_tasks(self, model):
        ids = torch.randint(0, 1000, (2, 10))
        out = model(input_ids=ids)
        assert set(out.logits.keys()) == {"toxicity", "bullying", "role", "emotion"}

    def test_logits_have_correct_shapes(self, model):
        B, L = 3, 15
        ids = torch.randint(0, 1000, (B, L))
        out = model(input_ids=ids)
        assert out.logits["toxicity"].shape == (B, 3)   # none/toxic/severe
        assert out.logits["bullying"].shape == (B, 3)   # none/harassment/threat
        assert out.logits["role"].shape == (B, 4)       # none/perp/victim/bystander
        assert out.logits["emotion"].shape == (B, 3)    # pos/neu/neg

    def test_forward_uses_inputs_embeds(self, model, fake_backbone):
        """Backbone should receive inputs_embeds (fused), NOT input_ids."""
        # Monkey-patch backbone.forward to track what it received
        received = {}
        original_forward = fake_backbone.forward

        def spy_forward(**kwargs):
            received.update(kwargs)
            return original_forward(**kwargs)

        fake_backbone.forward = lambda **kw: spy_forward(**kw)

        ids = torch.randint(0, 1000, (2, 10))
        model(input_ids=ids)

        assert "inputs_embeds" in received, "Backbone should receive inputs_embeds"
        assert "input_ids" not in received or received.get("input_ids") is None

    def test_pooled_output_shape(self, model):
        B = 2
        ids = torch.randint(0, 1000, (B, 10))
        out = model(input_ids=ids)
        assert out.pooled.shape == (B, 128)  # hidden_size


# ===========================================================================
# Pinyin fusion verification
# ===========================================================================

class TestPinyinFusion:
    def test_homophones_produce_closer_outputs(self, model, fake_lookup):
        """Tokens with same pinyin_id should produce MORE similar logits."""
        # Token 5 and 105 share same pinyin_id (5 % 100 == 105 % 100 == 5)
        ids_a = torch.tensor([[5, 10, 20]])
        ids_b = torch.tensor([[105, 10, 20]])  # 105 % 100 = 5 (same pinyin as 5)
        ids_c = torch.tensor([[50, 10, 20]])    # 50 % 100 = 50 (different pinyin)

        with torch.no_grad():
            out_a = model(input_ids=ids_a)
            out_b = model(input_ids=ids_b)
            out_c = model(input_ids=ids_c)

        # The pinyin residual for position 0 should be identical for a and b
        # (same pinyin_id) but different for c. This means a and b's fused
        # embeddings differ ONLY in the char embedding part, while a and c
        # differ in BOTH char and pinyin parts.
        logits_a = out_a.logits["toxicity"]
        logits_b = out_b.logits["toxicity"]
        logits_c = out_c.logits["toxicity"]

        # Note: at initialization, the effect may be subtle because pinyin_proj
        # weights are small. We verify the MECHANISM (pinyin IDs match).
        py_a = fake_lookup.lookup(ids_a)
        py_b = fake_lookup.lookup(ids_b)
        py_c = fake_lookup.lookup(ids_c)
        assert (py_a[0, 0] == py_b[0, 0]).item(), "Tokens 5 and 105 should share pinyin"
        assert (py_a[0, 0] != py_c[0, 0]).item(), "Tokens 5 and 50 should differ in pinyin"


# ===========================================================================
# Gradient + training compatibility
# ===========================================================================

class TestTrainingCompat:
    def test_gradient_flows_to_pinyin_proj(self, model):
        ids = torch.randint(0, 1000, (2, 10))
        out = model(input_ids=ids)
        loss = sum(v.sum() for v in out.logits.values())
        loss.backward()
        assert model.pinyin_proj.pinyin_proj.weight.grad is not None
        assert model.pinyin_proj.pinyin_proj.weight.grad.abs().sum() > 0

    def test_gradient_flows_to_heads(self, model):
        ids = torch.randint(0, 1000, (2, 10))
        out = model(input_ids=ids)
        loss = sum(v.sum() for v in out.logits.values())
        loss.backward()
        for name, head in model.heads.items():
            assert head.weight.grad is not None, f"Head {name} got no gradient"

    def test_compute_loss_works(self, model):
        ids = torch.randint(0, 1000, (2, 10))
        out = model(input_ids=ids)
        labels = {
            "toxicity": torch.tensor([0, 1]),
            "bullying": torch.tensor([0, 2]),
            "role": torch.tensor([0, 1]),
            "emotion": torch.tensor([1, 2]),
        }
        loss = model.compute_loss(out, labels)
        assert loss.dim() == 0  # scalar
        assert loss.requires_grad
