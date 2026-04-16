"""Integration tests for DualBranchMultiHead with mock backbone."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from cyberpuppy.models.pinyin_cnn_encoder import PinyinBranch
from cyberpuppy.models.qwen3_multihead import MultiTaskOutput
from cyberpuppy.models.dual_branch_multihead import DualBranchMultiHead


@dataclass
class FakeConfig:
    hidden_size: int = 128


@dataclass
class FakeOutput:
    last_hidden_state: torch.Tensor


class FakeBackbone(nn.Module):
    def __init__(self, hidden_size=128, vocab_size=1000):
        super().__init__()
        self.config = FakeConfig(hidden_size=hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.transform = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        h = self.embed_tokens(input_ids)
        return FakeOutput(last_hidden_state=self.transform(h))


@pytest.fixture
def model():
    backbone = FakeBackbone(hidden_size=128, vocab_size=1000)
    pinyin = PinyinBranch(vocab_size=39, embed_dim=32, hidden_dim=64)
    return DualBranchMultiHead(backbone, pinyin, hidden_size=128)


class TestConstruction:
    def test_creates_instance(self, model):
        assert model is not None

    def test_has_pinyin_components(self, model):
        assert hasattr(model, "pinyin_branch")
        assert hasattr(model, "pinyin_proj")
        assert isinstance(model.pinyin_branch, PinyinBranch)

    def test_inherits_4_heads(self, model):
        assert set(model.heads.keys()) == {"toxicity", "bullying", "role", "emotion"}

    def test_pinyin_proj_maps_to_hidden_size(self, model):
        assert model.pinyin_proj.out_features == 128


class TestForward:
    def test_with_pinyin(self, model):
        ids = torch.randint(0, 1000, (2, 10))
        py = torch.randint(0, 39, (2, 15))
        out = model(input_ids=ids, pinyin_ids=py)
        assert isinstance(out, MultiTaskOutput)
        assert set(out.logits.keys()) == {"toxicity", "bullying", "role", "emotion"}
        assert out.logits["toxicity"].shape == (2, 3)

    def test_without_pinyin_fallback(self, model):
        """When pinyin_ids is None, should work like parent (text-only)."""
        ids = torch.randint(0, 1000, (2, 10))
        out = model(input_ids=ids, pinyin_ids=None)
        assert isinstance(out, MultiTaskOutput)
        assert out.extra["pinyin_pooled"] is None

    def test_extra_contains_both_pooled(self, model):
        ids = torch.randint(0, 1000, (2, 10))
        py = torch.randint(0, 39, (2, 15))
        out = model(input_ids=ids, pinyin_ids=py)
        assert "text_pooled" in out.extra
        assert "pinyin_pooled" in out.extra
        assert out.extra["text_pooled"].shape == (2, 128)
        assert out.extra["pinyin_pooled"].shape == (2, 64)


class TestContrastiveLoss:
    def test_returns_scalar(self, model):
        ids = torch.randint(0, 1000, (4, 10))
        py = torch.randint(0, 39, (4, 15))
        out = model(input_ids=ids, pinyin_ids=py)
        loss = model.compute_contrastive_loss(out)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_zero_when_no_pinyin(self, model):
        ids = torch.randint(0, 1000, (4, 10))
        out = model(input_ids=ids, pinyin_ids=None)
        loss = model.compute_contrastive_loss(out)
        assert loss.item() == 0.0


class TestTraining:
    def test_gradient_flows_to_pinyin(self, model):
        ids = torch.randint(0, 1000, (2, 10))
        py = torch.randint(1, 39, (2, 15))
        out = model(input_ids=ids, pinyin_ids=py)
        loss = sum(v.sum() for v in out.logits.values())
        loss += model.compute_contrastive_loss(out)
        loss.backward()
        assert model.pinyin_branch.embed.weight.grad is not None
        assert model.pinyin_branch.embed.weight.grad.abs().sum() > 0
        assert model.pinyin_proj.weight.grad is not None

    def test_gradient_flows_to_heads(self, model):
        ids = torch.randint(0, 1000, (2, 10))
        py = torch.randint(0, 39, (2, 15))
        out = model(input_ids=ids, pinyin_ids=py)
        loss = sum(v.sum() for v in out.logits.values())
        loss.backward()
        for name, head in model.heads.items():
            assert head.weight.grad is not None, f"Head {name} got no gradient"

    def test_compute_loss_works(self, model):
        ids = torch.randint(0, 1000, (2, 10))
        py = torch.randint(0, 39, (2, 15))
        out = model(input_ids=ids, pinyin_ids=py)
        labels = {
            "toxicity": torch.tensor([0, 1]),
            "bullying": torch.tensor([0, 2]),
            "role": torch.tensor([0, 1]),
            "emotion": torch.tensor([1, 2]),
        }
        loss = model.compute_loss(out, labels)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_combined_loss(self, model):
        """Full training loss: task + contrastive + consistency."""
        ids = torch.randint(0, 1000, (4, 10))
        py = torch.randint(0, 39, (4, 15))
        out = model(input_ids=ids, pinyin_ids=py)
        labels = {
            "toxicity": torch.tensor([0, 1, 0, 1]),
            "bullying": torch.tensor([0, 2, 0, 1]),
            "role": torch.tensor([0, 1, 0, 1]),
            "emotion": torch.tensor([1, 2, 1, 0]),
        }
        task_loss = model.compute_loss(out, labels)
        cont_loss = model.compute_contrastive_loss(out)
        total = task_loss + 0.1 * cont_loss
        total.backward()
        # All params should have gradients
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
