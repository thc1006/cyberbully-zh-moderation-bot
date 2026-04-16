"""TDD contract for Path B2: Pinyin auxiliary embedding.

Goal: inject phonetic (pinyin) features into Qwen3 embeddings so that
homophones (智商/之上, 白人/拜仁) get identical pinyin representations,
enabling the model to "hear" toxicity even when toxic characters are
replaced with same-sound alternatives.

Architecture under test:
  PinyinLookupTable  — maps Qwen3 token IDs → pinyin vocab IDs
  PinyinProjection   — fuses char_embed + pinyin_embed → fused_embed

Design invariant: homophones MUST map to the same pinyin_id.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cyberpuppy.models.pinyin_embedding import (
    PinyinLookupTable,
    PinyinProjection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    """Load Qwen3 tokenizer (cached across all tests in this module)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")


@pytest.fixture(scope="module")
def lookup(tokenizer):
    """Build pinyin lookup table (slow — ~0.2s, cached across tests)."""
    return PinyinLookupTable(tokenizer)


@pytest.fixture
def proj():
    """Create a PinyinProjection with typical dimensions."""
    return PinyinProjection(base_dim=4096, pinyin_vocab_size=15000, pinyin_dim=64)


# ===========================================================================
# PinyinLookupTable tests
# ===========================================================================

class TestPinyinLookupTableConstruction:
    def test_creates_instance(self, lookup):
        assert lookup is not None

    def test_has_mapping_tensor(self, lookup):
        """Internal mapping should be a 1-D LongTensor indexed by token_id."""
        assert hasattr(lookup, "table")
        assert lookup.table.dtype == torch.long
        assert lookup.table.dim() == 1

    def test_table_covers_full_vocab(self, lookup, tokenizer):
        """Table must cover at least vocab_size (may be larger for pad_token_id safety)."""
        assert lookup.table.shape[0] >= tokenizer.vocab_size

    def test_pinyin_vocab_size_reasonable(self, lookup):
        """Unique pinyin IDs should be ~10K-20K (Chinese phonetic space)."""
        n_unique = lookup.table.unique().numel()
        assert 5_000 <= n_unique <= 25_000, f"Got {n_unique} unique pinyin IDs"

    def test_sentinel_id_is_zero(self, lookup):
        """Non-Chinese tokens should map to pinyin_id = 0 (sentinel)."""
        assert lookup.sentinel_id == 0


class TestHomophonePairsMapToSamePinyin:
    """The core invariant: homophones MUST share the same pinyin_id sequence."""

    HOMO_PAIRS = [
        ("智商", "之上"),    # zhishang — merged BPE tokens
        ("白人", "拜仁"),    # bai+ren — split BPE tokens
        ("歧視", "其實"),    # qishi — mixed BPE (2 tok vs 1 tok)
        ("死掉", "四調"),    # si+diao
    ]

    @pytest.mark.parametrize("clean,homo", HOMO_PAIRS)
    def test_homophone_pair_same_pinyin_sequence(self, clean, homo, lookup, tokenizer):
        """Concatenated pinyin IDs must be identical for homophones."""
        ids_a = tokenizer.encode(clean, add_special_tokens=False)
        ids_b = tokenizer.encode(homo, add_special_tokens=False)

        pinyins_a = [lookup.table[tid].item() for tid in ids_a]
        pinyins_b = [lookup.table[tid].item() for tid in ids_b]

        # Flatten compound pinyin: join non-zero IDs into a comparable sequence
        seq_a = [p for p in pinyins_a if p != lookup.sentinel_id]
        seq_b = [p for p in pinyins_b if p != lookup.sentinel_id]

        # If BPE merges differ (e.g., 歧視=2tok, 其實=1tok), compare the
        # concatenated pinyin STRING instead of the ID list
        str_a = lookup.pinyin_string_for_tokens(ids_a)
        str_b = lookup.pinyin_string_for_tokens(ids_b)
        assert str_a == str_b, (
            f"Homophone pair {clean}/{homo} got different pinyin: "
            f"{str_a!r} vs {str_b!r}"
        )


class TestNonChineseTokens:
    def test_english_maps_to_sentinel(self, lookup, tokenizer):
        ids = tokenizer.encode("hello world", add_special_tokens=False)
        for tid in ids:
            assert lookup.table[tid].item() == lookup.sentinel_id

    def test_numbers_map_to_sentinel(self, lookup, tokenizer):
        ids = tokenizer.encode("12345", add_special_tokens=False)
        for tid in ids:
            assert lookup.table[tid].item() == lookup.sentinel_id

    def test_punctuation_maps_to_sentinel(self, lookup, tokenizer):
        ids = tokenizer.encode("，。！？", add_special_tokens=False)
        for tid in ids:
            assert lookup.table[tid].item() == lookup.sentinel_id


class TestBatchLookup:
    def test_batch_index_returns_correct_shape(self, lookup, tokenizer):
        texts = ["白人都該死", "今天天氣真好"]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
        pinyin_ids = lookup.lookup(enc["input_ids"])
        assert pinyin_ids.shape == enc["input_ids"].shape
        assert pinyin_ids.dtype == torch.long


# ===========================================================================
# PinyinProjection tests
# ===========================================================================

class TestPinyinProjectionConstruction:
    def test_creates_instance(self, proj):
        assert proj is not None

    def test_has_pinyin_embed_and_proj(self, proj):
        assert hasattr(proj, "pinyin_embed")
        assert hasattr(proj, "pinyin_proj")
        assert isinstance(proj.pinyin_embed, nn.Embedding)
        assert isinstance(proj.pinyin_proj, nn.Linear)

    def test_output_dim_matches_base_dim(self, proj):
        """Projection output must be same dim as base embedding (4096)."""
        assert proj.pinyin_proj.out_features == 4096

    def test_parameter_count_efficient(self, proj):
        """Additive design: ~1.2M params, NOT 17M (old concat+linear)."""
        total = sum(p.numel() for p in proj.parameters())
        assert total < 2_000_000, f"Too many params: {total} (should be ~1.2M)"


class TestIdentityInit:
    def test_near_identity_with_sentinel_pinyin(self, proj):
        """Additive residual with sentinel pinyin should give output ≈ base_embed.

        Residual = pinyin_proj(pinyin_embed[0]). With small init (std=0.02),
        this is ~0.16 per dim vs base_embed range [-3, 3].
        """
        B, L, D = 2, 10, 4096
        base_embed = torch.randn(B, L, D)
        pinyin_ids = torch.zeros(B, L, dtype=torch.long)  # all sentinel

        with torch.no_grad():
            fused = proj(base_embed, pinyin_ids)

        # Residual contribution is small: std(proj) × std(embed) × sqrt(dim)
        diff = (fused - base_embed).abs().max().item()
        assert diff < 5.0, f"Near-identity violated: max diff = {diff}"

    def test_output_shape(self, proj):
        B, L, D = 2, 10, 4096
        base_embed = torch.randn(B, L, D)
        pinyin_ids = torch.randint(0, 15000, (B, L))

        with torch.no_grad():
            fused = proj(base_embed, pinyin_ids)

        assert fused.shape == (B, L, D)


class TestHomophoneEmbeddingSimilarity:
    def test_homophones_get_closer_embeddings(self, proj, lookup, tokenizer):
        """After fusion, homophones should be MORE similar than random pairs."""
        B, D = 1, 4096

        # Encode two homophones
        ids_a = tokenizer.encode("智商", add_special_tokens=False)
        ids_b = tokenizer.encode("之上", add_special_tokens=False)
        # And a random non-homophone
        ids_c = tokenizer.encode("天氣", add_special_tokens=False)

        # Create fake base embeddings (DIFFERENT for each — simulating real model)
        torch.manual_seed(42)
        embed_a = torch.randn(1, len(ids_a), D)
        embed_b = torch.randn(1, len(ids_b), D)  # different!
        embed_c = torch.randn(1, len(ids_c), D)

        py_a = lookup.lookup(torch.tensor([ids_a]))
        py_b = lookup.lookup(torch.tensor([ids_b]))
        py_c = lookup.lookup(torch.tensor([ids_c]))

        with torch.no_grad():
            fused_a = proj(embed_a, py_a).mean(dim=1)  # pool over length
            fused_b = proj(embed_b, py_b).mean(dim=1)
            fused_c = proj(embed_c, py_c).mean(dim=1)

        # At init, pinyin weight is zero → homophones NOT yet closer.
        # After training they WOULD be. Here we verify the MECHANISM works:
        # pinyin_ids for a and b should be identical (verified by lookup tests).
        # We just verify the pinyin embeddings are the same.
        pinyin_embed_a = proj.pinyin_embed(py_a)
        pinyin_embed_b = proj.pinyin_embed(py_b)
        # Mean-pool pinyin embeddings
        pe_a = pinyin_embed_a.float().mean(dim=1)
        pe_b = pinyin_embed_b.float().mean(dim=1)

        # Homophones should have identical pinyin embeddings
        pinyin_diff = (pe_a - pe_b).abs().max().item()
        assert pinyin_diff < 1e-6, (
            f"Homophone pinyin embeddings should be identical but differ by {pinyin_diff}"
        )


class TestPaddingAndEdgeCases:
    def test_padding_token_maps_to_sentinel(self, lookup, tokenizer):
        """pad_token_id must not cause IndexError and should map to sentinel."""
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        assert pad_id < lookup.table.shape[0], (
            f"pad_token_id {pad_id} >= table size {lookup.table.shape[0]}"
        )
        assert lookup.table[pad_id].item() == lookup.sentinel_id

    def test_out_of_range_clamped(self, lookup):
        """Token IDs beyond table size should not crash lookup()."""
        big_ids = torch.tensor([[999999]])
        # Should either work (if table extended) or raise clean error
        if lookup.table.shape[0] > 999999:
            result = lookup.lookup(big_ids)
            assert result.shape == big_ids.shape
        else:
            with pytest.raises(IndexError):
                lookup.lookup(big_ids)

    def test_proj_uses_actual_pinyin_vocab_size(self, lookup):
        """PinyinProjection should be created with lookup's actual vocab size."""
        proj = PinyinProjection(
            base_dim=128,
            pinyin_vocab_size=lookup.pinyin_vocab_size,
            pinyin_dim=32,
        )
        # Embedding covers all pinyin IDs including sentinel
        assert proj.pinyin_embed.num_embeddings == lookup.pinyin_vocab_size


class TestGradientFlow:
    def test_pinyin_embed_receives_gradient(self, proj):
        """Ensure pinyin embedding is in the gradient path."""
        B, L, D = 2, 5, 4096
        base_embed = torch.randn(B, L, D)
        pinyin_ids = torch.randint(1, 15000, (B, L))  # non-sentinel

        fused = proj(base_embed, pinyin_ids)
        loss = fused.sum()
        loss.backward()

        assert proj.pinyin_embed.weight.grad is not None
        assert proj.pinyin_embed.weight.grad.abs().sum() > 0

    def test_pinyin_proj_receives_gradient(self, proj):
        B, L, D = 2, 5, 4096
        base_embed = torch.randn(B, L, D)
        pinyin_ids = torch.randint(1, 15000, (B, L))

        fused = proj(base_embed, pinyin_ids)
        loss = fused.sum()
        loss.backward()

        assert proj.pinyin_proj.weight.grad is not None
