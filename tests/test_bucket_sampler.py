"""TDD tests for LengthBucketSampler — eliminates pathological-batch OOM by
grouping similar-length sequences (HF `LengthGroupedSampler` style)."""
from __future__ import annotations

import random

import pytest

from cyberpuppy.training.bucket_sampler import LengthBucketSampler

pytestmark = pytest.mark.unit


def test_sampler_covers_every_index_exactly_once() -> None:
    lengths = [10, 50, 30, 80, 20, 60, 90, 40, 70, 100, 15, 25]
    sampler = LengthBucketSampler(lengths, batch_size=3, shuffle=False)
    seen = []
    for batch in sampler:
        seen.extend(batch)
    assert sorted(seen) == list(range(len(lengths)))


def test_sampler_groups_similar_lengths_per_batch() -> None:
    """Within each batch, max-min length should be small."""
    rng = random.Random(0)
    lengths = [rng.randint(10, 200) for _ in range(200)]
    sampler = LengthBucketSampler(lengths, batch_size=8, mega_batch_mult=10,
                                   shuffle=True, seed=0)
    spreads = []
    for batch in sampler:
        bls = [lengths[i] for i in batch]
        spreads.append(max(bls) - min(bls))
    avg_spread = sum(spreads) / len(spreads)
    # Without bucketing the avg spread is ~150-160 (uniform 10..200).
    # With mega_batch=80 and batch=8, spread should be << 50.
    assert avg_spread < 50, f"avg spread {avg_spread} too large; bucketing not effective"


def test_sampler_len_matches_batch_count() -> None:
    lengths = [5] * 17
    s = LengthBucketSampler(lengths, batch_size=4, shuffle=False)
    # 17 items / 4 batch = 5 batches (last has 1)
    assert len(s) == 5
    batches = list(s)
    assert len(batches) == 5
    assert sum(len(b) for b in batches) == 17


def test_sampler_shuffle_changes_order() -> None:
    lengths = list(range(50))
    a = list(LengthBucketSampler(lengths, batch_size=4, shuffle=True, seed=1))
    b = list(LengthBucketSampler(lengths, batch_size=4, shuffle=True, seed=2))
    # Different seeds → different batch order
    assert a != b


def test_sampler_no_shuffle_is_deterministic() -> None:
    lengths = list(range(50))
    a = list(LengthBucketSampler(lengths, batch_size=4, shuffle=False))
    b = list(LengthBucketSampler(lengths, batch_size=4, shuffle=False))
    assert a == b


def test_sampler_handles_uneven_last_batch() -> None:
    lengths = [1] * 13
    s = LengthBucketSampler(lengths, batch_size=5, shuffle=False)
    batches = list(s)
    sizes = [len(b) for b in batches]
    assert sum(sizes) == 13
    assert max(sizes) == 5
