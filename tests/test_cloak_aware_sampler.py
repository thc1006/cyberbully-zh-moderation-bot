"""Cloak-aware batch sampler: ensures ToxiCloakCN triplets (base/homo/emoji of
same pair_id) land in the SAME batch so consistency_loss can fire."""
from __future__ import annotations

import random

import pytest

from cyberpuppy.training.cloak_aware_sampler import CloakAwareBatchSampler

pytestmark = pytest.mark.unit


def _build_records(n_clean: int, n_pairs: int, clean_base_len: int = 50) -> list[dict]:
    """Fake records: n_clean singletons (pair_id=-1) + n_pairs × 3 cloak samples."""
    records = []
    for i in range(n_clean):
        records.append({
            "text": "x" * (clean_base_len + i % 20),
            "metadata": {"cloak_pair_id": -1},
        })
    for p in range(n_pairs):
        for variant in ("base", "homo", "emoji"):
            records.append({
                "text": "y" * (30 + p % 10),
                "metadata": {"cloak_pair_id": p, "cloak_variant": variant},
            })
    return records


def test_sampler_groups_pair_triplets_together() -> None:
    """Each cloak triplet must land in the SAME batch."""
    records = _build_records(n_clean=48, n_pairs=20)
    sampler = CloakAwareBatchSampler(records, batch_size=8, shuffle=False)
    all_batches = list(sampler)

    # Track which batch each pair_id's samples land in
    pair_batch_ids: dict[int, set[int]] = {}
    for bi, batch in enumerate(all_batches):
        for idx in batch:
            pid = records[idx]["metadata"]["cloak_pair_id"]
            if pid >= 0:
                pair_batch_ids.setdefault(pid, set()).add(bi)

    # Every pair should be in exactly ONE batch
    for pid, bids in pair_batch_ids.items():
        assert len(bids) == 1, f"pair {pid} split across batches {bids}"


def test_sampler_covers_every_sample_once() -> None:
    records = _build_records(n_clean=100, n_pairs=30)
    sampler = CloakAwareBatchSampler(records, batch_size=8, shuffle=True, seed=1)
    seen = []
    for batch in sampler:
        seen.extend(batch)
    assert sorted(seen) == list(range(len(records)))


def test_sampler_handles_zero_pairs() -> None:
    """Pure clean dataset: degrades to regular length-bucket behavior."""
    records = _build_records(n_clean=50, n_pairs=0)
    sampler = CloakAwareBatchSampler(records, batch_size=8, shuffle=False)
    batches = list(sampler)
    assert sum(len(b) for b in batches) == 50


def test_sampler_handles_zero_clean() -> None:
    """Pure cloak dataset: all batches are triplet batches."""
    records = _build_records(n_clean=0, n_pairs=10)
    sampler = CloakAwareBatchSampler(records, batch_size=9, shuffle=False)
    batches = list(sampler)
    total = sum(len(b) for b in batches)
    assert total == 30  # 10 pairs × 3


def test_sampler_shuffle_changes_order() -> None:
    records = _build_records(n_clean=60, n_pairs=10)
    a = list(CloakAwareBatchSampler(records, batch_size=8, shuffle=True, seed=1))
    b = list(CloakAwareBatchSampler(records, batch_size=8, shuffle=True, seed=2))
    assert a != b


def test_sampler_set_epoch_changes_order() -> None:
    records = _build_records(n_clean=60, n_pairs=10)
    s = CloakAwareBatchSampler(records, batch_size=8, shuffle=True, seed=1)
    s.set_epoch(0)
    first = list(s)
    s.set_epoch(1)
    second = list(s)
    assert first != second


def test_sampler_incomplete_triplets_ignored_for_pairing() -> None:
    """If a pair has only 1-2 variants (shouldn't happen in practice but be safe),
    they're treated as singletons (not triplet-grouped)."""
    records = _build_records(n_clean=10, n_pairs=0)
    # Manually add an incomplete pair (only 2 variants)
    records.append({"text": "aa", "metadata": {"cloak_pair_id": 99, "cloak_variant": "base"}})
    records.append({"text": "bb", "metadata": {"cloak_pair_id": 99, "cloak_variant": "homo"}})
    sampler = CloakAwareBatchSampler(records, batch_size=4, shuffle=False)
    seen = []
    for b in sampler:
        seen.extend(b)
    # All 12 samples should still be covered
    assert sorted(seen) == list(range(12))


def test_sampler_len_matches_actual_batches() -> None:
    records = _build_records(n_clean=40, n_pairs=15)
    sampler = CloakAwareBatchSampler(records, batch_size=8, shuffle=False)
    assert len(sampler) == len(list(sampler))
