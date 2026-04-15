"""Cloak-aware batch sampler for v2.2 adversarial training.

Guarantees that all variants of a ToxiCloakCN triplet (base + homo + emoji of
same pair_id) land in the SAME batch, so `consistency_loss` can actually fire.

Non-cloak samples use length-bucket sampling as before (minimizes padding
waste). Cloak triplets get dedicated "triplet batches" where each batch is
composed entirely of complete triplets — batch size rounded to nearest
multiple of 3.

See tests/test_cloak_aware_sampler.py for the contract.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, List, Sequence

from .bucket_sampler import LengthBucketSampler


class CloakAwareBatchSampler:
    def __init__(
        self,
        records: Sequence[dict],
        batch_size: int,
        mega_batch_mult: int = 50,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        # Partition: complete triplets (pair_id shared by all 3 variants) vs singletons
        pair_map: dict[int, list[int]] = defaultdict(list)
        clean_indices: list[int] = []
        clean_lengths: list[int] = []
        for i, r in enumerate(records):
            pid = r.get("metadata", {}).get("cloak_pair_id", -1)
            if pid is not None and pid >= 0:
                pair_map[pid].append(i)
            else:
                clean_indices.append(i)
                clean_lengths.append(len(r.get("text", "")))

        # Only complete triplets get paired; incomplete pairs go to clean.
        self.triplets: list[list[int]] = []
        for pid, idxs in pair_map.items():
            if len(idxs) == 3:
                self.triplets.append(sorted(idxs))
            else:
                for i in idxs:
                    clean_indices.append(i)
                    clean_lengths.append(len(records[i].get("text", "")))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Round down to a multiple of 3 so triplet-batches are always complete.
        self.triplets_per_batch = max(1, batch_size // 3)

        # Reusable sampler for the clean portion.
        self.clean_indices = clean_indices
        self._clean_sampler = LengthBucketSampler(
            clean_lengths,
            batch_size=batch_size,
            mega_batch_mult=mega_batch_mult,
            shuffle=shuffle,
            seed=seed,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._clean_sampler.set_epoch(epoch)

    def __len__(self) -> int:
        n_triplet_batches = (
            (len(self.triplets) + self.triplets_per_batch - 1) // self.triplets_per_batch
            if self.triplets else 0
        )
        return len(self._clean_sampler) + n_triplet_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)

        # Triplet batches: each batch = K complete triplets (K*3 samples).
        triplet_order = list(range(len(self.triplets)))
        if self.shuffle:
            rng.shuffle(triplet_order)
        triplet_batches: list[list[int]] = []
        for i in range(0, len(triplet_order), self.triplets_per_batch):
            batch: list[int] = []
            for j in triplet_order[i:i + self.triplets_per_batch]:
                batch.extend(self.triplets[j])
            if batch:
                triplet_batches.append(batch)

        # Clean batches via length-bucket sampler, mapped back to absolute indices.
        clean_batches: list[list[int]] = []
        for rel_batch in self._clean_sampler:
            clean_batches.append([self.clean_indices[r] for r in rel_batch])

        # Interleave so gradient updates alternate between adversarial pressure
        # and clean distribution — avoids catastrophic forgetting of either.
        all_batches = triplet_batches + clean_batches
        if self.shuffle:
            rng.shuffle(all_batches)

        for b in all_batches:
            yield b
