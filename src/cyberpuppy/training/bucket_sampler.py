"""Length-grouped batch sampler — eliminates pathological-batch OOM
for variable-length sequence training.

Inspired by HuggingFace's LengthGroupedSampler. Within each "mega-batch"
of `batch_size * mega_batch_mult` indices, items are sorted by length so
that adjacent batches contain similar-length sequences. Padding waste
collapses; peak activation memory becomes bounded by the LONGEST sample
in a batch (not the longest sample in the dataset).

Use:
    sampler = LengthBucketSampler(lengths, batch_size=8, shuffle=True, seed=42)
    DataLoader(dataset, batch_sampler=sampler, ...)
"""
from __future__ import annotations

import random
from typing import Iterator, List, Sequence


class LengthBucketSampler:
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        mega_batch_mult: int = 50,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.mega_batch_size = batch_size * max(1, mega_batch_mult)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Optional — called by trainer between epochs to vary shuffle order."""
        self._epoch = epoch

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        n = len(self.lengths)
        rng = random.Random(self.seed + self._epoch)
        indices = list(range(n))
        if self.shuffle:
            rng.shuffle(indices)
        # Carve into mega-batches; sort each by length DESC (longest first
        # within mega keeps padding tight; first batch determines peak mem).
        batches: List[List[int]] = []
        for i in range(0, n, self.mega_batch_size):
            mega = indices[i:i + self.mega_batch_size]
            mega.sort(key=lambda idx: self.lengths[idx], reverse=True)
            for j in range(0, len(mega), self.batch_size):
                batch = mega[j:j + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        # Shuffle order of batches so model sees alternating mem peaks
        if self.shuffle:
            rng.shuffle(batches)
        for b in batches:
            yield b
