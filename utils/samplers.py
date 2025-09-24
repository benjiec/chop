#!/usr/bin/env python3

from typing import Callable, Iterable, Iterator, List, Optional, Sequence
import random


class ClassAwareBatchSampler:
    """A generic batch sampler that attempts to include at least one example
    for each target class in every batch, when available.

    Dataset-agnostic: relies on `index_to_classset(index) -> Iterable[int]`.
    Recycling is allowed to meet per-class minimums.
    """

    def __init__(
        self,
        indices: Sequence[int],
        batch_size: int,
        target_class_ids: Sequence[int],
        index_to_classset: Callable[[int], Iterable[int]],
        seed: int = 17,
        drop_last: bool = False,
        min_per_class_per_batch: int = 1,
    ) -> None:
        self.indices: List[int] = list(indices)
        self.batch_size: int = int(batch_size)
        self.target_class_ids: List[int] = list(target_class_ids)
        self.index_to_classset = index_to_classset
        self.seed: int = int(seed)
        self.drop_last: bool = bool(drop_last)
        self.min_per_class_per_batch: int = max(1, int(min_per_class_per_batch or 1))

        # Pre-index candidates per class
        self._class_to_indices: dict[int, List[int]] = {c: [] for c in self.target_class_ids}
        for idx in self.indices:
            classes = set(int(c) for c in self.index_to_classset(idx) or [])
            for c in self.target_class_ids:
                if c in classes:
                    self._class_to_indices[c].append(idx)

        # Deterministic order per class and global pool
        rng = random.Random(self.seed)
        for c in self.target_class_ids:
            rng.shuffle(self._class_to_indices[c])
        self._global_indices: List[int] = list(self.indices)
        rng.shuffle(self._global_indices)

    def __iter__(self) -> Iterator[List[int]]:
        used_once: set[int] = set()
        cls_ptr: dict[int, int] = {c: 0 for c in self.target_class_ids}
        global_ptr: int = 0

        total_batches = len(self)
        for _ in range(total_batches):
            batch: List[int] = []
            batch_set: set[int] = set()
            covered_counts: dict[int, int] = {c: 0 for c in self.target_class_ids}

            # Per-class minimum coverage with recycle fallback
            for c in self.target_class_ids:
                cand_list = self._class_to_indices[c]
                if not cand_list:
                    continue
                start = cls_ptr[c]
                scanned = 0
                i = 0
                needed = self.min_per_class_per_batch - covered_counts[c]
                limit = 2 * len(cand_list)
                while needed > 0 and len(batch) < self.batch_size and scanned < limit:
                    idx = cand_list[(start + i) % len(cand_list)]
                    scanned += 1
                    i += 1
                    if idx in batch_set:
                        continue
                    allow_recycle = scanned > len(cand_list)
                    if (idx not in used_once) or allow_recycle:
                        batch.append(idx)
                        batch_set.add(idx)
                        used_once.add(idx)
                        try:
                            classes_here = set(int(x) for x in self.index_to_classset(idx) or [])
                        except Exception:
                            classes_here = set()
                        for cls in self.target_class_ids:
                            if cls in classes_here:
                                covered_counts[cls] += 1
                        needed = self.min_per_class_per_batch - covered_counts[c]
                cls_ptr[c] = (start + i) % len(cand_list)

            # Fill remaining slots: first fresh, then recycled
            while len(batch) < self.batch_size and global_ptr < len(self._global_indices):
                idx = self._global_indices[global_ptr]
                global_ptr += 1
                if idx in batch_set or idx in used_once:
                    continue
                batch.append(idx)
                batch_set.add(idx)
                used_once.add(idx)

            if len(batch) < self.batch_size:
                for idx in self._global_indices:
                    if len(batch) >= self.batch_size:
                        break
                    if idx in batch_set:
                        continue
                    batch.append(idx)
                    batch_set.add(idx)

            if len(batch) == 0:
                break
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            from math import ceil
            return ceil(len(self.indices) / max(1, self.batch_size))


