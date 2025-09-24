#!/usr/bin/env python3

from typing import Callable, Iterable, Iterator, List, Optional, Sequence
import random


class ClassAwareBatchSampler:
    """A generic batch sampler that attempts to include at least one example
    for each target class in every batch, when available.

    This sampler is dataset-agnostic. It relies on a caller-supplied
    `index_to_classset` function that maps an item index to the set of class ids
    present for that item (e.g., window-level class presence).

    Parameters
    ----------
    indices : Sequence[int]
        The pool of dataset indices to draw from (e.g., Subset.indices)
    batch_size : int
        Desired batch size
    target_class_ids : Sequence[int]
        Class ids to cover in each batch when possible
    index_to_classset : Callable[[int], Iterable[int]]
        Function mapping item index -> iterable of class ids present for that item
    seed : int
        RNG seed for determinism
    drop_last : bool
        Drop the last batch if it has fewer than batch_size items
    """

    def __init__(
        self,
        indices: Sequence[int],
        batch_size: int,
        target_class_ids: Sequence[int],
        index_to_classset: Callable[[int], Iterable[int]],
        seed: int = 17,
        drop_last: bool = False,
        min_per_class_per_batch: int = 0,
    ) -> None:
        self.indices: List[int] = list(indices)
        self.batch_size: int = int(batch_size)
        self.target_class_ids: List[int] = list(target_class_ids)
        self.index_to_classset = index_to_classset
        self.seed: int = int(seed)
        self.drop_last: bool = bool(drop_last)
        # Cap the minimum per-class requirement to floor(batch_size / num_target_classes)
        num_targets = max(1, len(self.target_class_ids))
        max_min = self.batch_size // num_targets
        self.min_per_class_per_batch: int = max(0, min(int(min_per_class_per_batch), max_min))

        # Pre-index candidates per class for quick lookup
        self._class_to_indices: dict[int, List[int]] = {c: [] for c in self.target_class_ids}
        for idx in self.indices:
            classes = set(int(c) for c in self.index_to_classset(idx) or [])
            for c in self.target_class_ids:
                if c in classes:
                    self._class_to_indices[c].append(idx)

        # Deterministic shuffles per class list
        rng = random.Random(self.seed)
        for c in self.target_class_ids:
            rng.shuffle(self._class_to_indices[c])

        # Also prepare a global pool order for fill-ins
        self._global_indices: List[int] = list(self.indices)
        rng.shuffle(self._global_indices)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        used_once: set[int] = set()
        cls_ptr: dict[int, int] = {c: 0 for c in self.target_class_ids}
        global_ptr: int = 0

        total_batches = len(self)
        for _ in range(total_batches):
            batch: List[int] = []
            batch_set: set[int] = set()
            covered_counts: dict[int, int] = {c: 0 for c in self.target_class_ids}

            # Per-class minimum coverage phase
            if self.min_per_class_per_batch > 0 and len(self.target_class_ids) > 0:
                for c in self.target_class_ids:
                    needed = self.min_per_class_per_batch - covered_counts[c]
                    if needed <= 0:
                        continue
                    cand_list = self._class_to_indices[c]
                    if not cand_list:
                        continue
                    # First pass: fresh (unused_once) items for this class
                    start = cls_ptr[c]
                    scanned = 0
                    i = 0
                    while needed > 0 and len(batch) < self.batch_size and scanned < len(cand_list):
                        idx = cand_list[(start + i) % len(cand_list)]
                        scanned += 1
                        i += 1
                        if idx in batch_set:
                            continue
                        if idx not in used_once:
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

                    # Second pass: allow recycling (already used_once) but not within-batch duplicates
                    start2 = cls_ptr[c]
                    scanned2 = 0
                    i2 = 0
                    while needed > 0 and len(batch) < self.batch_size and scanned2 < len(cand_list):
                        idx = cand_list[(start2 + i2) % len(cand_list)]
                        scanned2 += 1
                        i2 += 1
                        if idx in batch_set:
                            continue
                        # recycled
                        batch.append(idx)
                        batch_set.add(idx)
                        try:
                            classes_here = set(int(x) for x in self.index_to_classset(idx) or [])
                        except Exception:
                            classes_here = set()
                        for cls in self.target_class_ids:
                            if cls in classes_here:
                                covered_counts[cls] += 1
                        needed = self.min_per_class_per_batch - covered_counts[c]
                    cls_ptr[c] = (start2 + i2) % len(cand_list)

            else:
                # Best-effort single coverage per class without minimum
                for c in self.target_class_ids:
                    cand_list = self._class_to_indices[c]
                    if not cand_list or len(batch) >= self.batch_size:
                        continue
                    start = cls_ptr[c]
                    scanned = 0
                    i = 0
                    picked = None
                    while scanned < len(cand_list):
                        idx = cand_list[(start + i) % len(cand_list)]
                        scanned += 1
                        i += 1
                        if idx in batch_set:
                            continue
                        picked = idx
                        break
                    cls_ptr[c] = (start + i) % len(cand_list)
                    if picked is not None and len(batch) < self.batch_size:
                        batch.append(picked)
                        batch_set.add(picked)
                        if picked not in used_once:
                            used_once.add(picked)
                        try:
                            classes_here = set(int(x) for x in self.index_to_classset(picked) or [])
                        except Exception:
                            classes_here = set()
                        for cls in self.target_class_ids:
                            if cls in classes_here:
                                covered_counts[cls] += 1

            # Fill remaining slots: first fresh global, then recycled
            while len(batch) < self.batch_size and global_ptr < len(self._global_indices):
                idx = self._global_indices[global_ptr]
                global_ptr += 1
                if idx in batch_set:
                    continue
                if idx in used_once:
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


