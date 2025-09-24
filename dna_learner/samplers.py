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
    ) -> None:
        self.indices: List[int] = list(indices)
        self.batch_size: int = int(batch_size)
        self.target_class_ids: List[int] = list(target_class_ids)
        self.index_to_classset = index_to_classset
        self.seed: int = int(seed)
        self.drop_last: bool = bool(drop_last)

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
        used: set[int] = set()
        # Pointers per class for round-robin
        cls_ptr: dict[int, int] = {c: 0 for c in self.target_class_ids}
        global_ptr: int = 0

        remaining = len(self.indices)
        while remaining > 0:
            batch: List[int] = []

            # First, try to cover each target class with one sample
            for c in self.target_class_ids:
                cand_list = self._class_to_indices[c]
                # advance pointer until we find an unused candidate or exhaust
                while cls_ptr[c] < len(cand_list) and cand_list[cls_ptr[c]] in used:
                    cls_ptr[c] += 1
                if cls_ptr[c] < len(cand_list):
                    idx = cand_list[cls_ptr[c]]
                    batch.append(idx)
                    used.add(idx)
                    cls_ptr[c] += 1
                    remaining -= 1
                # else: no candidate left for this class; skip

            # Fill remaining slots from the global pool
            while len(batch) < self.batch_size and remaining > 0:
                # advance global pointer
                while global_ptr < len(self._global_indices) and self._global_indices[global_ptr] in used:
                    global_ptr += 1
                if global_ptr >= len(self._global_indices):
                    break
                idx = self._global_indices[global_ptr]
                batch.append(idx)
                used.add(idx)
                global_ptr += 1
                remaining -= 1

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


