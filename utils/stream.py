#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import numpy as np
import pickle


@dataclass
class NumericalStreamSequence:
    sequence_id: str
    channels: List[str]
    data: np.ndarray  # shape [L, C], float32


class NumericalStream:
    """Loader and normalizer for per-sequence numerical streams.

    Expects a pickle file containing a list of objects with fields:
      - sequence_id: str
      - channels: List[str]
      - data: np.ndarray [L, C]
    """

    def __init__(self, pickle_path: str):
        self.pickle_path = str(pickle_path)
        self.channels: Optional[List[str]] = None
        self._raw_map: Dict[str, np.ndarray] = {}
        self._norm_map: Optional[Dict[str, np.ndarray]] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._load()

    @property
    def mean(self) -> Optional[np.ndarray]:
        return self._mean

    @property
    def std(self) -> Optional[np.ndarray]:
        return self._std

    @property
    def sequence_ids(self) -> List[str]:
        return list(self._raw_map.keys())

    def _load(self) -> None:
        with open(self.pickle_path, 'rb') as f:
            items = pickle.load(f)
        if not isinstance(items, list) or len(items) == 0:
            raise ValueError("aux stream pickle must be a non-empty list of objects")
        channel_list: Optional[List[str]] = None
        raw: Dict[str, np.ndarray] = {}
        for it in items:
            if isinstance(it, dict):
                seq_id = it.get('sequence_id')
                channels = it.get('channels')
                data = it.get('data')
            else:
                seq_id = getattr(it, 'sequence_id', None)
                channels = getattr(it, 'channels', None)
                data = getattr(it, 'data', None)
            if not isinstance(seq_id, str):
                raise ValueError("aux stream item missing sequence_id")
            if not isinstance(channels, (list, tuple)) or len(channels) == 0:
                raise ValueError(f"aux stream channels invalid for {seq_id}")
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                raise ValueError(f"aux stream data must be 2D ndarray for {seq_id}")
            if channel_list is None:
                channel_list = list(channels)
            elif list(channels) != channel_list:
                raise ValueError("aux stream channels mismatch across items")
            arr = np.array(data, dtype=np.float32)
            # sanitize NaN/inf
            arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32, copy=False)
            raw[seq_id] = arr
        self.channels = channel_list
        self._raw_map = raw
        self._norm_map = None
        self._mean = None
        self._std = None

    def normalize(self) -> None:
        """Compute per-channel z-score across all sequences and create normalized map.

        Padding is managed by the caller; statistics are computed on raw (unpadded) data.
        """
        if self.channels is None:
            raise ValueError("channels not loaded")
        C = int(len(self.channels))
        if C == 0:
            self._norm_map = {k: v.copy() for k, v in self._raw_map.items()}
            self._mean = None
            self._std = None
            return
        cat = np.concatenate(list(self._raw_map.values()), axis=0) if len(self._raw_map) > 0 else np.zeros((0, C), dtype=np.float32)
        mean = (cat.mean(axis=0) if cat.size > 0 else np.zeros((C,), dtype=np.float32))
        std = (cat.std(axis=0) if cat.size > 0 else np.ones((C,), dtype=np.float32))
        std = np.where(std <= 1e-8, 1.0, std).astype(np.float32)
        self._mean = mean.astype(np.float32)
        self._std = std.astype(np.float32)
        normed: Dict[str, np.ndarray] = {}
        for sid, arr in self._raw_map.items():
            normed[sid] = ((arr - self._mean) / self._std).astype(np.float32, copy=False)
        self._norm_map = normed

    def get(self, sequence_id: str) -> np.ndarray:
        """Return normalized data if available, else raw; shape [L,C]."""
        if self._norm_map is not None and sequence_id in self._norm_map:
            return self._norm_map[sequence_id]
        if sequence_id in self._raw_map:
            return self._raw_map[sequence_id]
        raise KeyError(f"sequence_id not found in NumericalStream: {sequence_id}")

    def pad(self, sequence_id: str, prefix_len: int) -> np.ndarray:
        """Return data with left zero-padding of length prefix_len added.

        Uses normalized data if available, otherwise raw.
        """
        base = self.get(sequence_id)
        if prefix_len <= 0:
            return base.astype(np.float32, copy=True)
        C = int(base.shape[1])
        return np.concatenate([np.zeros((prefix_len, C), dtype=np.float32), base], axis=0)


