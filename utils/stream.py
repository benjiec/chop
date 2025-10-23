#!/usr/bin/env python3
from typing import List, Dict, Optional
import numpy as np
import pickle
from pathlib import Path
import gzip


def load_fasta(fasta_path: str) -> Dict[str, str]:
    records: Dict[str, str] = {}
    p = Path(fasta_path)
    is_gz = any(suf == '.gz' for suf in p.suffixes)
    opener = gzip.open if is_gz else open
    mode = 'rt' if is_gz else 'r'
    sid = None
    buf: List[str] = []
    with opener(fasta_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sid is not None:
                    records[sid] = ''.join(buf).upper()
                sid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if sid is not None:
            records[sid] = ''.join(buf).upper()
    return records


def build_gc_generator(window_size: int):
    w = int(window_size)
    if w <= 0:
        raise ValueError("window size must be positive")
    if w % 2 == 1:
        x = y = w // 2
    else:
        y = w // 2
        x = y - 1

    def gc_generator(seq: str) -> np.ndarray:
        L = len(seq)
        seq_u = seq.upper()
        b = np.frombuffer(seq_u.encode('ascii'), dtype=np.uint8)
        gc_mask = ((b == 71) | (b == 67)).astype(np.float32)  # 'G' or 'C'
        pref = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(gc_mask, dtype=np.float32)])
        out = np.zeros(L, dtype=np.float32)
        for i in range(L):
            s = max(0, i - x)
            e = min(L, i + y + 1)
            cnt = pref[e] - pref[s]
            denom = float(e - s) if (e - s) > 0 else 1.0
            out[i] = cnt / denom
        return out

    return gc_generator


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

    @staticmethod
    def create_empty(pickle_path: str) -> "NumericalStream":
        """Create a new empty stream on disk and return the instance.

        The file will contain an empty list to indicate no sequences/channels yet.
        """
        path_obj = Path(pickle_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, 'wb') as f:
            pickle.dump([], f)
        return NumericalStream(str(path_obj))

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
        if not isinstance(items, list):
            raise ValueError("aux stream pickle must be a list of objects")
        if len(items) == 0:
            # empty stream
            self.channels = None
            self._raw_map = {}
            self._norm_map = None
            self._mean = None
            self._std = None
            return
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

    def save(self) -> None:
        """Write the current raw stream to pickle in the standard format."""
        items: List[Dict[str, object]] = []
        channel_list = self.channels if self.channels is not None else []
        for sid in sorted(self._raw_map.keys()):
            arr = self._raw_map[sid]
            items.append({
                'sequence_id': sid,
                'channels': list(channel_list),
                'data': arr.astype(np.float32, copy=False),
            })
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(items, f)

    def add_channel(self, fasta_path: str, channel_name: str, generate_data) -> None:
        """Add a new channel computed from sequences in fasta_path.

        - If stream is empty, sequences from fasta become the stream's sequence_ids and lengths.
        - If stream is non-empty, sequence IDs must match exactly and lengths must match.
        - generate_data(seq_str) -> np.ndarray of shape [L] or [L,].
        """
        if not isinstance(channel_name, str) or len(channel_name) == 0:
            raise ValueError("channel_name must be a non-empty string")
        recs = load_fasta(fasta_path)
        if len(recs) == 0:
            raise ValueError("FASTA contains no sequences")

        if self.channels is None:
            # Initialize empty stream with this single channel
            new_channels = [channel_name]
            new_map: Dict[str, np.ndarray] = {}
            for sid, seq in recs.items():
                arr = np.asarray(generate_data(seq), dtype=np.float32)
                if arr.ndim != 1:
                    raise ValueError(f"generate_data must return 1D array for {sid}")
                if int(arr.shape[0]) != len(seq):
                    raise ValueError(f"length mismatch for {sid}: data={int(arr.shape[0])} seq={len(seq)}")
                new_map[sid] = arr.reshape(-1, 1)
            self.channels = new_channels
            self._raw_map = new_map
            self._norm_map = None
            self._mean = None
            self._std = None
            return

        # Stream non-empty: validate channel uniqueness, sequence IDs and lengths
        if channel_name in self.channels:
            raise ValueError(f"channel '{channel_name}' already exists")
        existing_ids = set(self._raw_map.keys())
        fasta_ids = set(recs.keys())
        if existing_ids != fasta_ids:
            missing = existing_ids - fasta_ids
            extra = fasta_ids - existing_ids
            raise ValueError(f"FASTA sequence IDs must match existing stream; missing={sorted(list(missing))}, extra={sorted(list(extra))}")
        # Append new column per sequence
        for sid in self._raw_map.keys():
            seq = recs[sid]
            arr = np.asarray(generate_data(seq), dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"generate_data must return 1D array for {sid}")
            if int(arr.shape[0]) != len(seq):
                raise ValueError(f"length mismatch for {sid}: data={int(arr.shape[0])} seq={len(seq)}")
            cur = self._raw_map[sid]
            self._raw_map[sid] = np.concatenate([cur, arr.reshape(-1, 1)], axis=1)
        self.channels = list(self.channels) + [channel_name]
        self._norm_map = None
        self._mean = None
        self._std = None
