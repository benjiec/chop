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


def build_vienna_dg_generator(window_size: int, temp_celsius: float = 25.0, mode: str = 'mfe'):
    """Return a generator that computes per-position RNA ΔG using ViennaRNA.

    - window_size: centered window size N. If even, uses X=Y-1 per GC rule.
    - temp_celsius: folding temperature in Celsius (default 25).
    - mode: 'mfe' (minimum free energy) or 'pf' (ensemble free energy).
    """
    w = int(window_size)
    if w <= 0:
        raise ValueError("window size must be positive")
    mode = str(mode).lower()
    if mode not in ('mfe', 'pf'):
        raise ValueError("mode must be 'mfe' or 'pf'")
    if w % 2 == 1:
        x = y = w // 2
    else:
        y = w // 2
        x = y - 1

    # Lazy import to avoid hard dependency at module import time
    try:
        import RNA  # type: ignore
    except Exception as e:  # pragma: no cover - exercised in script/skip test
        raise ImportError("ViennaRNA Python bindings 'RNA' are required for build_vienna_dg_generator") from e

    def dg_generator(seq: str) -> np.ndarray:
        L = len(seq)
        # Convert DNA to RNA for ViennaRNA (T->U). Replace unknowns with A.
        seq_u = seq.upper().replace('T', 'U').replace('N', 'A')
        md = RNA.md()
        md.temperature = float(temp_celsius)
        out = np.zeros(L, dtype=np.float32)
        for i in range(L):
            s = max(0, i - x)
            e = min(L, i + y + 1)
            win_seq = seq_u[s:e]
            fc = RNA.fold_compound(win_seq, md)
            if mode == 'mfe':
                _, energy = fc.mfe()
                val = float(energy)
            else:
                # Partition function; extract ensemble free energy robustly across API variants
                pf_ret = fc.pf()
                val = None
                if isinstance(pf_ret, (list, tuple)):
                    # Prefer index 1 when numeric (common convention), fallback to last/first numeric
                    cand = None
                    if len(pf_ret) >= 2 and isinstance(pf_ret[1], (float, int, np.floating)):
                        cand = pf_ret[1]
                    else:
                        # search last numeric, then first numeric
                        for item in reversed(pf_ret):
                            if isinstance(item, (float, int, np.floating)):
                                cand = item
                                break
                        if cand is None:
                            for item in pf_ret:
                                if isinstance(item, (float, int, np.floating)):
                                    cand = item
                                    break
                    if cand is None:
                        raise TypeError("ViennaRNA pf() did not return a numeric free energy component")
                    val = float(cand)
                else:
                    val = float(pf_ret)
            out[i] = val
        return out

    return dg_generator


# Turner nearest-neighbor stacking free energies at ~37°C (kcal/mol) for dinucleotide steps (proxy values)
# Keyed by DNA dinucleotide in 5'->3' (as bytes). More negative indicates greater stability.
Turner2004NNdeltaG37C = {
    b'AA': -0.9,
    b'AT': -1.1,  # AU/UA
    b'AC': -2.1,  # AC/UG
    b'AG': -2.1,  # AG/UC
    b'TA': -1.3,  # UA/AU
    b'TT': -0.9,  # UU/AA
    b'TC': -2.4,  # UC/AG
    b'TG': -2.1,  # UG/AC
    b'CA': -1.4,  # CA/GU
    b'CT': -2.1,  # CU/GA
    b'CC': -2.0,  # CC/GG
    b'CG': -3.4,  # CG/GC
    b'GA': -2.4,  # GA/CU
    b'GT': -1.5,  # GU/CA
    b'GC': -3.3,  # GC/CG
    b'GG': -2.0,  # GG/CC
}

# RNA nearest-neighbor enthalpy (kcal/mol) and entropy (cal/mol·K) from Xia et al. 1998 (U mapped to T)
Turner2004NN_dH = {
    b'AA': -7.6,  # AA/UU
    b'AT': -7.8,  # AU/UA
    b'AC': -8.1,  # AC/UG
    b'AG': -10.4, # AG/UC
    b'TA': -7.7,  # UA/AU
    b'TT': -7.6,  # UU/AA
    b'TC': -10.4, # UC/AG
    b'TG': -8.1,  # UG/AC
    b'CA': -8.4,  # CA/GU
    b'CT': -10.4, # CU/GA
    b'CC': -8.0,  # CC/GG
    b'CG': -10.6, # CG/GC
    b'GA': -10.5, # GA/CU
    b'GT': -8.2,  # GU/CA
    b'GC': -14.2, # GC/CG
    b'GG': -8.0,  # GG/CC
}

Turner2004NN_dS = {
    b'AA': -21.3,
    b'AT': -21.9,
    b'AC': -22.6,
    b'AG': -26.9,
    b'TA': -21.2,
    b'TT': -21.3,
    b'TC': -26.9,
    b'TG': -22.6,
    b'CA': -22.4,
    b'CT': -26.9,
    b'CC': -19.9,
    b'CG': -27.2,
    b'GA': -27.8,
    b'GT': -22.2,
    b'GC': -34.9,
    b'GG': -19.9,
}


def build_dinuc_generator(window_size: int, mode: str = 'count', temp_celsius: float = 25.0):
    """Return a generator computing per-position dinucleotide-based stability proxy.

    Modes:
      - 'count': fraction of strong-stacking dinucleotides in the window (GG, GC, CG, CC).
      - 'weighted': frequency-weighted sum using approximate stacking free energies (kcal/mol).

    Output is float32 array length L. 'count' yields values in [0,1]; 'weighted' yields negative numbers for stronger structure.
    """
    w = int(window_size)
    if w <= 0:
        raise ValueError("window size must be positive")
    if w % 2 == 1:
        x = y = w // 2
    else:
        y = w // 2
        x = y - 1

    strong_set = {b'GG', b'GC', b'CG', b'CC'}
    # Approximate stacking free energies (proxy, kcal/mol) per dinucleotide step; more negative = more stable
    # Values are illustrative; if precise values are needed, this can be overridden in a future extension.
    def dinuc_generator(seq: str) -> np.ndarray:
        L = len(seq)
        seq_u = seq.upper().replace('N', 'A')
        b = np.frombuffer(seq_u.encode('ascii'), dtype=np.uint8)
        out = np.zeros(L, dtype=np.float32)
        T = 273.15 + float(temp_celsius)
        for i in range(L):
            s = max(0, i - x)
            e = min(L, i + y + 1)
            total_pairs = max(0, e - s - 1)
            if total_pairs <= 0:
                out[i] = 0.0
                continue
            window = b[s:e]
            # Count all dinucleotides in window (overlapping)
            if mode == 'count':
                cnt = 0
                for j in range(len(window) - 1):
                    din = bytes(window[j:j+2])
                    if din in strong_set:
                        cnt += 1
                out[i] = float(cnt) / float(total_pairs)
            elif mode == 'weighted':
                acc = 0.0
                for j in range(len(window) - 1):
                    din = bytes(window[j:j+2])
                    wv = Turner2004NNdeltaG37C.get(din, -1.0)
                    acc += float(wv)
                # convert to frequency-weighted sum per window
                out[i] = float(acc) / float(total_pairs)
            elif mode == 'dg_at_temp':
                # Compute average ΔG at a given temperature using ΔH and ΔS
                acc = 0.0
                for j in range(len(window) - 1):
                    din = bytes(window[j:j+2])
                    dH = Turner2004NN_dH.get(din)
                    dS = Turner2004NN_dS.get(din)
                    if dH is None or dS is None:
                        # fallback to 37C table if missing
                        acc += float(Turner2004NNdeltaG37C.get(din, -1.0))
                    else:
                        g = float(dH) - T * (float(dS) / 1000.0)
                        acc += g
                out[i] = float(acc) / float(total_pairs)
            else:
                raise ValueError("invalid mode for dinuc generator")
        return out

    return dinuc_generator


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
        is_gz = any(suf == '.gz' for suf in path_obj.suffixes)
        opener = gzip.open if is_gz else open
        with opener(path_obj, 'wb') as f:
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
        p = Path(self.pickle_path)
        is_gz = any(suf == '.gz' for suf in p.suffixes)
        opener = gzip.open if is_gz else open
        with opener(self.pickle_path, 'rb') as f:
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
        p = Path(self.pickle_path)
        is_gz = any(suf == '.gz' for suf in p.suffixes)
        opener = gzip.open if is_gz else open
        with opener(self.pickle_path, 'wb') as f:
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
