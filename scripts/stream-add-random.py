#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np

from utils.stream import NumericalStream


def build_random_generator():
    def randgen(seq: str) -> np.ndarray:
        L = len(seq)
        # Uniform [0,1) per position; seed is intentionally not set per repo rules.
        return np.random.random(size=(L,)).astype(np.float32)
    return randgen


def main():
    p = argparse.ArgumentParser(description="Add a random channel to a NumericalStream")
    p.add_argument('--stream', required=True, help='Path to stream pickle file; created if missing')
    p.add_argument('--fna', required=True, help='FASTA/FNA file to compute sequences from')
    args = p.parse_args()

    stream_path = Path(args.stream)
    if stream_path.exists():
        ns = NumericalStream(str(stream_path))
    else:
        ns = NumericalStream.create_empty(str(stream_path))

    channel_name = "random"
    gen = build_random_generator()
    ns.add_channel(args.fna, channel_name, gen)
    ns.save()
    print(f"✓ Added channel '{channel_name}' and saved to {stream_path}")


if __name__ == '__main__':
    main()


