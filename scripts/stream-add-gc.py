#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np

from utils.stream import NumericalStream, build_gc_generator


def main():
    p = argparse.ArgumentParser(description="Add GC sliding-window channel to a NumericalStream")
    p.add_argument('--stream', required=True, help='Path to stream pickle file; created if missing')
    p.add_argument('--fna', required=True, help='FASTA/FNA file to compute sequences from')
    p.add_argument('--win', type=int, required=True, help='Sliding window size (centered)')
    args = p.parse_args()

    stream_path = Path(args.stream)
    if stream_path.exists():
        ns = NumericalStream(str(stream_path))
    else:
        ns = NumericalStream.create_empty(str(stream_path))

    win = int(args.win)
    channel_name = f"gc_win_{win}"
    gc_fn = build_gc_generator(win)
    ns.add_channel(args.fna, channel_name, gc_fn)
    ns.save()
    print(f"✓ Added channel '{channel_name}' and saved to {stream_path}")


if __name__ == '__main__':
    main()


