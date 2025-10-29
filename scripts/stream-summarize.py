#!/usr/bin/env python3

import argparse
from pathlib import Path

from utils.stream import NumericalStream


def main():
    p = argparse.ArgumentParser(description="Summarize a NumericalStream pickle (.pkl or .pkl.gz)")
    p.add_argument('--stream', required=True, help='Path to stream file')
    args = p.parse_args()

    sp = Path(args.stream)
    if not sp.exists():
        print(f"Stream not found: {sp}")
        return

    ns = NumericalStream(str(sp))
    channels = ns.channels or []
    seq_ids = ns.sequence_ids

    # Validate per-sequence channel dimension matches
    mismatches = []
    for sid in seq_ids:
        arr = ns.get(sid)
        if arr.ndim != 2 or (len(channels) and int(arr.shape[1]) != len(channels)):
            mismatches.append((sid, arr.shape))

    print(f"Stream: {sp}")
    print(f"Num channels: {len(channels)}")
    if channels:
        print(f"Channels: {', '.join(channels)}")
    print(f"Num sequences: {len(seq_ids)}")
    if mismatches:
        print("WARNING: Channel dimension mismatches detected:")
        for sid, shape in mismatches:
            print(f"  {sid}: shape={shape}, expected channel dim={len(channels)}")

    for sid in seq_ids[0:10]:
        print(sid)
        arr = ns.get(sid)
        print(arr[0:25,])

if __name__ == '__main__':
    main()


