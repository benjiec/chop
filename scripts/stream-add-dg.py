#!/usr/bin/env python3

import argparse
from pathlib import Path

from utils.stream import NumericalStream, build_vienna_dg_generator


def main():
    p = argparse.ArgumentParser(description="Add RNA ΔG (ViennaRNA) sliding-window channel to a NumericalStream")
    p.add_argument('--stream', required=True, help='Path to stream pickle (.pkl or .pkl.gz); created if missing')
    p.add_argument('--fna', required=True, help='FASTA/FNA file to compute sequences from')
    p.add_argument('--win', type=int, required=True, help='Sliding window size (centered)')
    p.add_argument('--temp', type=float, default=25.0, help='Temperature in Celsius (default 25)')
    p.add_argument('--mode', choices=['mfe', 'pf'], default='mfe', help='Energy mode: mfe (default) or pf')
    args = p.parse_args()

    stream_path = Path(args.stream)
    if stream_path.exists():
        ns = NumericalStream(str(stream_path))
    else:
        ns = NumericalStream.create_empty(str(stream_path))

    win = int(args.win)
    channel_name = f"dg_win_{win}"
    try:
        dg_fn = build_vienna_dg_generator(win, temp_celsius=float(args.temp), mode=str(args.mode))
    except ImportError as e:
        print("ViennaRNA (RNA) not available:", e)
        print("Install ViennaRNA Python bindings to use this script.")
        return

    ns.add_channel(args.fna, channel_name, dg_fn)
    ns.save()
    print(f"✓ Added channel '{channel_name}' and saved to {stream_path}")


if __name__ == '__main__':
    main()


