#!/usr/bin/env python3

import argparse
from pathlib import Path

from utils.stream import NumericalStream, build_dinuc_generator


def main():
    p = argparse.ArgumentParser(description='Add dinucleotide-based stability proxy channel to a NumericalStream')
    p.add_argument('--fna-fn', required=True, help='FASTA file with sequences (can be .gz)')
    p.add_argument('--stream-path', required=True, help='Path to aux stream pickle (.pkl or .pkl.gz) to create or update')
    p.add_argument('--window-size', type=int, default=50, help='Centered window size (e.g., 50)')
    p.add_argument('--mode', choices=['count', 'weighted'], default='count', help='count: fraction of strong dinucs; weighted: frequency-weighted Turner-like energies')
    p.add_argument('--channel-name', type=str, default=None, help='Optional custom channel name')
    args = p.parse_args()

    # Load or create stream
    sp = Path(args.stream_path)
    if not sp.exists():
        ns = NumericalStream.create_empty(str(sp))
    else:
        ns = NumericalStream(str(sp))

    gen = build_dinuc_generator(window_size=int(args.window_size), mode=str(args.mode))

    if args.channel_name:
        ch_name = args.channel_name
    else:
        suffix = f"w{int(args.window_size)}"
        ch_name = f"dinuc_{args.mode}_{suffix}"

    ns.add_channel(str(args.fna_fn), ch_name, gen)
    ns.save()
    print(f"Added channel '{ch_name}' to stream: {sp}")


if __name__ == '__main__':
    main()


