#!/usr/bin/env python3

import argparse
import csv
import sys
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pivot epochs.csv (wide) into tall format: epoch,batch,metric,value"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input epochs.csv",
    )
    parser.add_argument(
        "--output",
        required=False,
        default="-",
        help="Path to output CSV (default: stdout)",
    )
    return parser.parse_args()


def detect_metric_columns(headers: List[str]) -> List[Tuple[str, str]]:
    """
    Return list of (batch_prefix, header) for headers starting with val_ or train_.
    batch_prefix will be either 'val' or 'train'.
    """
    metric_columns: List[Tuple[str, str]] = []
    for h in headers:
        if h.startswith("val_"):
            metric_columns.append(("val", h))
        elif h.startswith("train_"):
            metric_columns.append(("train", h))
    return metric_columns


def strip_prefix(batch: str, header: str) -> str:
    prefix = f"{batch}_"
    if header.startswith(prefix):
        return header[len(prefix) :]
    return header


def pivot_epochs_csv(in_path: str, out_path: str) -> None:
    with open(in_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if "epoch" not in headers:
            raise ValueError("Input CSV must include an 'epoch' column")

        metric_columns = detect_metric_columns(headers)

        out_file = sys.stdout if out_path == "-" else open(out_path, "w", newline="")
        try:
            writer = csv.writer(out_file)
            writer.writerow(["epoch", "batch", "metric", "value"])

            for row in reader:
                epoch = row["epoch"]
                for batch, header in metric_columns:
                    metric = strip_prefix(batch, header)
                    value = row.get(header, "")
                    if value is None or value == "":
                        continue
                    writer.writerow([epoch, batch, metric, value])
        finally:
            if out_file is not sys.stdout:
                out_file.close()


def main() -> None:
    args = parse_args()
    pivot_epochs_csv(args.input, args.output)


if __name__ == "__main__":
    main()


