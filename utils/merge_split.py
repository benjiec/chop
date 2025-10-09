#!/usr/bin/env python3

from __future__ import annotations

from typing import Dict, List, Tuple, Iterable
from pathlib import Path
import csv
import random

# Reuse existing FASTA/FNA loader to avoid duplication and to support .gz
from utils.genome import _load_fasta


REQUIRED_TSV_COLUMNS: List[str] = [
    'sequence_id', 'gene_id', 'gene_start', 'gene_end', 'exon_start', 'exon_end', 'strand'
]


def _index_for_column(header: List[str], name: str) -> int:
    for i, h in enumerate(header):
        if h.lower() == name.lower():
            return i
    raise ValueError(f"Missing column '{name}' in TSV header: {header}")


def load_all_fasta(fasta_paths: Iterable[str]) -> Dict[str, str]:
    """Load and merge multiple FASTA/FNA files into a single sequence map.

    If a sequence_id appears multiple times with different sequences, raise ValueError.
    """
    merged: Dict[str, str] = {}
    for p in fasta_paths:
        records = _load_fasta(str(p))
        for sid, seq in records.items():
            if sid in merged:
                if merged[sid] != seq:
                    raise ValueError(f"Conflicting sequences for id '{sid}' across FASTA inputs")
                continue
            merged[sid] = seq
    return merged


def load_all_tsv(tsv_paths: Iterable[str]) -> Tuple[List[str], List[List[str]]]:
    """Load and concatenate row-per-exon TSVs.

    Returns a (header, rows) tuple. Validates required columns are present.
    """
    header: List[str] = []
    rows: List[List[str]] = []
    for tsv_path in tsv_paths:
        with open(tsv_path, 'r') as f:
            local_header = f.readline().rstrip('\n').split('\t')
            # Initialize common header or validate compatibility
            if not header:
                header = local_header
                # Ensure required columns exist
                for col in REQUIRED_TSV_COLUMNS:
                    _index_for_column(header, col)
            else:
                # We allow additional/superset columns but require the required ones to be present
                for col in REQUIRED_TSV_COLUMNS:
                    _index_for_column(local_header, col)
            for line in f:
                if not line:
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) < len(header):
                    # If some files have fewer columns, we still accept if they cover required columns;
                    # pad trailing empties to align with header length for consistent writing
                    parts = parts + [''] * (len(header) - len(parts))
                rows.append(parts)
    if not header:
        # If no TSVs or empty files, return default header to allow writing empty outputs
        header = list(REQUIRED_TSV_COLUMNS)
    return header, rows


def group_rows_by_sequence(rows: List[List[str]], header: List[str]) -> Dict[str, List[List[str]]]:
    """Group TSV rows by sequence_id column."""
    sid_idx = _index_for_column(header, 'sequence_id')
    grouped: Dict[str, List[List[str]]] = {}
    for r in rows:
        if len(r) <= sid_idx:
            continue
        sid = r[sid_idx]
        if not sid:
            continue
        grouped.setdefault(sid, []).append(r)
    return grouped


def pick_splits(
    available_sequence_ids: List[str],
    num_train: int,
    num_test: int,
) -> Tuple[List[str], List[str]]:
    """Shuffle sequence ids and select disjoint train and test subsets.

    Raises ValueError if there are not enough sequence ids to satisfy the counts.
    """
    if num_train < 0 or num_test < 0:
        raise ValueError("num_train and num_test must be non-negative")
    seq_ids = list(available_sequence_ids)
    random.shuffle(seq_ids)
    if len(seq_ids) < (num_train + num_test):
        raise ValueError(
            f"Requested num_train+num_test={num_train + num_test} exceeds available sequences={len(seq_ids)}"
        )
    train_ids = seq_ids[:num_train]
    test_ids = seq_ids[num_train:num_train + num_test]
    return train_ids, test_ids


def write_fasta(sequences: Dict[str, str], sequence_ids: Iterable[str], output_path: str) -> None:
    """Write a FASTA file containing the specified sequence ids."""
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w') as f:
        for sid in sequence_ids:
            seq = sequences.get(sid)
            if seq is None:
                # Skip missing to be defensive; caller should have intersected sets already
                continue
            f.write(f">{sid}\n")
            f.write(f"{seq}\n")


def write_tsv(header: List[str], rows: Iterable[List[str]], output_path: str) -> None:
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


def merge_and_split(
    tsv_inputs: Iterable[str],
    fasta_inputs: Iterable[str],
    num_train: int,
    num_test: int,
) -> Tuple[
    Dict[str, str],
    List[str],
    List[str],
    List[str],
    List[str],
]:
    """Merge inputs and compute train/test sequence id lists and filtered TSV rows.

    Returns tuple: (all_sequences, header, train_rows, test_rows, all_valid_sequence_ids)
    """
    sequences = load_all_fasta(fasta_inputs)
    header, all_rows = load_all_tsv(tsv_inputs)
    rows_by_sid = group_rows_by_sequence(all_rows, header)
    valid_sids = sorted(set(rows_by_sid.keys()) & set(sequences.keys()))
    train_ids, test_ids = pick_splits(valid_sids, num_train=num_train, num_test=num_test)
    # Build row lists in original order
    train_sid_set = set(train_ids)
    test_sid_set = set(test_ids)
    train_rows: List[List[str]] = []
    test_rows: List[List[str]] = []
    for r in all_rows:
        sid = r[_index_for_column(header, 'sequence_id')]
        if sid in train_sid_set:
            train_rows.append(r)
        elif sid in test_sid_set:
            test_rows.append(r)
    return sequences, header, train_rows, test_rows, valid_sids



