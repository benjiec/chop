#!/usr/bin/env python3

import unittest
import tempfile
from pathlib import Path
import gzip

import numpy as np

from utils.genome import AnnotatedGenomeDataset, build_class_windows
from utils.constants import GenePredictionClass as P


def write_temp(path: Path, content: str):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'wt') as f:
            f.write(content)
    else:
        with open(path, 'w') as f:
            f.write(content)


def build_single_exon_contig(name: str, total_len: int, start_pos0: int, stop_pos0: int) -> str:
    # Build a sequence of Ns with ATG at start_pos0 and TAA at stop_pos0
    seq = list('N' * total_len)
    seq[start_pos0:start_pos0+3] = list('ATG')
    seq[stop_pos0:stop_pos0+3] = list('TAA')
    return f">{name}\n{''.join(seq)}\n"


class TestAnnotatedGenomeDatasetSampling(unittest.TestCase):
    def _make_two_contigs_dataset(self, window: int = 16, stride: int = 8,
                                  class_weights=None, num_windows=None, seed: int = 17,
                                  window_incl_classes=None, gene_class=P.GENE,
                                  exclude_margin_bps: int = 200):
        # Two contigs; exon spans create windows containing START-only, GENE-only, STOP-only, and combos
        total_len = 200
        # Contig c1: exon ~ [20, 73) with START at 20..22 and STOP at 70..72
        fasta_c1 = build_single_exon_contig('c1', total_len, 20, 70)
        # Contig c2: exon ~ [100, 153)
        fasta_c2 = build_single_exon_contig('c2', total_len, 100, 150)
        fasta = fasta_c1 + fasta_c2
        # TSV rows (1-based inclusive starts/ends). exon_end is inclusive; parser converts to exclusive
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            f"c1\tg1\t21\t73\t21\t73\t+\n"
            f"c2\tg2\t101\t153\t101\t153\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / 'seq.fna.gz'
            tsv_path = fp / 'ann.tsv'
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)
            ds = AnnotatedGenomeDataset(
                str(fasta_path), str(tsv_path), window=window, stride=stride,
                num_windows=num_windows, class_weights=class_weights, seed=seed,
                window_incl_classes=window_incl_classes,
                gene_class=gene_class,
                exclude_margin_bps=exclude_margin_bps,
                random_prefix_ns=False
            )
            return ds

    # classset_to_contigs accounting removed; no longer tested

    def test_excludes_weight_one_classes_from_classsets(self):
        # This assertion relied on classset accounting which has been removed; keep behavior check minimal
        cw = [1.0] * 8
        cw[P.START] = 2.0
        ds = self._make_two_contigs_dataset(class_weights=cw)
        self.assertGreater(len(ds.windows), 0)

    def test_num_windows_none_keeps_all(self):
        ds = self._make_two_contigs_dataset(num_windows=None)
        # Manual recompute total windows
        total = len(ds.windows)
        self.assertEqual(len(ds), total)

    def test_num_windows_geq_total_keeps_all(self):
        ds = self._make_two_contigs_dataset(num_windows=10_000)
        self.assertEqual(len(ds), len(ds.windows))

    def test_balanced_sampling_roughly_equal_counts(self):
        # Balance START and STOP (ignore INTERGENIC)
        cw = [1.0] * 8
        cw[P.START] = 2.0
        cw[P.STOP] = 2.0
        ds = self._make_two_contigs_dataset(class_weights=cw, num_windows=12, seed=123)
        # Compute per-class counts across selected windows using helper mapping
        sel = ds._selected_window_indices
        self.assertIsNotNone(sel)
        cw_map = build_class_windows(ds.windows, ds.targets, [P.START, P.STOP], exclude_margin_bps=ds._exclude_margin_bps, class_weights=ds.class_weights)
        class_counts = {P.START: 0, P.STOP: 0}
        for wi in sel:
            for c in class_counts.keys():
                if wi in cw_map.get(c, []):
                    class_counts[c] += 1
        diff = abs(class_counts[P.START] - class_counts[P.STOP])
        self.assertLessEqual(diff, 2, f"Unbalanced selection: {class_counts}")

    def test_includes_only_classes_specified(self):
        # Exclude everything except START
        cw = [1.0] * 8
        cw[P.START] = 2.0
        cw[P.STOP] = 2.0
        ds = self._make_two_contigs_dataset(class_weights=cw, window_incl_classes=[P.START], num_windows=2)
        sel = ds._selected_window_indices
        self.assertIsNotNone(sel)
        cw_map = build_class_windows(ds.windows, ds.targets, [P.START], exclude_margin_bps=ds._exclude_margin_bps, class_weights=ds.class_weights)
        start_set = set(cw_map.get(P.START, []))
        # All selected windows must be START-assigned
        self.assertTrue(all(wi in start_set for wi in sel))

    def test_reproducibility_with_seed(self):
        cw = [1.0] * 8
        cw[P.START] = 2.0
        cw[P.STOP] = 2.0
        ds1 = self._make_two_contigs_dataset(class_weights=cw, num_windows=10, seed=42)
        ds2 = self._make_two_contigs_dataset(class_weights=cw, num_windows=10, seed=42)
        self.assertEqual(ds1._selected_window_indices, ds2._selected_window_indices)

    def test_excludes_edge_only_weighted_windows(self):
        # Build a small contig of length 32 with START at positions 0..2 and STOP at 29..31
        # With window=16 and margin_fraction=0.2 -> margin=3, both events lie in edge bands only
        total_len = 32
        fasta = (
            f">ctg\n"
            + ''.join(['N'] * total_len)
        )
        # Insert ATG at 0..2 and TAA at 29..31
        seq_list = list('N' * total_len)
        seq_list[0:3] = list('ATG')
        seq_list[29:32] = list('TAA')
        fasta = f">ctg\n{''.join(seq_list)}\n"
        # Single + strand exon covering the whole region so START/STOP labels are set accordingly
        # Use exon_start=1 (1-based) and exon_end=32 to get last_exon[1]=32 -> stop_pos=29
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "ctg\tg\t1\t32\t1\t32\t+\n"
        )
        cw = [1.0] * 8
        cw[P.START] = 2.0
        cw[P.STOP] = 2.0
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / 's.fna.gz'
            tsv_path = fp / 'a.tsv'
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)
            ds = AnnotatedGenomeDataset(
                str(fasta_path), str(tsv_path), window=16, stride=16,
                class_weights=cw, seed=17, exclude_margin_bps=3,
                random_prefix_ns=False
            )
            # Expect two windows [0:16], [16:32]
            self.assertEqual(len(ds.windows), 2)
            # Both should be excluded for weighted-only-on-edges: no assignments produced
            cw_map = build_class_windows(ds.windows, ds.targets, [P.START, P.STOP], exclude_margin_bps=3, class_weights=cw)
            self.assertEqual(cw_map, {})


if __name__ == '__main__':
    unittest.main()


