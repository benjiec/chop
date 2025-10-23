#!/usr/bin/env python3

import unittest
import tempfile
import pickle
import os
from pathlib import Path
import numpy as np

from utils.genome import AnnotatedGenomeDataset
from utils.constants import GenePredictionClass as P




if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

import unittest
import tempfile
from pathlib import Path
import gzip

import numpy as np
from torch.utils.data import Subset

from utils.genome import (
    AnnotatedGenomeDataset,
    build_class_windows,
    add_random_n_prefix,
    GeneAnnotation,
    build_targets_for_annotation,
)
from utils.windowing import compute_window_slices
from utils.constants import GenePredictionClass as P, DNAEmbed


def write_temp(path: Path, content: str):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'wt') as f:
            f.write(content)
    else:
        with open(path, 'w') as f:
            f.write(content)


class TestAnnotatedGenomeDataset(unittest.TestCase):
    def test_single_exon_plus_strand(self):
        # Build sequence with exon: start ATG at idx 4, stop TAA ending at idx 19 (0-based)
        # 1-based exon coordinates: start=5, end=20
        exon = "ATG" + "G"*10 + "TAA"  # 16 bases
        fasta = f">contig1\nNNNN{exon}NN\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "contig1\tgene1\t5\t20\t5\t20\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "seq.fna.gz"
            tsv_path = fp / "ann.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)
            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), gene_class=P.GENE, random_prefix_ns=False)
            self.assertEqual(len(ds), 1)
            seq_enc, tgt = ds[0]
            seq = f"NNNN{exon}NN"
            self.assertEqual(len(seq_enc), len(seq))
            # START at pos 4..6, STOP at pos 17..19
            self.assertTrue(all(tgt[4:7] == P.START))
            self.assertTrue(all(tgt[17:20] == P.STOP))
            # GENE between
            self.assertTrue(all(tgt[7:17] == P.GENE))
            # INTERGENIC outside
            self.assertTrue(all(tgt[:4] == P.INTERGENIC))
            self.assertTrue(all(tgt[20:] == P.INTERGENIC))

    def test_single_exon_plus_strand_intergenic_gene_class(self):
        exon = "ATG" + "G"*10 + "TAA"
        fasta = f">contig1\nNNNN{exon}NN\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "contig1\tgene1\t5\t20\t5\t20\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "seq.fna.gz"
            tsv_path = fp / "ann.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)
            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), random_prefix_ns=False)
            _, tgt = ds[0]
            self.assertTrue(all(tgt[4:7] == P.START))
            self.assertTrue(all(tgt[17:20] == P.STOP))
            self.assertTrue(all(tgt[7:17] == P.INTERGENIC))

    def test_single_exon_minus_strand(self):
        # Minus strand exon; forward first 3 bases 'TTA' (RC->TAA STOP), last 3 bases 'CAT' (RC->ATG START)
        exon_fwd = "TTA" + "G"*10 + "CAT"
        seq = f"{'N'*10}{exon_fwd}{'N'*4}"
        fasta = f">contig2\n{seq}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "contig2\tgene2\t11\t26\t11\t26\t-\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "seq.fna.gz"
            tsv_path = fp / "ann.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)
            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), gene_class=P.GENE, random_prefix_ns=False)
            _, tgt = ds[0]
            self.assertTrue(all(tgt[23:26] == P.START))
            self.assertTrue(all(tgt[10:13] == P.STOP))

    def test_multi_exon_donor_acceptor(self):
        # Two exons (+)
        seq_list = list('N'*35)
        seq_list[5:8] = list('ATG')
        seq_list[12:14] = list('GT')
        seq_list[18:20] = list('AG')
        seq_list[27:30] = list('TAA')
        fasta = f">contig3\n{''.join(seq_list)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "contig3\tgene3\t6\t30\t6\t12\t+\n"
            "contig3\tgene3\t6\t30\t21\t30\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "seq.fna.gz"
            tsv_path = fp / "ann.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)
            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), gene_class=P.GENE, random_prefix_ns=False)
            _, tgt = ds[0]
            self.assertTrue(all(tgt[12:14] == P.DSS))
            self.assertTrue(all(tgt[18:20] == P.ASS))

    def test_dss_variants_plus_strand_GA(self):
        seq = list('N'*40)
        seq[5:8] = list('ATG')
        seq[12:14] = list('GA')
        seq[18:20] = list('AG')
        seq[30:33] = list('TAA')
        fasta = f">cGA\n{''.join(seq)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "cGA\tgeneGA\t6\t33\t6\t12\t+\n"
            "cGA\tgeneGA\t6\t33\t21\t33\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            write_temp(fp/"seq.fna.gz", fasta)
            write_temp(fp/"ann.tsv", tsv)
            ds = AnnotatedGenomeDataset(str(fp/"seq.fna.gz"), str(fp/"ann.tsv"), gene_class=P.GENE, random_prefix_ns=False)
            _, tgt = ds[0]
            self.assertTrue(all(tgt[12:14] == P.DSS))
            self.assertTrue(all(tgt[18:20] == P.ASS))

    def test_dss_variants_plus_strand_GC(self):
        seq = list('N'*40)
        seq[5:8] = list('ATG')
        seq[12:14] = list('GC')
        seq[18:20] = list('AG')
        seq[30:33] = list('TAA')
        fasta = f">cGC\n{''.join(seq)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "cGC\tgeneGC\t6\t33\t6\t12\t+\n"
            "cGC\tgeneGC\t6\t33\t21\t33\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            write_temp(fp/"seq.fna.gz", fasta)
            write_temp(fp/"ann.tsv", tsv)
            ds = AnnotatedGenomeDataset(str(fp/"seq.fna.gz"), str(fp/"ann.tsv"), random_prefix_ns=False)
            _, tgt = ds[0]
            self.assertTrue(all(tgt[12:14] == P.DSS))
            self.assertTrue(all(tgt[18:20] == P.ASS))

    def test_multi_exon_reverse_strand(self):
        # Two exons on minus strand; RC should satisfy START/STOP and donor/acceptor
        seq = list('N'*50)
        seq[33:36] = list('CAT')
        seq[10:13] = list('TTA')
        seq[24:26] = list('AC')
        seq[18:20] = list('CT')
        fasta = f">cMinus\n{''.join(seq)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "cMinus\tgn\t11\t36\t11\t18\t-\n"
            "cMinus\tgn\t11\t36\t27\t36\t-\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            write_temp(fp/"seq.fna.gz", fasta)
            write_temp(fp/"ann.tsv", tsv)
            ds = AnnotatedGenomeDataset(str(fp/"seq.fna.gz"), str(fp/"ann.tsv"), random_prefix_ns=False)
            _, tgt = ds[0]
            self.assertTrue(all(tgt[33:36] == P.START))
            self.assertTrue(all(tgt[10:13] == P.STOP))

    def test_invalid_gene_end_without_stop(self):
        seq = list('N'*30)
        seq[5:8] = list('ATG')
        seq[17:20] = list('AAA')
        fasta = f">cBad\n{''.join(seq)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "cBad\tbad\t6\t20\t6\t20\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            write_temp(fp/"seq.fna.gz", fasta)
            write_temp(fp/"ann.tsv", tsv)
            ds = AnnotatedGenomeDataset(str(fp/"seq.fna.gz"), str(fp/"ann.tsv"), gene_class=P.GENE, random_prefix_ns=False)
            self.assertEqual(len(ds), 0)


class TestAnnotatedGenomeDatasetSplit(unittest.TestCase):
    def _build_two_contigs_dataset(self, window: int = None, stride: int = None):
        exon = "ATG" + ("G" * 10) + "TAA"
        seq1 = f"NNNN{exon}NN"
        seq2 = f"NNNNN{exon}N"
        fasta = f">c1\n{seq1}\n>c2\n{seq2}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "c1\tg1\t5\t20\t5\t20\t+\n"
            "c2\tg2\t6\t21\t6\t21\t+\n"
        )
        td = tempfile.TemporaryDirectory()
        fp = Path(td.name)
        fasta_path = fp / "seq.fna.gz"
        tsv_path = fp / "ann.tsv"
        write_temp(fasta_path, fasta)
        write_temp(tsv_path, tsv)
        ds = AnnotatedGenomeDataset(
            str(fasta_path), str(tsv_path),
            window=window, stride=stride,
            random_prefix_ns=False,
        )
        return td, ds

    def test_split_by_contig_when_windowing(self):
        td, ds = self._build_two_contigs_dataset(window=16, stride=8)
        try:
            total = len(ds)
            train_size = total // 2
            val_size = total - train_size
            import random
            random.seed(42)
            train_ds, val_ds = ds.split(train_size, val_size)
            self.assertIsInstance(train_ds, Subset)
            self.assertIsInstance(val_ds, Subset)
            self.assertEqual(len(train_ds) + len(val_ds), total)

            def contigs_of(subset: Subset):
                base = subset.dataset
                idxs = subset.indices
                contigs = set()
                for wi in idxs:
                    contig_idx, _, _ = base.windows[wi]
                    contigs.add(base.contig_ids[contig_idx])
                return contigs

            train_contigs = contigs_of(train_ds)
            val_contigs = contigs_of(val_ds)
            self.assertTrue(train_contigs.isdisjoint(val_contigs))
        finally:
            td.cleanup()

    def test_split_without_windowing_by_sequence_index(self):
        td, ds = self._build_two_contigs_dataset(window=None, stride=None)
        try:
            total = len(ds)
            train_size = 1
            val_size = total - train_size
            import random
            random.seed(17)
            train_ds, val_ds = ds.split(train_size, val_size)
            self.assertEqual(len(train_ds), 1)
            self.assertEqual(len(val_ds), total - 1)
            train_idxs = set(train_ds.indices)
            val_idxs = set(val_ds.indices)
            self.assertTrue(train_idxs.isdisjoint(val_idxs))
        finally:
            td.cleanup()

    def test_split_windowing_no_selection_uses_absolute_indices(self):
        td, ds = self._build_two_contigs_dataset(window=8, stride=8)
        try:
            contig_to_indices = {}
            for w_idx, (cid, _, _) in enumerate(ds.windows):
                contig_to_indices.setdefault(cid, []).append(w_idx)
            import random
            random.seed(42)
            contig_ids = list(contig_to_indices.keys())
            random.shuffle(contig_ids)
            total = len(ds)
            train_target = len(contig_to_indices[contig_ids[0]])
            val_target = total - train_target
            train_expected, val_expected = [], []
            for cid in contig_ids:
                group = contig_to_indices[cid]
                if len(train_expected) + len(group) <= train_target or (len(train_expected) < train_target and len(val_expected) >= val_target):
                    train_expected.extend(group)
                else:
                    val_expected.extend(group)

            random.seed(42)
            train_ds, val_ds = ds.split(train_target, val_target)
            self.assertEqual(len(train_ds) + len(val_ds), total)
            self.assertTrue(set(train_ds.indices).isdisjoint(set(val_ds.indices)))

            def contigs_from_indices(indices):
                return { ds.contig_ids[ ds.windows[i][0] ] for i in indices }
            self.assertTrue(contigs_from_indices(train_ds.indices).isdisjoint(contigs_from_indices(val_ds.indices)))
        finally:
            td.cleanup()

    def test_split_windowing_with_selection_uses_selected_positions(self):
        td, _ = self._build_two_contigs_dataset(window=8, stride=8)
        try:
            import random
            random.seed(7)
            fp = Path(td.name)
            fasta_path = fp / "seq.fna.gz"
            tsv_path = fp / "ann.tsv"
            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), window=8, stride=8, num_windows=6, random_prefix_ns=False)

            contig_to_positions = {}
            for pos, w_idx in enumerate(ds._selected_window_indices):
                cid, _, _ = ds.windows[w_idx]
                contig_to_positions.setdefault(cid, []).append(pos)
            random.seed(99)
            contig_ids = list(contig_to_positions.keys())
            random.shuffle(contig_ids)
            train_target = len(contig_to_positions[contig_ids[0]])
            total_positions = sum(len(v) for v in contig_to_positions.values())
            val_target = total_positions - train_target
            train_expected, val_expected = [], []
            for cid in contig_ids:
                group = contig_to_positions[cid]
                if len(train_expected) + len(group) <= train_target or (len(train_expected) < train_target and len(val_expected) >= val_target):
                    train_expected.extend(group)
                else:
                    val_expected.extend(group)

            random.seed(99)
            train_ds, val_ds = ds.split(train_target, val_target)
            self.assertEqual(train_ds.indices, train_expected)
            self.assertEqual(val_ds.indices, val_expected)

            def contigs_from_positions(positions):
                return { ds.contig_ids[ ds.windows[ ds._selected_window_indices[p] ][0] ] for p in positions }
            self.assertTrue(contigs_from_positions(train_ds.indices).isdisjoint(contigs_from_positions(val_ds.indices)))
        finally:
            td.cleanup()


def build_single_exon_contig(name: str, total_len: int, start_pos0: int, stop_pos0: int) -> str:
    seq = list('N' * total_len)
    seq[start_pos0:start_pos0+3] = list('ATG')
    seq[stop_pos0:stop_pos0+3] = list('TAA')
    return f">{name}\n{''.join(seq)}\n"


class TestAnnotatedGenomeDatasetSampling(unittest.TestCase):
    def _make_two_contigs_dataset(self, window: int = 16, stride: int = 8,
                                  class_weights=None, num_windows=None, seed: int = 17,
                                  gene_class=P.GENE, exclude_margin_bps: int = 200):
        total_len = 200
        fasta_c1 = build_single_exon_contig('c1', total_len, 20, 70)
        fasta_c2 = build_single_exon_contig('c2', total_len, 100, 150)
        fasta = fasta_c1 + fasta_c2
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
            import random as _r
            _r.seed(seed)
            ds = AnnotatedGenomeDataset(
                str(fasta_path), str(tsv_path), window=window, stride=stride,
                num_windows=num_windows, class_weights=class_weights,
                gene_class=gene_class,
                exclude_margin_bps=exclude_margin_bps,
                random_prefix_ns=False
            )
            return ds

    def test_excludes_weight_one_classes_from_classsets(self):
        cw = [1.0] * 8
        cw[P.START] = 2.0
        ds = self._make_two_contigs_dataset(class_weights=cw)
        self.assertGreater(len(ds.windows), 0)

    def test_num_windows_none_keeps_all(self):
        ds = self._make_two_contigs_dataset(num_windows=None)
        total = len(ds.windows)
        self.assertEqual(len(ds), total)

    def test_num_windows_geq_total_keeps_all(self):
        ds = self._make_two_contigs_dataset(num_windows=10000)
        self.assertEqual(len(ds), len(ds.windows))

    def test_balanced_sampling_roughly_equal_counts(self):
        cw = [1.0] * 8
        cw[P.START] = 2.0
        cw[P.STOP] = 2.0
        ds = self._make_two_contigs_dataset(class_weights=cw, num_windows=12, seed=123)
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

    def test_reproducibility_with_seed(self):
        cw = [1.0] * 8
        cw[P.START] = 2.0
        cw[P.STOP] = 2.0
        ds1 = self._make_two_contigs_dataset(class_weights=cw, num_windows=10, seed=42)
        ds2 = self._make_two_contigs_dataset(class_weights=cw, num_windows=10, seed=42)
        self.assertEqual(ds1._selected_window_indices, ds2._selected_window_indices)

    def test_excludes_edge_only_weighted_windows(self):
        total_len = 32
        seq_list = list('N' * total_len)
        seq_list[0:3] = list('ATG')
        seq_list[29:32] = list('TAA')
        fasta = f">ctg\n{''.join(seq_list)}\n"
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
            import random as _r
            _r.seed(17)
            ds = AnnotatedGenomeDataset(
                str(fasta_path), str(tsv_path), window=16, stride=16,
                class_weights=cw, exclude_margin_bps=3,
                random_prefix_ns=False
            )
            self.assertEqual(len(ds.windows), 2)
            cw_map = build_class_windows(ds.windows, ds.targets, [P.START, P.STOP], exclude_margin_bps=3, class_weights=cw)
            self.assertEqual(cw_map, {})


class TestAnnotatedGenomeDatasetWindowing(unittest.TestCase):
    def test_windowing_splits_contig(self):
        seq_list = list('N' * 50)
        seq_list[9:12] = list('ATG')
        seq_list[37:40] = list('TAA')
        fasta = f">ctg\n{''.join(seq_list)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "ctg\tg\t10\t40\t10\t40\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "s.fna.gz"
            tsv_path = fp / "a.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)

            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), window=16, stride=8, random_prefix_ns=False)
            expected = compute_window_slices(50, window=16, stride=8)
            self.assertEqual(len(ds), len(expected))

            for i, (s, e) in enumerate(expected):
                x, y = ds[i]
                self.assertEqual(x.shape[0], e - s)
                self.assertEqual(y.shape[0], e - s)

            ds_full = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), random_prefix_ns=False)
            self.assertEqual(len(ds_full), 1)
            x_full, y_full = ds_full[0]
            self.assertEqual(x_full.shape[0], 50)
            self.assertEqual(y_full.shape[0], 50)

    def test_windowing_pads_last_short_window_to_fixed_length(self):
        seq_list = list('N' * 30)
        seq_list[0:3] = list('ATG')
        seq_list[27:30] = list('TAA')
        fasta = f">ctg\n{''.join(seq_list)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "ctg\tg\t1\t30\t1\t30\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "s.fna.gz"
            tsv_path = fp / "a.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)

            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), window=16, stride=16, random_prefix_ns=False)
            self.assertEqual(len(ds), 2)
            x0, y0 = ds[0]
            x1, y1 = ds[1]
            self.assertEqual(x0.shape[0], 16)
            self.assertEqual(y0.shape[0], 16)
            self.assertEqual(x1.shape[0], 16)
            self.assertEqual(y1.shape[0], 16)

    def test_window_filter_skips_windows_without_weighted_targets(self):
        # Sequence with START and STOP present, but all class weights are 1.0
        seq_list = list('N' * 50)
        seq_list[9:12] = list('ATG')
        seq_list[37:40] = list('TAA')
        fasta = f">ctg\n{''.join(seq_list)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "ctg\tg\t10\t40\t10\t40\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "s.fna.gz"
            tsv_path = fp / "a.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)

            cw = [1.0] * 8
            ds = AnnotatedGenomeDataset(
                str(fasta_path), str(tsv_path), window=16, stride=8,
                class_weights=cw, random_prefix_ns=False
            )
            # With all class weights == 1.0, no window should be kept
            self.assertEqual(len(ds.windows), 0)
            self.assertEqual(len(ds), 0)

            # Increasing a class weight (>1) should admit windows containing that class
            cw2 = [1.0] * 8
            cw2[P.START] = 2.0
            ds2 = AnnotatedGenomeDataset(
                str(fasta_path), str(tsv_path), window=16, stride=8,
                class_weights=cw2, random_prefix_ns=False
            )
            self.assertGreater(len(ds2.windows), 0)


class TestAnnotatedGenomeDatasetPadding(unittest.TestCase):
    def test_random_n_prefix_added_with_seed(self):
        # Build a simple + strand exon with START/STOP to make a valid contig
        exon = "ATG" + ("G" * 10) + "TAA"  # 16 bases
        seq = f"NNNN{exon}NN"  # total 4 + 16 + 2 = 22
        fasta = f">ctgX\n{seq}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "ctgX\tg\t5\t20\t5\t20\t+\n"
        )

        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "seq.fna.gz"
            tsv_path = fp / "ann.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)

            # first, don't pad
            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), random_prefix_ns=False)
            seq_enc, tgt = ds[0]

            delta = len(seq_enc) - len(seq)
            self.assertEqual(delta, 0)

            # now, default behavior is to pad
            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path))
            seq_enc, tgt = ds[0]

            # Length must increase by at least 100 and at most 400
            delta = len(seq_enc) - len(seq)
            self.assertGreaterEqual(delta, 100)
            self.assertLessEqual(delta, 400)

            # First delta tokens are Ns; targets INTERGENIC
            self.assertTrue(np.all(seq_enc[:delta] == DNAEmbed.N))
            self.assertTrue(np.all(tgt[:delta] == P.INTERGENIC))

class DummyRng:
    def __init__(self, val: int):
        self._val = int(val)

    def randint(self, a: int, b: int) -> int:
        return self._val


class TestNPrefix(unittest.TestCase):
    def test_disabled_no_change(self):
        seq = 'ATGC'
        tgt = np.array([P.INTERGENIC]*4, dtype=np.int64)
        new_seq, new_tgt, pad = add_random_n_prefix(seq, tgt, enabled=False, rng=DummyRng(123))
        self.assertEqual(seq, new_seq)
        self.assertTrue(np.array_equal(tgt, new_tgt))
        self.assertEqual(pad, 0)

    def test_enabled_with_fixed_rng(self):
        seq = 'ATGC'
        tgt = np.array([P.INTERGENIC]*4, dtype=np.int64)
        new_seq, new_tgt, pad = add_random_n_prefix(seq, tgt, enabled=True, rng=DummyRng(5), min_len=5, max_len=5)
        self.assertEqual(pad, 5)
        self.assertTrue(new_seq.startswith('N'*5))
        self.assertEqual(new_seq[-4:], 'ATGC')
        self.assertEqual(len(new_tgt), 9)
        self.assertTrue(np.all(new_tgt[:5] == P.INTERGENIC))
        self.assertTrue(np.all(new_tgt[5:] == tgt))


class TestBuildTargetsForAnnotation(unittest.TestCase):
    def test_plus_strand_conforming_returns_targets(self):
        seq = list('A' * 30)
        seq[5:8] = list('ATG')
        seq[10:12] = list('GT')
        seq[13:15] = list('AG')
        seq[22:25] = list('TAA')
        seq = ''.join(seq)

        ann = GeneAnnotation(sequence_id='ctg', gene_id='g1', strand='+', exons=[(5, 10), (15, 25)])
        fails = {}
        tgt = build_targets_for_annotation(
            seq=seq,
            ann=ann,
            gene_class=P.GENE,
            failure_counts=fails,
        )
        self.assertIsNotNone(tgt)
        self.assertEqual(fails.get('start', 0), 0)
        self.assertEqual(fails.get('stop', 0), 0)
        self.assertEqual(fails.get('ass', 0), 0)
        self.assertEqual(fails.get('dss', 0), 0)
        self.assertTrue(np.all(tgt[5:8] == P.START))
        self.assertTrue(np.all(tgt[22:25] == P.STOP))
        self.assertTrue(np.all(tgt[10:12] == P.DSS))
        self.assertTrue(np.all(tgt[13:15] == P.ASS))

    def test_nonconforming_start_disallowed_returns_none_and_records(self):
        seq = list('A' * 30)
        seq[5:8] = list('AAA')
        seq[10:12] = list('GT')
        seq[13:15] = list('AG')
        seq[22:25] = list('TAA')
        seq = ''.join(seq)
        ann = GeneAnnotation(sequence_id='ctg', gene_id='g1', strand='+', exons=[(5, 10), (15, 25)])
        fails = {}
        tgt = build_targets_for_annotation(
            seq=seq,
            ann=ann,
            gene_class=P.GENE,
            allow_nonconforming_start=False,
            failure_counts=fails,
        )
        self.assertIsNone(tgt)
        self.assertEqual(fails.get('start', 0), 1)

    def test_nonconforming_start_allowed_returns_targets_and_records(self):
        seq = list('A' * 30)
        seq[5:8] = list('AAA')
        seq[10:12] = list('GT')
        seq[13:15] = list('AG')
        seq[22:25] = list('TAA')
        seq = ''.join(seq)
        ann = GeneAnnotation(sequence_id='ctg', gene_id='g1', strand='+', exons=[(5, 10), (15, 25)])
        fails = {}
        tgt = build_targets_for_annotation(
            seq=seq,
            ann=ann,
            gene_class=P.GENE,
            allow_nonconforming_start=True,
            failure_counts=fails,
        )
        self.assertIsNotNone(tgt)
        self.assertEqual(fails.get('start', 0), 1)
        self.assertTrue(np.all(tgt[5:8] == P.START))

    def test_nonconforming_dss_disallowed_returns_none_and_records(self):
        seq = list('A' * 30)
        seq[5:8] = list('ATG')
        seq[10:12] = list('AA')
        seq[13:15] = list('AG')
        seq[22:25] = list('TAA')
        seq = ''.join(seq)
        ann = GeneAnnotation(sequence_id='ctg', gene_id='g1', strand='+', exons=[(5, 10), (15, 25)])
        fails = {}
        tgt = build_targets_for_annotation(
            seq=seq,
            ann=ann,
            gene_class=P.GENE,
            allow_nonconforming_dss=False,
            failure_counts=fails,
        )
        self.assertIsNone(tgt)
        self.assertEqual(fails.get('dss', 0), 1)

    def test_minus_strand_conforming_returns_targets(self):
        seq = list('A' * 20)
        exon = (5, 15)
        seq[exon[1]-3:exon[1]] = list('CAT')
        seq[exon[0]:exon[0]+3] = list('TTA')
        seq = ''.join(seq)
        ann = GeneAnnotation(sequence_id='ctg', gene_id='g1', strand='-', exons=[exon])
        fails = {}
        tgt = build_targets_for_annotation(
            seq=seq,
            ann=ann,
            gene_class=P.GENE,
            failure_counts=fails,
        )
        self.assertIsNotNone(tgt)
        self.assertEqual(fails, {})
        self.assertTrue(np.all(tgt[exon[1]-3:exon[1]] == P.START))
        self.assertTrue(np.all(tgt[exon[0]:exon[0]+3] == P.STOP))


class TestClassWindowHelpers(unittest.TestCase):
    def test_build_class_windows_excludes_edges(self):
        windows = [(0, 0, 10)]
        targets = [np.array([P.START, P.START, 0, 0, 0, 0, 0, 0, P.STOP, P.STOP], dtype=np.int64)]
        classes = [P.START, P.STOP]
        class_windows = build_class_windows(windows, targets, classes_to_balance=classes, exclude_margin_bps=2)
        self.assertEqual(class_windows, {}, "Windows with weighted classes only on edges should be excluded")

    def test_build_class_windows_assigns_highest_weight(self):
        windows = [(0, 0, 10)]
        targets = [np.array([0, 0, P.START, 0, P.STOP, 0, 0, 0, 0, 0], dtype=np.int64)]
        classes = [P.START, P.STOP]
        cw = [1.0]*8
        cw[P.START] = 2.0
        cw[P.STOP] = 5.0
        class_windows = build_class_windows(windows, targets, classes_to_balance=classes, exclude_margin_bps=1, class_weights=cw)
        self.assertIn(P.STOP, class_windows)
        self.assertEqual(class_windows[P.STOP], [0])

        cw2 = [1.0]*8
        cw2[P.START] = 7.0
        cw2[P.STOP] = 2.0
        class_windows = build_class_windows(windows, targets, classes_to_balance=classes, exclude_margin_bps=1, class_weights=cw2)
        self.assertIn(P.START, class_windows)
        self.assertEqual(class_windows[P.START], [0])
        self.assertNotIn(P.STOP, class_windows)

    def test_build_class_windows_tie_breaker_smallest_class(self):
        windows = [(0, 0, 10)]
        targets = [np.array([0, 0, P.START, P.START, P.STOP, P.STOP, 0, 0, 0, 0], dtype=np.int64)]
        classes = [P.START, P.STOP]
        class_windows = build_class_windows(windows, targets, classes_to_balance=classes, exclude_margin_bps=1, class_weights=[1.0]*8)
        self.assertEqual(class_windows, {P.START: [0]})

    def test_edge_only_higher_weight_ignored_middle_next_highest_selected(self):
        windows = [(0, 0, 10)]
        targets = [np.array([P.STOP, 0, 0, 0, P.START, 0, 0, 0, 0, P.STOP], dtype=np.int64)]
        classes = [P.START, P.STOP]
        cw = [1.0]*8
        cw[P.START] = 2.0
        cw[P.STOP] = 5.0
        class_windows = build_class_windows(windows, targets, classes_to_balance=classes, exclude_margin_bps=2, class_weights=cw)
        self.assertEqual(class_windows, {P.START: [0]})


if __name__ == '__main__':
    unittest.main()


