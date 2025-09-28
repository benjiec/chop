#!/usr/bin/env python3

import unittest
import tempfile
from pathlib import Path
import gzip

from torch.utils.data import Subset

from utils.genome import AnnotatedGenomeDataset


def write_temp(path: Path, content: str):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'wt') as f:
            f.write(content)
    else:
        with open(path, 'w') as f:
            f.write(content)


class TestAnnotatedGenomeDatasetSplit(unittest.TestCase):
    def _build_two_contigs_dataset(self, window: int = None, stride: int = None):
        # Two simple + strand single-exon contigs with valid START/STOP motifs
        # c1 exon: 1-based 5..20 => START at 4..6 (0-based) and STOP at 17..19
        exon = "ATG" + ("G" * 10) + "TAA"
        seq1 = f"NNNN{exon}NN"  # length 22
        # c2 exon shifted
        seq2 = f"NNNNN{exon}N"  # length 22 as well
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

            # No contig overlap across splits
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
            # Ensure subsets are disjoint by indices
            train_idxs = set(train_ds.indices)
            val_idxs = set(val_ds.indices)
            self.assertTrue(train_idxs.isdisjoint(val_idxs))
        finally:
            td.cleanup()

    def test_split_windowing_no_selection_uses_absolute_indices(self):
        # Windowing enabled, no internal selection: split should operate on absolute window indices
        td, ds = self._build_two_contigs_dataset(window=8, stride=8)
        try:
            # Build expected split deterministically by contig group sizes
            contig_to_indices = {}
            for w_idx, (cid, _, _) in enumerate(ds.windows):
                contig_to_indices.setdefault(cid, []).append(w_idx)
            import random
            random.seed(42)
            contig_ids = list(contig_to_indices.keys())
            random.shuffle(contig_ids)
            total = len(ds)
            # target: take all windows from first contig as train, rest as val
            train_target = len(contig_to_indices[contig_ids[0]])
            val_target = total - train_target
            train_expected, val_expected = [], []
            for cid in contig_ids:
                group = contig_to_indices[cid]
                if len(train_expected) + len(group) <= train_target or (len(train_expected) < train_target and len(val_expected) >= val_target):
                    train_expected.extend(group)
                else:
                    val_expected.extend(group)

            # Perform split under same RNG state
            random.seed(42)
            train_ds, val_ds = ds.split(train_target, val_target)
            self.assertEqual(train_ds.indices, train_expected)
            self.assertEqual(val_ds.indices, val_expected)
            # Ensure no contig overlap
            def contigs_from_indices(indices):
                return { ds.contig_ids[ ds.windows[i][0] ] for i in indices }
            self.assertTrue(contigs_from_indices(train_ds.indices).isdisjoint(contigs_from_indices(val_ds.indices)))
        finally:
            td.cleanup()

    def test_split_windowing_with_selection_uses_selected_positions(self):
        # With internal selection active, split should operate in the coordinate space of selected positions
        td, _ = self._build_two_contigs_dataset(window=8, stride=8)
        try:
            # Rebuild dataset with internal selection enabled
            # Use deterministic RNG for selection
            import random
            random.seed(7)
            fp = Path(td.name)
            fasta_path = fp / "seq.fna.gz"
            tsv_path = fp / "ann.tsv"
            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), window=8, stride=8, num_windows=6, random_prefix_ns=False)

            # Build expected positions per contig in the selected list
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

            # Perform split under same RNG state
            random.seed(99)
            train_ds, val_ds = ds.split(train_target, val_target)
            self.assertEqual(train_ds.indices, train_expected)
            self.assertEqual(val_ds.indices, val_expected)

            # Extra: map positions to absolute windows and ensure contig disjointness
            def contigs_from_positions(positions):
                return { ds.contig_ids[ ds.windows[ ds._selected_window_indices[p] ][0] ] for p in positions }
            self.assertTrue(contigs_from_positions(train_ds.indices).isdisjoint(contigs_from_positions(val_ds.indices)))
        finally:
            td.cleanup()


if __name__ == '__main__':
    unittest.main()


