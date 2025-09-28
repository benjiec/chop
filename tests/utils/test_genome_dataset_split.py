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


if __name__ == '__main__':
    unittest.main()


