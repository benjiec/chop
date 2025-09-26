#!/usr/bin/env python3

import unittest
import tempfile
from pathlib import Path
import gzip
import random

import numpy as np

from utils.genome import AnnotatedGenomeDataset
from utils.constants import GenePredictionClass as P, DNAEmbed


def write_temp(path: Path, content: str):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'wt') as f:
            f.write(content)
    else:
        with open(path, 'w') as f:
            f.write(content)


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

            ds = AnnotatedGenomeDataset(str(fasta_path), str(tsv_path), random_prefix_ns=True)
            seq_enc, tgt = ds[0]

            # Length must increase by at least 100 and at most 400
            delta = len(seq_enc) - len(seq)
            self.assertGreaterEqual(delta, 100)
            self.assertLessEqual(delta, 400)

            # First delta tokens are Ns; targets INTERGENIC
            self.assertTrue(np.all(seq_enc[:delta] == DNAEmbed.N))
            self.assertTrue(np.all(tgt[:delta] == P.INTERGENIC))


if __name__ == '__main__':
    unittest.main()


