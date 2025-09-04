#!/usr/bin/env python3

import unittest
import numpy as np

from utils.dataset import (
    GenomicSyntheticTestingDataset,
    RandomBasesGenerator,
    RandomChoiceGenerator,
    RandomUTR5Generator,
    AddATGGenerator,
)
from utils.constants import GenePredictionClass as P


class TestGenomicSyntheticDataset(unittest.TestCase):
    def test_max_length_enforced(self):
        max_len = 200
        layouts = [
            RandomBasesGenerator(length=50, target=P.INTERGENIC),
            RandomUTR5Generator(choices=["AAAAAATG"], target=P.UTR5),
            AddATGGenerator(),
            RandomBasesGenerator(length=100, target=P.INTERGENIC),
        ]
        ds = GenomicSyntheticTestingDataset(
            max_sequence_length=max_len,
            num_contigs=10,
            layouts_per_contig=1,
            layouts=layouts,
        )
        for i in range(len(ds)):
            seq, tgt = ds[i]
            self.assertLessEqual(len(seq), max_len)
            self.assertEqual(len(seq), len(tgt))

    def test_random_choice_targets_length(self):
        gen = RandomChoiceGenerator(choices=["ACGTAC"], target=P.UTR5)
        seq, tgt = gen.generate(None)
        self.assertEqual(len(seq), len(tgt))
        self.assertEqual(len(seq), 6)

    def test_random_utr5_marks_start(self):
        gen = RandomUTR5Generator(choices=["AAAATG"], target=P.UTR5)
        seq, tgt = gen.generate(None)
        self.assertEqual(seq[-3:], "ATG")
        self.assertEqual(tgt[-3:], [P.START, P.START, P.START])

    def test_add_atg_generator(self):
        g = AddATGGenerator()
        # When last ends with ATG, no addition
        seq, tgt = g.generate("CCCATG")
        self.assertEqual(seq, "")
        self.assertEqual(tgt, [])
        # When last does not end with ATG
        seq, tgt = g.generate("CCCCCC")
        self.assertEqual(seq, "ATG")
        self.assertEqual(tgt, [P.START, P.START, P.START])

    def test_decoy_insertion(self):
        # Ensure decoys can be inserted and length preserved
        gen = RandomBasesGenerator(length=100, target=P.INTERGENIC, decoy="ATG", max_decoy=3)
        seq, tgt = gen.generate(None)
        self.assertEqual(len(seq), 100)
        self.assertEqual(len(tgt), 100)
        # At least one ATG likely present (probabilistic), but we won't assert it strictly


if __name__ == '__main__':
    unittest.main()
