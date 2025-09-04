#!/usr/bin/env python3

import unittest
import numpy as np
import random

from utils.dataset import (
    GenomicSyntheticTestingDataset,
    RandomBasesGenerator,
    RandomChoiceGenerator,
    RandomUTR5Generator,
    AddATGGenerator,
)
from utils.constants import GenePredictionClass as P


class TestGenomicSyntheticDataset(unittest.TestCase):
    def setUp(self):
        # Seed RNGs for deterministic tests
        random.seed(0)
        np.random.seed(0)
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

    def test_random_min_length_range(self):
        # RandomBasesGenerator should produce lengths within [random_min_length, length]
        gen = RandomBasesGenerator(length=120, target=P.INTERGENIC, random_min_length=80)
        lengths = set()
        for _ in range(100):
            seq, tgt = gen.generate(None)
            lengths.add(len(seq))
            self.assertEqual(len(seq), len(tgt))
            self.assertGreaterEqual(len(seq), 80)
            self.assertLessEqual(len(seq), 120)
        # We should observe multiple distinct lengths in the range
        self.assertGreaterEqual(len(lengths), 5)

    def test_random_choice_selects_multiple(self):
        gen = RandomChoiceGenerator(choices=["AAAA", "TTTT"], target=P.UTR5)
        seen = set()
        for _ in range(50):
            seq, tgt = gen.generate(None)
            seen.add(seq)
            self.assertEqual(len(seq), len(tgt))
        self.assertGreaterEqual(len(seen), 2, "RandomChoiceGenerator should select among multiple choices")

    def test_full_sequence_targets_from_generators(self):
        # Layout: 5 intergenic, UTR5 (ends with ATG), AddATG (no-op), 4 intergenic
        layouts = [
            RandomBasesGenerator(length=5, target=P.INTERGENIC),
            RandomUTR5Generator(choices=["CCCCTGATG"], target=P.UTR5),  # length 9, ends with ATG
            AddATGGenerator(),
            RandomBasesGenerator(length=4, target=P.INTERGENIC),
        ]
        ds = GenomicSyntheticTestingDataset(
            max_sequence_length=32,
            num_contigs=1,
            layouts_per_contig=1,
            layouts=layouts,
        )
        seq, tgt = ds[0]
        self.assertEqual(len(seq), len(tgt))
        # Check segments
        self.assertTrue(all(t == P.INTERGENIC for t in tgt[:5]))
        # UTR5 region of length 6 prior to ATG
        self.assertTrue(all(t == P.UTR5 for t in tgt[5:5+6]))
        # START labels for last 3 of UTR5 segment
        self.assertEqual(list(tgt[5+6:5+9]), [P.START, P.START, P.START])
        # Trailing intergenic 4
        self.assertTrue(all(t == P.INTERGENIC for t in tgt[-4:]))

    def test_decoy_arbitrary_token_inserted(self):
        # Use a non-ATGCN decoy and verify it can appear
        gen = RandomBasesGenerator(length=60, target=P.INTERGENIC, decoy="MMM", max_decoy=3)
        found = False
        for _ in range(30):
            seq, _ = gen.generate(None)
            if "MMM" in seq:
                found = True
                break
        self.assertTrue(found, "Expected at least one generated sequence to contain the decoy 'MMM'")

    def test_assertion_on_max_sequence_violation(self):
        # Force violation by setting max length too small for layout
        layouts = [
            RandomBasesGenerator(length=10, target=P.INTERGENIC),
            RandomUTR5Generator(choices=["AAAAATG"], target=P.UTR5),  # length 7
            RandomBasesGenerator(length=10, target=P.INTERGENIC),
        ]
        with self.assertRaises(AssertionError):
            GenomicSyntheticTestingDataset(
                max_sequence_length=10,  # too small
                num_contigs=1,
                layouts_per_contig=1,
                layouts=layouts,
            )


if __name__ == '__main__':
    unittest.main()
