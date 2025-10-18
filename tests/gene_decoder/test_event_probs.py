#!/usr/bin/env python3

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gene_decoder import PredictedSequence
from gene_decoder.event_probs import compute_event_rows
from utils.constants import GenePredictionClass as P


class TestEventProbs(unittest.TestCase):
    def test_compute_event_rows_simple_gene(self):
        # Construct a simple sequence with one start, one intron (DSS/ASS), and one stop
        # Positions (0-based):
        # 0-2 ATG (START)
        # 3-5 CCC
        # 6-7 GT  (DSS donor)
        # 8-9 CC
        # 10-11 AG (ASS acceptor)
        # 12-14 CCC
        # 15-17 TAA (STOP)
        seq = "ATG" + "CCC" + "GT" + "CC" + "AG" + "CCC" + "TAA"
        L = len(seq)
        num_classes = 8
        probs = np.zeros((L, num_classes), dtype=np.float32)

        # Set event probabilities at spans
        probs[0:3, int(P.START)] = 0.9
        probs[6:8, int(P.DSS)] = 0.7
        probs[10:12, int(P.ASS)] = 0.6
        probs[15:18, int(P.STOP)] = 0.8

        items = [PredictedSequence(
            sequence_index=0,
            sequence=seq,
            probabilities=probs,
            class_order=['INTERGENIC','UTR5','START','GENE','STOP','UTR3','DSS','ASS'],
            sequence_id='seq1',
        )]

        # Build expected TSV (row-per-exon), two exons spanning the above layout on '+' strand
        # Exons are 0-based half-open in code, but TSV expects 1-based inclusive
        # Exon1: [0, 6) -> exon_start=1, exon_end=6
        # Exon2: [12, 18) -> exon_start=13, exon_end=18
        with tempfile.TemporaryDirectory() as td:
            tsv_path = Path(td) / 'expected.tsv'
            with open(tsv_path, 'w') as f:
                f.write('\t'.join(['sequence_id','gene_id','gene_start','gene_end','exon_start','exon_end','strand']) + '\n')
                f.write('\t'.join(['seq1','g1','1','18','1','6','+']) + '\n')
                f.write('\t'.join(['seq1','g1','1','18','13','18','+']) + '\n')

            rows = compute_event_rows(items, str(tsv_path), dss_motifs_mode='standard')

        # Filter rows for this sequence
        rows = [r for r in rows if r[0] == 'seq1']
        self.assertTrue(len(rows) >= 4)

        # Map by (class, pos, type)
        by_key = {(cls, pos, typ): prob for (_, cls, typ, pos, prob) in rows}

        # Expected positive events and positions (1-based)
        self.assertIn(('START', 1, 'positive'), by_key)
        self.assertIn(('DSS', 7, 'positive'), by_key)
        self.assertIn(('ASS', 11, 'positive'), by_key)
        self.assertIn(('STOP', 16, 'positive'), by_key)

        # Check probabilities are close to the set means
        self.assertAlmostEqual(by_key[('START', 1, 'positive')], 0.9, places=6)
        self.assertAlmostEqual(by_key[('DSS', 7, 'positive')], 0.7, places=6)
        self.assertAlmostEqual(by_key[('ASS', 11, 'positive')], 0.6, places=6)
        self.assertAlmostEqual(by_key[('STOP', 16, 'positive')], 0.8, places=6)

        # Ensure there are no negatives for unintended motifs in this synthetic sequence
        negatives = [r for r in rows if r[2] == 'negative']
        self.assertEqual(len(negatives), 0)


if __name__ == '__main__':
    unittest.main()


