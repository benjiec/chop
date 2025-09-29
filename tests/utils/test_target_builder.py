import unittest
import numpy as np

from utils.constants import GenePredictionClass as P
from utils.genome import GeneAnnotation, build_targets_for_annotation


class TestBuildTargetsForAnnotation(unittest.TestCase):
    def test_plus_strand_conforming_returns_targets(self):
        # Build a simple sequence with valid motifs
        seq = list('A' * 30)
        # START at exon start [5:8]
        seq[5:8] = list('ATG')
        # Donor at e1[1] == 10 → [10:12]
        seq[10:12] = list('GT')
        # Acceptor at e2[0]-2 == 13 → [13:15]
        seq[13:15] = list('AG')
        # STOP at last_exon[1]-3 == 22 → [22:25]
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
        seq[5:8] = list('AAA')  # non-ATG start
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
        seq[5:8] = list('AAA')  # non-ATG start
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
        seq[10:12] = list('AA')  # invalid donor
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
        # Single exon on minus strand; ensure reverse-complement motifs are set
        # For START at last 3 of exon: seq[end-3:end] should be 'CAT' so RC is 'ATG'
        # For STOP at start: seq[start:start+3] should be 'TTA' so RC is 'TAA'
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
        # START should cover the last 3 bases of exon
        self.assertTrue(np.all(tgt[exon[1]-3:exon[1]] == P.START))
        # STOP should cover the first 3 bases
        self.assertTrue(np.all(tgt[exon[0]:exon[0]+3] == P.STOP))


if __name__ == '__main__':
    unittest.main()


