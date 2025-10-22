#!/usr/bin/env python3

import os
import tempfile
import unittest
from importlib.machinery import SourceFileLoader

import numpy as np

from gene_decoder import PredictedSequence
from utils.constants import GenePredictionClass as P
from utils.genome import AnnotatedGenomeDataset


def _load_script_module():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts', 'metrics-from-pickle.py')
    return SourceFileLoader('metrics_from_pickle_script', path).load_module()


class TestMetricsFromPickleScript(unittest.TestCase):
    def _write_fna(self, dirpath: str, records):
        fna_path = os.path.join(dirpath, 'test.fa')
        with open(fna_path, 'w') as f:
            for sid, seq in records.items():
                f.write(f'>{sid}\n')
                f.write(seq + '\n')
        return fna_path

    def _write_tsv(self, dirpath: str, rows):
        tsv_path = os.path.join(dirpath, 'test.tsv')
        with open(tsv_path, 'w') as f:
            f.write('\t'.join(['sequence_id','gene_id','gene_start','gene_end','exon_start','exon_end','strand']) + '\n')
            for r in rows:
                f.write('\t'.join(map(str, r)) + '\n')
        return tsv_path

    def test_build_sequence_results_alignment_and_preds(self):
        mod = _load_script_module()
        with tempfile.TemporaryDirectory() as d:
            # Simple plus-strand single-exon gene: ATG ... TAA
            sid = 'contig1'
            seq = 'ATGAAATAA'  # len=9, START at 0..2, STOP at 6..8
            fna = self._write_fna(d, {sid: seq})
            # 1-based coordinates in TSV
            rows = [
                (sid, 'g1', 1, 9, 1, 9, '+'),
            ]
            tsv = self._write_tsv(d, rows)

            ds = AnnotatedGenomeDataset(fna, tsv, window=None, random_prefix_ns=False)
            self.assertEqual(len(ds), 1)
            tokens0, targets0 = ds[0]
            self.assertEqual(len(tokens0), len(seq))

            # Scrambled class order covering all canonical classes
            class_order = ['STOP', 'START', 'ASS', 'DSS', 'GENE', 'UTR5', 'INTERGENIC', 'UTR3']
            L = len(seq)
            C = len(class_order)
            probs = np.full((L, C), 0.01, dtype=np.float32)
            # Make INTERGENIC the default winner
            intergenic_col = class_order.index('INTERGENIC')
            probs[:, intergenic_col] = 0.90
            # Boost START at positions 0..2
            start_col = class_order.index('START')
            probs[0:3, start_col] = 0.95
            # Boost STOP at positions 6..8
            stop_col = class_order.index('STOP')
            probs[6:9, stop_col] = 0.96

            items = [PredictedSequence(sequence_index=0, sequence=seq, probabilities=probs, class_order=class_order, sequence_id=sid)]

            results = mod._build_sequence_results(items, ds)
            self.assertEqual(len(results), 1)
            r0 = results[0]

            # Check tokens match dataset
            np.testing.assert_array_equal(r0.sequence_tokens, tokens0)
            # Check targets match dataset
            np.testing.assert_array_equal(r0.targets, targets0)

            # Verify probabilities are aligned to canonical order
            self.assertEqual(r0.probabilities.shape, (L, len(P.idx_to_cls)))
            # Predictions should reflect boosted spans
            expected = np.full(L, P.INTERGENIC, dtype=np.int64)
            expected[0:3] = P.START
            expected[6:9] = P.STOP
            np.testing.assert_array_equal(r0.predictions, expected)

    def test_sanity_check_fna_sequences(self):
        mod = _load_script_module()
        with tempfile.TemporaryDirectory() as d:
            sid = 'contigX'
            seq = 'ATGANATAA'  # includes 'N'
            fna = self._write_fna(d, {sid: 'ATGAUATAA'})  # FASTA contains 'U' which should normalize to 'N'

            class_order = ['INTERGENIC','START','STOP','GENE','UTR5','UTR3','DSS','ASS']
            probs = np.zeros((len(seq), len(class_order)), dtype=np.float32)
            ps_ok = PredictedSequence(sequence_index=0, sequence=seq, probabilities=probs, class_order=class_order, sequence_id=sid)

            # Should not raise
            mod._sanity_check_fna_sequences([ps_ok], fna)

            # Mismatch should raise AssertionError
            ps_bad = PredictedSequence(sequence_index=0, sequence='ATGAAAAGG', probabilities=probs, class_order=class_order, sequence_id=sid)
            with self.assertRaises(AssertionError):
                mod._sanity_check_fna_sequences([ps_bad], fna)


if __name__ == '__main__':
    unittest.main()


