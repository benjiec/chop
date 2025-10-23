#!/usr/bin/env python3

import unittest
import tempfile
import pickle
from pathlib import Path
import numpy as np

from utils.genome import AnnotatedGenomeDataset
from utils.stream import NumericalStream


class TestNumericalStreamIntegration(unittest.TestCase):
    def _write_simple_fasta(self, path: Path, seqs):
        with open(path, 'w') as f:
            for sid, s in seqs.items():
                f.write(f">{sid}\n")
                f.write(s + "\n")

    def _write_simple_tsv(self, path: Path, rows):
        # sequence_id, gene_id, gene_start, gene_end, exon_start, exon_end, strand
        with open(path, 'w') as f:
            f.write("sequence_id\tgene_id\tstrand\texon_start\texon_end\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")

    def test_aux_stream_load_and_align_and_window(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            seqs = {'ctg1': 'A' * 30, 'ctg2': 'C' * 50}
            fa = td / 's.fa'
            self._write_simple_fasta(fa, seqs)

            rows = [['ctg1', 'g1', '+', 1, 3], ['ctg2', 'g2', '+', 1, 3]]
            tsv = td / 'a.tsv'
            self._write_simple_tsv(tsv, rows)

            channels = ['dg50', 'gc50']
            aux_items = [
                {'sequence_id': 'ctg1', 'channels': channels, 'data': np.random.randn(30, 2).astype(np.float32)},
                {'sequence_id': 'ctg2', 'channels': channels, 'data': np.random.randn(50, 2).astype(np.float32)},
            ]
            aux_pkl = td / 'aux.pkl'
            with open(aux_pkl, 'wb') as f:
                pickle.dump(aux_items, f)

            ds = AnnotatedGenomeDataset(str(fa), str(tsv), window=20, stride=15, aux_stream_path=str(aux_pkl), random_prefix_ns=False,
                                        allow_nonconforming_start=True, allow_nonconforming_stop=True, allow_nonconforming_ass=True, allow_nonconforming_dss=True)
            self.assertGreater(len(ds), 0)
            x, y, a = ds[0]
            self.assertEqual(x.shape[0], y.shape[0])
            self.assertEqual(int(a.shape[0]), int(x.shape[0]))
            self.assertEqual(int(a.shape[1]), 2)

    def test_aux_stream_length_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            seqs = {'ctg1': 'A' * 10}
            fa = td / 's.fa'
            self._write_simple_fasta(fa, seqs)
            rows = [['ctg1', 'g1', '+', 1, 3]]
            tsv = td / 'a.tsv'
            self._write_simple_tsv(tsv, rows)

            channels = ['dg50']
            aux_items = [{'sequence_id': 'ctg1', 'channels': channels, 'data': np.random.randn(9, 1).astype(np.float32)}]
            aux_pkl = td / 'aux.pkl'
            with open(aux_pkl, 'wb') as f:
                pickle.dump(aux_items, f)

            with self.assertRaises(ValueError):
                _ = AnnotatedGenomeDataset(str(fa), str(tsv), aux_stream_path=str(aux_pkl), random_prefix_ns=False)

    def test_aux_stream_prefix_padding(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            seqs = {'ctg1': 'A' * 25}
            fa = td / 's.fa'
            self._write_simple_fasta(fa, seqs)
            rows = [['ctg1', 'g1', '+', 1, 3]]
            tsv = td / 'a.tsv'
            self._write_simple_tsv(tsv, rows)

            channels = ['dg50']
            data = np.ones((25, 1), dtype=np.float32)
            aux_items = [{'sequence_id': 'ctg1', 'channels': channels, 'data': data}]
            aux_pkl = td / 'aux.pkl'
            with open(aux_pkl, 'wb') as f:
                pickle.dump(aux_items, f)

            ds = AnnotatedGenomeDataset(str(fa), str(tsv), aux_stream_path=str(aux_pkl), random_prefix_ns=True,
                                        allow_nonconforming_start=True, allow_nonconforming_stop=True, allow_nonconforming_ass=True, allow_nonconforming_dss=True,
                                        aux_normalize=False)
            _, _, a = ds[0]
            self.assertTrue(np.any(a == 0.0))
            self.assertTrue(np.any(a != 0.0))

    def test_create_empty_add_channel_and_save_reload(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pkl = td / 'empty.pkl'
            ns = NumericalStream.create_empty(str(pkl))
            # Add channel from fasta
            fa = td / 's.fa'
            seqs = {'ctg1': 'ATGC', 'ctg2': 'NNNNN'}
            self._write_simple_fasta(fa, seqs)
            def gen_len(seq: str):
                return np.arange(len(seq), dtype=np.float32)
            ns.add_channel(str(fa), 'pos', gen_len)
            # Add second channel
            def gen_ones(seq: str):
                return np.ones(len(seq), dtype=np.float32)
            ns.add_channel(str(fa), 'ones', gen_ones)
            # Save and reload
            ns.save()
            ns2 = NumericalStream(str(pkl))
            self.assertEqual(ns2.channels, ['pos', 'ones'])
            for sid in ['ctg1', 'ctg2']:
                arr = ns2.get(sid)
                self.assertEqual(arr.shape[1], 2)
                self.assertEqual(arr.shape[0], len(seqs[sid]))


if __name__ == '__main__':
    unittest.main()


