#!/usr/bin/env python3

import unittest
import tempfile
import pickle
from pathlib import Path
import numpy as np

from utils.genome import AnnotatedGenomeDataset
from utils.stream import NumericalStream, build_gc_generator


class TestGCGenerator(unittest.TestCase):
    def _gc_ratio(self, s: str) -> float:
        s = s.upper()
        if len(s) == 0:
            return 0.0
        gc = sum(1 for ch in s if ch in ('G', 'C'))
        return gc / float(len(s))

    def test_gc_generator_centering_rules(self):
        # Odd window (5): X=2,Y=2 centered on token
        seq = 'ATGCGT'  # length 6
        gen5 = build_gc_generator(5)
        out5 = gen5(seq)
        # Position 2 (0-based) -> window [0..4)
        self.assertAlmostEqual(out5[2], self._gc_ratio(seq[0:5]), places=6)
        # Position 3 -> window [1..6)
        self.assertAlmostEqual(out5[3], self._gc_ratio(seq[1:6]), places=6)

        # Even window (4): X=1,Y=2
        gen4 = build_gc_generator(4)
        out4 = gen4(seq)
        # Position 2 -> window [1..5)
        self.assertAlmostEqual(out4[2], self._gc_ratio(seq[1:5]), places=6)
        # Position 3 -> window [2..6)
        self.assertAlmostEqual(out4[3], self._gc_ratio(seq[2:6]), places=6)

    def test_gc_generator_edges(self):
        # Document behavior at sequence edges: window is clipped to sequence bounds
        seq = 'GCAA'  # L=4
        # Odd window (3): X=1,Y=1
        gen3 = build_gc_generator(3)
        out3 = gen3(seq)
        # i=0 -> [0..1)
        self.assertAlmostEqual(out3[0], self._gc_ratio(seq[0:1]), places=6)
        # i=1 -> [0..3)
        self.assertAlmostEqual(out3[1], self._gc_ratio(seq[0:3]), places=6)
        # i=3 -> [2..4)
        self.assertAlmostEqual(out3[3], self._gc_ratio(seq[2:4]), places=6)
        # Even window (4): X=1,Y=2 -> clipped
        gen4 = build_gc_generator(4)
        out4 = gen4(seq)
        # i=0 -> [0..3)
        self.assertAlmostEqual(out4[0], self._gc_ratio(seq[0:3]), places=6)
        # i=3 -> [2..4)
        self.assertAlmostEqual(out4[3], self._gc_ratio(seq[2:4]), places=6)


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

    def test_gz_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            pkl = td / 's.pkl.gz'
            # create empty gz
            with open(td / 'tmp.pkl', 'wb') as f:
                pickle.dump([], f)
            import gzip as _gz
            with _gz.open(pkl, 'wb') as f:
                pickle.dump([], f)
            ns = NumericalStream(str(pkl))
            # Add channel from fasta
            fa = td / 's.fa'
            seqs = {'ctg1': 'ATGCA', 'ctg2': 'GGGG'}
            self._write_simple_fasta(fa, seqs)
            def gen_zero(seq: str):
                return np.zeros(len(seq), dtype=np.float32)
            ns.add_channel(str(fa), 'z', gen_zero)
            ns.save()
            ns2 = NumericalStream(str(pkl))
            self.assertEqual(ns2.channels, ['z'])
            for sid in ['ctg1', 'ctg2']:
                arr = ns2.get(sid)
                self.assertEqual(arr.shape, (len(seqs[sid]), 1))


class TestViennaDG(unittest.TestCase):
    def test_vienna_dg_generator_import_and_shape(self):
        # Skip if ViennaRNA is not installed
        try:
            import RNA  # noqa: F401
        except Exception:
            self.skipTest("ViennaRNA 'RNA' bindings not installed")
        from utils.stream import build_vienna_dg_generator
        gen = build_vienna_dg_generator(5, temp_celsius=37.0, mode='mfe')
        seq = 'ATGCAU'  # includes U; DNA T->U conversion is handled in generator
        out = gen(seq)
        self.assertEqual(out.shape, (len(seq),))
        self.assertEqual(out.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()


