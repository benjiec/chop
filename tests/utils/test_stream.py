#!/usr/bin/env python3

import unittest
import tempfile
import pickle
from pathlib import Path
import numpy as np
import sys
from importlib.machinery import SourceFileLoader

from utils.genome import AnnotatedGenomeDataset
from utils.stream import NumericalStream, build_gc_generator, build_dinuc_generator, Turner2004NNdeltaG37C, Turner2004NN_dH, Turner2004NN_dS


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


class TestStreamAddTargetsScript(unittest.TestCase):
    def _write_simple_fasta(self, path: Path, seqs):
        with open(path, 'w') as f:
            for sid, s in seqs.items():
                f.write(f">{sid}\n")
                f.write(s + "\n")

    def _write_simple_tsv(self, path: Path, rows):
        with open(path, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")

    def test_stream_add_targets_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # Build a simple gene: ATG.....TAA over full length
            sid = 'contig1'
            seq = 'T'*10+'ATG'+'CTC'*4+'GTTTTTAG'+'GGG'+'TAA'+'C'*50
            fna = td / 'seq.fa'
            self._write_simple_fasta(fna, {sid: seq})
            tsv = td / 'ann.tsv'
            # 1-based inclusive
            rows = [
              (sid, 'g1', 11, 39, 11, 25, '+'),
              (sid, 'g1', 11, 39, 34, 39, '+'),
            ]
            self._write_simple_tsv(tsv, rows)

            # Target stream path
            pkl = td / 'stream.pkl.gz'

            # Load and run the script via importlib
            script_path = str((Path(__file__).parents[2] / 'scripts' / 'stream-add-targets.py').resolve())
            mod = SourceFileLoader('stream_add_targets', script_path).load_module()
            argv_backup = sys.argv
            try:
                sys.argv = ['stream-add-targets.py', '--stream', str(pkl), '--fna', str(fna), '--tsv', str(tsv)]
                mod.main()
            finally:
                sys.argv = argv_backup

            # Verify stream has the target channel and matches dataset targets
            from utils.stream import NumericalStream
            ns = NumericalStream(str(pkl))
            self.assertIn('target', ns.channels)
            arr = ns.get(sid)
            self.assertEqual(arr.shape[0], len(seq))
            # Rebuild dataset to compare
            from utils.genome import AnnotatedGenomeDataset
            ds = AnnotatedGenomeDataset(str(fna), str(tsv), random_prefix_ns=False)
            # Only one contig
            tgt = ds.targets[0].astype(np.float32)
            # Only channel should be target
            np.testing.assert_array_equal(arr[:, 0], tgt)


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


class TestDinucGenerator(unittest.TestCase):
    def test_dinuc_count_fraction(self):
        seq = 'GGCCAT'  # L=6
        gen = build_dinuc_generator(4, mode='count')
        out = gen(seq)
        # For i=2 with w=4 (even): window [1..5) = 'GCCA'; pairs: GC, CC, CA -> 2/3 strong
        self.assertAlmostEqual(float(out[2]), 2.0/3.0, places=6)

    def test_dinuc_weighted_sum(self):
        seq = 'GGCCAT'
        gen = build_dinuc_generator(4, mode='weighted')
        out = gen(seq)
        # For i=2, window 'GCCA'; weights from Turner2004NNdeltaG37C
        expected = (
            Turner2004NNdeltaG37C[b'GC'] +
            Turner2004NNdeltaG37C[b'CC'] +
            Turner2004NNdeltaG37C[b'CA']
        ) / 3.0
        self.assertAlmostEqual(float(out[2]), expected, places=6)

    def test_dinuc_weighted_at_temp_25C(self):
        # Window 'GCCA' at i=2 (w=4) → pairs: GC, CC, CA
        seq = 'GGCCAT'
        gen = build_dinuc_generator(4, mode='weighted', temp_celsius=25.0)
        out = gen(seq)
        # expected mean of ΔH - T*ΔS over pairs
        pairs = [b'GC', b'CC', b'CA']
        exp_vals = []
        for p in pairs:
            dH = Turner2004NN_dH[p]
            dS = Turner2004NN_dS[p]
            exp_vals.append(dH - 298.15 * (dS / 1000.0))
        expected = sum(exp_vals) / len(exp_vals)
        self.assertAlmostEqual(float(out[2]), expected, places=6)

    def test_dinuc_dg37_close_to_table(self):
        # For each dinucleotide, compute ΔG37 from dH/dS and compare to the 37C table
        T = 310.15
        max_diff = 0.0
        for din, g37 in Turner2004NNdeltaG37C.items():
            dH = Turner2004NN_dH.get(din)
            dS = Turner2004NN_dS.get(din)
            if dH is None or dS is None:
                continue
            calc = dH - T * (dS / 1000.0)
            diff = abs(calc - g37)
            max_diff = max(max_diff, diff)
            # Allow tolerance due to parameter set differences; within ~1.3 kcal/mol
            self.assertLessEqual(diff, 1.3, msg=f"{din} diff {diff}")
        # Sanity: at least one pair compared
        self.assertGreater(max_diff, 0.0)

    def test_dinuc_repeats_and_non_gc(self):
        # Sequence designed so a window contains repeated strong 'GG' and an A/T-only dinuc 'AT'
        # seq: GGGGAT (L=6); with w=5 (odd), at i=3 → window [1..6) = 'GGGAT'
        # pairs: GG, GG, GA, AT → strong count = 2/4; weighted = (GG+GG+GA+AT)/4
        seq = 'GGGGAT'
        gen_count = build_dinuc_generator(5, mode='count')
        gen_weighted = build_dinuc_generator(5, mode='weighted')
        out_c = gen_count(seq)
        out_w = gen_weighted(seq)
        # i=3
        self.assertAlmostEqual(float(out_c[3]), 2.0/4.0, places=6)
        expected_w = (
            Turner2004NNdeltaG37C[b'GG'] +
            Turner2004NNdeltaG37C[b'GG'] +
            Turner2004NNdeltaG37C[b'GA'] +
            Turner2004NNdeltaG37C[b'AT']
        ) / 4.0
        self.assertAlmostEqual(float(out_w[3]), expected_w, places=6)


class TestStreamAddDinucScript(unittest.TestCase):
    def _write_fasta(self, path: Path, seqs):
        with open(path, 'w') as f:
            for sid, s in seqs.items():
                f.write(f">{sid}\n")
                f.write(s + "\n")

    def test_stream_add_dinuc_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            sid = 'ctg'
            seq = 'GGCCAT'
            fa = td / 's.fa'
            self._write_fasta(fa, {sid: seq})
            pkl = td / 'dinuc.pkl'

            script_path = str((Path(__file__).parents[2] / 'scripts' / 'stream-add-dinuc.py').resolve())
            mod = SourceFileLoader('stream_add_dinuc', script_path).load_module()
            argv_backup = sys.argv
            try:
                sys.argv = ['stream-add-dinuc.py', '--fna-fn', str(fa), '--stream-path', str(pkl), '--win', '4', '--mode', 'count']
                mod.main()
            finally:
                sys.argv = argv_backup

            ns = NumericalStream(str(pkl))
            # Default channel name
            self.assertIn('dinuc_count_win_4', ns.channels)
            arr = ns.get(sid)
            self.assertEqual(arr.shape, (len(seq), 1))
            # i=2 expected 2/3
            self.assertAlmostEqual(float(arr[2, 0]), 2.0/3.0, places=6)


class TestStreamExportMotifTableScript(unittest.TestCase):
    def _write_fasta(self, path: Path, seqs):
        with open(path, 'w') as f:
            for sid, s in seqs.items():
                f.write(f">{sid}\n")
                f.write(s + "\n")

    def _write_tsv(self, path: Path, rows):
        with open(path, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")

    def test_stream_export_motif_table_end_to_end(self):
        # Build two contigs with clear START/STOP and splice motifs; annotate one gene per contig
        # c1: NNNN ATG GGG TAA NNN -> START at 4, STOP at 11 (1-based start 5, end 14)
        c1 = "NNNNATGGGTAANNN"
        # c2: donor 'GT' at 6 (0-based), acceptor 'AG' at 10
        c2 = "NNNNNGTNNAGNNN"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fa = td / 's.fa'
            self._write_fasta(fa, {'c1': c1, 'c2': c2})
            # TSV with one plus-strand exon on c1 covering the ATG..TAA, and two exons on c2 separated by intron
            tsv = td / 'a.tsv'
            rows = [
                ('c1', 'g1', 5, 14, 5, 14, '+'),
                ('c2', 'g2', 5, 13, 5, 8, '+'),
                ('c2', 'g2', 5, 13, 11, 13, '+'),
            ]
            self._write_tsv(tsv, rows)

            # Build a simple stream with two channels: c0=position index, c1=2.0
            pkl = td / 'aux.pkl'
            ns = NumericalStream.create_empty(str(pkl))
            def gen_c0(seq: str):
                return np.arange(len(seq), dtype=np.float32)
            def gen_c1(seq: str):
                return np.full(len(seq), 2.0, dtype=np.float32)
            ns.add_channel(str(fa), 'c0_idx', gen_c0)
            ns.add_channel(str(fa), 'c1_two', gen_c1)
            ns.save()

            # Run export script
            out_csv = td / 'motifs.csv'
            script_path = str((Path(__file__).parents[2] / 'scripts' / 'stream-export-motif-table.py').resolve())
            mod = SourceFileLoader('stream_export_motif', script_path).load_module()
            argv_backup = sys.argv
            try:
                sys.argv = ['stream-export-motif-table.py', '--fna', str(fa), '--tsv', str(tsv), '--stream', str(pkl), '--output-csv', str(out_csv)]
                mod.main()
            finally:
                sys.argv = argv_backup

            # Load CSV and check header
            import csv as _csv
            with open(out_csv, 'r', newline='') as f:
                rdr = _csv.reader(f)
                header = next(rdr)
            self.assertEqual(header[:4], ['sequence_id', 'pos', 'motif_type', 'is_positive'])
            self.assertEqual(header[4:], ['c0_idx', 'c1_two'])


    def test_stream_export_with_decoder_probs(self):
        # Single contig with START at 0, STOP at 3, ASS at 5, DSS at 6
        seq = "ATGTAAGT"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fa = td / 's.fa'
            self._write_fasta(fa, {'c1': seq})
            # Simple TSV with one exon covering entire sequence on plus strand
            tsv = td / 'a.tsv'
            rows = [
                ('c1', 'g1', 1, len(seq), 1, len(seq), '+'),
            ]
            self._write_tsv(tsv, rows)

            # Aux stream with constant channel for easy checking
            pkl = td / 'aux.pkl'
            ns = NumericalStream.create_empty(str(pkl))
            def gen_c0(s: str):
                return np.ones(len(s), dtype=np.float32)
            ns.add_channel(str(fa), 'ones', gen_c0)
            ns.save()

            # Build decoder pickle with explicit probabilities per class
            from gene_decoder import PredictedSequence
            class_order = ['INTERGENIC','UTR5','START','GENE','STOP','UTR3','DSS','ASS']
            L = len(seq)
            C = len(class_order)
            probs = np.zeros((L, C), dtype=np.float32)
            # START span 0..2 set to 0.9
            probs[0:3, class_order.index('START')] = 0.9
            # STOP span 3..5 set to 0.8
            probs[3:6, class_order.index('STOP')] = 0.8
            # ASS span 5..6 set to 0.6 (2bp)
            probs[5:7, class_order.index('ASS')] = 0.6
            # DSS span 6..7 set to 0.7 (2bp)
            probs[6:8, class_order.index('DSS')] = 0.7
            items = [PredictedSequence(sequence_index=0, sequence=seq, probabilities=probs, class_order=class_order, sequence_id='c1')]
            dec_pkl = td / 'dec.pkl'
            with open(dec_pkl, 'wb') as f:
                pickle.dump(items, f)

            # Run export with decoder
            out_csv = td / 'motifs.csv'
            script_path = str((Path(__file__).parents[2] / 'scripts' / 'stream-export-motif-table.py').resolve())
            mod = SourceFileLoader('stream_export_motif_with_probs', script_path).load_module()
            argv_backup = sys.argv
            try:
                sys.argv = ['stream-export-motif-table.py', '--fna', str(fa), '--tsv', str(tsv), '--stream', str(pkl), '--decoder-pkl', str(dec_pkl), '--output-csv', str(out_csv)]
                mod.main()
            finally:
                sys.argv = argv_backup

            # Parse CSV and check header only
            import csv as _csv
            with open(out_csv, 'r', newline='') as f:
                rdr = _csv.reader(f)
                header = next(rdr)
            # Header should include motif_prob after is_positive
            self.assertEqual(header[:5], ['sequence_id', 'pos', 'motif_type', 'is_positive', 'motif_prob'])

if __name__ == '__main__':
    unittest.main()


