#!/usr/bin/env python3

import unittest
import tempfile
import os
from pathlib import Path

import numpy as np

from utils.gff import GFFDataset
from utils.constants import GenePredictionClass as P


def write_temp(path: Path, content: str):
    with open(path, 'w') as f:
        f.write(content)


class TestGFFDataset(unittest.TestCase):
    def test_single_exon_plus_strand(self):
        # Build sequence with exon: start ATG at idx 4, stop TAA ending at idx 19 (0-based)
        # 1-based exon coordinates: start=5, end=20
        exon = "ATG" + "G"*10 + "TAA"  # 16 bases
        fasta = f">contig1\nNNNN{exon}NN\n"
        tsv = "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n" \
              "contig1\tgene1\t5\t20\t5\t20\t+\n"
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "seq.fna"
            tsv_path = fp / "ann.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)
            ds = GFFDataset(str(fasta_path), str(tsv_path))
            self.assertEqual(len(ds), 1)
            seq_enc, tgt = ds[0]
            seq = f"NNNN{exon}NN"
            self.assertEqual(len(seq_enc), len(seq))
            # START at pos 4..6, STOP at pos 17..19
            self.assertTrue(all(tgt[4:7] == P.START))
            self.assertTrue(all(tgt[17:20] == P.STOP))
            # GENE between
            self.assertTrue(all(tgt[7:17] == P.GENE))
            # INTERGENIC outside
            self.assertTrue(all(tgt[:4] == P.INTERGENIC))
            self.assertTrue(all(tgt[20:] == P.INTERGENIC))

    def test_single_exon_minus_strand(self):
        # Minus strand exon; forward first 3 bases 'TTA' (RC->TAA STOP), last 3 bases 'CAT' (RC->ATG START)
        # 1-based exon: start=11, end=26 (length 16)
        exon_fwd = "TTA" + "G"*10 + "CAT"
        seq = f"{'N'*10}{exon_fwd}{'N'*4}"
        fasta = f">contig2\n{seq}\n"
        tsv = "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n" \
              "contig2\tgene2\t11\t26\t11\t26\t-\n"
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "seq.fna"
            tsv_path = fp / "ann.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)
            ds = GFFDataset(str(fasta_path), str(tsv_path))
            seq_enc, tgt = ds[0]
            # For minus strand, START occupies last 3 bases of exon (indices 23..25), STOP occupies first 3 (10..12)
            self.assertTrue(all(tgt[23:26] == P.START))
            self.assertTrue(all(tgt[10:13] == P.STOP))

    def test_multi_exon_donor_acceptor(self):
        # Two exons (+): exon1 1-based 6..12 (0-based [5,12)), exon2 21..30 ([20,30))
        # START at 5..7 'ATG'; STOP at 27..29 'TAA'
        # Donor at 12..13 must be 'GT'; acceptor at 19..20 must be 'AG'
        seq_list = list('N'*35)
        # Fill motifs
        seq_list[5:8] = list('ATG')
        seq_list[17:20] = list('TAA')  # careful: stop should be at 27..29; we'll set later
        seq_list[12:14] = list('GT')
        seq_list[18:20] = list('AG')
        # Place STOP at 27..29
        seq_list[27:30] = list('TAA')
        fasta = f">contig3\n{''.join(seq_list)}\n"
        tsv = "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n" \
              "contig3\tgene3\t6\t30\t6\t12\t+\n" \
              "contig3\tgene3\t6\t30\t21\t30\t+\n"
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fasta_path = fp / "seq.fna"
            tsv_path = fp / "ann.tsv"
            write_temp(fasta_path, fasta)
            write_temp(tsv_path, tsv)
            ds = GFFDataset(str(fasta_path), str(tsv_path))
            _, tgt = ds[0]
            # Donor at 12..13, Acceptor at 18..19 (we label 2bp windows)
            self.assertTrue(all(tgt[12:14] == P.DSS))
            self.assertTrue(all(tgt[18:20] == P.ASS))

    def test_dss_variants_plus_strand_GA(self):
        # Donor 'GA' at exon1 end, acceptor 'AG' at exon2 start-2
        seq = list('N'*40)
        # Exon1: 1-based 6..12 (0-based 5..12); START at 5..7
        seq[5:8] = list('ATG')
        # Donor at 12..13 = 'GA'
        seq[12:14] = list('GA')
        # Acceptor at 18..19 = 'AG'
        seq[18:20] = list('AG')
        # Exon2: 1-based 21..33; STOP at 31..33
        seq[30:33] = list('TAA')
        fasta = f">cGA\n{''.join(seq)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "cGA\tgeneGA\t6\t33\t6\t12\t+\n"
            "cGA\tgeneGA\t6\t33\t21\t33\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            write_temp(fp/"seq.fna", fasta)
            write_temp(fp/"ann.tsv", tsv)
            ds = GFFDataset(str(fp/"seq.fna"), str(fp/"ann.tsv"))
            _, tgt = ds[0]
            self.assertTrue(all(tgt[12:14] == P.DSS))
            self.assertTrue(all(tgt[18:20] == P.ASS))

    def test_dss_variants_plus_strand_GC(self):
        # Donor 'GC' at exon1 end, acceptor 'AG' at exon2 start-2
        seq = list('N'*40)
        seq[5:8] = list('ATG')
        seq[12:14] = list('GC')
        seq[18:20] = list('AG')
        seq[30:33] = list('TAA')
        fasta = f">cGC\n{''.join(seq)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "cGC\tgeneGC\t6\t33\t6\t12\t+\n"
            "cGC\tgeneGC\t6\t33\t21\t33\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            write_temp(fp/"seq.fna", fasta)
            write_temp(fp/"ann.tsv", tsv)
            ds = GFFDataset(str(fp/"seq.fna"), str(fp/"ann.tsv"))
            _, tgt = ds[0]
            self.assertTrue(all(tgt[12:14] == P.DSS))
            self.assertTrue(all(tgt[18:20] == P.ASS))

    def test_multi_exon_reverse_strand(self):
        # Two exons on minus strand; RC should satisfy START/STOP and donor/acceptor
        # Exons (1-based): 11..18 and 27..36 (0-based [10,18), [26,36))
        seq = list('N'*50)
        # Place forward strand such that RC at last exon end = ATG, RC at first exon start = TAA
        # For minus strand: last exon end (33..35) slice RC must be ATG => forward must be CAT at 33..35
        seq[33:36] = list('CAT')
        # First exon start (10..12 RC window) must be TAA => forward must be TTA at 10..12
        seq[10:13] = list('TTA')
        # Intron donor/acceptor for minus: donor at e2[0]-2 (26-2=24..25) RC in {GT,GC,GA} -> forward must be complement of those reversed.
        # Choose donor RC='GT' => forward at 24..26 should be 'AC' (RC of 'GT' is 'AC') but careful: RC('AC')='GT'.
        seq[24:26] = list('AC')
        # Acceptor at e1[1] (18..20) RC='AG' => forward at 18..20 must be 'CT' (RC of 'AG' is 'CT')
        seq[18:20] = list('CT')
        fasta = f">cMinus\n{''.join(seq)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "cMinus\tgn\t11\t36\t11\t18\t-\n"
            "cMinus\tgn\t11\t36\t27\t36\t-\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            write_temp(fp/"seq.fna", fasta)
            write_temp(fp/"ann.tsv", tsv)
            ds = GFFDataset(str(fp/"seq.fna"), str(fp/"ann.tsv"))
            _, tgt = ds[0]
            # START at last exon end: last exon [26,36) -> start_pos=33, labels 33..35 (slice 33:36)
            self.assertTrue(all(tgt[33:36] == P.START))
            # STOP at first exon start indices 10..12
            self.assertTrue(all(tgt[10:13] == P.STOP))

    def test_invalid_gene_end_without_stop(self):
        # Last exon does not end with a STOP codon -> expect assertion error
        seq = list('N'*30)
        # Place START ok
        seq[5:8] = list('ATG')
        # Last exon end will be 'AAA' not a stop
        seq[17:20] = list('AAA')
        fasta = f">cBad\n{''.join(seq)}\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "cBad\tbad\t6\t20\t6\t20\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            write_temp(fp/"seq.fna", fasta)
            write_temp(fp/"ann.tsv", tsv)
            with self.assertRaises(AssertionError):
                GFFDataset(str(fp/"seq.fna"), str(fp/"ann.tsv"))


if __name__ == '__main__':
    unittest.main()
