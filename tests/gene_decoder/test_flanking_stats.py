import os
import tempfile
import unittest

from gene_decoder.flanking_stats import compute_flanking_motif_stats


class TestFlankingStats(unittest.TestCase):
    def _write_fasta(self, path: str, seq_id: str, seq: str) -> None:
        with open(path, 'w') as f:
            f.write(f">{seq_id}\n")
            f.write(seq + "\n")

    def _write_tsv(self, path: str, seq_id: str) -> None:
        # Two-exon gene on + strand: exons [10,15), [25,30)
        # Header: sequence_id, gene_id, gene_start, gene_end, exon_start, exon_end, strand
        with open(path, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            # 1-based inclusive starts, and exon_end is read as exclusive by loader
            # First exon [10,15) -> start 11, end 15
            f.write(f"{seq_id}\tgene1\t11\t30\t11\t15\t+\n")
            # Second exon [25,30) -> start 26, end 30
            f.write(f"{seq_id}\tgene1\t11\t30\t26\t30\t+\n")

    def test_compute_flanking_motif_stats_basic(self):
        # Build a simple 40bp sequence with known motifs and exons
        seq = list('A' * 40)
        # Negative DSS motif at positions 5..7: 'GT'
        seq[5:7] = list('GT')
        # START at 10..13: 'ATG'
        seq[10:13] = list('ATG')
        # Positive DSS at 15..17: 'GT'
        seq[15:17] = list('GT')
        # Negative ASS motif at 8..10: 'AG'
        seq[8:10] = list('AG')
        # Positive ASS at 23..25: 'AG'
        seq[23:25] = list('AG')
        # STOP at 27..30: 'TAA'
        seq[27:30] = list('TAA')
        sequence = ''.join(seq)

        with tempfile.TemporaryDirectory() as d:
            fna = os.path.join(d, 'test.fna')
            tsv = os.path.join(d, 'test.tsv')
            self._write_fasta(fna, 'contig1', sequence)
            self._write_tsv(tsv, 'contig1')

            dss_counts, ass_counts = compute_flanking_motif_stats(
                fna_fn=fna,
                tsv_fn=tsv,
                flank=3,
                site='both',
                dss_motifs_mode='standard',
                num_contigs=0,
            )

            # Expected flanking strings (flank=3): [3bp upstream][2bp motif][3bp downstream]
            # Positive DSS at s=15 -> upstream 12..14: 'GAA', motif 'GT', downstream 17..19: 'AAA'
            self.assertIn('GAAGTAAA', dss_counts)
            self.assertEqual(dss_counts['GAAGTAAA'].total, 1)
            self.assertEqual(dss_counts['GAAGTAAA'].positives, 1)
            self.assertEqual(dss_counts['GAAGTAAA'].negatives, 0)

            # Negative DSS at s=5 -> upstream 2..4: 'AAA', motif 'GT', downstream 7..9 overlaps ASS at 8..9 => 'AAG'
            self.assertIn('AAAGTAAG', dss_counts)
            self.assertEqual(dss_counts['AAAGTAAG'].total, 1)
            self.assertEqual(dss_counts['AAAGTAAG'].positives, 0)
            self.assertEqual(dss_counts['AAAGTAAG'].negatives, 1)

            # Positive ASS at s=23 -> upstream 20..22: 'AAA', motif 'AG', downstream 25..27: 'AAT'
            self.assertIn('AAAAGAAT', ass_counts)
            self.assertEqual(ass_counts['AAAAGAAT'].total, 1)
            self.assertEqual(ass_counts['AAAAGAAT'].positives, 1)
            self.assertEqual(ass_counts['AAAAGAAT'].negatives, 0)

            # Negative ASS at s=8 -> upstream 5..7: 'GTA', motif 'AG', downstream 10..12: 'ATG'
            self.assertIn('GTAAGATG', ass_counts)
            self.assertEqual(ass_counts['GTAAGATG'].total, 1)
            self.assertEqual(ass_counts['GTAAGATG'].positives, 0)
            self.assertEqual(ass_counts['GTAAGATG'].negatives, 1)


if __name__ == '__main__':
    unittest.main()


