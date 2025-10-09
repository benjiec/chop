#!/usr/bin/env python3

import unittest

from utils.sequences import validate_start_stop_codons_from_exons, reverse_complement


class TestCodonValidation(unittest.TestCase):
    def _validate_single_exon_gene(self, sequence, start, end, strand='+'):
        exons = [{'start': start, 'end': end}]
        return validate_start_stop_codons_from_exons(sequence, exons, strand)

    def test_reverse_complement(self):
        self.assertEqual(reverse_complement("ATG"), "CAT")
        self.assertEqual(reverse_complement("TAA"), "TTA")
        self.assertEqual(reverse_complement("TAG"), "CTA")
        self.assertEqual(reverse_complement("TGA"), "TCA")
        self.assertEqual(reverse_complement("ATGAAATAG"), "CTATTTCAT")
        self.assertEqual(reverse_complement("ATGN"), "NCAT")
        self.assertEqual(reverse_complement("atg"), "CAT")

    def test_forward_strand_valid_codons(self):
        sequence = "NNATGAAACCCTAAGG"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 14, '+'))
        sequence = "NNATGAAACCCTAGGG"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 14, '+'))
        sequence = "NNATGAAACCCTGAGG"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 14, '+'))

    def test_forward_strand_invalid_codons(self):
        sequence = "NNTTGAAACCCTAAGG"
        self.assertFalse(self._validate_single_exon_gene(sequence, 2, 14, '+'))
        sequence = "NNATGAAACCCTATGG"
        self.assertFalse(self._validate_single_exon_gene(sequence, 2, 14, '+'))
        sequence = "NNTTGAAACCCTATGG"
        self.assertFalse(self._validate_single_exon_gene(sequence, 2, 14, '+'))

    def test_reverse_strand_valid_codons(self):
        sequence = "NNTCAAAACCCATGG"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 13, '-'))
        sequence = "NNTTAAAACCCATGG"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 13, '-'))
        sequence = "NNCTAAAACCCATGG"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 13, '-'))

    def test_reverse_strand_invalid_codons(self):
        sequence = "NNTCAAAACCCCTTGG"
        self.assertFalse(self._validate_single_exon_gene(sequence, 2, 14, '-'))
        sequence = "NNGAAAAACCCATGG"
        self.assertFalse(self._validate_single_exon_gene(sequence, 2, 13, '-'))

    def test_boundary_conditions(self):
        sequence = "AT"
        self.assertFalse(self._validate_single_exon_gene(sequence, 0, 2, '+'))
        sequence = "ATGAA"
        self.assertFalse(self._validate_single_exon_gene(sequence, 3, 5, '+'))
        sequence = "ATGAA"
        self.assertFalse(self._validate_single_exon_gene(sequence, 0, 2, '+'))

    def test_edge_cases(self):
        sequence = "ATGTAA"
        self.assertTrue(self._validate_single_exon_gene(sequence, 0, 6, '+'))
        sequence = "NNATGTAANN"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 8, '+'))
        sequence = "ATGTAA"
        self.assertTrue(self._validate_single_exon_gene(sequence, 0, 6))


class TestIntegration(unittest.TestCase):
    def _validate_single_exon_gene(self, sequence, start, end, strand='+'):
        exons = [{'start': start, 'end': end}]
        return validate_start_stop_codons_from_exons(sequence, exons, strand)

    def test_real_gene_validation(self):
        upstream = "N" * 2000
        gene_seq = "ATG" + "AAA" * 100 + "TAA"
        downstream = "N" * 2000
        full_sequence = upstream + gene_seq + downstream
        gene_start = 2000
        gene_end = 2000 + len(gene_seq)
        self.assertTrue(self._validate_single_exon_gene(full_sequence, gene_start, gene_end, '+'))
        self.assertFalse(self._validate_single_exon_gene(full_sequence, gene_start, gene_end, '-'))

    def test_multiple_strand_combinations(self):
        test_cases = [
            ("ATGTAA", 0, 6, '+', True),
            ("ATGTAG", 0, 6, '+', True),
            ("ATGTGA", 0, 6, '+', True),
            ("TTACAT", 0, 6, '-', True),
            ("CTACAT", 0, 6, '-', True),
            ("TCACAT", 0, 6, '-', True),
            ("ATGTTT", 0, 6, '+', False),
            ("TTACTT", 0, 6, '-', False),
        ]
        for sequence, start, end, strand, expected in test_cases:
            with self.subTest(seq=sequence, strand=strand):
                result = self._validate_single_exon_gene(sequence, start, end, strand)
                self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()


