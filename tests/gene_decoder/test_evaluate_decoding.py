import os
import tempfile
import unittest

from gene_decoder.evaluate_decoding import evaluate_decoding


def _write(path: str, text: str):
    with open(path, 'w') as f:
        f.write(text)


class TestEvaluateDecoding(unittest.TestCase):
    def test_single_exon_exact_match_top_rank_only(self):
        with tempfile.TemporaryDirectory() as td:
            expected = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
                "contig1\tgeneA\t5\t20\t5\t20\t+\n"
            )
            decoded = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\tk\tstart_rank\tboundary_score\ttransition_score\tcodon_penalty\n"
                "contig1\tgene_1\t5\t20\t5\t20\t+\t1\t1\t1.0\t0.0\t\n"
            )
            exp_path = os.path.join(td, 'exp.tsv')
            dec_path = os.path.join(td, 'dec.tsv')
            _write(exp_path, expected)
            _write(dec_path, decoded)

            res = evaluate_decoding(dec_path, exp_path, topk_starts=1, top_start_rank_only=True, per_sequence=True)
            self.assertEqual(res['exon']['tp'], 1)
            self.assertEqual(res['exon']['fp'], 0)
            self.assertEqual(res['exon']['fn'], 0)
            self.assertEqual(res['gene']['tp'], 1)
            self.assertEqual(res['gene']['fp'], 0)
            self.assertEqual(res['gene']['fn'], 0)

    def test_multi_exon_match_all_candidates_vs_top_only(self):
        with tempfile.TemporaryDirectory() as td:
            # expected gene with two exons on '+'
            expected = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
                "contig2\tgeneB\t6\t30\t6\t12\t+\n"
                "contig2\tgeneB\t6\t30\t21\t30\t+\n"
            )
            # decoded: same start position with two candidate genes under the start
            # - rank 1 matches expected
            # - rank 2 is a spurious different gene
            decoded = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\tk\tstart_rank\tboundary_score\ttransition_score\tcodon_penalty\n"
                "contig2\tgene_1\t6\t30\t6\t12\t+\t1\t1\t5.0\t0.0\t\n"
                "contig2\tgene_1\t6\t30\t21\t30\t+\t1\t1\t5.0\t0.0\t\n"
                "contig2\tgene_2\t6\t18\t6\t18\t+\t2\t2\t4.0\t0.0\t\n"
            )
            exp_path = os.path.join(td, 'exp.tsv')
            dec_path = os.path.join(td, 'dec.tsv')
            _write(exp_path, expected)
            _write(dec_path, decoded)

            res_all = evaluate_decoding(dec_path, exp_path, topk_starts=1, top_start_rank_only=False)
            # exons: predicted set includes both gene_1 exons and gene_2 exon; gene_2 adds an FP exon
            self.assertEqual(res_all['exon']['tp'], 2)
            self.assertEqual(res_all['exon']['fp'], 1)
            self.assertEqual(res_all['exon']['fn'], 0)
            # gene-level: one TP (gene_1), one FP (gene_2)
            self.assertEqual(res_all['gene']['tp'], 1)
            self.assertEqual(res_all['gene']['fp'], 1)
            self.assertEqual(res_all['gene']['fn'], 0)

            res_top = evaluate_decoding(dec_path, exp_path, topk_starts=1, top_start_rank_only=True)
            self.assertEqual(res_top['exon']['tp'], 2)
            self.assertEqual(res_top['exon']['fp'], 0)
            self.assertEqual(res_top['exon']['fn'], 0)
            self.assertEqual(res_top['gene']['tp'], 1)
            self.assertEqual(res_top['gene']['fp'], 0)
            self.assertEqual(res_top['gene']['fn'], 0)

    def test_strand_aware_matching(self):
        with tempfile.TemporaryDirectory() as td:
            expected = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
                "contig3\tgeneC\t11\t26\t11\t26\t-\n"
            )
            # decoded predicts same span but '+' strand; should not match
            decoded = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\tk\tstart_rank\tboundary_score\ttransition_score\tcodon_penalty\n"
                "contig3\tgene_1\t11\t26\t11\t26\t+\t1\t1\t2.0\t0.0\t\n"
            )
            exp_path = os.path.join(td, 'exp.tsv')
            dec_path = os.path.join(td, 'dec.tsv')
            _write(exp_path, expected)
            _write(dec_path, decoded)

            res = evaluate_decoding(dec_path, exp_path, topk_starts=1, top_start_rank_only=True)
            self.assertEqual(res['exon']['tp'], 0)
            self.assertEqual(res['exon']['fp'], 1)
            self.assertEqual(res['exon']['fn'], 1)
            self.assertEqual(res['gene']['tp'], 0)
            self.assertEqual(res['gene']['fp'], 1)
            self.assertEqual(res['gene']['fn'], 1)


if __name__ == '__main__':
    unittest.main()


