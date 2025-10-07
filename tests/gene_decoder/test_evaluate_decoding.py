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

    def test_multiple_sequences_with_identical_coordinates_counted_per_sequence(self):
        with tempfile.TemporaryDirectory() as td:
            # Two different sequence_ids share identical coordinates for the true gene
            expected = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
                "contig7\ttrue\t11\t26\t11\t26\t+\n"
                "contig8\ttrue\t11\t26\t11\t26\t+\n"
            )
            # Decoded: both contigs have the correct gene; contig8 also has an extra FP start
            decoded = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\tk\tstart_rank\tboundary_score\ttransition_score\tcodon_penalty\n"
                # contig7 correct gene at start 11, boundary 10.0
                "contig7\tgene_1\t11\t26\t11\t26\t+\t1\t1\t10.0\t0.0\t\n"
                # contig8 correct gene at start 11, boundary 10.0
                "contig8\tgene_1\t11\t26\t11\t26\t+\t1\t1\t10.0\t0.0\t\n"
                # contig8 extra FP gene at a different start 201, boundary 9.0
                "contig8\tgene_2\t201\t240\t201\t240\t+\t2\t1\t9.0\t0.0\t\n"
            )
            exp_path = os.path.join(td, 'exp.tsv')
            dec_path = os.path.join(td, 'dec.tsv')
            _write(exp_path, expected)
            _write(dec_path, decoded)

            # With topk_starts=1, only the highest-boundary start per sequence is included
            res1 = evaluate_decoding(dec_path, exp_path, topk_starts=1, top_start_rank_only=True)
            # Each sequence contributes its own counts; duplicates across sequence_ids are not collapsed
            self.assertEqual(res1['exon']['tp'], 2)
            self.assertEqual(res1['exon']['fp'], 0)
            self.assertEqual(res1['exon']['fn'], 0)
            self.assertEqual(res1['gene']['tp'], 2)
            self.assertEqual(res1['gene']['fp'], 0)
            self.assertEqual(res1['gene']['fn'], 0)

            # With topk_starts=2, contig8 includes its FP start as well
            res2 = evaluate_decoding(dec_path, exp_path, topk_starts=2, top_start_rank_only=True)
            self.assertEqual(res2['exon']['tp'], 2)
            self.assertEqual(res2['exon']['fp'], 1)
            self.assertEqual(res2['exon']['fn'], 0)
            self.assertEqual(res2['gene']['tp'], 2)
            self.assertEqual(res2['gene']['fp'], 1)
            self.assertEqual(res2['gene']['fn'], 0)
            # start-level: contig7 TP start; contig8 TP+FP starts
            self.assertEqual(res2['start']['tp'], 2)
            self.assertEqual(res2['start']['fp'], 1)
            self.assertEqual(res2['start']['fn'], 0)

    def test_topk_starts_selection_affects_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            # expected: only one correct gene at start 101
            expected = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
                "contig4\ttrue\t101\t120\t101\t120\t+\n"
            )
            # decoded: two starts
            # - start 101 (boundary 10.0): correct
            # - start 201 (boundary 9.0): incorrect gene -> FP when included
            decoded = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\tk\tstart_rank\tboundary_score\ttransition_score\tcodon_penalty\n"
                "contig4\tgene_1\t101\t120\t101\t120\t+\t1\t1\t10.0\t0.0\t\n"
                "contig4\tgene_1\t201\t240\t201\t240\t+\t2\t1\t9.0\t0.0\t\n"
            )
            exp_path = os.path.join(td, 'exp.tsv')
            dec_path = os.path.join(td, 'dec.tsv')
            _write(exp_path, expected)
            _write(dec_path, decoded)

            # topk_starts=1 selects only start 101 -> perfect match
            res1 = evaluate_decoding(dec_path, exp_path, topk_starts=1, top_start_rank_only=True)
            self.assertEqual(res1['exon']['tp'], 1)
            self.assertEqual(res1['exon']['fp'], 0)
            self.assertEqual(res1['exon']['fn'], 0)
            self.assertEqual(res1['gene']['tp'], 1)
            self.assertEqual(res1['gene']['fp'], 0)
            self.assertEqual(res1['gene']['fn'], 0)

            # topk_starts=2 includes both starts -> adds one FP exon and FP gene
            res2 = evaluate_decoding(dec_path, exp_path, topk_starts=2, top_start_rank_only=True)
            self.assertEqual(res2['exon']['tp'], 1)
            self.assertEqual(res2['exon']['fp'], 1)
            self.assertEqual(res2['exon']['fn'], 0)
            self.assertEqual(res2['gene']['tp'], 1)
            self.assertEqual(res2['gene']['fp'], 1)
            self.assertEqual(res2['gene']['fn'], 0)
            # start-level: two predicted starts, one expected -> one TP, one FP
            self.assertEqual(res2['start']['tp'], 1)
            self.assertEqual(res2['start']['fp'], 1)
            self.assertEqual(res2['start']['fn'], 0)

    def test_gene_tp_requires_exact_splice_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            # expected two-exon plus strand gene
            expected = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
                "contig5\tgeneD\t6\t30\t6\t12\t+\n"
                "contig5\tgeneD\t6\t30\t21\t30\t+\n"
            )
            # decoded: same start, but second exon end off by +1 (31) -> gene should NOT match
            decoded = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\tk\tstart_rank\tboundary_score\ttransition_score\tcodon_penalty\n"
                "contig5\tgene_1\t6\t31\t6\t12\t+\t1\t1\t5.0\t0.0\t\n"
                "contig5\tgene_1\t6\t31\t21\t31\t+\t1\t1\t5.0\t0.0\t\n"
            )
            exp_path = os.path.join(td, 'exp.tsv')
            dec_path = os.path.join(td, 'dec.tsv')
            _write(exp_path, expected)
            _write(dec_path, decoded)

            res = evaluate_decoding(dec_path, exp_path, topk_starts=1, top_start_rank_only=True)
            # exon-level: one TP (first exon), one FP (mismatched second), one FN (missing expected second)
            self.assertEqual(res['exon']['tp'], 1)
            self.assertEqual(res['exon']['fp'], 1)
            self.assertEqual(res['exon']['fn'], 1)
            # gene-level: no TP; predicted gene doesn't exactly match expected exon set
            self.assertEqual(res['gene']['tp'], 0)
            self.assertEqual(res['gene']['fp'], 1)
            self.assertEqual(res['gene']['fn'], 1)

    def test_emulated_topk_splicing_multiple_candidates_under_start(self):
        with tempfile.TemporaryDirectory() as td:
            # expected single-exon gene at a start
            expected = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
                "contig6\tgeneE\t51\t80\t51\t80\t+\n"
            )
            # decoded under the same start has two candidates (emulates top_k_splicing=2):
            # rank 1 is incorrect, rank 2 is the correct gene
            decoded = (
                "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\tk\tstart_rank\tboundary_score\ttransition_score\tcodon_penalty\n"
                "contig6\tgene_1\t51\t79\t51\t79\t+\t1\t1\t6.0\t0.0\t\n"
                "contig6\tgene_2\t51\t80\t51\t80\t+\t2\t2\t5.5\t0.0\t\n"
            )
            exp_path = os.path.join(td, 'exp.tsv')
            dec_path = os.path.join(td, 'dec.tsv')
            _write(exp_path, expected)
            _write(dec_path, decoded)

            # top_start_rank_only: include only rank 1 -> wrong gene only
            res_top = evaluate_decoding(dec_path, exp_path, topk_starts=1, top_start_rank_only=True)
            self.assertEqual(res_top['exon']['tp'], 0)
            self.assertEqual(res_top['exon']['fp'], 1)
            self.assertEqual(res_top['exon']['fn'], 1)
            self.assertEqual(res_top['gene']['tp'], 0)
            self.assertEqual(res_top['gene']['fp'], 1)
            self.assertEqual(res_top['gene']['fn'], 1)
            # start-level: predicted start exists and matches expected
            self.assertEqual(res_top['start']['tp'], 1)
            self.assertEqual(res_top['start']['fp'], 0)
            self.assertEqual(res_top['start']['fn'], 0)

            # include all candidates under the start -> both genes considered
            res_all = evaluate_decoding(dec_path, exp_path, topk_starts=1, top_start_rank_only=False)
            self.assertEqual(res_all['exon']['tp'], 1)
            self.assertEqual(res_all['exon']['fp'], 1)
            self.assertEqual(res_all['exon']['fn'], 0)
            self.assertEqual(res_all['gene']['tp'], 1)
            self.assertEqual(res_all['gene']['fp'], 1)
            self.assertEqual(res_all['gene']['fn'], 0)
            # start-level: still a single start -> TP only
            self.assertEqual(res_all['start']['tp'], 1)
            self.assertEqual(res_all['start']['fp'], 0)
            self.assertEqual(res_all['start']['fn'], 0)


if __name__ == '__main__':
    unittest.main()


