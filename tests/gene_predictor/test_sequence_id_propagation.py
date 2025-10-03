import unittest
import tempfile
from pathlib import Path
import pickle
import numpy as np
import torch

from gene_predictor.predict_and_analyze import generate_test_data, run_predictions
from gene_decoder import PredictedSequence


class DummyModel(torch.nn.Module):
    def __init__(self, max_len: int = 64, num_classes: int = 3):
        super().__init__()
        self.model = torch.nn.Module()
        class Emb:
            def __init__(self, m):
                self.max_seq_length = m
        self.model.embedding = Emb(max_len)
        self.num_classes = num_classes
    def forward(self, x, return_attention: bool = False):
        B, L = x.shape
        logits = torch.zeros((B, L, self.num_classes), dtype=torch.float32)
        logits[..., 1] = 1.0
        if return_attention:
            return logits, {}
        return logits


class TestSequenceIdPropagation(unittest.TestCase):
    def test_sequence_id_in_results_and_decoder_pickle(self):
        exon = "ATG" + ("G" * 10) + "TAA"  # 16 bases
        fasta = f">accX|random description\nNNNN{exon}NN\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "accX|random\tgene1\t5\t20\t5\t20\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fna = fp / "seq.fna.gz"
            ann = fp / "ann.tsv"
            # Write gzip FASTA
            import gzip
            with gzip.open(fna, "wt") as f:
                f.write(fasta)
            with open(ann, "w") as f:
                f.write(tsv)

            # Build dataset and run predictions
            dl, ds = generate_test_data(str(fna), str(ann), num_contigs=0)
            model = DummyModel(max_len=64, num_classes=3)
            results = run_predictions(model, dl, device='cpu', return_attention=False,
                                      blending_window_margin_bp=200, aggregator='blend',
                                      random_prefix_ns=False)

            self.assertGreaterEqual(len(results), 1)
            r0 = results[0]
            # sequence_id should come from FASTA header first token: 'accX|random'
            self.assertIn('sequence_id', r0)
            self.assertEqual(r0['sequence_id'], 'accX|random')

            # Create decoder items and pickle, ensure sequence_id is carried
            class_order = ['intergenic', 'gene', 'start']
            items = [
                PredictedSequence(
                    sequence_index=r['sequence_index'],
                    sequence=''.join(['N'] * len(r['sequence_tokens'])),
                    probabilities=r['probabilities'],
                    class_order=class_order,
                    sequence_id=r.get('sequence_id'),
                ) for r in results
            ]
            out_pkl = fp / "items.pkl"
            with open(out_pkl, 'wb') as f:
                pickle.dump(items, f)
            with open(out_pkl, 'rb') as f:
                loaded = pickle.load(f)
            self.assertIsInstance(loaded[0], PredictedSequence)
            self.assertEqual(loaded[0].sequence_id, 'accX|random')


if __name__ == '__main__':
    unittest.main()


