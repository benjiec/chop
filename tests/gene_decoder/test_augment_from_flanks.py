import os
import tempfile
import unittest
import numpy as np
import csv

from gene_decoder import PredictedSequence
from gene_decoder.augment_from_flanks import augment_items_from_flanks


class TestAugmentFromFlanks(unittest.TestCase):
    def test_override_ass_dss_probs(self):
        # Build simple sequence with DSS at 10..12 ('GT') and ASS at 20..22 ('AG')
        seq = list('A' * 40)
        seq[10:12] = list('GT')
        seq[20:22] = list('AG')
        sequence = ''.join(seq)

        L = len(sequence)
        class_order = ['INTERGENIC','UTR5','START','GENE','STOP','UTR3','DSS','ASS']
        probs = np.zeros((L, len(class_order)), dtype=np.float32)
        # initial predictions
        probs[10:12, class_order.index('DSS')] = 0.2
        probs[20:22, class_order.index('ASS')] = 0.3

        items = [PredictedSequence(0, sequence, probs, class_order, 'contigX')]

        # Build a flank CSV that matches our motifs with flank=3
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, 'flanks.csv')
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['site','motif','t','p','n','p_over_t'])
                # DSS flank: upstream 7..9 (AAA), motif GT, downstream 12..14 (AAA) => 'AAAGTAAA'
                w.writerow(['DSS','AAAGTAAA',1,1,0,0.8])
                # ASS flank: upstream 17..19 (AAA), motif AG, downstream 22..24 (AAA) => 'AAAAGAAA'
                w.writerow(['ASS','AAAAGAAA',1,1,0,0.9])

            out = augment_items_from_flanks(
                items,
                flank_counts_csv=csv_path,
                flank=3,
                dss_motifs_mode='standard',
                mode='override',
            )

            self.assertEqual(len(out), 1)
            r = out[0]
            dss_col = class_order.index('DSS')
            ass_col = class_order.index('ASS')

            # Expect overrides to ~1.0 on spans
            self.assertTrue(np.allclose(r.probabilities[10:12, dss_col], 0.8, atol=1e-6))
            self.assertTrue(np.allclose(r.probabilities[20:22, ass_col], 0.9, atol=1e-6))


if __name__ == '__main__':
    unittest.main()


