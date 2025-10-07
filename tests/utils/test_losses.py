#!/usr/bin/env python3

import unittest
import torch

from utils.losses import event_based_ce_loss_factory, event_based_bce_loss_factory
from utils.constants import GenePredictionClass as P


class TestEventBasedCELoss(unittest.TestCase):
    def test_loss_masks_to_events_only(self):
        # Build loss with DSS motifs: only 'GT'
        loss_fn = event_based_ce_loss_factory({'GT'})

        # Construct a batch with one obvious START at pos 0: ATG, and avoid STOP at pos1
        # Map DNAEmbed: A=0,T=1,G=2,C=3,N=4
        A, T, G, C = 0, 1, 2, 3
        sequences = torch.tensor([[A, T, G, C, C, C]])  # B=1, L=6; triplets: ATG, TGC, GCC, CCC -> no STOP
        # With span inclusion, event is positions 0..2
        START = 2
        targets = torch.tensor([[START, START, START, 0, 0, 0]])

        # Logits: make correct class at pos0 be high; others arbitrary
        Cn = 8
        logits = torch.zeros((1, 6, Cn), dtype=torch.float32)
        logits[0, 0:3, START] = 5.0  # correct across span
        logits[0, 3:, :] = 0.0  # non-events shouldn't affect loss

        comp = {}
        loss = loss_fn(sequences, targets, logits, comp)
        self.assertTrue(torch.is_tensor(loss))
        # Loss should be close to CE at event pos only
        import torch.nn.functional as F
        ce_span = F.cross_entropy(logits[0, 0:3, :], targets[0, 0:3], reduction='mean')
        self.assertAlmostEqual(float(loss.detach().cpu().item()), float(ce_span.detach().cpu().item()), places=6)

    def test_counts_all_event_types_and_targets_used(self):
        # Loss with DSS motifs set to {'GT'}
        loss_fn = event_based_ce_loss_factory({'GT'})

        # Build a sequence likely containing multiple events due to overlaps
        # Bases: A=0, T=1, G=2, C=3
        A, T, G, C = 0, 1, 2, 3
        seq = [A, T, G,  T, A, A,  G, T,  A, G,  C, C]
        sequences = torch.tensor([seq])

        # Targets: mark each event span with its class (others default 0)
        START, STOP, DSS, ASS = 2, 4, 6, 7
        targets = torch.zeros((1, len(seq)), dtype=torch.long)
        # START span 0..2
        targets[0, 0:3] = START
        # STOP span 3..5 (TAA)
        targets[0, 3:6] = STOP
        # DSS span 6..7 (GT)
        targets[0, 6:8] = DSS
        # ASS span 8..9 (AG)
        targets[0, 8:10] = ASS

        Cn = 8
        logits = torch.zeros((1, len(seq), Cn), dtype=torch.float32)
        # Make the correct classes confident at intended event spans
        logits[0, 0:3, START] = 5.0
        logits[0, 3:6, STOP] = 5.0
        logits[0, 6:8, DSS] = 5.0
        logits[0, 8:10, ASS] = 5.0

        # Compute loss
        comp = {}
        loss = loss_fn(sequences, targets, logits, comp)

        # Independently compute expected by re-scanning events using the same motif rules
        from utils.constants import DNAEmbed, ConventionalStopCodons, ConventionalAcceptorDinucleotides
        bases = [DNAEmbed.idx_to_bp[x] for x in seq]
        event_positions = set()
        for i in range(0, len(seq) - 2):
            tri = ''.join(bases[i:i+3])
            if tri == 'ATG' or tri in ConventionalStopCodons:
                event_positions.update(range(i, i+3))
        for i in range(0, len(seq) - 1):
            di = ''.join(bases[i:i+2])
            if di in {'GT'} or di in ConventionalAcceptorDinucleotides:
                event_positions.update(range(i, i+2))

        import torch.nn.functional as F
        if event_positions:
            idxs = torch.tensor(sorted(event_positions))
            rows = logits[0, idxs, :]
            y = targets[0, idxs]
            expected = F.cross_entropy(rows, y, reduction='mean')
            self.assertAlmostEqual(float(loss.detach().cpu().item()), float(expected.detach().cpu().item()), places=6)
            self.assertEqual(comp.get('event_count'), len(event_positions))
        else:
            self.assertAlmostEqual(float(loss.detach().cpu().item()), 0.0, places=8)

    def test_motif_span_inclusion_for_triplets_and_dimers(self):
        # Ensure CE is accumulated over full spans of motifs (3 for START/STOP, 2 for DSS/ASS)
        loss_fn = event_based_ce_loss_factory({'GT'})
        A, T, G, C = 0, 1, 2, 3
        # ATG at 0..2, GT at 4..5, AG at 7..8
        seq = [A, T, G,  C,  G, T,  C,  A, G]
        sequences = torch.tensor([seq])
        Cn = 8
        logits = torch.zeros((1, len(seq), Cn), dtype=torch.float32)
        # Targets: set distinct correct classes per span
        START, DSS, ASS = 2, 6, 7
        targets = torch.zeros((1, len(seq)), dtype=torch.long)
        # Encode START span 0..2
        targets[0, 0] = START
        targets[0, 1] = START
        targets[0, 2] = START
        # DSS span 4..5
        targets[0, 4] = DSS
        targets[0, 5] = DSS
        # ASS span 7..8
        targets[0, 7] = ASS
        targets[0, 8] = ASS
        # Make logits correct on these spans; other positions arbitrary
        logits[0, 0:3, START] = 5.0
        logits[0, 4:6, DSS] = 5.0
        logits[0, 7:9, ASS] = 5.0

        comp = {}
        loss = loss_fn(sequences, targets, logits, comp)
        # Expect small loss since event-span positions match their targets with strong logits
        self.assertLess(float(loss.detach().cpu().item()), 0.1)

    def test_variable_length_stop_and_dss_motifs_span(self):
        # Monkeypatch STOP to include a 5bp motif and DSS to include a 7bp motif
        from utils import constants as K
        orig_stops = K.ConventionalStopCodons.copy()
        orig_acc = K.ConventionalAcceptorDinucleotides.copy()
        try:
            K.ConventionalStopCodons.clear()
            K.ConventionalStopCodons.update({'ATGCC'})  # 5bp stop-like motif

            # DSS: pass 7bp motif via factory
            dss_motif = 'GTAAGGG'
            loss_fn = event_based_ce_loss_factory({dss_motif})

            # Build sequence embedding both motifs with clear separation
            # A=0,T=1,G=2,C=3
            A, T, G, C = 0, 1, 2, 3
            # Positions: 0..4 -> ATGCC, 10..16 -> GTAAGGG
            seq = [A, T, G, C, C,  C, C, C, C, C,  G, T, A, A, G, G, G]
            sequences = torch.tensor([seq])
            L = len(seq)

            Cn = 8
            logits = torch.zeros((1, L, Cn), dtype=torch.float32)
            targets = torch.zeros((1, L), dtype=torch.long)
            STOP, DSS = 4, 6
            # STOP span 0..4
            targets[0, 0:5] = STOP
            logits[0, 0:5, STOP] = 5.0
            # DSS span 10..16
            targets[0, 10:17] = DSS
            logits[0, 10:17, DSS] = 5.0

            comp = {}
            loss = loss_fn(sequences, targets, logits, comp)
            # Expect event_count equals 5 + 7 = 12
            self.assertEqual(comp.get('event_count'), 12)
            # Expect low loss since both spans are correct
            self.assertLess(float(loss.detach().cpu().item()), 0.1)
        finally:
            # restore
            K.ConventionalStopCodons.clear()
            K.ConventionalStopCodons.update(orig_stops)

    def test_dss_motifs_selection_changes_loss(self):
        # Two versions: GT-only vs GT+GA
        loss_gt = event_based_ce_loss_factory({'GT'})
        loss_dino = event_based_ce_loss_factory({'GT', 'GA'})

        # Sequence with GA at pos0 and GT at pos4; no other events
        # G=2, A=0, C=3, T=1
        G, A, C, T = 2, 0, 3, 1
        seq = [G, A,  C, C,  G, T,  C, C,  C, C,  C, C]
        sequences = torch.tensor([seq])
        L = len(seq)
        # Targets: all INTERGENIC (0) so events count as FP positions but still included
        targets = torch.zeros((1, L), dtype=torch.long)

        Cn = 8
        logits = torch.zeros((1, L, Cn), dtype=torch.float32)
        # Make GT event (pos4) correct for class 0 => very low CE
        logits[0, 4, 0] = 5.0
        # Make GA event (pos0) wrong (class 1 high) => higher CE
        logits[0, 0, 1] = 5.0

        comp1 = {}
        v1 = loss_gt(sequences, targets, logits, comp1)
        self.assertEqual(comp1.get('event_count'), 2)  # GT span counted

        comp2 = {}
        v2 = loss_dino(sequences, targets, logits, comp2)
        self.assertEqual(comp2.get('event_count'), 4)  # GA span now included
        # Including the worse GA event should increase the mean loss
        self.assertGreater(float(v2.detach().cpu().item()), float(v1.detach().cpu().item()))


class TestEventBasedWeightedLosses(unittest.TestCase):
    def test_weighted_ce_increases_when_high_ce_class_weight_is_higher(self):
        # Sequence with START at 0..2 and STOP at 5..7
        A, T, G, C = 0, 1, 2, 3
        seq = [A, T, G,  C, C,  T, A, A,  C]
        sequences = torch.tensor([seq])
        L = len(seq)
        Cn = 8

        targets = torch.zeros((1, L), dtype=torch.long)
        targets[0, 0:3] = P.START
        targets[0, 5:8] = P.STOP

        logits = torch.zeros((1, L, Cn), dtype=torch.float32)
        # Make START tokens relatively wrong (higher CE): p(true) ~ 0.4
        logits[0, 0:3, P.START] = 0.0
        logits[0, 0:3, 0] = 0.5  # distractor
        # Make STOP tokens relatively correct (lower CE): p(true) ~ 0.9
        logits[0, 5:8, P.STOP] = 5.0

        # Unweighted CE
        loss_unweighted = event_based_ce_loss_factory({'GT'})(sequences, targets, logits, {})
        # Weighted CE: emphasize START class more
        cw = {P.START: 2.0, P.STOP: 1.0}
        loss_weighted = event_based_ce_loss_factory({'GT'}, class_weights=cw)(sequences, targets, logits, {})
        self.assertGreater(float(loss_weighted.detach().cpu().item()), float(loss_unweighted.detach().cpu().item()))

    def test_bce_neg_weight_increases_loss_for_false_positives(self):
        # Two ATG spans: first is true START, second is not START
        A, T, G, C = 0, 1, 2, 3
        # Positions 0..2 -> ATG, 6..8 -> ATG
        seq = [A, T, G,  C, C,  C,  A, T, G]
        sequences = torch.tensor([seq])
        L = len(seq)
        Cn = 8

        targets = torch.zeros((1, L), dtype=torch.long)
        targets[0, 0:3] = P.START   # positive START span
        # second ATG remains INTERGENIC (negative for START)

        logits = torch.zeros((1, L, Cn), dtype=torch.float32)
        # Positive span: make START likely (low loss)
        logits[0, 0:3, P.START] = 4.0
        # Negative span: make START somewhat likely (false positive-like)
        logits[0, 6:9, P.START] = 1.5

        # pos weights from class weights; set START pos=2, neg weight varies
        pos_w = {P.START: 2.0}
        bce_low_neg = event_based_bce_loss_factory({'GT'}, pos_weights=pos_w, neg_weights={P.START: 1.0})
        bce_high_neg = event_based_bce_loss_factory({'GT'}, pos_weights=pos_w, neg_weights={P.START: 3.0})

        v_low = bce_low_neg(sequences, targets, logits, {})
        v_high = bce_high_neg(sequences, targets, logits, {})
        self.assertGreater(float(v_high.detach().cpu().item()), float(v_low.detach().cpu().item()))

    def test_str_vs_int_keyed_weights_equivalence_ce(self):
        # Simple sequence with one START span to exercise weighting
        A, T, G, C = 0, 1, 2, 3
        seq = [A, T, G, C, C]
        sequences = torch.tensor([seq])
        L = len(seq)
        Cn = 8
        targets = torch.zeros((1, L), dtype=torch.long)
        targets[0, 0:3] = P.START
        logits = torch.zeros((1, L, Cn), dtype=torch.float32)
        logits[0, 0:3, P.START] = 2.0

        ce_int = event_based_ce_loss_factory({'GT'}, class_weights={P.START: 2.0})
        ce_str = event_based_ce_loss_factory({'GT'}, class_weights={'START': 2.0})
        v_int = float(ce_int(sequences, targets, logits, {}).detach().cpu().item())
        v_str = float(ce_str(sequences, targets, logits, {}).detach().cpu().item())
        self.assertAlmostEqual(v_int, v_str, places=8)

    def test_str_vs_int_keyed_weights_equivalence_bce(self):
        # Two ATG spans: first positive, second negative
        A, T, G, C = 0, 1, 2, 3
        seq = [A, T, G,  C, C,  A, T, G]
        sequences = torch.tensor([seq])
        L = len(seq)
        Cn = 8
        targets = torch.zeros((1, L), dtype=torch.long)
        targets[0, 0:3] = P.START
        logits = torch.zeros((1, L, Cn), dtype=torch.float32)
        logits[0, 0:3, P.START] = 3.0
        logits[0, 5:8, P.START] = 1.0

        bce_int = event_based_bce_loss_factory({'GT'}, pos_weights={P.START: 2.0}, neg_weights={P.START: 1.5})
        bce_str = event_based_bce_loss_factory({'GT'}, pos_weights={'START': 2.0}, neg_weights={'START': 1.5})
        v_int = float(bce_int(sequences, targets, logits, {}).detach().cpu().item())
        v_str = float(bce_str(sequences, targets, logits, {}).detach().cpu().item())
        self.assertAlmostEqual(v_int, v_str, places=8)



if __name__ == '__main__':
    unittest.main()


