#!/usr/bin/env python3

import unittest
import torch
import numpy as np

from dna_learner.model import (
    DNAEmbedding,
    MaskedTransformerLayer,
    GenePredictorModel,
    GenePredictorModule,
    create_base_config,
)


class TestGenePredictorModel(unittest.TestCase):
    def test_embedding_shapes_and_maxlen(self):
        emb = DNAEmbedding(vocab_size=5, d_model=16, max_seq_length=10, kmer_size=0)
        x = torch.randint(0, 5, (2, 8))  # <= max len
        y = emb(x)
        self.assertEqual(y.shape, (2, 8, 16))
        x2 = torch.randint(0, 5, (2, 12))  # > max len, still should run
        y2 = emb(x2)
        self.assertEqual(y2.shape, (2, 12, 16))

    def test_model_forward_shapes(self):
        model = GenePredictorModel(
            max_seq_length=50,
            num_classes=3,
            vocab_size=5,
            d_model=16,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
            attention_masks={0: 2},
            kmer_size=0,
        )
        x = torch.randint(0, 5, (2, 30))
        logits = model(x)
        self.assertEqual(logits.shape, (2, 30, 3))
        logits2, attn = model(x, return_attention=True)
        self.assertEqual(logits2.shape, (2, 30, 3))
        self.assertIn('layer_0', attn)
        self.assertEqual(attn['layer_0'].shape, (2, 2, 30, 30))

    def test_module_training_step(self):
        config = create_base_config(
            max_seq_length=64,
            num_classes=3,
            class_names=['INTERGENIC', 'UTR5', 'START'],
            d_model=18,  # not divisible by n_heads; will be adjusted
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            class_weights=[1.0, 1.0, 1.0],
            attention_masks={0: 2},
            kmer_size=0,
        )
        module = GenePredictorModule(config)
        # Verify d_model adjusted
        self.assertEqual(module.model.embedding.d_model % module.model.transformer_layers[0].n_heads, 0)
        x = torch.randint(0, 5, (2, 40))
        y = torch.randint(0, 3, (2, 40))
        loss = module.training_step((x, y), 0)
        self.assertTrue(torch.is_tensor(loss))

    def test_focal_loss_toggle(self):
        # Base config without focal
        base_cfg = create_base_config(
            max_seq_length=32,
            num_classes=3,
            class_names=['INTERGENIC', 'UTR5', 'START'],
            d_model=16,
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            class_weights=[1.0, 1.0, 1.0],
            attention_masks={0: 2},
            kmer_size=0,
        )

        focal_cfg = create_base_config(
            max_seq_length=32,
            num_classes=3,
            class_names=['INTERGENIC', 'UTR5', 'START'],
            d_model=16,
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            class_weights=[1.0, 1.0, 1.0],
            attention_masks={0: 2},
            kmer_size=0,
            use_focal=True,
            focal_gamma=2.0,
        )

        # Create modules
        ce_module = GenePredictorModule(base_cfg)
        focal_module = GenePredictorModule(focal_cfg)

        # Construct logits and targets: make them confidently correct to compare magnitude differences
        N, C = 10, 3
        logits = torch.zeros(N, C)
        targets = torch.randint(0, C, (N,))
        for i in range(N):
            # Give correct class a higher logit
            logits[i, targets[i]] = 4.0
            # Others lower
            for c in range(C):
                if c != targets[i]:
                    logits[i, c] = -1.0

        # Compute losses
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(logits, targets, reduction='mean')
        # Use module's focal implementation directly
        focal_loss = focal_module._focal_loss(logits, targets, focal_module.focal_gamma, focal_module.focal_alpha, per_token_weights=None)

        # With confident correct predictions, focal loss should be smaller than CE
        self.assertLess(focal_loss.item(), ce_loss.item() + 1e-6)

    def test_cc_readouts_disabled_no_attrs(self):
        model = GenePredictorModel(
            max_seq_length=32,
            num_classes=3,
            vocab_size=5,
            d_model=16,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
            attention_masks={0: 2},
            kmer_size=0,
            class_conditional_readouts={'enabled': False}
        )
        self.assertFalse(model.use_cc_readouts)
        self.assertEqual(model.class_conditional_readouts.get('enabled', False), False)

    def test_cc_readouts_bias_injection_changes_only_target_class(self):
        # Use 6 classes to include START (2) and STOP (4)
        model = GenePredictorModel(
            max_seq_length=20,
            num_classes=6,
            vocab_size=5,
            d_model=24,
            n_layers=1,
            n_heads=3,
            dropout=0.0,
            attention_masks={0: 2},
            kmer_size=0,
            class_conditional_readouts={
                'enabled': True,
                'entries': [
                    {'class': 2, 'before': 10, 'after': 0},  # START upstream-only
                    {'class': 4, 'before': 0, 'after': 10},  # STOP downstream-only
                ]
            }
        )
        x = torch.randint(0, 5, (1, 20))
        # Zero cc projections to establish a baseline equal to classifier-only
        for proj in model.cc_proj:
            torch.nn.init.zeros_(proj.weight)
            torch.nn.init.zeros_(proj.bias)
        baseline = model(x).detach()
        # Inject bias into first entry (class 2)
        bias = 0.5
        model.cc_proj[0].bias.data.fill_(bias)
        out1 = model(x).detach()
        # Class 2 column should increase by ~bias; others unchanged
        delta = out1 - baseline
        self.assertTrue(torch.allclose(delta[0, :, 2], torch.full((20,), bias), atol=1e-6))
        # Other class columns ~0
        other_cols = [c for c in range(6) if c != 2]
        self.assertLess(torch.abs(delta[0, :, other_cols]).max().item(), 1e-6)
        # Reset, now inject into second entry (class 4)
        model.cc_proj[0].bias.data.zero_()
        model.cc_proj[1].bias.data.fill_(0.3)
        out2 = model(x).detach()
        delta2 = out2 - baseline
        self.assertTrue(torch.allclose(delta2[0, :, 4], torch.full((20,), 0.3), atol=1e-6))
        other_cols2 = [c for c in range(6) if c != 4]
        self.assertLess(torch.abs(delta2[0, :, other_cols2]).max().item(), 1e-6)

    def test_cc_readouts_string_class_resolution(self):
        # Same as above but specify class by name
        model = GenePredictorModel(
            max_seq_length=16,
            num_classes=6,
            vocab_size=5,
            d_model=16,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
            attention_masks={0: 2},
            kmer_size=0,
            class_conditional_readouts={
                'enabled': True,
                'entries': [
                    {'class': 'START', 'before': 8, 'after': 0},
                ]
            }
        )
        x = torch.randint(0, 5, (1, 16))
        # Zero proj then add bias for the single entry
        torch.nn.init.zeros_(model.cc_proj[0].weight)
        torch.nn.init.zeros_(model.cc_proj[0].bias)
        baseline = model(x).detach()
        model.cc_proj[0].bias.data.fill_(0.2)
        out = model(x).detach()
        delta = out - baseline
        # START index is 2 per utils.constants
        self.assertTrue(torch.allclose(delta[0, :, 2], torch.full((16,), 0.2), atol=1e-6))
        other_cols = [c for c in range(6) if c != 2]
        self.assertLess(torch.abs(delta[0, :, other_cols]).max().item(), 1e-6)

    def test_fp_beta_penalty_increases_loss_on_background_fps(self):
        # Config with 3 classes, class 1 weighted > 1
        cfg = create_base_config(
            max_seq_length=4,
            num_classes=3,
            class_names=['GENE','START','STOP'],
            d_model=16,
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=1,
            class_weights=[1.0, 5.0, 1.0],
            attention_masks={0: 2},
            kmer_size=0,
        )
        cfg['loss']['entropy_lambda'] = 0.0
        # First, set fp_beta=0
        cfg['loss']['fp_beta'] = 0.0
        mod_no_fp = GenePredictorModule(cfg)
        # Now same with fp_beta>0
        cfg2 = dict(cfg)
        cfg2['loss'] = dict(cfg['loss'])
        cfg2['loss']['fp_beta'] = 0.5
        mod_with_fp = GenePredictorModule(cfg2)

        # Construct logits to create FP toward START on background tokens
        # batch=1, L=4, C=3
        # targets: all GENE (0)
        targets = torch.zeros((1, 4), dtype=torch.long)
        # logits: high prob for START (class 1) on last two positions
        logits = torch.tensor([
            [[ 3.0, -1.0, -2.0],  # clearly GENE
             [ 2.0, -0.5, -1.0],  # mostly GENE
             [-1.0,  2.0, -1.0],  # FP toward START
             [-1.0,  2.5, -1.5],  # stronger FP toward START
            ]
        ], dtype=torch.float32)

        # Monkeypatch model forward for both modules
        mod_no_fp.model.forward = lambda x: logits
        mod_with_fp.model.forward = lambda x: logits

        loss_no_fp = mod_no_fp.validation_step((torch.zeros_like(targets), targets), 0)
        loss_with_fp = mod_with_fp.validation_step((torch.zeros_like(targets), targets), 0)
        # With FP penalty, loss should be higher
        self.assertGreater(float(loss_with_fp), float(loss_no_fp))


if __name__ == '__main__':
    unittest.main()


