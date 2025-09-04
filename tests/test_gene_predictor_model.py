#!/usr/bin/env python3

import unittest
import torch
import numpy as np

from gene_predictor.model import (
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
        ce_loss = ce_module._compute_loss(logits, targets)
        focal_loss = focal_module._compute_loss(logits, targets)

        # With confident correct predictions, focal loss should be smaller than CE
        self.assertLess(focal_loss.item(), ce_loss.item() + 1e-6)


if __name__ == '__main__':
    unittest.main()


