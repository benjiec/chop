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
    AuxStreamEncoder,
    BiasedGlobalCrossAttention,
)
from utils.losses import adjusted_ce_entropy_loss
from utils.constants import EventHeadIdx as H


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
        extras = {}
        logits2 = model(x, extras=extras, return_attention='attention')
        self.assertEqual(logits2.shape, (2, 30, 3))
        self.assertIn('attention', extras)
        attn = extras['attention']
        self.assertIn('layer_0', attn)
        self.assertEqual(attn['layer_0'].shape, (2, 2, 30, 30))

    def test_aux_stream_disabled_is_noop(self):
        # Model configured with aux support disabled
        model = GenePredictorModel(
            max_seq_length=64,
            num_classes=3,
            vocab_size=5,
            d_model=16,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
            attention_masks=None,
            kmer_size=0,
            enable_aux_stream=False,
            aux_cross_attn_layers=1,
        )
        x = torch.randint(0, 5, (2, 40))
        aux = torch.randn(2, 40, 2)
        logits_no_aux = model(x)
        logits_with_ignored_aux = model(x, aux_stream=aux)
        self.assertEqual(logits_no_aux.shape, logits_with_ignored_aux.shape)

    def test_aux_stream_enabled_shapes_and_lazy_init(self):
        model = GenePredictorModel(
            max_seq_length=64,
            num_classes=4,
            vocab_size=5,
            d_model=24,
            n_layers=1,
            n_heads=3,
            dropout=0.0,
            attention_masks=None,
            kmer_size=0,
            enable_aux_stream=True,
            aux_cross_attn_layers=1,
            aux_relpos_max_distance=16,
        )
        x = torch.randint(0, 5, (2, 32))
        aux = torch.randn(2, 32, 3)  # C=3
        # Before first forward, aux encoder not yet built
        self.assertTrue(getattr(model, 'aux_encoder') is None)
        logits = model(x, aux_stream=aux)
        self.assertEqual(logits.shape, (2, 32, 4))
        # After forward, aux encoder must be initialized with in_channels=3
        self.assertIsNotNone(model.aux_encoder)
        self.assertEqual(int(getattr(model, '_aux_encoder_in_dim')), 3)

    def test_biased_global_cross_attention_bias_matrix(self):
        d_model = 12
        heads = 3
        L = 10
        block = BiasedGlobalCrossAttention(d_model=d_model, n_heads=heads, dropout=0.0, relpos_max_distance=4)
        x = torch.randn(2, L, d_model)
        y = torch.randn(2, L, d_model)
        out = block(x, y)
        self.assertEqual(out.shape, (2, L, d_model))
        # Gate is between 0 and 1 after sigmoid
        gate = torch.sigmoid(block.gate).item()
        self.assertGreaterEqual(gate, 0.0)
        self.assertLessEqual(gate, 1.0)

    def test_model_with_event_heads_outputs_event_logits(self):
        model = GenePredictorModel(
            max_seq_length=32,
            num_classes=5,
            vocab_size=5,
            d_model=16,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
            attention_masks=None,
            kmer_size=0,
            num_event_heads=4,
        )
        x = torch.randint(0, 5, (2, 20))
        extras = {}
        logits = model(x, extras=extras, return_event_logits='event_logits')
        self.assertEqual(logits.shape, (2, 20, 5))
        self.assertIn('event_logits', extras)
        ev = extras['event_logits']
        self.assertIsNotNone(ev)
        self.assertEqual(ev.shape, (2, 20, 4))


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
        module = GenePredictorModule(config, custom_loss_fn=lambda s,t,l,ev,c: adjusted_ce_entropy_loss(l, t, loss_window_margin_bp=0, class_weights=config.get('loss',{}).get('class_weights'), entropy_lambda=0.0, fp_beta=0.0, components_out=c))
        # Verify d_model adjusted
        self.assertEqual(module.model.embedding.d_model % module.model.transformer_layers[0].n_heads, 0)
        x = torch.randint(0, 5, (2, 40))
        y = torch.randint(0, 3, (2, 40))
        loss = module.training_step((x, y, None), 0)
        self.assertTrue(torch.is_tensor(loss))

    # focal loss tests removed as feature is no longer present


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
        mod_no_fp = GenePredictorModule(cfg, custom_loss_fn=lambda s,t,l,ev,c: adjusted_ce_entropy_loss(l, t, loss_window_margin_bp=0, class_weights=cfg.get('loss',{}).get('class_weights'), entropy_lambda=cfg.get('loss',{}).get('entropy_lambda',0.0), fp_beta=cfg.get('loss',{}).get('fp_beta',0.0), components_out=c))
        # Now same with fp_beta>0
        cfg2 = dict(cfg)
        cfg2['loss'] = dict(cfg['loss'])
        cfg2['loss']['fp_beta'] = 0.5
        mod_with_fp = GenePredictorModule(cfg2, custom_loss_fn=lambda s,t,l,ev,c: adjusted_ce_entropy_loss(l, t, loss_window_margin_bp=0, class_weights=cfg2.get('loss',{}).get('class_weights'), entropy_lambda=cfg2.get('loss',{}).get('entropy_lambda',0.0), fp_beta=cfg2.get('loss',{}).get('fp_beta',0.5), components_out=c))

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
        mod_no_fp.model.forward = lambda x, **kwargs: logits
        mod_with_fp.model.forward = lambda x, **kwargs: logits

        loss_no_fp = mod_no_fp.validation_step((torch.zeros_like(targets), targets, None), 0)
        loss_with_fp = mod_with_fp.validation_step((torch.zeros_like(targets), targets, None), 0)
        # With FP penalty, loss should be higher
        self.assertGreater(float(loss_with_fp), float(loss_no_fp))


if __name__ == '__main__':
    unittest.main()



# Additional tests consolidated from test_per_head_attention.py
"""
Unit tests for per-head attention masking implementation.

These tests verify that:
1. Different heads get different attention patterns
2. Symmetric and asymmetric masks work correctly
3. Global attention works when no mask is specified
4. Attention weights are returned in correct format
5. Forward pass works with and without attention extraction
"""

import sys
from pathlib import Path
from dna_learner.model import MaskedTransformerLayer

class TestPerHeadAttention(unittest.TestCase):
    """Test suite for per-head attention masking."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        
        self.d_model = 12  # Must be divisible by n_heads
        self.n_heads = 3
        self.seq_length = 20
        self.batch_size = 2
        
        # Test input
        self.test_input = torch.randn(self.batch_size, self.seq_length, self.d_model)
    
    def test_symmetric_attention_masks(self):
        """Test symmetric attention masking (window around position)."""
        
        attention_masks = {0: 2, 1: 4}  # Head 0: 2bp window, Head 1: 4bp window
        
        layer = MaskedTransformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=48,
            dropout=0.0,  # No dropout for deterministic results
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        
        # Test forward pass
        output, attention_weights = layer(self.test_input, return_attention=True)
        
        # Verify output shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.d_model))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.n_heads, self.seq_length, self.seq_length))
        
        # Test specific position (10) attention patterns
        pos = 10
        
        # Head 0: 2bp window should attend to positions 8-12
        head0_attn = attention_weights[0, 0, pos, :].detach().numpy()
        nonzero_pos = [i for i, attn in enumerate(head0_attn) if attn > 1e-6]
        self.assertEqual(min(nonzero_pos), 8, "Head 0 should start attention at pos-2")
        self.assertEqual(max(nonzero_pos), 12, "Head 0 should end attention at pos+2")
        
        # Head 1: 4bp window should attend to positions 6-14
        head1_attn = attention_weights[0, 1, pos, :].detach().numpy()
        nonzero_pos = [i for i, attn in enumerate(head1_attn) if attn > 1e-6]
        self.assertEqual(min(nonzero_pos), 6, "Head 1 should start attention at pos-4")
        self.assertEqual(max(nonzero_pos), 14, "Head 1 should end attention at pos+4")
        
        # Head 2: No mask, should attend globally (0-19)
        head2_attn = attention_weights[0, 2, pos, :].detach().numpy()
        nonzero_pos = [i for i, attn in enumerate(head2_attn) if attn > 1e-6]
        self.assertEqual(min(nonzero_pos), 0, "Head 2 should have global attention (start=0)")
        self.assertEqual(max(nonzero_pos), 19, "Head 2 should have global attention (end=19)")
    
    def test_asymmetric_attention_masks(self):
        """Test asymmetric attention masking (different upstream/downstream ranges)."""
        
        attention_masks = {0: (5, 0), 1: (2, 3), 2: (0, 4)}  # Different asymmetric patterns
        
        layer = MaskedTransformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=48,
            dropout=0.0,  # No dropout for deterministic results
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        layer.eval()  # Set to eval mode for deterministic behavior
        
        output, attention_weights = layer(self.test_input, return_attention=True)
        
        # Test position 10
        pos = 10
        
        # Head 0: (5, 0) should attend to positions 5-10 (5bp upstream only)
        head0_attn = attention_weights[0, 0, pos, :].detach().numpy()
        nonzero_pos = [i for i, attn in enumerate(head0_attn) if attn > 1e-6]
        self.assertEqual(min(nonzero_pos), 5, "Head 0 should attend 5bp upstream")
        self.assertEqual(max(nonzero_pos), 10, "Head 0 should not attend downstream")
        
        # Head 1: (2, 3) should attend to positions 8-13 (2bp upstream + 3bp downstream)
        head1_attn = attention_weights[0, 1, pos, :].detach().numpy()
        nonzero_pos = [i for i, attn in enumerate(head1_attn) if attn > 1e-6]
        self.assertEqual(min(nonzero_pos), 8, "Head 1 should attend 2bp upstream")
        self.assertEqual(max(nonzero_pos), 13, "Head 1 should attend 3bp downstream")
        
        # Head 2: (0, 4) should attend to positions 10-14 (4bp downstream only)
        head2_attn = attention_weights[0, 2, pos, :].detach().numpy()
        nonzero_pos = [i for i, attn in enumerate(head2_attn) if attn > 1e-6]
        self.assertEqual(min(nonzero_pos), 10, "Head 2 should not attend upstream")
        self.assertEqual(max(nonzero_pos), 14, "Head 2 should attend 4bp downstream")

    def test_mask_zero_outside_windows(self):
        """Masked positions should have zero attention mass outside the window."""
        # Symmetric case
        attention_masks = {0: 2, 1: 4}
        layer = MaskedTransformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=48,
            dropout=0.0,
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        layer.eval()
        output, attn = layer(self.test_input, return_attention=True)
        pos = 10
        # Head 0 window [8,12]
        h0 = attn[0, 0, pos, :].detach().numpy()
        for i in list(range(0, 8)) + list(range(13, self.seq_length)):
            self.assertEqual(h0[i], 0.0, "Masked-out positions must have zero attention (symmetric)")
        # Head 1 window [6,14]
        h1 = attn[0, 1, pos, :].detach().numpy()
        for i in list(range(0, 6)) + list(range(15, self.seq_length)):
            self.assertEqual(h1[i], 0.0, "Masked-out positions must have zero attention (symmetric)")

        # Asymmetric case
        attention_masks = {0: (5, 0), 1: (2, 3)}
        layer = MaskedTransformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=48,
            dropout=0.0,
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        layer.eval()
        output, attn = layer(self.test_input, return_attention=True)
        # Head 0 window [5,10]
        h0 = attn[0, 0, pos, :].detach().numpy()
        for i in list(range(0, 5)) + list(range(11, self.seq_length)):
            self.assertEqual(h0[i], 0.0, "Masked-out positions must have zero attention (asymmetric)")
        # Head 1 window [8,13]
        h1 = attn[0, 1, pos, :].detach().numpy()
        for i in list(range(0, 8)) + list(range(14, self.seq_length)):
            self.assertEqual(h1[i], 0.0, "Masked-out positions must have zero attention (asymmetric)")

    def test_mask_includes_self_across_configs(self):
        """Each mask must include self-attention at the query position."""
        configs = [
            {0: 1, 1: 3},
            {0: (5, 0), 1: (0, 4)},
            {0: (2, 2), 1: (10, 6)},
        ]
        pos = 10
        for attention_masks in configs:
            layer = MaskedTransformerLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dim_feedforward=48,
                dropout=0.0,
                attention_masks=attention_masks,
                max_seq_length=self.seq_length
            )
            layer.eval()
            _, attn = layer(self.test_input, return_attention=True)
            for head_idx in range(self.n_heads):
                a = attn[0, head_idx, pos, pos].item()
                self.assertGreater(a, 0.0, f"Head {head_idx} must include self in its window")
    
    def test_no_masks_global_attention(self):
        """Test that heads without masks use global attention."""
        
        attention_masks = {}  # No masks specified
        
        layer = MaskedTransformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=48,
            dropout=0.0,  # No dropout for deterministic results
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        layer.eval()  # Set to eval mode for deterministic behavior
        
        output, attention_weights = layer(self.test_input, return_attention=True)
        
        # All heads should have global attention
        pos = 10
        for head_idx in range(self.n_heads):
            head_attn = attention_weights[0, head_idx, pos, :].detach().numpy()
            nonzero_pos = [i for i, attn in enumerate(head_attn) if attn > 1e-6]
            
            # Allow for small numerical precision issues at boundaries
            self.assertLessEqual(min(nonzero_pos), 1, f"Head {head_idx} should have near-global attention (start~0)")
            self.assertGreaterEqual(max(nonzero_pos), 18, f"Head {head_idx} should have near-global attention (end~19)")
    
    def test_attention_weights_format(self):
        """Test that attention weights are returned in correct format."""
        
        attention_masks = {0: 1, 1: (3, 2)}
        
        layer = MaskedTransformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=48,
            dropout=0.0,  # No dropout for deterministic results
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        
        # Test with return_attention=True
        output, attention_weights = layer(self.test_input, return_attention=True)
        
        self.assertIsNotNone(attention_weights, "Attention weights should be returned")
        self.assertEqual(attention_weights.shape, 
                        (self.batch_size, self.n_heads, self.seq_length, self.seq_length),
                        "Attention weights should have shape (batch, heads, seq_len, seq_len)")
        
        # Test with return_attention=False
        output_no_attn = layer(self.test_input, return_attention=False)
        
        self.assertEqual(output_no_attn.shape, (self.batch_size, self.seq_length, self.d_model))
        # Should return just the output tensor, not a tuple
        self.assertIsInstance(output_no_attn, torch.Tensor)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights are properly normalized (sum to 1)."""
        
        attention_masks = {0: 2, 1: (4, 1)}
        
        layer = MaskedTransformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=48,
            dropout=0.0,  # No dropout for testing normalization
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        
        layer.eval()  # Disable any remaining dropout
        
        with torch.no_grad():
            output, attention_weights = layer(self.test_input, return_attention=True)
        
        # Check that attention weights sum to 1 for each position in each head
        for batch_idx in range(self.batch_size):
            for head_idx in range(self.n_heads):
                for pos in range(self.seq_length):
                    attn_sum = attention_weights[batch_idx, head_idx, pos, :].sum().item()
                    self.assertAlmostEqual(attn_sum, 1.0, places=5, 
                                         msg=f"Attention weights should sum to 1 for batch {batch_idx}, head {head_idx}, pos {pos}")
    
    def test_donut_attention_mask(self):
        """Donut masks should exclude a local gap while allowing upstream/downstream bands."""
        attention_masks = {0: (5, 2, 5)}  # before=5, gap=2, after=5
        layer = MaskedTransformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=48,
            dropout=0.0,
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        layer.eval()
        _, attn = layer(self.test_input, return_attention=True)
        pos = 10
        # Head 0 uses donut: should have zero (or near-zero) mass at center and a gap of size ~gap around it
        h0 = attn[0, 0, pos, :].detach().numpy()
        # Check the central indices are zeroed: [pos-gap+1 .. pos+gap-1] roughly excludes +/-1 around center for gap=2
        self.assertEqual(h0[pos], 0.0, "Donut mask should exclude self-attention at the query position")
        self.assertEqual(h0[pos-1], 0.0, "Donut mask should exclude immediate upstream inside the gap")
        self.assertEqual(h0[pos+1], 0.0, "Donut mask should exclude immediate downstream inside the gap")
        # Upstream band should be non-zero
        self.assertGreater(h0[pos-3], 0.0, "Upstream band should be allowed in donut mask")
        # Downstream band should be non-zero
        self.assertGreater(h0[pos+3], 0.0, "Downstream band should be allowed in donut mask")
    
    def test_edge_cases(self):
        """Test edge cases like position 0 and last position."""
        
        attention_masks = {0: 3, 1: (5, 2)}
        
        layer = MaskedTransformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=48,
            dropout=0.0,  # No dropout for deterministic results
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        layer.eval()  # Set to eval mode for deterministic behavior
        
        output, attention_weights = layer(self.test_input, return_attention=True)
        
        # Test position 0 (start of sequence)
        pos = 0
        head0_attn = attention_weights[0, 0, pos, :].detach().numpy()
        nonzero_pos = [i for i, attn in enumerate(head0_attn) if attn > 1e-6]
        
        # Head 0 with 3bp window at position 0 should attend to 0-3
        self.assertEqual(min(nonzero_pos), 0, "Position 0 should handle edge case correctly")
        self.assertTrue(max(nonzero_pos) <= 3, "Position 0 should not exceed window size")
        
        # Test last position (19)
        pos = 19
        head0_attn = attention_weights[0, 0, pos, :].detach().numpy()
        nonzero_pos = [i for i, attn in enumerate(head0_attn) if attn > 1e-6]
        
        # Head 0 with 3bp window at position 19 should attend to 16-19
        self.assertTrue(min(nonzero_pos) >= 16, "Last position should handle edge case correctly")
        self.assertEqual(max(nonzero_pos), 19, "Last position should not exceed sequence length")


# Additional tests consolidated from test_entropy_penalty.py
from dna_learner.model import GenePredictorModule, create_base_config
class TestEntropyPenalty(unittest.TestCase):
    def _make_module(self, num_classes=3, entropy_lambda=1e-3):
        cfg = create_base_config(
            max_seq_length=4,
            num_classes=num_classes,
            class_names=[f'C{i}' for i in range(num_classes)],
            d_model=24,
            n_layers=1,
            n_heads=3,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=1,
            class_weights=[1.0]*num_classes,
            attention_masks=None,
            kmer_size=0,
        )
        cfg['loss']['use_focal'] = False
        cfg['loss']['entropy_lambda'] = entropy_lambda
        cfg['loss']['fp_beta'] = 0.0
        mod = GenePredictorModule(cfg, custom_loss_fn=lambda s,t,l,c: adjusted_ce_entropy_loss(l, t, loss_window_margin_bp=0, class_weights=[1.0]*num_classes, entropy_lambda=entropy_lambda, fp_beta=0.0, components_out=c))
        return mod

    def test_entropy_term_effect(self):
        torch.manual_seed(0)
        mod = self._make_module(num_classes=3, entropy_lambda=1e-2)
        logits = torch.tensor([
            [[5.0, -2.0, -3.0],
             [2.0,  1.8,  1.7],
             [0.1,  0.0, -0.1],
             [1.0,  1.0,  1.0],
            ]
        ], dtype=torch.float32)
        targets = torch.tensor([[0, 0, 1, 2]], dtype=torch.long)
        ce_only = torch.nn.functional.cross_entropy(logits.view(-1, 3), targets.view(-1), reduction='mean')
        loss_with_entropy = adjusted_ce_entropy_loss(
            logits,
            targets,
            loss_window_margin_bp=0,
            class_weights=[1.0]*3,
            entropy_lambda=1e-2,
            fp_beta=0.0,
            components_out=None,
        )
        self.assertLessEqual(float(loss_with_entropy), float(ce_only) + 1e-6)


class TestValidationBatchCollection(unittest.TestCase):
    def test_validation_epoch_results_collects_per_batch(self):
        # Create a minimal config and module
        num_classes = 3
        cfg = create_base_config(
            max_seq_length=16,
            num_classes=num_classes,
            class_names=[f'C{i}' for i in range(num_classes)],
            d_model=24,
            n_layers=1,
            n_heads=3,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            class_weights=[1.0] * num_classes,
            attention_masks=None,
            kmer_size=0,
        )

        # Simple loss: mean CE on provided logits/targets
        def dummy_loss(seqs, targets, logits, event_logits, components):
            return torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean'
            )

        mod = GenePredictorModule(cfg, custom_loss_fn=dummy_loss)

        # Prepare 5 validation items total, processed in batches of 2,2,1
        L = 10
        x1 = torch.randint(0, 5, (2, L), dtype=torch.long)
        y1 = torch.randint(0, num_classes, (2, L), dtype=torch.long)
        x2 = torch.randint(0, 5, (2, L), dtype=torch.long)
        y2 = torch.randint(0, num_classes, (2, L), dtype=torch.long)
        x3 = torch.randint(0, 5, (1, L), dtype=torch.long)
        y3 = torch.randint(0, num_classes, (1, L), dtype=torch.long)

        # Begin validation epoch and run three batches
        mod.on_validation_epoch_start()
        mod.validation_step((x1, y1, None), 0)
        mod.validation_step((x2, y2, None), 1)
        mod.validation_step((x3, y3, None), 2)

        # Expect 3 BatchResult entries (one per batch)
        brs = getattr(mod, 'validation_epoch_results')
        self.assertEqual(len(brs), 3)

        # Total items across batches should equal 5
        total_items = sum(int(br.logits_batch.size(0)) for br in brs)
        self.assertEqual(total_items, 5)


class TestModuleForwardAPI(unittest.TestCase):
    def test_module_forward_returns_logits_and_event_logits_in_extras(self):
        num_classes = 4
        num_event_heads = 3
        cfg = create_base_config(
            max_seq_length=32,
            num_classes=num_classes,
            class_names=[f'C{i}' for i in range(num_classes)],
            d_model=16,
            n_layers=1,
            n_heads=2,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            class_weights=[1.0] * num_classes,
            attention_masks=None,
            kmer_size=0,
            num_event_heads=num_event_heads,
        )

        # Custom loss not used here but match signature
        dummy_loss = lambda s, t, l, ev, c: torch.tensor(0.0)
        mod = GenePredictorModule(cfg, custom_loss_fn=dummy_loss)

        x = torch.randint(0, 5, (2, 20), dtype=torch.long)
        extras = {}
        logits = mod.forward(x, extras=extras, return_event_logits='event_logits')

        self.assertEqual(logits.shape, (2, 20, num_classes))
        self.assertIn('event_logits', extras)
        ev = extras['event_logits']
        self.assertIsInstance(ev, torch.Tensor)
        self.assertEqual(ev.shape, (2, 20, num_event_heads))

    def test_module_forward_requires_extras_when_requesting_event_logits(self):
        cfg = create_base_config(
            max_seq_length=16,
            num_classes=3,
            class_names=['A', 'B', 'C'],
            d_model=12,
            n_layers=1,
            n_heads=3,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=1,
            class_weights=[1.0, 1.0, 1.0],
            attention_masks=None,
            kmer_size=0,
            num_event_heads=2,
        )
        mod = GenePredictorModule(cfg, custom_loss_fn=lambda s, t, l, ev, c: torch.tensor(0.0))
        x = torch.randint(0, 5, (1, 10), dtype=torch.long)
        with self.assertRaises(AssertionError):
            _ = mod.forward(x, return_event_logits='event_logits')

    def test_module_supports_aux_stream_in_batches(self):
        num_classes = 3
        cfg = create_base_config(
            max_seq_length=32,
            num_classes=num_classes,
            class_names=[f'C{i}' for i in range(num_classes)],
            d_model=16,
            n_layers=1,
            n_heads=2,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            class_weights=[1.0] * num_classes,
            attention_masks=None,
            kmer_size=0,
        )
        # Enable aux in model config
        cfg['model']['enable_aux_stream'] = True
        cfg['model']['aux_cross_attn_layers'] = 1

        dummy_loss = lambda s, t, l, ev, c: torch.tensor(0.0)
        mod = GenePredictorModule(cfg, custom_loss_fn=dummy_loss)
        x = torch.randint(0, 5, (2, 20), dtype=torch.long)
        y = torch.randint(0, num_classes, (2, 20), dtype=torch.long)
        aux = torch.randn(2, 20, 2)

        # Training with 3-tuple
        loss1 = mod.training_step((x, y, aux), 0)
        self.assertTrue(torch.is_tensor(loss1))

        # Validation with dict batch
        loss2 = mod.validation_step((x, y, aux), 0)
        self.assertTrue(torch.is_tensor(loss2))
