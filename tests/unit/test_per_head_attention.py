#!/usr/bin/env python3
"""
Unit tests for per-head attention masking implementation.

These tests verify that:
1. Different heads get different attention patterns
2. Symmetric and asymmetric masks work correctly
3. Global attention works when no mask is specified
4. Attention weights are returned in correct format
5. Forward pass works with and without attention extraction
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.gene_predictor import MaskedTransformerLayer

class TestPerHeadAttention(unittest.TestCase):
    """Test suite for per-head attention masking."""
    
    def setUp(self):
        """Set up test fixtures."""
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
            dropout=0.1,
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
            dropout=0.1,
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        
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
        
        # Head 2: (0, 4) should attend to positions 10-13 (4bp downstream only)
        head2_attn = attention_weights[0, 2, pos, :].detach().numpy()
        nonzero_pos = [i for i, attn in enumerate(head2_attn) if attn > 1e-6]
        self.assertEqual(min(nonzero_pos), 10, "Head 2 should not attend upstream")
        self.assertEqual(max(nonzero_pos), 13, "Head 2 should attend 4bp downstream")
    
    def test_no_masks_global_attention(self):
        """Test that heads without masks use global attention."""
        
        attention_masks = {}  # No masks specified
        
        layer = MaskedTransformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=48,
            dropout=0.1,
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        
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
            dropout=0.1,
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
    
    def test_edge_cases(self):
        """Test edge cases like position 0 and last position."""
        
        attention_masks = {0: 3, 1: (5, 2)}
        
        layer = MaskedTransformerLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_feedforward=48,
            dropout=0.1,
            attention_masks=attention_masks,
            max_seq_length=self.seq_length
        )
        
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

if __name__ == '__main__':
    unittest.main()
