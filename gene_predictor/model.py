"""
Gene Prediction Model using Transformer Architecture

This module implements a transformer-based model for predicting gene structures
from DNA sequences, incorporating biological constraints and multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import math
import sys
from pathlib import Path

DEFAULT_VOCAB_SIZE = 5
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_D_MODEL = 512
DEFAULT_N_LAYERS = 6
DEFAULT_N_HEADS = 8
DEFAULT_DROPOUT = 0.1
DEFAULT_THRESHOLD = 0.5


class DNAEmbedding(nn.Module):
    """DNA sequence embedding layer with k-mer features and positional encoding."""
    
    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE, d_model: int = DEFAULT_D_MODEL, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, kmer_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.kmer_size = kmer_size
        
        # DNA token embedding (A, C, G, T, N)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        
        # K-mer features (configurable size, 0 = disabled)
        if kmer_size > 0:
            # Use 'same' padding to maintain sequence length
            padding = (kmer_size - 1) // 2  # For odd kernel sizes
            self.kmer_conv = nn.Conv1d(d_model, d_model, kernel_size=kmer_size, padding=padding)
        else:
            self.kmer_conv = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_length)
        batch_size, seq_length = x.shape
        
        # Token embeddings
        embeddings = self.token_embedding(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        if seq_length <= self.max_seq_length:
            embeddings = embeddings + self.pos_encoding[:, :seq_length, :]
        
        # K-mer context features (if enabled)
        if self.kmer_conv is not None:
            kmer_features = self.kmer_conv(embeddings.transpose(1, 2)).transpose(1, 2)
            embeddings = embeddings + kmer_features
        
        # Layer normalization
        embeddings = self.layer_norm(embeddings)
        
        return embeddings


class MaskedTransformerLayer(nn.Module):
    """
    Transformer layer with per-head attention masking support.
    
    This enables different attention heads to focus on different regions:
    - Local patterns (3-5bp windows for codons)
    - Upstream context (50-200bp for regulatory elements)
    - Downstream context (20-50bp for coding sequence)
    - Global context (full sequence)
    
    Usage:
        attention_masks = {0: 4, 1: (20, 5), 2: (50, 0)}
        # Head 0: 4bp symmetric window
        # Head 1: 20bp upstream + 5bp downstream  
        # Head 2: 50bp upstream only
        # Other heads: Global attention
    """
    
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float,
                 attention_masks: Dict[int, int], max_seq_length: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention_masks = attention_masks
        self.max_seq_length = max_seq_length
        self.head_dim = d_model // n_heads
        
        # Custom multi-head attention with per-head masking
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def _create_per_head_masks(self, seq_length: int) -> torch.Tensor:
        """Create individual masks for each head."""
        if not self.attention_masks:
            return None
            
        device = next(self.parameters()).device
        masks = torch.zeros(self.n_heads, seq_length, seq_length, device=device, dtype=torch.bool)
        
        for head_idx in range(self.n_heads):
            if head_idx in self.attention_masks:
                mask_config = self.attention_masks[head_idx]
                
                # Support both symmetric and asymmetric masking
                if isinstance(mask_config, int):
                    # Symmetric: head -> window_size
                    window_size = mask_config
                    for i in range(seq_length):
                        start = max(0, i - window_size)
                        end = min(seq_length, i + window_size + 1)
                        masks[head_idx, i, start:end] = True
                        
                elif isinstance(mask_config, tuple) and len(mask_config) == 2:
                    # Asymmetric: head -> (before, after)
                    before, after = mask_config
                    for i in range(seq_length):
                        start = max(0, i - before) if before > 0 else i
                        end = min(seq_length, i + after + 1) if after > 0 else i + 1
                        masks[head_idx, i, start:end] = True
            else:
                # No mask - global attention
                masks[head_idx, :, :] = True
        
        return masks
    
    def _custom_multihead_attention(self, x: torch.Tensor, per_head_masks: torch.Tensor, return_attention: bool):
        """Custom multi-head attention with per-head masking."""
        batch_size, seq_length, d_model = x.shape
        
        # Linear projections
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head: (batch, seq_len, heads, head_dim) -> (batch, heads, seq_len, head_dim)
        q = q.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation per head
        head_outputs = []
        all_attention_weights = []
        
        for head_idx in range(self.n_heads):
            # Compute attention scores for this head
            scores = torch.matmul(q[:, head_idx], k[:, head_idx].transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply per-head mask
            if per_head_masks is not None:
                head_mask = per_head_masks[head_idx].unsqueeze(0).expand(batch_size, -1, -1)
                scores = scores.masked_fill(~head_mask, float('-inf'))
            
            # Softmax and dropout (ensure proper normalization after masking)
            attention_weights = F.softmax(scores, dim=-1)
            
            # Handle NaN values that can occur with extreme masking
            attention_weights = torch.where(torch.isnan(attention_weights), 
                                          torch.zeros_like(attention_weights), 
                                          attention_weights)
            
            attention_weights = self.attn_dropout(attention_weights)
            
            # Apply attention to values
            head_output = torch.matmul(attention_weights, v[:, head_idx])
            head_outputs.append(head_output)
            
            if return_attention:
                all_attention_weights.append(attention_weights)
        
        # Concatenate heads: (batch, heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        concat_output = torch.stack(head_outputs, dim=1).transpose(1, 2).contiguous()
        concat_output = concat_output.view(batch_size, seq_length, d_model)
        
        # Final linear projection
        output = self.out_proj(concat_output)
        
        if return_attention:
            # Stack attention weights: (heads, batch, seq_len, seq_len) -> (batch, heads, seq_len, seq_len)
            attention_tensor = torch.stack(all_attention_weights, dim=1)
            return output, attention_tensor
        else:
            return output, None
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Forward pass with per-head masked attention."""
        batch_size, seq_length, d_model = x.shape
        
        # Create per-head attention masks
        per_head_masks = self._create_per_head_masks(seq_length)
        
        # Custom multi-head attention with per-head masking
        attn_output, attention_weights = self._custom_multihead_attention(x, per_head_masks, return_attention)
        
        x = self.norm1(x + attn_output)
        
        # Feed forward with residual connection  
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        if return_attention:
            return x, attention_weights
        else:
            return x
