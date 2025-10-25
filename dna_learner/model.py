"""
Gene Prediction Model using Transformer Architecture

This module implements a transformer-based model for predicting gene structures
from DNA sequences, incorporating biological constraints and multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from typing import Any, Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import math
import sys
from pathlib import Path
from utils.metrics import SequenceResult

DEFAULT_VOCAB_SIZE = 5
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_D_MODEL = 512
DEFAULT_N_LAYERS = 6
DEFAULT_N_HEADS = 8
DEFAULT_DROPOUT = 0.1


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
                 attention_masks: Dict[int, Any], max_seq_length: int):
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
                
                # Support symmetric (int), asymmetric (before, after), and donut (before, gap, after)
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
                elif isinstance(mask_config, tuple) and len(mask_config) == 3:
                    # Donut: (before, gap, after) leaves a gap around the center
                    before, gap, after = mask_config
                    for i in range(seq_length):
                        # upstream window
                        if before > 0:
                            s1 = max(0, i - before)
                            e1 = max(0, i - max(0, gap))
                            if e1 > s1:
                                masks[head_idx, i, s1:e1] = True
                        # downstream window
                        if after > 0:
                            s2 = min(seq_length, i + max(1, gap))
                            e2 = min(seq_length, i + after + 1)
                            if e2 > s2:
                                masks[head_idx, i, s2:e2] = True
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


class GenePredictorModel(nn.Module):
    
    def __init__(self, max_seq_length, num_classes,
                 vocab_size: int = 5,
                 d_model: int = 504,
                 n_layers: int = 3, 
                 n_heads: int = 6,
                 dropout: float = 0.1, 
                 attention_masks: Optional[Dict[int, int]] = None,
                 kmer_size: int = 0,
                 num_event_heads: int = 0):

        super().__init__()

        # Validate that d_model is divisible by n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        # Store attention mask configuration
        self.attention_masks = attention_masks or {}
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        self.num_event_heads = int(num_event_heads) if num_event_heads is not None else 0
        
        # DNA embedding
        self.embedding = DNAEmbedding(vocab_size=vocab_size, d_model=d_model, max_seq_length=max_seq_length, kmer_size=kmer_size)
        
        # Custom transformer with attention masking
        self.transformer_layers = nn.ModuleList([
            MaskedTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                attention_masks=self.attention_masks,
                max_seq_length=max_seq_length
            ) for _ in range(n_layers)
        ])

        # Classification head - configurable number of classes
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Optional per-event heads (each outputs a single logit)
        if self.num_event_heads and self.num_event_heads > 0:
            self.event_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(self.num_event_heads)])
        else:
            self.event_heads = None

    def forward(self, x, extras: Optional[Dict[str, bool]] = None, return_attention: str = None, return_event_logits: str = None):
        # Embed DNA sequence
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        
        # Apply transformer layers with masking
        layer_attention_weights = {}
        for i, layer in enumerate(self.transformer_layers):
            if return_attention:
                x, attention_weights = layer(x, return_attention=True)
                layer_attention_weights[f'layer_{i}'] = attention_weights
            else:
                x = layer(x)  # (batch_size, seq_length, d_model)

        # Apply dropout
        x = self.dropout(x)

        # Determine what to return
        want_attention_key: Optional[str] = None
        want_event_key: Optional[str] = None
        # Allow booleans for back-compat: True -> default key name
        if isinstance(return_attention, bool) and return_attention:
            want_attention_key = 'attention'
        elif isinstance(return_attention, str) and len(return_attention) > 0:
            want_attention_key = return_attention
        if isinstance(return_event_logits, bool) and return_event_logits:
            want_event_key = 'event_logits'
        elif isinstance(return_event_logits, str) and len(return_event_logits) > 0:
            want_event_key = return_event_logits

        # If event heads are enabled, the caller must request event logits via extras
        event_logits: Optional[torch.Tensor] = None
        if self.event_heads is not None:
            assert want_event_key is not None, "Model configured with event heads but caller did not request event logits via extras; pass return_event_logits to forward()"
            event_logits_list = [head(x).squeeze(-1) for head in self.event_heads]  # list of (B,L)
            if len(event_logits_list) > 0:
                event_logits = torch.stack(event_logits_list, dim=-1)  # (B,L,H)
            else:
                event_logits = None

        # Classify each position (base logits)
        logits = self.classifier(x)  # (batch_size, seq_length, num_classes)

        # Build return according to requested keys (require caller-provided extras dict)
        if (want_attention_key is not None) or (want_event_key is not None):
            assert isinstance(extras, dict), "extras dict must be provided when requesting return_attention or return_event_logits"
            if want_attention_key is not None:
                extras[want_attention_key] = layer_attention_weights
            if want_event_key is not None:
                extras[want_event_key] = event_logits
        return logits


@dataclass
class BatchResult:
    sequence_tokens_batch: torch.Tensor
    targets_batch: Optional[torch.Tensor]
    logits_batch: torch.Tensor
    event_logits_batch: Optional[torch.Tensor]
    sequence_index_start: int


class GenePredictorModule(pl.LightningModule):
    
    def __init__(self, config: Dict[str, Any], custom_loss_fn: Callable):
        super().__init__()
        
        # Store config
        self.config = config
        self.learning_rate = config['training']['learning_rate']
        self.num_classes = config['model']['num_classes']
        self.class_names = config.get('class_names', [f'Class_{i}' for i in range(self.num_classes)])

        # Create model
        model_config = config['model']
        self.model = GenePredictorModel(
            max_seq_length=model_config['max_seq_length'],
            num_classes=model_config['num_classes'],
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            dropout=model_config['dropout'],
            attention_masks=model_config.get('attention_masks'),
            kmer_size=model_config.get('kmer_size', 0),
            num_event_heads=model_config.get('num_event_heads', 0),
        )

        # Loss function configuration (class weights remain in config for datasets/metrics)
        loss_config = config.get('loss', {})

        # Save hyperparameters for logging
        self.save_hyperparameters(config)

        # Require externally provided custom loss function
        self.custom_loss_fn: Callable = custom_loss_fn

        # Initialize per-epoch validation collection for callbacks (populated in validation_step)
        # Collect BatchResult entries for metrics callback
        self.validation_epoch_results: List[BatchResult] = []

    def forward(self, x, extras: Optional[Dict[str, Any]] = None,
                return_attention: Optional[str] = None,
                return_event_logits: Optional[str] = None):
        return self.model(
            x,
            extras=extras,
            return_attention=return_attention,
            return_event_logits=return_event_logits,
        )
    
    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        extras: Dict[str, Any] = {}
        logits = self.model(sequences, extras=extras, return_event_logits='event_logits')
        event_logits = extras['event_logits'] if 'event_logits' in extras else None
        
        comp = {}
        # Call custom loss always with event_logits (may be None)
        loss = self.custom_loss_fn(sequences, targets, logits, event_logits, components_out=comp)
        
        # Log metrics - average across epoch not just last batch
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        extras: Dict[str, Any] = {}
        logits = self.model(sequences, extras=extras, return_event_logits='event_logits')
        event_logits = extras['event_logits'] if 'event_logits' in extras else None
        
        comp = {}
        loss = self.custom_loss_fn(sequences, targets, logits, event_logits, components_out=comp)
        
        # Log metrics - average across epoch not just last batch
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        # Also log per-head validation losses for epoch aggregation
        for k, v in (comp.items() if isinstance(comp, dict) else []):
            key = str(k)
            if key.startswith('loss_head_') and not key.endswith('_weighted'):
                self.log(f"val_{key}", float(v), prog_bar=False, on_step=False, on_epoch=True)
        
        # Collect batch results for metrics callback conversion later
        coll = self.validation_epoch_results
        start_idx = sum(int(br.logits_batch.size(0)) for br in coll) if coll else 0
        coll.append(BatchResult(
            sequence_tokens_batch=sequences.detach(),
            targets_batch=targets.detach() if targets is not None else None,
            logits_batch=logits.detach(),
            event_logits_batch=event_logits.detach() if isinstance(event_logits, torch.Tensor) else None,
            sequence_index_start=start_idx,
        ))
        
        return loss

    def on_validation_epoch_start(self) -> None:
        # Reset per-epoch collection container for callback
        self.validation_epoch_results = []
    
    def test_step(self, batch, batch_idx):
        # Same as validation step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def create_base_config(
    max_seq_length: int, num_classes: int, class_names: list,
    d_model: int = 504,
    n_layers: int = 3,
    n_heads: int = 6,
    learning_rate: float = 5e-5,
    max_epochs: int = 25,
    batch_size: int = 8,
    class_weights: Optional[list] = None,
    # (removed: loss_window_margin_fraction, loss_window_margin_bp)
    attention_masks: Optional[Dict[int, int]] = None,
    kmer_size: int = 0,
    accumulate_grad_batches: Optional[int] = None,
    num_event_heads: Optional[int] = None,
) -> dict:

    # Validate d_model is divisible by n_heads
    if d_model % n_heads != 0:
        # Adjust d_model to nearest multiple
        d_model = (d_model // n_heads) * n_heads
        print(f"Warning: Adjusted d_model to {d_model} (divisible by {n_heads} heads)")
    
    cfg = {
        'model': {
            'vocab_size': 5,      # A, T, G, C, N
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'num_classes': num_classes,
            'dropout': 0.1,
            'max_seq_length': max_seq_length,
            'attention_masks': attention_masks,
            'kmer_size': kmer_size
        },
        'training': {
            'learning_rate': learning_rate,
            'max_epochs': max_epochs,
            'batch_size': batch_size
        },
        'loss': {
            'class_weights': class_weights,
        },
        'class_names': class_names
    }

    if accumulate_grad_batches is not None:
        cfg['training']['accumulate_grad_batches'] = int(accumulate_grad_batches)

    if num_event_heads is not None:
        cfg['model']['num_event_heads'] = int(num_event_heads)

    return cfg


def set_class_conditional_readout_config_with_head(*args, **kwargs) -> None:
    raise RuntimeError("class_conditional_readouts have been removed; configure directional attention via attention_masks instead")
