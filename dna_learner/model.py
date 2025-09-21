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
from typing import Any, Dict, List, Tuple, Optional
import math
import sys
from pathlib import Path

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
                 class_conditional_readouts: Optional[Dict[str, Any]] = None):

        super().__init__()

        # Validate that d_model is divisible by n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        # Store attention mask configuration
        self.attention_masks = attention_masks or {}
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        self.class_conditional_readouts = class_conditional_readouts or {'enabled': False}
        
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

        # Optional generic class-conditional readouts
        self.use_cc_readouts: bool = bool(self.class_conditional_readouts.get('enabled', False))
        self.cc_entries: List[Dict[str, Any]] = []
        if self.use_cc_readouts:
            entries = self.class_conditional_readouts.get('entries')
            # Require explicit entries; if missing or empty, no readouts are created
            if not isinstance(entries, list):
                entries = []
            # Build modules per entry
            self.cc_q = nn.ModuleList()
            self.cc_k = nn.ModuleList()
            self.cc_v = nn.ModuleList()
            self.cc_proj = nn.ModuleList()
            for e in entries or []:
                self.cc_q.append(nn.Linear(d_model, d_model))
                self.cc_k.append(nn.Linear(d_model, d_model))
                self.cc_v.append(nn.Linear(d_model, d_model))
                self.cc_proj.append(nn.Linear(2 * d_model, 1))
                # Store params per entry
                self.cc_entries.append({
                    'class': e.get('class'),
                    'before': int(e.get('before', 0)),
                    'after': int(e.get('after', 0)),
                    'gap': int(e.get('gap', 0)),
                })
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
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
        
        # Classify each position (base logits)
        logits = self.classifier(x)  # (batch_size, seq_length, num_classes)

        # If enabled, compute class-conditional readout logits and add to respective class columns
        if self.use_cc_readouts and self.cc_entries:
            B, L, D = x.shape
            scale = math.sqrt(D)

            # Helper: build relative donut/asymmetric mask [L, L]
            def build_rel_mask(before: int, gap: int, after: int) -> torch.Tensor:
                m = torch.zeros(L, L, dtype=torch.bool, device=x.device)
                for i in range(L):
                    # upstream
                    if before > 0:
                        s1 = max(0, i - before)
                        e1 = max(0, i - max(0, gap))
                        if e1 > s1:
                            m[i, s1:e1] = True
                    # downstream
                    if after > 0:
                        s2 = min(L, i + max(1, gap))
                        e2 = min(L, i + after + 1)
                        if e2 > s2:
                            m[i, s2:e2] = True
                    # pure local (no before/after)
                    if before == 0 and after == 0:
                        m[i, i] = True
                return m

            # Resolve class indices map if provided via config
            name_to_idx = {}
            try:
                # Prefer config class_names if present
                # self.class_names is available only in LightningModule, not here. So use constants if available.
                from utils.constants import GenePredictionClass as P
                name_to_idx = {v: k for k, v in P.idx_to_cls.items()}
            except Exception:
                name_to_idx = {}

            for idx, entry in enumerate(self.cc_entries):
                cls_spec = entry.get('class')
                if isinstance(cls_spec, str):
                    cls_idx = name_to_idx.get(cls_spec.upper(), None)
                else:
                    try:
                        cls_idx = int(cls_spec)
                    except Exception:
                        cls_idx = None
                if cls_idx is None or cls_idx >= logits.size(-1):
                    continue
                before = int(entry.get('before', 0))
                after = int(entry.get('after', 0))
                gap = int(entry.get('gap', 0))
                mask = build_rel_mask(before, gap, after)

                q = self.cc_q[idx](x)
                k = self.cc_k[idx](x)
                v = self.cc_v[idx](x)
                scores = torch.matmul(q, k.transpose(1, 2)) / scale
                scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
                attn = F.softmax(scores, dim=-1)
                # Replace NaNs (can happen if mask row has no allowed positions)
                attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
                c = torch.matmul(attn, v)
                logit_delta = self.cc_proj[idx](torch.cat([x, c], dim=-1)).squeeze(-1)  # (B,L)
                logits[..., cls_idx] = logits[..., cls_idx] + logit_delta
        
        if return_attention:
            return logits, layer_attention_weights
        else:
            return logits


class GenePredictorModule(pl.LightningModule):
    
    def __init__(self, config: Dict[str, Any]):
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
            class_conditional_readouts=model_config.get('class_conditional_readouts')
        )

        # Loss function configuration
        loss_config = config.get('loss', {})
        class_weights = loss_config.get('class_weights')
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # Keep a reference to class weights for manual per-token loss when needed
        self._class_weights_tensor = class_weights

        # Optional edge masking for loss within each window: fraction of sequence length per side
        # Default to 0.2 (20%) if not provided
        try:
            self.loss_window_margin_fraction: float = float(loss_config.get('loss_window_margin_fraction', 0.2))
        except Exception:
            self.loss_window_margin_fraction = 0.2

        # Focal loss options
        self.use_focal: bool = bool(loss_config.get('use_focal', False))
        self.focal_gamma: float = float(loss_config.get('focal_gamma', 2.0))
        focal_alpha = loss_config.get('focal_alpha')
        # If explicit focal_alpha is not provided, fall back to class_weights as alphas
        if focal_alpha is None:
            focal_alpha = class_weights.tolist() if class_weights is not None else None
        self.focal_alpha = torch.tensor(focal_alpha, dtype=torch.float32) if focal_alpha is not None else None
        
        # Save hyperparameters for logging
        self.save_hyperparameters(config)

        # Entropy regularization strength (lambda). Default 1e-3 if not specified
        try:
            self.entropy_lambda: float = float(loss_config.get('entropy_lambda', 1e-3))
        except Exception:
            self.entropy_lambda = 1e-3

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        return self.model(x, return_attention=return_attention)
    
    def _calculate_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate accuracy metrics for each class."""
        predictions = torch.argmax(logits, dim=-1)
        
        # Overall accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean()
        
        # Per-class accuracy
        metrics = {'accuracy': accuracy}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = (targets == class_idx)
            if class_mask.any():
                class_accuracy = correct[class_mask].mean()
                metrics[f'{class_name.lower()}_accuracy'] = class_accuracy
            else:
                metrics[f'{class_name.lower()}_accuracy'] = torch.tensor(0.0)
        
        return metrics
    
    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        logits = self.model(sequences)
        
        # Reshape for loss calculation
        batch_size, seq_length, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)

        # Build per-token weights to down-weight/zero window edges
        margin = int(max(0, min(seq_length // 2, round(self.loss_window_margin_fraction * seq_length))))
        if margin > 0 and seq_length > 2 * margin:
            center = torch.ones(seq_length, dtype=torch.float32, device=logits.device)
            center[:margin] = 0.0
            center[-margin:] = 0.0
            per_token_weights = center.unsqueeze(0).expand(batch_size, -1).contiguous().view(-1)
        else:
            per_token_weights = None
        
        # Calculate loss
        loss = self._compute_loss(logits_flat, targets_flat, per_token_weights)
        
        # Calculate metrics
        metrics = self._calculate_metrics(logits, targets)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', metrics['accuracy'], prog_bar=True)
        
        # Log per-class accuracies
        for class_name in self.class_names:
            metric_name = f'train_{class_name.lower()}_accuracy'
            if metric_name in metrics:
                self.log(metric_name, metrics[metric_name], prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        logits = self.model(sequences)
        
        # Reshape for loss calculation
        batch_size, seq_length, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)

        # Edge mask per token (1 for center, 0 for edges if margin>0)
        margin = int(max(0, min(seq_length // 2, round(self.loss_window_margin_fraction * seq_length))))
        if margin > 0 and seq_length > 2 * margin:
            center = torch.ones(seq_length, dtype=torch.float32, device=logits.device)
            center[:margin] = 0.0
            center[-margin:] = 0.0
            edge_mask = center.unsqueeze(0).expand(batch_size, -1).contiguous().view(-1) > 0
        else:
            edge_mask = torch.ones_like(targets_flat, dtype=torch.bool)
        
        # Class weight filter: only classes with weight>1.0 are included
        try:
            cw = self._class_weights_tensor
            if cw is not None:
                allowed = (cw > 1.0).to(dtype=torch.bool, device=logits.device)
                class_mask = allowed[targets_flat]
            else:
                class_mask = torch.ones_like(targets_flat, dtype=torch.bool)
        except Exception:
            class_mask = torch.ones_like(targets_flat, dtype=torch.bool)
        
        include_mask = edge_mask & class_mask
        if include_mask.any():
            # Per-token CE (unweighted for validation reporting)
            ce_vec = torch.nn.functional.cross_entropy(
                logits_flat, targets_flat,
                weight=None,
                reduction='none'
            )
            loss = ce_vec[include_mask].mean()
        else:
            loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        
        # Calculate metrics
        metrics = self._calculate_metrics(logits, targets)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', metrics['accuracy'], prog_bar=True)
        
        # Log per-class accuracies
        for class_name in self.class_names:
            metric_name = f'val_{class_name.lower()}_accuracy'
            if metric_name in metrics:
                self.log(metric_name, metrics[metric_name], prog_bar=True)
        
        return loss
    
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

    def _compute_loss(self, logits_flat: torch.Tensor, targets_flat: torch.Tensor, per_token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss with optional per-token weighting (edge masking)."""
        if not self.use_focal:
            # Base CE per-token
            if per_token_weights is None:
                ce_loss = self.criterion(logits_flat, targets_flat)
                # Entropy term (maximize entropy => subtract lambda * H)
                if self.entropy_lambda and self.entropy_lambda != 0.0:
                    probs = torch.softmax(logits_flat, dim=-1)
                    # -sum p log p
                    entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1).mean()
                    ce_loss = ce_loss - float(self.entropy_lambda) * entropy
                return ce_loss
            # Per-token CE with weights
            ce_vec = torch.nn.functional.cross_entropy(
                logits_flat, targets_flat,
                weight=self._class_weights_tensor.to(logits_flat.device) if self._class_weights_tensor is not None else None,
                reduction='none'
            )
            # Entropy per-token
            if self.entropy_lambda and self.entropy_lambda != 0.0:
                probs = torch.softmax(logits_flat, dim=-1)
                ent_vec = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)
                ce_vec = ce_vec - float(self.entropy_lambda) * ent_vec
            weights = per_token_weights.to(ce_vec.dtype)
            denom = torch.clamp((weights > 0).to(ce_vec.dtype).sum(), min=1.0)
            return (ce_vec * weights).sum() / denom
        # Focal loss path: compute per-token and apply weights if provided (no entropy reg with focal)
        return self._focal_loss(logits_flat, targets_flat, self.focal_gamma, self.focal_alpha, per_token_weights)

    @staticmethod
    def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float, alpha: Optional[torch.Tensor], per_token_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-class focal loss using logits.
        - logits: (N, C)
        - targets: (N,)
        - gamma: focusing parameter
        - alpha: Optional tensor of shape (C,) for per-class weighting
        """
        # Log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)  # (N, C)
        probs = torch.exp(log_probs)  # (N, C)

        # Gather true-class probabilities
        indices = torch.arange(logits.size(0), device=logits.device)
        log_pt = log_probs[indices, targets]  # (N,)
        pt = probs[indices, targets]          # (N,)

        # Alpha weighting
        if alpha is not None:
            alpha = alpha.to(logits.device)
            alpha_t = alpha[targets]
        else:
            alpha_t = 1.0

        focal_factor = (1.0 - pt).clamp(min=0.0) ** gamma
        loss = -alpha_t * focal_factor * log_pt
        if per_token_weights is not None:
            w = per_token_weights.to(loss.dtype)
            denom = torch.clamp(w.sum(), min=1.0)
            return (loss * w).sum() / denom
        return loss.mean()


def create_base_config(
    max_seq_length: int, num_classes: int, class_names: list,
    d_model: int = 504,
    n_layers: int = 3,
    n_heads: int = 6,
    learning_rate: float = 5e-5,
    max_epochs: int = 25,
    batch_size: int = 8,
    class_weights: Optional[list] = None,
    loss_window_margin_fraction: Optional[float] = 0.2,
    attention_masks: Optional[Dict[int, int]] = None,
    kmer_size: int = 0,
    use_focal: Optional[bool] = None,
    focal_gamma: Optional[float] = None,
    focal_alpha: Optional[list] = None,
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
            'loss_window_margin_fraction': loss_window_margin_fraction,
        },
        'class_names': class_names
    }

    # Optionally add focal loss settings
    if use_focal is not None:
        cfg['loss']['use_focal'] = use_focal
    if focal_gamma is not None:
        cfg['loss']['focal_gamma'] = focal_gamma
    if focal_alpha is not None:
        cfg['loss']['focal_alpha'] = focal_alpha

    return cfg
