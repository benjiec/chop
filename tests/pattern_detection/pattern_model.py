#!/usr/bin/env python3
"""
Shared Pattern Detection Model for genomic sequence analysis.

This model can be configured for different tasks:
- Simple ATG detection (2 classes)
- UTR pattern detection (3 classes)
- Other genomic pattern recognition tasks

The model architecture and training logic are shared, while datasets
and task-specific configurations are handled by individual test drivers.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Dict, Any, Optional
import sys
from pathlib import Path
import math

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.gene_predictor import DNAEmbedding

class MaskedTransformerLayer(nn.Module):
    """
    Transformer layer with per-head attention masking support.
    """
    
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float,
                 attention_masks: Dict[int, int], max_seq_length: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention_masks = attention_masks
        self.max_seq_length = max_seq_length
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
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
        
    def _create_attention_mask(self, seq_length: int) -> Optional[torch.Tensor]:
        """Create combined attention mask for all heads."""
        if not self.attention_masks:
            return None
            
        # Create mask for each head
        device = next(self.parameters()).device
        mask = torch.zeros(self.n_heads, seq_length, seq_length, device=device, dtype=torch.bool)
        
        for head_idx in range(self.n_heads):
            if head_idx in self.attention_masks:
                window_size = self.attention_masks[head_idx]
                
                # Create local window mask for this head
                for i in range(seq_length):
                    start = max(0, i - window_size)
                    end = min(seq_length, i + window_size + 1)
                    mask[head_idx, i, start:end] = True
            else:
                # No mask - global attention
                mask[head_idx, :, :] = True
        
        # Convert to attention mask format (True = attend, False = ignore)
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with masked attention."""
        seq_length = x.size(1)
        
        # Create attention mask
        attn_mask = self._create_attention_mask(seq_length)
        
        # Self-attention with residual connection
        if attn_mask is not None:
            # For now, use the first head's mask as a global mask
            # TODO: Implement proper per-head masking
            global_mask = attn_mask[0]  # Use head 0's mask for all heads
            # Convert to attention mask format (False = ignore)
            attention_mask = ~global_mask  # Invert for PyTorch format
            attn_output, _ = self.self_attn(x, x, x, attn_mask=attention_mask)
        else:
            attn_output, _ = self.self_attn(x, x, x)
        
        x = self.norm1(x + attn_output)
        
        # Feed forward with residual connection  
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x
    


class PatternDetectionModel(nn.Module):
    """
    Configurable transformer model for genomic pattern detection.
    
    Can handle different numbers of classes and model sizes based on configuration.
    Supports per-head attention masking for local pattern detection.
    """
    
    def __init__(self, vocab_size: int = 5, d_model: int = 504, n_layers: int = 3, 
                 n_heads: int = 6, num_classes: int = 3, dropout: float = 0.1, 
                 max_seq_length: int = 1000, attention_masks: Optional[Dict[int, int]] = None,
                 kmer_size: int = 3):
        super().__init__()
        
        # Validate that d_model is divisible by n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        # Store attention mask configuration
        self.attention_masks = attention_masks or {}
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length) with DNA indices
            
        Returns:
            Logits of shape (batch_size, seq_length, num_classes)
        """
        # Embed DNA sequence
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        
        # Apply transformer layers with masking
        for layer in self.transformer_layers:
            x = layer(x)  # (batch_size, seq_length, d_model)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classify each position
        logits = self.classifier(x)  # (batch_size, seq_length, num_classes)
        
        return logits

class PatternDetectionModule(pl.LightningModule):
    """
    PyTorch Lightning module for pattern detection.
    
    Handles training, validation, and testing for any number of classes.
    Class names and metrics are configurable.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Store config
        self.config = config
        self.learning_rate = config['training']['learning_rate']
        self.num_classes = config['model']['num_classes']
        self.class_names = config.get('class_names', [f'Class_{i}' for i in range(self.num_classes)])
        
        # Create model
        model_config = config['model']
        self.model = PatternDetectionModel(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout'],
            max_seq_length=model_config['max_seq_length'],
            attention_masks=model_config.get('attention_masks'),
            kmer_size=model_config.get('kmer_size', 3)
        )
        
        # Loss function
        loss_config = config.get('loss', {})
        class_weights = loss_config.get('class_weights')
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Save hyperparameters for logging
        self.save_hyperparameters(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
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
        
        # Calculate loss
        loss = self.criterion(logits_flat, targets_flat)
        
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
        
        # Calculate loss
        loss = self.criterion(logits_flat, targets_flat)
        
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

def create_base_config(
    num_classes: int,
    class_names: list,
    d_model: int = 504,
    n_layers: int = 3,
    n_heads: int = 6,
    max_seq_length: int = 1000,
    learning_rate: float = 5e-5,
    max_epochs: int = 25,
    batch_size: int = 8,
    class_weights: Optional[list] = None,
    attention_masks: Optional[Dict[int, int]] = None,
    kmer_size: int = 3
) -> dict:
    """
    Create a base configuration that can be customized by individual tests.
    
    Args:
        num_classes: Number of output classes
        class_names: List of class names for logging
        d_model: Model dimension (must be divisible by n_heads)
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        max_seq_length: Maximum sequence length
        learning_rate: Learning rate
        max_epochs: Maximum training epochs
        batch_size: Batch size
        class_weights: Optional class weights for loss function
        attention_masks: Optional dict mapping head_idx -> window_size for local attention
    
    Returns:
        Configuration dictionary
    """
    # Validate d_model is divisible by n_heads
    if d_model % n_heads != 0:
        # Adjust d_model to nearest multiple
        d_model = (d_model // n_heads) * n_heads
        print(f"Warning: Adjusted d_model to {d_model} (divisible by {n_heads} heads)")
    
    return {
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
            'class_weights': class_weights
        },
        'class_names': class_names
    }
