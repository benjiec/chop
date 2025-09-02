#!/usr/bin/env python3
"""
Shared Layout Detection Model for genomic sequence analysis.

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

from gene_predictor.model import DNAEmbedding, MaskedTransformerLayer


class LayoutDetectionModel(nn.Module):
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
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length) with DNA indices
            return_attention: Whether to return attention weights
            
        Returns:
            Logits of shape (batch_size, seq_length, num_classes)
            If return_attention=True, also returns dict of attention weights
        """
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
        
        # Classify each position
        logits = self.classifier(x)  # (batch_size, seq_length, num_classes)
        
        if return_attention:
            return logits, layer_attention_weights
        else:
            return logits

class LayoutDetectionModule(pl.LightningModule):
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
        self.model = LayoutDetectionModel(
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
