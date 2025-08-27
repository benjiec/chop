#!/usr/bin/env python3
"""
Training script for gene prediction (gene boundary detection).

This script trains a transformer-based model to predict gene boundaries:
INTERGENIC, UTR5, START, GENE_BODY, STOP, UTR3
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from models.gene_predictor import DNAEmbedding
from utils.gene_prediction_processor import (
    GenePredictionTargetGenerator, encode_dna_sequence, load_gene_contexts_gene_prediction
)
from utils.constants import GenePredictionClass, DNA_VOCAB


class GenePredictionModel(nn.Module):
    """Transformer model for gene boundary prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model parameters
        vocab_size = config.get('vocab_size', 5)  # A, C, G, T, N
        d_model = config.get('d_model', 256)
        n_layers = config.get('n_layers', 4)
        n_heads = config.get('n_heads', 8)
        dropout = config.get('dropout', 0.1)
        max_seq_length = config.get('max_seq_length', 12288)
        
        # DNA embedding
        self.embedding = DNAEmbedding(vocab_size, d_model, max_seq_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Gene boundary prediction head (6 classes)
        self.gene_boundary_head = nn.Linear(d_model, 6)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for gene prediction.
        
        Args:
            x: DNA sequence tensor (batch_size, seq_length)
            
        Returns:
            Dictionary with gene_boundaries predictions
        """
        # Embeddings
        hidden = self.embedding(x)  # (batch_size, seq_length, d_model)
        
        # Transformer layers
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        
        # Layer norm and dropout
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        
        # Gene boundary predictions
        gene_boundaries = self.gene_boundary_head(hidden)  # (batch_size, seq_length, 6)
        
        return {
            'gene_boundaries': gene_boundaries
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss is designed to address class imbalance by down-weighting 
    easy examples and focusing on hard examples.
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) where C = number of classes
            targets: (N) where each value is 0 <= targets[i] <= C-1
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GenePredictionLoss(nn.Module):
    """Loss function for gene prediction with class weighting or focal loss."""
    
    def __init__(self, class_weights: Optional[Dict[int, float]] = None, 
                 use_focal_loss: bool = False, focal_alpha: float = 1.0, focal_gamma: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        
        if use_focal_loss:
            # Set up focal loss
            alpha = None
            if class_weights:
                alpha = torch.zeros(6)
                for class_idx, weight in class_weights.items():
                    if 0 <= class_idx < 6:
                        alpha[class_idx] = weight * focal_alpha
                alpha = alpha / alpha.sum() * 6  # Normalize
            
            self.loss_fn = FocalLoss(alpha=alpha, gamma=focal_gamma, reduction='mean')
            self.weight_tensor = None
        else:
            # Set up weighted cross-entropy
            if class_weights:
                weights = torch.zeros(6)
                for class_idx, weight in class_weights.items():
                    if 0 <= class_idx < 6:
                        weights[class_idx] = weight
                self.register_buffer('weight_tensor', weights)
            else:
                self.weight_tensor = None
            self.loss_fn = None
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss for gene prediction.
        
        Args:
            predictions: Model predictions with 'gene_boundaries' key
            targets: Ground truth with 'gene_boundaries' key
            
        Returns:
            total_loss, loss_dict
        """
        pred_boundaries = predictions['gene_boundaries']  # (batch, seq_len, 6)
        true_boundaries = targets['gene_boundaries']      # (batch, seq_len)
        
        # Reshape for loss computation
        pred_boundaries = pred_boundaries.view(-1, 6)     # (batch*seq_len, 6)
        true_boundaries = true_boundaries.view(-1)        # (batch*seq_len,)
        
        # Compute loss based on loss type
        if self.use_focal_loss:
            boundary_loss = self.loss_fn(pred_boundaries, true_boundaries.long())
        else:
            boundary_loss = F.cross_entropy(
                pred_boundaries, 
                true_boundaries.long(),
                weight=self.weight_tensor,
                reduction='mean'
            )
        
        loss_dict = {
            'gene_boundary_loss': boundary_loss,
            'total_loss': boundary_loss
        }
        
        return boundary_loss, loss_dict


class GenePredictionDataset(torch.utils.data.Dataset):
    """Dataset for gene prediction training."""
    
    def __init__(self, fasta_file: Path, tsv_file: Path, max_seq_length: int = 4096):
        self.max_seq_length = max_seq_length
        self.target_generator = GenePredictionTargetGenerator()
        
        # Load sequences and annotations
        self.sequences = {}
        self.genes_data = {}
        
        # Load data
        self._load_fasta_and_annotations(fasta_file, tsv_file)
        
    def _load_genes_from_tsv(self, tsv_file: Path) -> List[Dict]:
        """Convert TSV annotations to gene list format for target generation."""
        import pandas as pd
        from collections import defaultdict
        
        df = pd.read_csv(tsv_file, sep='\t')
        
        # Group by sequence_id and gene_id to create gene entries
        genes_list = []
        gene_groups = df.groupby(['sequence_id', 'gene_id'])
        
        for (sequence_id, gene_id), group in gene_groups:
            # TSV coordinates are 1-based, convert to 0-based for processing
            gene_start = group['gene_start'].min() - 1  # Convert to 0-based
            gene_end = group['gene_end'].max()          # End is already exclusive in TSV format
            strand = group['strand'].iloc[0]
            
            # Extract exons (CDS entries)
            exons = []
            for _, row in group.iterrows():
                exon_start = row['exon_start'] - 1  # Convert to 0-based
                exon_end = row['exon_end']           # End is already exclusive
                exons.append({'start': exon_start, 'end': exon_end})
            
            # Sort exons by start position
            exons = sorted(exons, key=lambda x: x['start'])
            
            genes_list.append({
                'sequence_id': sequence_id,
                'gene_id': gene_id,
                'start': gene_start,
                'end': gene_end,
                'strand': strand,
                'exons': exons
            })
        
        return genes_list

    def _load_fasta_and_annotations(self, fasta_file: Path, tsv_file: Path):
        """Load FASTA sequences and TSV annotations."""
        # Load FASTA
        with open(fasta_file, 'r') as f:
            current_seq_id = None
            current_seq = []
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq_id:
                        self.sequences[current_seq_id] = ''.join(current_seq)
                    current_seq_id = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)
            
            if current_seq_id:
                self.sequences[current_seq_id] = ''.join(current_seq)
        
        # Load TSV annotations (preprocessed gene contexts)
        genes_list = self._load_genes_from_tsv(tsv_file)
        
        # Group genes by sequence
        for seq_id in self.sequences.keys():
            seq_genes = [gene for gene in genes_list if gene['sequence_id'] == seq_id]
            self.genes_data[seq_id] = seq_genes
        
        # Create windows for training
        self.windows = []
        for seq_id, sequence in self.sequences.items():
            seq_len = len(sequence)
            stride = self.max_seq_length // 4  # 75% overlap for smaller sequences
            
            for start in range(0, seq_len, stride):
                end = min(start + self.max_seq_length, seq_len)
                if end - start >= self.max_seq_length // 4:  # Minimum window size
                    self.windows.append({
                        'sequence_id': seq_id,
                        'start': start,
                        'end': end,
                        'sequence': sequence[start:end],
                        'genes': self.genes_data[seq_id]
                    })
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[idx]
        sequence = window['sequence']
        genes_list = window['genes']
        window_start = window['start']
        
        # Adjust gene coordinates to window coordinates
        adjusted_genes = []
        for gene in genes_list:
            gene_start = max(0, gene['start'] - window_start)
            gene_end = min(len(sequence) - 1, gene['end'] - window_start)
            
            # Only include genes that overlap with window
            if gene_start < len(sequence) and gene_end >= 0:
                # Adjust exon coordinates to window coordinates
                adjusted_exons = []
                for exon in gene.get('exons', []):
                    exon_start = max(0, exon['start'] - window_start)
                    exon_end = min(len(sequence) - 1, exon['end'] - window_start)
                    
                    # Only include exons that overlap with window
                    if exon_start < len(sequence) and exon_end >= 0:
                        adjusted_exons.append({
                            'start': exon_start,
                            'end': exon_end
                        })
                
                if adjusted_exons:  # Only include genes with exons in the window
                    adjusted_genes.append({
                        'sequence_id': gene['sequence_id'],
                        'gene_id': gene['gene_id'],
                        'start': gene_start,
                        'end': gene_end,
                        'strand': gene['strand'],
                        'exons': adjusted_exons
                    })
        
        # Generate targets
        targets = self.target_generator.generate_targets(sequence, adjusted_genes)
        
        # Encode sequence
        encoded_seq = encode_dna_sequence(sequence)
        
        # Pad to max_seq_length
        if len(encoded_seq) < self.max_seq_length:
            pad_length = self.max_seq_length - len(encoded_seq)
            encoded_seq = np.pad(encoded_seq, (0, pad_length), constant_values=DNA_VOCAB['N'])
            targets = np.pad(targets, (0, pad_length), constant_values=GenePredictionClass.INTERGENIC)
        
        return torch.tensor(encoded_seq, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class GenePredictionModule(pl.LightningModule):
    """PyTorch Lightning module for gene prediction training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create model
        self.model = GenePredictionModel(config['model'])
        
        # Store loss configuration
        self.loss_config = config.get('loss', {})
        
        # Create loss function (will set class weights after seeing data)
        self.loss_fn = GenePredictionLoss()
        
        # Learning rate
        self.learning_rate = float(config.get('training', {}).get('learning_rate', 1e-4))
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
    
    def set_class_weights(self, class_weights: Dict[int, float]):
        """Set class weights for loss function."""
        use_focal_loss = self.loss_config.get('use_focal_loss', False)
        focal_alpha = self.loss_config.get('focal_alpha', 1.0)
        focal_gamma = self.loss_config.get('focal_gamma', 2.0)
        
        self.loss_fn = GenePredictionLoss(
            class_weights=class_weights,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        
        # Forward pass
        predictions = self(sequences)
        
        # Compute loss
        loss, loss_dict = self.loss_fn(predictions, {'gene_boundaries': targets})
        
        # Log metrics
        for key, value in loss_dict.items():
            self.log(f'train_{key}', value, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        
        # Forward pass
        predictions = self(sequences)
        
        # Compute loss
        loss, loss_dict = self.loss_fn(predictions, {'gene_boundaries': targets})
        
        # Log metrics
        for key, value in loss_dict.items():
            self.log(f'val_{key}', value, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=float(self.config.get('training', {}).get('weight_decay', 0.01))
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.1
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_total_loss'
            }
        }


def create_data_loaders(config: Dict[str, Any], fasta_file: Path, tsv_file: Path) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Create dataset
    dataset = GenePredictionDataset(
        fasta_file=fasta_file,
        tsv_file=tsv_file,
        max_seq_length=config['model']['max_seq_length']
    )
    
    # Calculate class weights from dataset
    all_targets = []
    for i in range(len(dataset)):
        _, targets = dataset[i]
        all_targets.append(targets.numpy())
    
    all_targets = np.concatenate(all_targets)
    target_generator = GenePredictionTargetGenerator()
    
    # Choose class weighting strategy based on configuration
    weight_strategy = config.get('loss', {}).get('weight_strategy', 'capped')
    if weight_strategy == 'sqrt':
        class_weights = target_generator.get_class_weights_sqrt(all_targets)
    elif weight_strategy == 'capped':
        max_ratio = config.get('loss', {}).get('max_weight_ratio', 50.0)
        class_weights = target_generator.get_class_weights(all_targets, max_ratio)
    else:
        # Original uncapped weights (not recommended)
        class_weights = target_generator.get_class_weights(all_targets, max_weight_ratio=float('inf'))
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = config.get('training', {}).get('batch_size', 4)
    num_workers = config.get('data', {}).get('num_workers', 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, class_weights


def main():
    # Set environment variables to ensure progress bars show up
    import os
    os.environ.setdefault('TERM', 'xterm')
    if 'COLUMNS' not in os.environ:
        os.environ['COLUMNS'] = '80'
    if 'LINES' not in os.environ:
        os.environ['LINES'] = '24'
    
    parser = argparse.ArgumentParser(description='Train gene prediction model')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--fasta', required=True, help='Path to FASTA file')
    parser.add_argument('--tsv', required=True, help='Path to TSV annotation file')
    parser.add_argument('--output-dir', required=True, help='Output directory for model checkpoints')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, class_weights = create_data_loaders(
        config, Path(args.fasta), Path(args.tsv)
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Class weights: {class_weights}")
    
    # Create model
    model = GenePredictionModule(config)
    model.set_class_weights(class_weights)
    
    print(f"Model configuration:")
    print(f"  - Model size: {config['model']['d_model']}")
    print(f"  - Layers: {config['model']['n_layers']}")
    print(f"  - Heads: {config['model']['n_heads']}")
    print(f"  - Max epochs: {config['training']['max_epochs']}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Loss strategy: {model.loss_config}")
    print("Starting training...")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename='gene_prediction_model_{epoch:02d}_{val_total_loss:.3f}',
            monitor='val_total_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_total_loss',
            patience=config.get('training', {}).get('early_stopping_patience', 10),
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Setup logger
    logger = None
    if config.get('logging', {}).get('use_wandb', False):
        logger = WandbLogger(
            project=config['logging']['wandb_project'],
            name=f"gene_prediction_{config['model']['d_model']}_{config['model']['n_layers']}layers"
        )
    else:
        logger = TensorBoardLogger(
            save_dir=output_dir,
            name='gene_prediction_logs'
        )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config.get('hardware', {}).get('accelerator', 'auto'),
        devices=config.get('hardware', {}).get('devices', 'auto'),
        accumulate_grad_batches=config.get('training', {}).get('accumulate_grad_batches', 1),
        precision=config.get('training', {}).get('precision', 32),
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=10,  # More frequent logging
        enable_model_summary=True,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=2  # Quick validation check before training
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training completed. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
