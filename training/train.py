#!/usr/bin/env python3
"""
Training script for the gene prediction model.

This script trains the transformer-based gene prediction model using PyTorch Lightning.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from models.gene_predictor import GenePredictor, BiologicalLoss, create_model
from utils.dna_processor import DNADataset, load_fasta_sequences, load_gff_annotations
from torch.utils.data import DataLoader, random_split


class GenePredictionModule(pl.LightningModule):
    """PyTorch Lightning module for gene prediction training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create model
        self.model = create_model(config['model'])
        
        # Create loss function
        self.loss_fn = BiologicalLoss(**config.get('loss', {}))
        
        # Learning rate
        self.learning_rate = config.get('training', {}).get('learning_rate', 1e-4)
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.model(x, mask)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['targets']
        
        # Forward pass
        predictions = self.model(input_ids, attention_mask)
        
        # Calculate loss
        loss = self.loss_fn(predictions, targets)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        # Update metrics
        self._update_metrics(predictions, targets, 'train')
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['targets']
        
        # Forward pass
        predictions = self.model(input_ids, attention_mask)
        
        # Calculate loss
        loss = self.loss_fn(predictions, targets)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        
        # Update metrics
        self._update_metrics(predictions, targets, 'val')
        
        return loss
    
    def on_train_epoch_end(self):
        """Log training metrics at the end of each epoch."""
        for metric_name, metric_value in self.train_metrics.items():
            self.log(f'train_{metric_name}', metric_value, on_epoch=True)
        self.train_metrics = {}
    
    def on_validation_epoch_end(self):
        """Log validation metrics at the end of each epoch."""
        for metric_name, metric_value in self.val_metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_epoch=True)
        self.val_metrics = {}
    
    def _update_metrics(self, predictions: Dict[str, torch.Tensor], 
                        targets: Dict[str, torch.Tensor], split: str):
        """Update metrics for the current batch."""
        metrics = self.train_metrics if split == 'train' else self.val_metrics
        
        # Gene boundary accuracy
        if 'gene_boundaries' in predictions and 'gene_boundaries' in targets:
            pred_labels = torch.argmax(predictions['gene_boundaries'], dim=-1)
            true_labels = torch.argmax(targets['gene_boundaries'], dim=-1)
            accuracy = (pred_labels == true_labels).float().mean()
            metrics['gene_boundary_accuracy'] = accuracy
        
        # Exon/intron accuracy
        if 'exon_intron' in predictions and 'exon_intron' in targets:
            pred_labels = torch.argmax(predictions['exon_intron'], dim=-1)
            true_labels = torch.argmax(targets['exon_intron'], dim=-1)
            accuracy = (pred_labels == true_labels).float().mean()
            metrics['exon_intron_accuracy'] = accuracy
        
        # Splice site accuracy
        if 'splice_sites' in predictions and 'splice_sites' in targets:
            pred_labels = torch.argmax(predictions['splice_sites'], dim=-1)
            true_labels = torch.argmax(targets['splice_sites'], dim=-1)
            accuracy = (pred_labels == true_labels).float().mean()
            metrics['splice_site_accuracy'] = accuracy
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.get('training', {}).get('weight_decay', 0.01)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get('training', {}).get('max_epochs', 100)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


def create_data_loaders(config: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Load sequences and annotations
    sequences = load_fasta_sequences(config['data']['sequences_path'])
    
    annotations = []
    if 'annotations_path' in config['data']:
        annotations = load_gff_annotations(config['data']['annotations_path'])
    
    # Create dataset
    dataset = DNADataset(
        sequences=sequences,
        annotations=annotations,
        max_length=config['model']['max_seq_length']
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train gene prediction model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    if config.get('logging', {}).get('use_wandb', False):
        logger = WandbLogger(
            project=config['logging']['project_name'],
            name=config['logging']['run_name']
        )
    else:
        logger = TensorBoardLogger('logs', name='gene_prediction')
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Create model
    model = GenePredictionModule(config)
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config['training']['checkpoint_dir'],
            filename='gene_predictor-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training'].get('patience', 10),
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='auto',
        devices='auto',
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=0.25,
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        deterministic=args.debug
    )
    
    # Train model
    if args.resume:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    else:
        trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_model_path = os.path.join(config['training']['checkpoint_dir'], 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    main()
