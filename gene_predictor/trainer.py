#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Optional, Callable, List

from gene_predictor.model import GenePredictorModule


def train(dataset, config, output_dir: Path, additional_callback_generator: Optional[Callable[[DataLoader], List[pl.Callback]]] = None):

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=0
    )
    
    print(f"training samples: {len(train_dataset)}")
    print(f"validation samples: {len(val_dataset)}")
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module = GenePredictorModule(config)
    
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"model parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")
    print(f"full configuration: {config}")
    
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename='model_{epoch:02d}_{val_loss:.3f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            mode='min'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]

    if additional_callback_generator:
        callbacks.extend(additional_callback_generator(val_loader))

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='auto',
        devices='auto',
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        default_root_dir=output_dir
    )

    # Train model
    trainer.fit(module, train_loader, val_loader)
    
    # Test model
    trainer.test(module, val_loader)

    return module, val_loader
