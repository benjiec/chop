#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Optional, Callable, List
from dna_learner.model import GenePredictorModule
import shutil


def train(
    dataset,
    config,
    output_dir: Path,
    additional_callback_generator: Optional[Callable[[DataLoader], List[pl.Callback]]] = None,
    monitor_metric: str = 'val_loss',
    monitor_mode: str = 'min',
    custom_loss_fn: Optional[Callable] = None,
):

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # Prefer dataset-provided split (e.g., contig-aware) if available
    if hasattr(dataset, 'split') and callable(getattr(dataset, 'split')):
        print("Customized dataset splitting for training and validation")
        train_dataset, val_dataset = dataset.split(train_size, val_size)
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
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

    if custom_loss_fn is None:
        raise ValueError("custom_loss_fn is required for training; please provide one")
    module = GenePredictorModule(config, custom_loss_fn=custom_loss_fn)
    
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"model parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")
    print(f"full configuration: {config}")
    
    # Build filename that reflects the monitored metric for clarity/DRYness
    metric_key = monitor_metric
    filename_primary = f"model_epoch={{epoch:02d}}_{metric_key}={{{metric_key}:.3f}}"

    callbacks: List[pl.Callback] = []
    # Primary checkpoint uses the requested monitor_metric
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=filename_primary,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    ))

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
        default_root_dir=output_dir,
        accumulate_grad_batches=int(config.get('training', {}).get('accumulate_grad_batches', 1)),
    )

    # Train model
    trainer.fit(module, train_loader, val_loader)
    
    # Test model
    trainer.test(module, val_loader)

    return module, val_loader
