#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Optional, Callable, List

from dna_learner.model import GenePredictorModule
import shutil


class BestCheckpointAlias(pl.Callback):
    """Maintain an alias checkpoints/best.ckpt pointing to the current best model.

    This avoids adding a second ModelCheckpoint while ensuring consumers can load
    the best checkpoint deterministically.
    """

    def __init__(self, checkpoints_dir: Path):
        super().__init__()
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        self._update_best_alias(trainer)

    def on_fit_end(self, trainer: pl.Trainer, pl_module):
        self._update_best_alias(trainer)

    def _update_best_alias(self, trainer: pl.Trainer):
        # Find primary ModelCheckpoint callback and get its best_model_path
        best_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                path = getattr(cb, 'best_model_path', '')
                if path:
                    best_path = Path(path)
                    break
        if best_path and best_path.exists():
            alias = self.checkpoints_dir / 'best.ckpt'
            try:
                shutil.copyfile(best_path, alias)
            except Exception:
                pass


def train(
    dataset,
    config,
    output_dir: Path,
    additional_callback_generator: Optional[Callable[[DataLoader], List[pl.Callback]]] = None,
    monitor_metric: str = 'val_loss',
    monitor_mode: str = 'min',
):

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

    # Neutral alias and early stopping on the primary metric only
    callbacks.append(BestCheckpointAlias(output_dir / "checkpoints"))
    callbacks.append(pl.callbacks.EarlyStopping(monitor=monitor_metric, patience=8, mode=monitor_mode))

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
