#!/usr/bin/env python3

import sys
import os
from pathlib import Path

from dna_learner.trainer import train as run_trainer
import argparse
from datetime import datetime
import numpy as np
from typing import Optional, Dict, Callable

import pytorch_lightning as pl

from utils.constants import GenePredictionClass as P
from utils.constants import StandardDonorDinucleotides, DinoDonorDinucleotides
from utils.losses import event_based_ce_loss_factory, event_based_bce_loss_factory
from dna_learner.model import GenePredictorModule, create_base_config
from utils.constants import GenePredictionClass as P
from gene_predictor.metrics_callback import F1Callback
from gene_predictor.metrics_callback import LossComponentsCallback
from gene_predictor.metrics_callback import DualMetricEarlyStopping
from utils.genome import AnnotatedGenomeDataset


def create_config(d_model: int = 512, n_layers: int = 4, n_heads: int = 8,
                  learning_rate: float = 5e-5, max_epochs: int = 25, batch_size: int = 8,
                  use_class_weights: bool = True,
                  start_weight: float = 5.0, stop_weight: float = 3.0, dss_weight: float = 3.0, ass_weight: float = 2.0,
                  start_neg_weight: float = 1.0, stop_neg_weight: float = 1.0, dss_neg_weight: float = 1.5, ass_neg_weight: float = 1.5,
                  utr_weight: float = 3.0,
                  attention_masks: Optional[Dict[int, int]] = None, kmer_size: int = 3,
                  max_seq_length: int = 1000,
                  use_focal: bool = False, focal_gamma: float = 1.5,
                  focal_alpha: Optional[list] = None,
                  cc_enabled: bool = False,
                  start_before: int = 300, start_after: int = 0,
                  stop_before: int = 0, stop_after: int = 300,
                  cc_gap: int = 0,
                  entropy_lambda: float = 0.0,
                  fp_beta: float = 5.0,
                  accumulate_grad_batches: int = 1,
                  min_per_class_per_batch: int = 1) -> dict:

    # Class weights for START/STOP detection
    # START/STOP codons are rare and important, UTR5 regions provide context
    if use_class_weights:
        # Build full weights map for all defined classes
        weights_map = {
            P.INTERGENIC: 1.0,
            P.UTR5: utr_weight,
            P.START: start_weight,
            P.GENE: 1.0,
            P.STOP: stop_weight,
            P.UTR3: utr_weight,
            P.DSS: dss_weight,
            P.ASS: ass_weight,
        }
        class_weights = [weights_map.get(i, 1.0) for i in sorted(P.idx_to_cls.keys())]
    else:
        weights_map = None
        class_weights = None
    
    cfg = create_base_config(
        max_seq_length=max_seq_length,
        num_classes=len(P.idx_to_cls),
        class_names=[P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())],
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        class_weights=class_weights,
        attention_masks=attention_masks,
        kmer_size=kmer_size,
        use_focal=use_focal,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        entropy_lambda=entropy_lambda,
        fp_beta=fp_beta,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    if cc_enabled:
        cfg['model']['class_conditional_readouts'] = {
            'enabled': True,
            'entries': [
                {'class': 'START', 'before': int(start_before), 'after': int(start_after), 'gap': int(cc_gap)},
                {'class': 'STOP', 'before': int(stop_before), 'after': int(stop_after), 'gap': int(cc_gap)},
            ]
        }

    # Trainer-level sampler option
    cfg['training']['min_per_class_per_batch'] = int(min_per_class_per_batch)

    # Record loss weights in config for reference (not consumed by dna_learner yet)
    cfg.setdefault('loss', {})
    cfg['loss']['bce_pos_weights'] = {
        int(P.START): float(start_weight),
        int(P.STOP): float(stop_weight),
        int(P.DSS): float(dss_weight),
        int(P.ASS): float(ass_weight),
    }
    cfg['loss']['bce_neg_weights'] = {
        int(P.START): float(start_neg_weight),
        int(P.STOP): float(stop_neg_weight),
        int(P.DSS): float(dss_neg_weight),
        int(P.ASS): float(ass_neg_weight),
    }
    cfg['loss']['ce_class_weights'] = {
        int(P.INTERGENIC): 1.0,
        int(P.UTR5): float(utr_weight),
        int(P.START): float(start_weight),
        int(P.GENE): 1.0,
        int(P.STOP): float(stop_weight),
        int(P.UTR3): float(utr_weight),
        int(P.DSS): float(dss_weight),
        int(P.ASS): float(ass_weight),
    }
    return cfg


def train(fna_fn: str, tsv_fn: str,
          d_model: int = 512, n_layers: int = 4, n_heads: int = 8,
          learning_rate: float = 5e-5, max_epochs: int = 25, batch_size: int = 8,
          use_class_weights: bool = True,
          start_weight: float = 5.0, stop_weight: float = 3.0, dss_weight: float = 3.0, ass_weight: float = 2.0,
          start_neg_weight: float = 1.0, stop_neg_weight: float = 1.0, dss_neg_weight: float = 1.5, ass_neg_weight: float = 1.5,
          utr_weight: float = 3.0,
          attention_masks: Optional[Dict[int, int]] = None, kmer_size: int = 3,
          max_seq_length: int = 1000,
          stride: int = 500,
          num_windows: int = 5000,
          use_focal: bool = False, focal_gamma: float = 1.5,
          focal_alpha: Optional[list] = None,
          cc_enabled: bool = False,
          start_before: int = 300, start_after: int = 0,
          stop_before: int = 0, stop_after: int = 300,
          cc_gap: int = 0,
          entropy_lambda: float = 0.0,
          fp_beta: float = 5.0,
          accumulate_grad_batches: int = 1,
          min_per_class_per_batch: int = 1,
          custom_loss_fn: Optional[Callable] = None,
          dss_motifs: Optional[str] = None):

    # Create config
    config = create_config(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        learning_rate=learning_rate, max_epochs=max_epochs, batch_size=batch_size,
        use_class_weights=use_class_weights, start_weight=start_weight, stop_weight=stop_weight, utr_weight=utr_weight,
        dss_weight=dss_weight, ass_weight=ass_weight,
        attention_masks=attention_masks, kmer_size=kmer_size, max_seq_length=max_seq_length,
        use_focal=use_focal, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
        cc_enabled=cc_enabled,
        start_before=start_before, start_after=start_after,
        stop_before=stop_before, stop_after=stop_after,
        cc_gap=cc_gap,
        entropy_lambda=entropy_lambda,
        fp_beta=fp_beta,
        accumulate_grad_batches=accumulate_grad_batches,
        min_per_class_per_batch=min_per_class_per_batch,
        start_neg_weight=start_neg_weight,
        stop_neg_weight=stop_neg_weight,
        dss_neg_weight=dss_neg_weight,
        ass_neg_weight=ass_neg_weight,
    )

    # Pass class weights to dataset for sampling/accounting (format: list of floats indexed by class id)
    dataset_class_weights = config.get('loss', {}).get('class_weights')

    if num_windows:
        dataset = AnnotatedGenomeDataset(
            fna_fn,
            tsv_fn,
            window=max_seq_length,
            stride=stride,
            num_windows=num_windows,
            class_weights=dataset_class_weights
        )
    else:
        dataset = AnnotatedGenomeDataset(
            fna_fn,
            tsv_fn,
            window=max_seq_length,
            stride=stride,
            class_weights=dataset_class_weights,
        )

    # Create output directory early for saving sample data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"gene_predictor/gene_predictor_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def mk_training_cb(val_loader):
        # Save best checkpoints by F1 in addition to the primary loss-based checkpoint
        f1_ckpt = pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="model_epoch={epoch:02d}_val_f1={val_f1:.3f}",
            monitor='val_f1',
            mode='max',
            save_top_k=3,
            save_last=False,
            auto_insert_metric_name=False,
        )
        return [ F1Callback(val_loader), LossComponentsCallback(report_train_components=True, run_dir=output_dir), DualMetricEarlyStopping(patience=8), f1_ckpt ]
    
    model, val_loader = run_trainer(
        dataset,
        config,
        output_dir,
        mk_training_cb,
        monitor_metric='val_loss',
        monitor_mode='min',
        custom_loss_fn=custom_loss_fn,
    )

    print(f"results saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train gene boundary and splicing site detection")
    parser.add_argument('--fna-fn', type=str, required=True, help='File name for genome sequence in FASTA format')
    parser.add_argument('--tsv-fn', type=str, required=True, help='File name for annotations in TSV format')
    parser.add_argument('--max-seq-length', type=int, default=1000, help='Maximum sequence length (also used as dataset window size)')
    parser.add_argument('--stride', type=int, default=500, help='Window stride')
    parser.add_argument('--num-windows', type=int, default=5000, help='Number of windows to train with; if 0, do not sample windows')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=25, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--kmer', type=int, default=3, help='K-mer size for convolution (0=disabled, 3=codon sensitive)')
    parser.add_argument('--attention-masks', type=str,
                        help='Head attention masks: symmetric "head:window", asymmetric "head:before:after", or donut "head:before:gap:after" (e.g., "0:4,1:8:6,2:50:0,3:20:10:0")')

    # class weights
    parser.add_argument('--disable-class-weights', action='store_true', help='Disable class weights')
    parser.add_argument('--start-weight', type=float, default=5.0, help='Weight for START class')
    parser.add_argument('--stop-weight', type=float, default=3.0, help='Weight for STOP class')
    parser.add_argument('--dss-weight', type=float, default=3.0, help='Weight for DSS class')
    parser.add_argument('--ass-weight', type=float, default=2.0, help='Weight for ASS class')
    parser.add_argument('--utr-weight', type=float, default=3.0, help='Weight for UTR5/UTR3 class')
    # Negative weights for BCE-style losses (reuse class weights for positive weights)
    parser.add_argument('--start-neg-weight', type=float, default=1.0, help='Negative class weight for START (BCE)')
    parser.add_argument('--stop-neg-weight', type=float, default=1.0, help='Negative class weight for STOP (BCE)')
    parser.add_argument('--dss-neg-weight', type=float, default=1.5, help='Negative class weight for DSS (BCE)')
    parser.add_argument('--ass-neg-weight', type=float, default=1.5, help='Negative class weight for ASS (BCE)')

    # focal loss
    parser.add_argument('--use-focal', action='store_true', help='Enable focal loss instead of cross-entropy')
    parser.add_argument('--focal-gamma', type=float, default=1.5, help='Focal loss gamma (focusing parameter)')
    parser.add_argument('--focal-alpha', type=str, default=None,
                        help='Comma-separated per-class alpha weights for focal loss (e.g., "1.0,3.0,8.0"). Defaults to class-weights if omitted')

    # class conditional readout
    parser.add_argument('--enable-cc', action='store_true', help='Enable class-conditional readouts for START/STOP (disabled by default)')
    parser.add_argument('--start-before', type=int, default=300, help='CC upstream window for START')
    parser.add_argument('--start-after', type=int, default=0, help='CC downstream window for START')
    parser.add_argument('--stop-before', type=int, default=0, help='CC upstream window for STOP')
    parser.add_argument('--stop-after', type=int, default=300, help='CC downstream window for STOP')
    parser.add_argument('--cc-gap', type=int, default=0, help='Relative donut gap for CC masks')

    # optimization and loss tuning
    parser.add_argument('--accumulate-grad-batches', type=int, default=1, help='Accumulate gradients over this many steps')
    parser.add_argument('--entropy-lambda', type=float, default=0.0, help='Entropy regularization strength')
    parser.add_argument('--fp-beta', type=float, default=5.0, help='False positive penalty coefficient')
    parser.add_argument('--min-per-class-per-batch', type=int, default=1, help='Minimum items per target class per batch (recycling allowed)')
    # required DSS motifs choice
    parser.add_argument('--dss-motifs', type=str, required=True, choices=['standard', 'dino'], help='Donor splice site motifs to use for event-based loss: standard or dino')
    # loss selection
    parser.add_argument('--loss-type', type=str, default='event-ce', choices=['event-ce', 'event-bce'], help='Loss type: event-ce (masked cross-entropy) or event-bce (masked BCE per event class)')

    args = parser.parse_args()
    
    # Parse attention masks (support symmetric, asymmetric, and donut)
    attention_masks = None
    if args.attention_masks:
        attention_masks = {}
        for mask_spec in args.attention_masks.split(','):
            parts = mask_spec.split(':')
            head = int(parts[0])
            if len(parts) == 2:
                # Symmetric: head:window
                attention_masks[head] = int(parts[1])
            elif len(parts) == 3:
                # Asymmetric: head:before:after
                before, after = int(parts[1]), int(parts[2])
                attention_masks[head] = (before, after)
            elif len(parts) == 4:
                # Donut: head:before:gap:after
                before, gap, after = int(parts[1]), int(parts[2]), int(parts[3])
                attention_masks[head] = (before, gap, after)
            else:
                raise ValueError(f"Invalid attention mask format: {mask_spec}. Use 'head:window', 'head:before:after', or 'head:before:gap:after'")

    # Parse focal alpha list if provided
    focal_alpha = None
    if args.focal_alpha:
        try:
            focal_alpha = [float(x) for x in args.focal_alpha.split(',') if x.strip() != '']
        except Exception as e:
            raise ValueError(f"Invalid --focal-alpha '{args.focal_alpha}': must be comma-separated floats") from e
    
    # Resolve DSS motifs
    if args.dss_motifs == 'standard':
        dss_set = StandardDonorDinucleotides
    elif args.dss_motifs == 'dino':
        dss_set = DinoDonorDinucleotides
    else:
        raise ValueError('Invalid --dss-motifs')

    # Build custom loss from selection, wiring CLI weights
    if args.loss_type == 'event-ce':
        ce_weights = None
        if not args.disable_class_weights:
            ce_weights = {
                int(P.INTERGENIC): 1.0,
                int(P.UTR5): float(args.utr_weight),
                int(P.START): float(args.start_weight),
                int(P.GENE): 1.0,
                int(P.STOP): float(args.stop_weight),
                int(P.UTR3): float(args.utr_weight),
                int(P.DSS): float(args.dss_weight),
                int(P.ASS): float(args.ass_weight),
            }
        custom_loss = event_based_ce_loss_factory(dss_set, class_weights=ce_weights)
    else:
        pos_weights = {
            int(P.START): float(args.start_weight),
            int(P.STOP): float(args.stop_weight),
            int(P.DSS): float(args.dss_weight),
            int(P.ASS): float(args.ass_weight),
        }
        neg_weights = {
            int(P.START): float(args.start_neg_weight),
            int(P.STOP): float(args.stop_neg_weight),
            int(P.DSS): float(args.dss_neg_weight),
            int(P.ASS): float(args.ass_neg_weight),
        }
        custom_loss = event_based_bce_loss_factory(dss_set, pos_weights=pos_weights, neg_weights=neg_weights)

    # Run training
    output_dir = train(
        args.fna_fn, args.tsv_fn,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        use_class_weights=not args.disable_class_weights,
        start_weight=args.start_weight,
        stop_weight=args.stop_weight,
        utr_weight=args.utr_weight,
        dss_weight=args.dss_weight,
        ass_weight=args.ass_weight,
        attention_masks=attention_masks,
        kmer_size=args.kmer,
        max_seq_length=args.max_seq_length,
        stride=args.stride,
        num_windows=args.num_windows,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha,
        cc_enabled=args.enable_cc,
        start_before=args.start_before,
        start_after=args.start_after,
        stop_before=args.stop_before,
        stop_after=args.stop_after,
        cc_gap=args.cc_gap,
        entropy_lambda=args.entropy_lambda,
        fp_beta=args.fp_beta,
        accumulate_grad_batches=args.accumulate_grad_batches,
        min_per_class_per_batch=args.min_per_class_per_batch,
        custom_loss_fn=custom_loss,
        dss_motifs=args.dss_motifs,
        start_neg_weight=args.start_neg_weight,
        stop_neg_weight=args.stop_neg_weight,
        dss_neg_weight=args.dss_neg_weight,
        ass_neg_weight=args.ass_neg_weight,
    )

if __name__ == "__main__":
    main()
