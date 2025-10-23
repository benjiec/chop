#!/usr/bin/env python3

import sys
import os
from pathlib import Path

from dna_learner.trainer import train as run_trainer
import torch
import numpy as np
import argparse
from datetime import datetime
from typing import Optional, Dict, Callable

import pytorch_lightning as pl

from utils.constants import GenePredictionClass as P
from utils.constants import EventHeadIdx as H
from utils.constants import StandardDonorDinucleotides, DinoDonorDinucleotides
from utils.losses import event_based_ce_loss_factory, event_based_bce_loss_factory, event_head_bce_loss_factory
from utils.events import build_event_motifs, build_class_logits_from_event_head_logits
from utils.metrics import event_based_generic_metrics_factory, event_based_brier_factory
from dna_learner.model import GenePredictorModule, create_base_config
from gene_predictor.metrics_callback import MetricsCallback
from gene_predictor.metrics_callback import DualMetricEarlyStopping
from utils.genome import AnnotatedGenomeDataset


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
    parser.add_argument('--accumulate-grad-batches', type=int, default=1, help='Accumulate gradients over this many steps')
    parser.add_argument('--boost-start-stop', action='store_true')
    parser.add_argument('--aux-stream', type=str, default=None, help='Path to aux stream pickle file (per-sequence [L,C] features)')


    # class weights
    parser.add_argument('--disable-class-weights-for-loss', action='store_true', help='Disable class weights in loss function (still used for dataset)')
    parser.add_argument('--start-weight', type=float, default=5.0, help='Weight for START class')
    parser.add_argument('--stop-weight', type=float, default=3.0, help='Weight for STOP class')
    parser.add_argument('--dss-weight', type=float, default=4.0, help='Weight for DSS class')
    parser.add_argument('--ass-weight', type=float, default=2.0, help='Weight for ASS class')
    # Negative weights for BCE-style losses (reuse class weights for positive weights)
    parser.add_argument('--start-neg-weight', type=float, default=2.0, help='Negative class weight for START (BCE)')
    parser.add_argument('--stop-neg-weight', type=float, default=2.0, help='Negative class weight for STOP (BCE)')
    parser.add_argument('--dss-neg-weight', type=float, default=4.0, help='Negative class weight for DSS (BCE)')
    parser.add_argument('--ass-neg-weight', type=float, default=3.0, help='Negative class weight for ASS (BCE)')

    # Alpha weights per head/class (only used in event-head-bce)
    parser.add_argument('--start-alpha', type=float, default=1.0, help='Alpha weight for START head (event-head-bce)')
    parser.add_argument('--stop-alpha', type=float, default=1.0, help='Alpha weight for STOP head (event-head-bce)')
    parser.add_argument('--dss-alpha', type=float, default=1.0, help='Alpha weight for DSS head (event-head-bce)')
    parser.add_argument('--ass-alpha', type=float, default=1.0, help='Alpha weight for ASS head (event-head-bce)')

    # required DSS motifs choice
    parser.add_argument('--dss-motifs', type=str, required=True, choices=['standard', 'dino'], help='Donor splice site motifs to use for event-based loss: standard or dino')
    # loss selection
    parser.add_argument('--loss-type', type=str, default='event-ce', choices=['event-ce', 'event-bce', 'event-head-bce'],
                        help='Loss type: event-ce (masked cross-entropy), event-bce (masked BCE), or event-head-bce (masked BCE on separate event heads)')

    args = parser.parse_args()
    margin_bp = min(200, args.max_seq_length // 2)
    
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

    # Resolve DSS motifs
    if args.dss_motifs == 'standard':
        dss_set = StandardDonorDinucleotides
    elif args.dss_motifs == 'dino':
        dss_set = DinoDonorDinucleotides
    else:
        raise ValueError('Invalid --dss-motifs')

    class_weights_map = {
        P.INTERGENIC: 1.0,
        P.UTR5: 3.0,
        P.START: args.start_weight,
        P.GENE: 1.0,
        P.STOP: args.stop_weight,
        P.UTR3: 3.0,
        P.DSS: args.dss_weight,
        P.ASS: args.ass_weight,
    }
    class_weights = [class_weights_map.get(i, 1.0) for i in sorted(P.idx_to_cls.keys())]

    config = create_base_config(
        max_seq_length=args.max_seq_length,
        num_classes=len(P.idx_to_cls),
        class_names=[P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())],
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        class_weights=class_weights,
        attention_masks=attention_masks,
        kmer_size=args.kmer,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_event_heads=(4 if args.loss_type == 'event-head-bce' else None),
    )

    # CC readouts removed: configure directional attention via attention_masks and heads

    config.setdefault('custom', {})
    config['custom'].setdefault('loss', {})

    ce_weight_map = None
    bce_pos_weight_map = None
    bce_neg_weight_map = None
    bce_alpha_weight_map = None

    event_motifs_by_class = build_event_motifs(dss_set)
    # Persist motifs map always; head mapping only in event-head mode
    config['custom']['event_motifs_by_class'] = {int(k): sorted(list(v)) for k, v in event_motifs_by_class.items()}

    # Set weights in one place
    if not args.disable_class_weights_for_loss:
        if args.loss_type == 'event-ce':
            ce_weight_map = class_weights_map
            config['custom']['loss']['ce_class_weights'] = ce_weight_map
        else:
            bce_pos_weight_map = {
                int(P.START): float(args.start_weight),
                int(P.STOP): float(args.stop_weight),
                int(P.DSS): float(args.dss_weight),
                int(P.ASS): float(args.ass_weight),
            }
            bce_neg_weight_map = {
                int(P.START): float(args.start_neg_weight),
                int(P.STOP): float(args.stop_neg_weight),
                int(P.DSS): float(args.dss_neg_weight),
                int(P.ASS): float(args.ass_neg_weight),
            }
            config['custom']['loss']['bce_pos_weights'] = bce_pos_weight_map
            config['custom']['loss']['bce_neg_weights'] = bce_neg_weight_map

    # Alpha weights per class (used only by event-head BCE); persist regardless of class-weight toggle
    bce_alpha_weight_map = {
        int(P.START): float(args.start_alpha),
        int(P.STOP): float(args.stop_alpha),
        int(P.DSS): float(args.dss_alpha),
        int(P.ASS): float(args.ass_alpha),
    }
    config['custom']['loss']['bce_alpha_weights'] = bce_alpha_weight_map

    # Build event motifs map and custom loss
    head_class_ids = None
    if args.loss_type == 'event-ce':
        print("using event_based_ce_loss")
        custom_loss = event_based_ce_loss_factory(
            event_motifs_by_class,
            class_weights=ce_weight_map,
            loss_window_margin_bp=margin_bp,
        )

    elif args.loss_type == 'event-bce':
        print("using event_based_bce_loss")
        custom_loss = event_based_bce_loss_factory(
            event_motifs_by_class,
            pos_weights=bce_pos_weight_map,
            neg_weights=bce_neg_weight_map,
            loss_window_margin_bp=margin_bp,
        )

    else:
        # Event-head BCE using class-keyed maps and explicit head_class_ids order
        head_class_ids = [int(P.START), int(P.STOP), int(P.DSS), int(P.ASS)]
        config['custom']['head_class_ids'] = head_class_ids

        print("using event_head_bce_loss")
        custom_loss = event_head_bce_loss_factory(
            event_motifs_by_class,
            head_class_ids,
            pos_weights_by_class=bce_pos_weight_map,
            neg_weights_by_class=bce_neg_weight_map,
            alpha_weights_by_class=bce_alpha_weight_map,
            loss_window_margin_bp=margin_bp,
        )

    if args.num_windows:
        dataset = AnnotatedGenomeDataset(
            args.fna_fn,
            args.tsv_fn,
            window=args.max_seq_length,
            stride=args.stride,
            num_windows=args.num_windows,
            class_weights=class_weights
        )
    else:
        if args.boost_start_stop:
            print("boost START and STOP windows")
            window_boost_classes = [P.START, P.STOP]
        else:
            window_boost_classes = None
        dataset = AnnotatedGenomeDataset(
            args.fna_fn,
            args.tsv_fn,
            window=args.max_seq_length,
            stride=args.stride,
            class_weights=class_weights,
            window_boost_classes = window_boost_classes,
            aux_stream_path=args.aux_stream,
        )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"gene_predictor/gene_predictor_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build event-based metrics and brier functions (passed directly to callbacks)
    calc_metrics, calc_metrics_with_windows = event_based_generic_metrics_factory(event_motifs_by_class)
    calc_brier = event_based_brier_factory(event_motifs_by_class)

    def mk_training_cb(val_loader):
        # Build logits conversion fn for event-head mode if active
        num_event_heads = int(config.get('model', {}).get('num_event_heads') or 0)
        event_logits_conversion_fn = None
        if num_event_heads > 0:
            def _convert(seq_tokens_batch, event_logits_batch):
                if not (isinstance(event_logits_batch, torch.Tensor) and event_logits_batch.dim() == 3 and int(event_logits_batch.size(0)) >= 1):
                    raise RuntimeError('Event-head mode active but event logits are missing for conversion')
                B = int(event_logits_batch.size(0))
                outs = []
                for b in range(B):
                    wl = build_class_logits_from_event_head_logits(
                        seq_window_tokens=seq_tokens_batch[b:b+1, :],
                        event_logits_window=event_logits_batch[b:b+1, :, :],
                        event_motifs_by_class=event_motifs_by_class,
                        head_class_ids=head_class_ids,
                        num_classes=len(P.idx_to_cls),
                    )
                    outs.append(wl)
                return torch.from_numpy(np.stack(outs, axis=0))
            event_logits_conversion_fn = _convert

        # Event-head mode: don't monitor F1; still report metrics; early-stop on val_loss only (patience=8)
        if num_event_heads > 0:
            # Use the exact dict reference created above and passed to the loss
            alpha_by_class = bce_alpha_weight_map

            metrics_cb = MetricsCallback(val_loader, margin_bp=margin_bp, calculate_metrics_fn=calc_metrics, compute_brier_fn=calc_brier, run_dir=output_dir,
                                         event_logits_conversion_fn=event_logits_conversion_fn, event_motifs_by_class=event_motifs_by_class, head_class_ids=head_class_ids,
                                         alpha_by_class=alpha_by_class)

            cbs = [
                metrics_cb,
                pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=8),
            ]
            return cbs

        # Default: include F1 checkpoint and dual-metric early stopping
        f1_ckpt = pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="model_epoch={epoch:02d}_val_f1={val_f1:.3f}",
            monitor='val_f1',
            mode='max',
            save_top_k=3,
            save_last=False,
            auto_insert_metric_name=False,
        )
        # Non event-head mode retains original behavior
        return [
            MetricsCallback(val_loader, margin_bp=margin_bp, calculate_metrics_fn=calc_metrics, compute_brier_fn=calc_brier, run_dir=output_dir,
                            event_motifs_by_class=event_motifs_by_class, head_class_ids=head_class_ids),
            DualMetricEarlyStopping(patience=8),
            f1_ckpt,
        ]

    model, val_loader = run_trainer(
        dataset,
        config,
        output_dir,
        mk_training_cb,
        monitor_metric='val_loss',
        monitor_mode='min',
        custom_loss_fn=custom_loss,
    )

    print(f"results saved to: {output_dir}")

if __name__ == "__main__":
    main()
