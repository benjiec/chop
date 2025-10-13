#!/usr/bin/env python3

import sys
import os
from pathlib import Path

from dna_learner.trainer import train as run_trainer
import argparse
from datetime import datetime
from typing import Optional, Dict, Callable

import pytorch_lightning as pl

from utils.constants import GenePredictionClass as P
from utils.constants import EventHeadIdx as H
from utils.constants import StandardDonorDinucleotides, DinoDonorDinucleotides
from utils.losses import event_based_ce_loss_factory, event_based_bce_loss_factory, event_head_bce_loss_factory
from utils.events import build_event_motifs
from utils.metrics import event_based_generic_metrics_factory, event_based_brier_factory
from dna_learner.model import GenePredictorModule, create_base_config, set_class_conditional_readout_config, set_class_conditional_readout_config_with_head
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

    # class conditional readout
    parser.add_argument('--enable-cc', action='store_true', help='Enable class-conditional readouts for START/STOP (disabled by default)')
    parser.add_argument('--start-before', type=int, default=300, help='CC upstream window for START')
    parser.add_argument('--start-after', type=int, default=0, help='CC downstream window for START')
    parser.add_argument('--stop-before', type=int, default=0, help='CC upstream window for STOP')
    parser.add_argument('--stop-after', type=int, default=300, help='CC downstream window for STOP')
    parser.add_argument('--cc-gap', type=int, default=0, help='Relative donut gap for CC masks')

    # required DSS motifs choice
    parser.add_argument('--dss-motifs', type=str, required=True, choices=['standard', 'dino'], help='Donor splice site motifs to use for event-based loss: standard or dino')
    # loss selection
    parser.add_argument('--loss-type', type=str, default='event-ce', choices=['event-ce', 'event-bce', 'event-head-bce'],
                        help='Loss type: event-ce (masked cross-entropy), event-bce (masked BCE per event class on main logits), or event-head-bce (masked BCE on separate event heads)')

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

    ce_weight_map = None
    bce_pos_weight_map = None
    bce_neg_weight_map = None
    pos_weights_by_head_idx = None
    neg_weights_by_head_idx = None
    event_motifs_by_class = build_event_motifs(dss_set)

    # Build event motifs map and custom loss
    if args.loss_type == 'event-ce':
        if not args.disable_class_weights_for_loss:
            ce_weight_map = class_weights_map

        print("using event_based_ce_loss")
        custom_loss = event_based_ce_loss_factory(
            event_motifs_by_class,
            class_weights=ce_weight_map,
            loss_window_margin_bp=margin_bp,
        )

    elif args.loss_type == 'event-bce':
        if not args.disable_class_weights_for_loss:
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

        print("using event_based_bce_loss")
        custom_loss = event_based_bce_loss_factory(
            event_motifs_by_class,
            pos_weights=bce_pos_weight_map,
            neg_weights=bce_neg_weight_map,
            loss_window_margin_bp=margin_bp,
        )

    else:
        # Build head-indexed mapping for clarity and explicitness
        event_motifs_by_head_idx = {
            int(H.START): event_motifs_by_class[int(P.START)],
            int(H.STOP): event_motifs_by_class[int(P.STOP)],
            int(H.DSS): event_motifs_by_class[int(P.DSS)],
            int(H.ASS): event_motifs_by_class[int(P.ASS)],
        }
        # Defensive checks
        assert set(event_motifs_by_head_idx.keys()) == {0,1,2,3}, "event_motifs_by_head_idx must have keys 0..3"
        if not args.disable_class_weights_for_loss:
            pos_weights_by_head_idx = {
                int(H.START): float(args.start_weight),
                int(H.STOP): float(args.stop_weight),
                int(H.DSS): float(args.dss_weight),
                int(H.ASS): float(args.ass_weight),
            }
            neg_weights_by_head_idx = {
                int(H.START): float(args.start_neg_weight),
                int(H.STOP): float(args.stop_neg_weight),
                int(H.DSS): float(args.dss_neg_weight),
                int(H.ASS): float(args.ass_neg_weight),
            }
        else:
            pos_weights_by_head_idx = None
            neg_weights_by_head_idx = None

        print("using event_head_bce_loss")
        custom_loss = event_head_bce_loss_factory(
            event_motifs_by_head_idx,
            pos_weights_by_head_idx=pos_weights_by_head_idx,
            neg_weights_by_head_idx=neg_weights_by_head_idx,
            loss_window_margin_bp=margin_bp,
        )

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

    if args.enable_cc:
        # Bind CC readouts to both classifier classes and event head indices for START/STOP
        set_class_conditional_readout_config_with_head(config, int(P.START), int(args.start_before), int(args.start_after), int(args.cc_gap), int(H.START))
        set_class_conditional_readout_config_with_head(config, int(P.STOP), int(args.stop_before), int(args.stop_after), int(args.cc_gap), int(H.STOP))

    config.setdefault('custom', {})
    config['custom'].setdefault('loss', {})
    config['custom']['loss']['bce_pos_weights'] = bce_pos_weight_map
    config['custom']['loss']['bce_neg_weights'] = bce_neg_weight_map
    config['custom']['loss']['ce_class_weights'] = ce_weight_map
    config['custom']['loss']['event_head_bce_pos_weights'] = pos_weights_by_head_idx
    config['custom']['loss']['event_head_bce_neg_weights'] = neg_weights_by_head_idx

    # Persist event-head mappings when enabled, for inference routing
    if args.loss_type == 'event-head-bce':
        config['custom']['event_motifs_by_head_idx'] = {int(k): sorted(list(v)) for k, v in event_motifs_by_head_idx.items()}
        # Use EventHeadIdx constants to map to GenePredictionClass ids
        config['custom']['head_to_class_id'] = {
            int(H.START): int(P.START),
            int(H.STOP): int(P.STOP),
            int(H.DSS): int(P.DSS),
            int(H.ASS): int(P.ASS),
        }

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
        dataset = AnnotatedGenomeDataset(
            args.fna_fn,
            args.tsv_fn,
            window=args.max_seq_length,
            stride=args.stride,
            class_weights=class_weights
        )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"gene_predictor/gene_predictor_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build event-based metrics and brier functions (passed directly to callbacks)
    calc_metrics, calc_metrics_with_windows = event_based_generic_metrics_factory(event_motifs_by_class)
    calc_brier = event_based_brier_factory(event_motifs_by_class)

    def mk_training_cb(val_loader):
        f1_ckpt = pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="model_epoch={epoch:02d}_val_f1={val_f1:.3f}",
            monitor='val_f1',
            mode='max',
            save_top_k=3,
            save_last=False,
            auto_insert_metric_name=False,
        )
        return [ MetricsCallback(val_loader, margin_bp=margin_bp, calculate_metrics_fn=calc_metrics, compute_brier_fn=calc_brier, run_dir=output_dir),
                 DualMetricEarlyStopping(patience=8),
                 f1_ckpt ]

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
