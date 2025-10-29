#!/usr/bin/env python3

import sys
from pathlib import Path
import torch
import numpy as np
import argparse
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Set, Any

from utils.constants import GenePredictionClass
from utils.constants import StandardDonorDinucleotides, DinoDonorDinucleotides
from utils.events import build_event_motifs
from dna_learner.model import GenePredictorModule as ModelModule
from dna_learner.model import AuxStreamEncoder
from torch.utils.data import DataLoader
from utils.genome import AnnotatedGenomeDataset
from utils.metrics import SequenceResult, convert_tokens_to_sequence
from utils.metrics_report import compute_event_metrics, print_event_metrics_report
from utils.windowing import compute_window_slices, blend_logits


def load_trained_model(model_path: Path, device='cpu'):
    print(f"Loading model from: {model_path}")
    model = ModelModule.load_from_checkpoint(model_path, map_location=device, custom_loss_fn=None, strict=False)
    model.eval()
    model = model.to(device)
    return model


def predict_sequence_outputs(model, max_seq_len, seq_tokens_b: torch.Tensor,
                             stride: Optional[int] = None,
                             device: str = 'cpu',
                             return_attention: bool = False,
                             blending_window_margin_bp: int = 200,
                             aux_stream_full: Optional[torch.Tensor] = None,
                             event_motifs_by_class: Optional[Dict[int, Set[str]]] = None,
                             use_event_logits: bool = False) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    L = int(seq_tokens_b.size(1))
    if stride is None:
        stride = max(max_seq_len // 3, 1) if L > max_seq_len else max_seq_len
    slices = compute_window_slices(L, window=max_seq_len, stride=stride)

    window_logits_np = []
    _layer_attn_b = None

    for (s, e) in slices:
        win_tokens = seq_tokens_b[:, s:e]
        aux_win = None
        if aux_stream_full is not None:
            aux_win = aux_stream_full[:, s:e, :]
        extras: Dict[str, Any] = {}
        want_attn = bool(return_attention and len(slices) == 1)
        logits_b = model(
            win_tokens,
            extras=extras,
            return_attention=('attention' if want_attn else None),
            return_event_logits=('event_logits' if use_event_logits else None),
            aux_stream=aux_win,
        )
        if want_attn and 'attention' in extras:
            _layer_attn_b = extras['attention']
        wl = logits_b[0].detach().cpu().numpy()
        window_logits_np.append(wl)

    eff_margin = 0 if len(slices) == 1 else int(blending_window_margin_bp)
    if eff_margin > 0:
        eff_margin = max(0, min(eff_margin, max(0, stride // 2 - 1)))
    blended_logits_np = blend_logits(L, slices, window_logits_np, weight_mode='cosine', margin=eff_margin, exclude_edges=True)
    return blended_logits_np, _layer_attn_b


def run_predictions(model, data_loader, device='cpu', blending_window_margin_bp: int = 200,
                    disable_aux: bool = False) -> List[SequenceResult]:

    predicted_count = 0
    log_every = 5

    results: List[SequenceResult] = []
    max_len = int(model.model.embedding.max_seq_length)
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                sequences, targets, aux_stream = batch
            else:
                sequences, targets = batch
                aux_stream = None
            sequences = sequences.to(device)
            targets = targets.to(device)
            if aux_stream is not None:
                aux_stream = aux_stream.to(device)
            B = sequences.size(0)
            assert B == 1, "Expect batch_size==1 for evaluation"
            seq_tokens_b = sequences[0:1]
            targets_b = targets[0].detach().cpu()
            aux_full = None if (disable_aux or aux_stream is None) else aux_stream[0:1]

            logits_np, _ = predict_sequence_outputs(
                model, max_len, seq_tokens_b,
                device=device,
                blending_window_margin_bp=blending_window_margin_bp,
                aux_stream_full=aux_full,
                use_event_logits=False,
            )
            logits_trim_t = torch.from_numpy(logits_np)
            sr_list = SequenceResult.from_batch(
                sequence_tokens_batch=seq_tokens_b,
                targets_batch=targets_b.unsqueeze(0),
                logits_batch=logits_trim_t.unsqueeze(0),
                sequence_index_start=0,
                prob_activation='softmax',
                sequence_ids=None,
            )
            results.append(sr_list[0])

            predicted_count += 1
            if (predicted_count + 1) % log_every == 0:
                print(f"  Processed {predicted_count + 1} sequences...")

    return results


def summarize_parameters(model) -> Dict[str, Any]:
    rep: Dict[str, Any] = {}
    m = getattr(model, 'model', None)
    blocks = getattr(m, 'cross_attn_blocks', None)
    if not blocks:
        rep['has_cross_attn'] = False
        return rep
    rep['has_cross_attn'] = True
    gates = []
    diag_softmax = []
    centers = []
    for blk in blocks:
        with torch.no_grad():
            g = torch.sigmoid(blk.gate).item()
            gates.append(g)
            bias = blk.relpos_bias.detach().cpu().numpy()  # [H, 2K+1]
            K = int(blk.relpos_max_distance)
            centers.append(float(bias[:, K].mean()))
            # Softmax over distances to estimate diagonal prior mass
            b = bias
            b = b - b.max(axis=1, keepdims=True)
            e = np.exp(b)
            sm = e / np.sum(e, axis=1, keepdims=True)
            diag_softmax.append(float(sm[:, K].mean()))
    rep['gates'] = gates
    rep['diag_bias_center_mean'] = float(np.mean(centers))
    rep['diag_bias_softmax_center_mean'] = float(np.mean(diag_softmax))
    return rep


def main():
    p = argparse.ArgumentParser(description='Predict and inspect aux fusion contributions')
    p.add_argument('--fna-fn', required=True)
    p.add_argument('--tsv-fn', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--aux-stream', type=str, default=None)
    p.add_argument('--aux-normalize', action='store_true', default=True)
    p.add_argument('--aux-channels', type=str, default=None, help='Comma-separated channel indices to include (zero-based)')
    p.add_argument('--channel-ablation', action='store_true', help='Run per-channel ablation and rank channels by Δ metrics')
    # removed: gradient-based attribution option
    p.add_argument('--num-contigs', type=int, default=0)
    p.add_argument('--dss-motifs', type=str, default='standard', choices=['standard', 'dino'])
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    raw_model_path = Path(args.model_path)
    ckpt_path = raw_model_path if raw_model_path.is_absolute() else (run_dir / 'checkpoints' / raw_model_path)

    model = load_trained_model(ckpt_path, args.device)

    # Dataset
    # Parse aux channel selection
    aux_channels = None
    if args.aux_channels:
        aux_channels = [int(x) for x in str(args.aux_channels).split(',') if str(x).strip() != '']

    dataset = AnnotatedGenomeDataset(
        args.fna_fn,
        args.tsv_fn,
        window=None,
        num_contigs=args.num_contigs,
        random_prefix_ns=False,
        aux_stream_path=args.aux_stream,
        aux_normalize=bool(args.aux_normalize),
        aux_channels=aux_channels,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Backward-compat: if checkpoint had aux_encoder weights but model didn't eagerly build, construct and load them
    m = getattr(model, 'model', None)
    needs_aux = (args.aux_stream is not None) and (m is not None) and (getattr(m, 'aux_encoder', None) is None)
    if needs_aux:
        # Determine channel count C from dataset
        aux_c = None
        for arr in getattr(dataset, 'aux_by_contig', []) or []:
            if arr is not None and hasattr(arr, 'shape') and len(arr.shape) == 2:
                aux_c = int(arr.shape[1])
                break
        if aux_c is not None and aux_c > 0:
            d_model = int(getattr(m.embedding, 'd_model'))
            dropout = float(getattr(model, 'config', {}).get('model', {}).get('dropout', 0.1)) if isinstance(getattr(model, 'config', {}), dict) else 0.1
            m.aux_encoder = AuxStreamEncoder(in_channels=aux_c, d_model=d_model, dropout=dropout)
            m.aux_encoder = m.aux_encoder.to(args.device)
            # Load only aux sub-keys from checkpoint if present
            ckpt = torch.load(ckpt_path, map_location=args.device)
            sd = ckpt.get('state_dict', {}) if isinstance(ckpt, dict) else {}
            sub = {k[len('model.aux_encoder.'):]: v for k, v in sd.items() if k.startswith('model.aux_encoder.')}
            if sub:
                print("Augmenting AuxStreamEncoder,", aux_c, "channels, with loaded state", list(sub.keys()))
                m.aux_encoder.load_state_dict(sub, strict=False)
            else:
                print("Created AuxStreamEncoder with", aux_c, "channels, but cannot load state")
        else:
            print("Not creating AuxStreamEncoder eagerly - AuxStreamEncoder will be created later with random starting weights")

    # Motifs map
    cfg = getattr(model, 'config', {})
    ccfg = cfg.get('custom', {}) if isinstance(cfg, dict) else {}
    event_motifs_by_class = ccfg.get('event_motifs_by_class')
    if event_motifs_by_class is None:
        dss_set = DinoDonorDinucleotides if args.dss_motifs == 'dino' else StandardDonorDinucleotides
        event_motifs_by_class = build_event_motifs(dss_set)

    # Parameter summary
    rep = summarize_parameters(model)
    print("=== Cross-Attention Parameters ===")
    if rep.get('has_cross_attn'):
        print(f"gates(sigmoid): {rep['gates']}")
        print(f"diag_bias_center_mean: {rep['diag_bias_center_mean']:.3f}")
        print(f"diag_bias_softmax_center_mean: {rep['diag_bias_softmax_center_mean']:.3f}")
    else:
        print("No cross-attention blocks present.")

    # With-aux
    print("\n=== Inference WITH aux ===")
    res_with = run_predictions(model, loader, device=args.device, blending_window_margin_bp=200, disable_aux=False)
    met_with = compute_event_metrics(res_with, event_motifs_by_class)
    brier_with = met_with.get('brier_overall', None)
    print("")
    print_event_metrics_report(met_with, summary_only=True)

    # Without-aux
    print("\n=== Inference WITHOUT aux ===")
    res_no = run_predictions(model, loader, device=args.device, blending_window_margin_bp=200, disable_aux=True)
    met_no = compute_event_metrics(res_no, event_motifs_by_class)
    print("")
    print_event_metrics_report(met_no, summary_only=True)

    # Contribution proxy at output: mean L1 prob diff
    def _mean_prob_diff(a: List[SequenceResult], b: List[SequenceResult]) -> float:
        diffs = []
        for ra, rb in zip(a, b):
            pa = ra.probabilities
            pb = rb.probabilities
            if pa is None or pb is None:
                continue
            d = np.mean(np.abs(pa - pb))
            diffs.append(d)
        return float(np.mean(diffs)) if diffs else 0.0

    print(f"\nmean |Δprob| (with vs without aux): {_mean_prob_diff(res_with, res_no):.6f}")

    # Projection weight norms per channel (heuristic)
    m = getattr(model, 'model', None)
    enc = getattr(m, 'aux_encoder', None)
    if enc is not None and hasattr(enc, 'proj'):
        W = enc.proj.weight.detach().cpu().numpy()  # [D, C]
        per_ch = np.linalg.norm(W, axis=0)
        print("\n=== Aux Encoder proj weight L2 norms per channel ===")
        for c, v in enumerate(per_ch):
            print(f"channel[{c}]: {v:.6f}")

    # Channel ablation: zero one channel at a time and measure Δ metrics
    if args.channel_ablation and args.aux_stream:
        print("\n=== Channel Ablation (Δ brier and mean |Δprob|) ===")
        # Build a single pass cache of logits WITH aux for baseline
        # Reuse res_with as baseline
        # Now rerun with per-channel ablation by wrapping dataset loader
        # Load raw aux to inspect channel count
        ds = loader.dataset
        # Peek one sample's aux to get C
        tmp_loader = DataLoader(ds, batch_size=1, shuffle=False)
        first = next(iter(tmp_loader))
        aux0 = first[2]
        if isinstance(aux0, torch.Tensor):
            C = int(aux0.size(-1))
            ranks = []
            for c in range(C):
                print("channel", c)
                def _ablate_channel(batch):
                    seq, tgt, aux = batch
                    if aux is not None:
                        aux = aux.clone()
                        aux[:, :, c] = 0.0
                    return (seq, tgt, aux)
                # Build an ablated DataLoader view
                ablated_results: List[SequenceResult] = []
                with torch.no_grad():
                    for batch in DataLoader(ds, batch_size=1, shuffle=False):
                        seq, tgt, aux = batch if len(batch) == 3 else (batch[0], batch[1], None)
                        if aux is not None:
                            aux[:, :, c] = 0.0
                        seq = seq.to(args.device); tgt = tgt.to(args.device)
                        if aux is not None:
                            aux = aux.to(args.device)
                        logits_np, _ = predict_sequence_outputs(
                            model, int(m.embedding.max_seq_length), seq[0:1], device=args.device, aux_stream_full=(aux[0:1] if aux is not None else None))
                        logits_trim_t = torch.from_numpy(logits_np)
                        sr_list = SequenceResult.from_batch(
                            sequence_tokens_batch=seq[0:1], targets_batch=tgt[0:1], logits_batch=logits_trim_t.unsqueeze(0), sequence_index_start=0, prob_activation='softmax')
                        ablated_results.append(sr_list[0])
                met_ablate = compute_event_metrics(ablated_results, event_motifs_by_class)
                brier_ablate = met_ablate.get('brier_overall', None)
                mean_prob_delta = _mean_prob_diff(res_with, ablated_results)
                delta_brier = (float(brier_ablate) - float(brier_with)) if (brier_ablate is not None and brier_with is not None) else float('nan')
                ranks.append((c, delta_brier, mean_prob_delta))
            print("channel\tdelta_brier\tmean_abs_prob_delta")
            for c, db, mp in sorted(ranks, key=lambda t: (-np.nan_to_num(t[1], nan=-1e9), -t[2])):
                print(f"{c}\t{db:.6f}\t{mp:.6f}")

    # removed: gradient-based attribution block


if __name__ == '__main__':
    main()


