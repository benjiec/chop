#!/usr/bin/env python3

from typing import Dict, Tuple, List, Optional

from utils.constants import GenePredictionClass
from utils.metrics import (
    event_based_generic_metrics_factory,
    event_based_brier_factory,
    compute_event_span_mean_probability_metrics,
    SequenceResult,
)


def compute_event_metrics(
    results: List[SequenceResult],
    event_motifs_by_class: Dict[int, set],
    min_weight: float = 1.0,
) -> Dict[str, object]:
    calc_metrics, calc_metrics_with_windows = event_based_generic_metrics_factory(event_motifs_by_class)
    brier_fn = event_based_brier_factory(event_motifs_by_class)

    generic, events = calc_metrics_with_windows(results, min_weight=min_weight)
    brier = brier_fn(results, event_only=True)
    prob_metrics = compute_event_span_mean_probability_metrics(results, event_motifs_by_class)

    return {
        'generic': generic,
        'events': events,
        'brier_overall': brier.get('brier', 0.0),
        'brier_by_class': brier.get('brier_by_class', {}),
        'prob_metrics': prob_metrics,
    }


def print_event_metrics_report(
    metrics: Dict[str, object],
    classes_for_beta: Tuple[int, int, int, int] = (
        GenePredictionClass.START,
        GenePredictionClass.STOP,
        GenePredictionClass.DSS,
        GenePredictionClass.ASS,
    ),
) -> None:
    brier_overall = metrics.get('brier_overall', 0.0)
    brier_by_class = metrics.get('brier_by_class', {})
    generic = metrics.get('generic', {})
    prob_metrics = metrics.get('prob_metrics', {})

    print(f"Brier (overall): {float(brier_overall):.4f}")
    if brier_by_class:
        print("\nBrier by class:")
        for cls_idx in sorted(brier_by_class.keys()):
            name = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            print(f"  {name:>10s}: {float(brier_by_class[cls_idx]):.4f}")

    if generic:
        print("\nPer-class metrics:")
        for cls_idx in sorted(generic.keys()):
            name = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(cls_idx))
            m = generic[cls_idx]
            print(
                f"  {name:>10s}  TP={m['tp']} FP={m['fp']} FN={m['fn']}  "
                f"Sensitivity={m['sensitivity']:.1%} Precision={m['precision']:.1%} Specificity={m['specificity']:.1%}"
            )

    print("\nEvent-only probability distribution metrics (decoder span-mean, aggregated):")
    for cls_idx in classes_for_beta:
        cname = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
        fits = prob_metrics.get(int(cls_idx)) if isinstance(prob_metrics, dict) else None
        if fits:
            tp = fits['tp']
            tn = fits['tn']
            tail = fits.get('tail', {})
            print(
                f"  {cname:>5s} TP: n={int(tp['n'])} mean={tp['mean']:.4f} std={tp['std']:.4f} "
                f"beta(alpha={tp['beta_alpha']:.2f}, beta={tp['beta_beta']:.2f}) median={tp.get('median',0.0):.4f} iqr={tp.get('iqr',0.0):.4f}"
            )
            print(
                f"  {cname:>5s} TN: n={int(tn['n'])} mean={tn['mean']:.4f} std={tn['std']:.4f} "
                f"beta(alpha={tn['beta_alpha']:.2f}, beta={tn['beta_beta']:.2f}) median={tn.get('median',0.0):.4f} iqr={tn.get('iqr',0.0):.4f}"
            )
            if tail and int(tail.get('n', 0)) > 0:
                n = int(tail.get('n', 0))
                tp_tail = tail.get('tp', {})
                tn_tail = tail.get('tn', {})
                # Recompute robust on reported med/iqr for display consistency
                tp_med = float(tp_tail.get('median', 0.0))
                tp_iqr = float(tp_tail.get('iqr', 0.0))
                tn_med = float(tn_tail.get('median', 0.0))
                tn_iqr = float(tn_tail.get('iqr', 0.0))
                den = (tp_iqr + tn_iqr) / 2.0 if (tp_iqr > 0.0 or tn_iqr > 0.0) else 0.0
                tres = (tp_med - tn_med) / den if den > 0.0 else 0.0
                tauc = float(tail.get('auc', 0.0))
                print(
                    f"    tails (equal count): n={n} tp_tail(med/iqr)={tp_med:.4f}/{tp_iqr:.4f} "
                    f"tn_tail(med/iqr)={tn_med:.4f}/{tn_iqr:.4f} robust={tres:.2f} auc={tauc:.3f}"
                )
        else:
            print(f"  {cname:>5s} TP: n=0 mean=0.0000 std=0.0000 beta(alpha=0.00, beta=0.00)")
            print(f"  {cname:>5s} TN: n=0 mean=0.0000 std=0.0000 beta(alpha=0.00, beta=0.00)")

    if generic and prob_metrics and brier_by_class:
        import math
        print("\nSummary\ncls,sen/pre,brier,tp_tail_med/iqr-tn_tail_med/iqr,tail_robust,tn_m+2*tn_s")
        for cls_idx in classes_for_beta:
            cname = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            sen = generic[cls_idx]['sensitivity'] * 100
            pre = generic[cls_idx]['precision'] * 100
            b = brier_by_class[cls_idx]
            tp_m = prob_metrics[cls_idx]['tp']['mean']*100
            tp_s = prob_metrics[cls_idx]['tp']['std']*100
            tn_m = prob_metrics[cls_idx]['tn']['mean']*100
            tn_s = prob_metrics[cls_idx]['tn']['std']*100
            # Estimate min-prob to (tn_m/100)+2*(tn_s/100)
            min_prob=round(tn_m+2*tn_s,2)
            # Equal Tail Robust Effect Size using medians and IQR: (median_tp - median_tn) / ((iqr_tp + iqr_tn)/2)
            ttp_med = prob_metrics[cls_idx]['tail']['tp']['median'] * 100
            ttp_iqr = prob_metrics[cls_idx]['tail']['tp']['iqr'] * 100
            ttn_med = prob_metrics[cls_idx]['tail']['tn']['median'] * 100
            ttn_iqr = prob_metrics[cls_idx]['tail']['tn']['iqr'] * 100
            robust_den = (ttp_iqr + ttn_iqr) / 2.0
            robust = (ttp_med - ttn_med) / robust_den if robust_den > 0 else 0.0
            print(f"{cname:>5s},{b:.4f},{int(tp_m)}/{int(tp_s)}-{int(tn_m)}/{int(tn_s)},{int(ttp_med)}/{int(ttp_iqr)}-{int(ttn_med)}/{int(ttn_iqr)},{robust:.2f},{min_prob:.2f}")
