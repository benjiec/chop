#!/usr/bin/env python3

from typing import Dict, Tuple, List, Optional

from utils.constants import GenePredictionClass
from utils.metrics import (
    event_based_generic_metrics_factory,
    event_based_brier_factory,
    compute_event_span_mean_probability_beta_fits,
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
    beta_fits = compute_event_span_mean_probability_beta_fits(results, event_motifs_by_class)

    return {
        'generic': generic,
        'events': events,
        'brier_overall': brier.get('brier', 0.0),
        'brier_by_class': brier.get('brier_by_class', {}),
        'beta_fits': beta_fits,
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
    beta_fits = metrics.get('beta_fits', {})

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

    print("\nEvent-only probability Beta fits (decoder span-mean, aggregated):")
    for cls_idx in classes_for_beta:
        cname = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
        fits = beta_fits.get(int(cls_idx)) if isinstance(beta_fits, dict) else None
        if fits:
            tp = fits['tp']
            tn = fits['tn']
            print(
                f"  {cname:>5s} TP: n={int(tp['n'])} mean={tp['mean']:.4f} std={tp['std']:.4f} "
                f"beta(alpha={tp['beta_alpha']:.2f}, beta={tp['beta_beta']:.2f})"
            )
            print(
                f"  {cname:>5s} TN: n={int(tn['n'])} mean={tn['mean']:.4f} std={tn['std']:.4f} "
                f"beta(alpha={tn['beta_alpha']:.2f}, beta={tn['beta_beta']:.2f})"
            )
        else:
            print(f"  {cname:>5s} TP: n=0 mean=0.0000 std=0.0000 beta(alpha=0.00, beta=0.00)")
            print(f"  {cname:>5s} TN: n=0 mean=0.0000 std=0.0000 beta(alpha=0.00, beta=0.00)")

    if generic and beta_fits and brier_by_class:
        import math
        print("\nSummary\ncls,sen/pre,brier,tp_m/tp_s-tn_m/tn_s,ssmd")
        for cls_idx in classes_for_beta:
            cname = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            sen = generic[cls_idx]['sensitivity'] * 100
            pre = generic[cls_idx]['precision'] * 100
            b = brier_by_class[cls_idx]
            tp_m = beta_fits[cls_idx]['tp']['mean'] * 100
            tp_s = beta_fits[cls_idx]['tp']['std'] * 100
            tn_m = beta_fits[cls_idx]['tn']['mean'] * 100
            tn_s = beta_fits[cls_idx]['tn']['std'] * 100
            ssmd = (tp_m - tn_m) / math.sqrt(tp_s * tp_s + tn_s * tn_s) if (tp_s > 0 or tn_s > 0) else 0.0
            print(f"{cname:>5s},{int(sen)}/{int(pre)},{b:.4f},{int(tp_m)}/{int(tp_s)}-{int(tn_m)}/{int(tn_s)},{ssmd:.2f}")


