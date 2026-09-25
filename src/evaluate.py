"""
evaluate.py
-----------
Evaluation utilities using the competition metric: F_0.5 (macro-averaged).

F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)

Computed per Source-1 entity then averaged across all S1 entities
(including singletons — entities with no true matches).

Public API
----------
f05_score(precision, recall)              -> float
entity_f05(predicted_set, true_set)       -> float
macro_f05(predictions, ground_truth)      -> float
evaluate_predictions(predictions, gt_df)  -> dict   (full metrics report)
print_report(metrics)                     -> None
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from labels import parse_ground_truth

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def f05_score(precision: float, recall: float) -> float:
    """Compute F_0.5 from precision and recall.

    F_0.5 = (1 + 0.5²) × P × R / (0.5² × P + R)
           = (1.25 × P × R)     / (0.25 × P + R)
    """
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom


# ---------------------------------------------------------------------------
# Per-entity scoring
# ---------------------------------------------------------------------------

def entity_f05(predicted_set: set, true_set: set) -> float:
    """F_0.5 for a single Source-1 entity.

    Rules (from the problem statement):
    - If true_set is empty AND predicted_set is empty  → 1.0
    - If true_set is empty AND predicted_set non-empty → 0.0
    - Otherwise compute precision/recall on the sets.
    """
    if not true_set and not predicted_set:
        return 1.0
    if not true_set and predicted_set:
        return 0.0
    if not predicted_set:
        # true_set non-empty, predicted empty → recall = 0
        return 0.0

    tp        = len(predicted_set & true_set)
    precision = tp / len(predicted_set)
    recall    = tp / len(true_set)
    return f05_score(precision, recall)


# ---------------------------------------------------------------------------
# Macro-averaged F_0.5
# ---------------------------------------------------------------------------

def macro_f05(
    predictions: dict,
    ground_truth: dict,
) -> float:
    """Compute macro-averaged F_0.5 across all S1 entities.

    Parameters
    ----------
    predictions  : dict  { s1_entity_id: set(predicted_match_ids) }
    ground_truth : dict  { s1_entity_id: set(true_match_ids) }
                   (output of labels.parse_ground_truth)

    Returns
    -------
    float  macro-F_0.5 in [0, 1]
    """
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        scores.append(entity_f05(pred_set, true_set))

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions: dict,
    gt_df: pd.DataFrame,
) -> dict:
    """Compute a full suite of evaluation metrics.

    Parameters
    ----------
    predictions : dict  { s1_entity_id: set(predicted_match_ids) }
    gt_df       : ground-truth DataFrame (source1_entity_id, matched_entity_ids)

    Returns
    -------
    dict with keys:
        macro_f05, macro_precision, macro_recall,
        micro_precision, micro_recall, micro_f05,
        n_entities, n_singletons, n_matched,
        singleton_accuracy,
        per_entity_f05  (list of floats, same order as gt)
    """
    gt_dict = parse_ground_truth(gt_df)

    per_entity: list  = []
    per_prec:   list  = []
    per_rec:    list  = []

    total_tp  = 0
    total_pp  = 0   # total predicted positives
    total_rp  = 0   # total real positives

    n_singleton_correct = 0
    n_singletons        = 0

    for s1_id, true_set in gt_dict.items():
        pred_set = predictions.get(s1_id, set())

        tp = len(pred_set & true_set)
        pp = len(pred_set)
        rp = len(true_set)

        total_tp += tp
        total_pp += pp
        total_rp += rp

        prec = tp / pp if pp > 0 else 0.0
        rec  = tp / rp if rp > 0 else 0.0

        per_prec.append(prec)
        per_rec.append(rec)
        per_entity.append(entity_f05(pred_set, true_set))

        if rp == 0:
            n_singletons += 1
            if pp == 0:
                n_singleton_correct += 1

    macro_f   = float(np.mean(per_entity))
    macro_p   = float(np.mean(per_prec))
    macro_r   = float(np.mean(per_rec))

    micro_p   = total_tp / total_pp if total_pp > 0 else 0.0
    micro_r   = total_tp / total_rp if total_rp > 0 else 0.0
    micro_f   = f05_score(micro_p, micro_r)

    n_entities = len(gt_dict)
    n_matched  = n_entities - n_singletons
    singleton_acc = n_singleton_correct / max(n_singletons, 1)

    metrics = {
        "macro_f05":          round(macro_f,  6),
        "macro_precision":    round(macro_p,  6),
        "macro_recall":       round(macro_r,  6),
        "micro_f05":          round(micro_f,  6),
        "micro_precision":    round(micro_p,  6),
        "micro_recall":       round(micro_r,  6),
        "n_entities":         n_entities,
        "n_singletons":       n_singletons,
        "n_matched":          n_matched,
        "singleton_accuracy": round(singleton_acc, 6),
        "per_entity_f05":     per_entity,
    }

    logger.info(
        "Evaluation  macro_F0.5=%.4f  macro_P=%.4f  macro_R=%.4f  "
        "micro_F0.5=%.4f  singleton_acc=%.4f  entities=%d",
        macro_f, macro_p, macro_r, micro_f, singleton_acc, n_entities,
    )
    return metrics


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_report(metrics: dict) -> None:
    """Print a human-readable evaluation report to stdout."""
    sep = "-" * 50
    print(sep)
    print("EVALUATION REPORT")
    print(sep)
    print(f"  Macro F_0.5        : {metrics['macro_f05']:.4f}  ← leaderboard metric")
    print(f"  Macro Precision    : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall       : {metrics['macro_recall']:.4f}")
    print(f"  Micro F_0.5        : {metrics['micro_f05']:.4f}")
    print(f"  Micro Precision    : {metrics['micro_precision']:.4f}")
    print(f"  Micro Recall       : {metrics['micro_recall']:.4f}")
    print(sep)
    print(f"  Total entities     : {metrics['n_entities']}")
    print(f"  Matched entities   : {metrics['n_matched']}")
    print(f"  Singleton entities : {metrics['n_singletons']}")
    print(f"  Singleton accuracy : {metrics['singleton_accuracy']:.4f}")
    print(sep)

    # Distribution of per-entity F_0.5
    scores = metrics.get("per_entity_f05", [])
    if scores:
        arr = np.array(scores)
        print(f"  Per-entity F_0.5   : mean={arr.mean():.4f}  "
              f"med={np.median(arr):.4f}  "
              f"min={arr.min():.4f}  "
              f"max={arr.max():.4f}")
    print(sep)
