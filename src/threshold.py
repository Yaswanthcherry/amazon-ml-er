"""
threshold.py
------------
Finds and persists the optimal decision threshold for converting model
probability scores into binary match decisions.

Because the competition uses F_0.5 (precision-heavy), a higher threshold
generally improves precision at the cost of recall — which is the right
trade-off here.

Threshold search is done on the validation fold produced by train.py.

Public API
----------
tune_threshold(proba, pair_ids, gt_dict,
               metric, grid)            -> float
save_threshold(threshold)              -> None
load_threshold()                       -> float
"""

import logging
import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from evaluate import macro_f05, entity_f05

logger = logging.getLogger(__name__)

Metric = Literal["f05", "f1", "precision", "recall"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_threshold(threshold: float) -> None:
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.THRESHOLD_PATH, "wb") as f:
        pickle.dump(threshold, f)
    logger.info("Threshold %.4f saved to %s", threshold, cfg.THRESHOLD_PATH)


def load_threshold() -> float:
    if not cfg.THRESHOLD_PATH.exists():
        logger.warning(
            "No saved threshold found at %s — using default %.4f",
            cfg.THRESHOLD_PATH, cfg.DEFAULT_THRESHOLD,
        )
        return cfg.DEFAULT_THRESHOLD
    with open(cfg.THRESHOLD_PATH, "rb") as f:
        threshold = pickle.load(f)
    logger.info("Threshold %.4f loaded from %s", threshold, cfg.THRESHOLD_PATH)
    return threshold


# ---------------------------------------------------------------------------
# Core tuning logic
# ---------------------------------------------------------------------------

def tune_threshold(
    proba: np.ndarray,
    pair_ids: list,
    gt_dict: dict,
    metric: Metric = cfg.THRESHOLD_METRIC,
    grid: Optional[np.ndarray] = None,
) -> float:
    """Search for the threshold that maximises `metric` on the validation set.

    Parameters
    ----------
    proba    : 1-D array of match probabilities (one per candidate pair)
    pair_ids : list of (s1_entity_id, candidate_entity_id) tuples
    gt_dict  : dict  { s1_entity_id: set(true_match_ids) }
    metric   : optimisation target — "f05" | "f1" | "precision" | "recall"
    grid     : optional array of thresholds to search; uses config defaults

    Returns
    -------
    float  best threshold
    """
    if grid is None:
        grid = np.arange(
            cfg.THRESHOLD_GRID_MIN,
            cfg.THRESHOLD_GRID_MAX + 1e-9,
            cfg.THRESHOLD_GRID_STEP,
        )

    # Pre-group pair probabilities by s1_entity_id for fast evaluation
    # s1_to_pairs[s1_id] = list of (cand_id, prob)
    s1_to_pairs: dict = {}
    for (s1_id, c_id), p in zip(pair_ids, proba):
        s1_to_pairs.setdefault(s1_id, []).append((c_id, float(p)))

    best_threshold = cfg.DEFAULT_THRESHOLD
    best_score     = -1.0

    logger.info("Threshold search: metric=%s  grid=[%.2f, %.2f]  steps=%d",
                metric, grid[0], grid[-1], len(grid))

    for threshold in grid:
        predictions = _apply_threshold(s1_to_pairs, threshold, gt_dict)
        score = _evaluate_metric(predictions, gt_dict, metric)

        if score > best_score:
            best_score     = score
            best_threshold = float(threshold)

    logger.info(
        "Best threshold: %.4f  →  %s = %.4f",
        best_threshold, metric, best_score,
    )
    return best_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_threshold(
    s1_to_pairs: dict,
    threshold: float,
    gt_dict: dict,
) -> dict:
    """Convert probabilities to predictions given a threshold.

    Returns dict { s1_id: set(matched_ids) }
    Ensures every S1 entity in gt_dict appears in the output (even if empty).
    """
    predictions: dict = {s1_id: set() for s1_id in gt_dict}
    for s1_id, pairs in s1_to_pairs.items():
        matched = {c_id for (c_id, p) in pairs if p >= threshold}
        predictions[s1_id] = matched
    return predictions


def _evaluate_metric(predictions: dict, gt_dict: dict, metric: Metric) -> float:
    """Evaluate a given metric on predictions vs ground truth."""
    if metric == "f05":
        return macro_f05(predictions, gt_dict)

    # For f1 / precision / recall we compute macro-averages manually
    precisions: list = []
    recalls:    list = []

    for s1_id, true_set in gt_dict.items():
        pred_set = predictions.get(s1_id, set())
        tp = len(pred_set & true_set)
        pp = len(pred_set)
        rp = len(true_set)
        precisions.append(tp / pp if pp > 0 else (1.0 if rp == 0 else 0.0))
        recalls.append(tp / rp if rp > 0 else (1.0 if pp == 0 else 0.0))

    p = float(np.mean(precisions))
    r = float(np.mean(recalls))

    if metric == "precision":
        return p
    if metric == "recall":
        return r
    # f1
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


# ---------------------------------------------------------------------------
# Predict with a given threshold (used by predict.py)
# ---------------------------------------------------------------------------

def apply_threshold_to_scores(
    proba: np.ndarray,
    pair_ids: list,
    threshold: float,
    all_s1_ids: Optional[list] = None,
) -> dict:
    """Convert probability scores to a final predictions dict.

    Parameters
    ----------
    proba       : 1-D probability array
    pair_ids    : list of (s1_id, cand_id) tuples
    threshold   : decision threshold
    all_s1_ids  : if given, every ID in this list will have an entry in the
                  output (even if no candidates scored above threshold)

    Returns
    -------
    dict  { s1_entity_id: set(predicted_match_ids) }
    """
    predictions: dict = {}

    # Seed with empty sets for all known S1 entities
    if all_s1_ids:
        for eid in all_s1_ids:
            predictions[eid] = set()

    for (s1_id, c_id), p in zip(pair_ids, proba):
        if p >= threshold:
            predictions.setdefault(s1_id, set()).add(c_id)
        else:
            predictions.setdefault(s1_id, set())   # ensure entry exists

    return predictions
