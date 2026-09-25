"""
predict.py
----------
Runs inference on the test set (or any split) to produce final match
predictions.

Pipeline
--------
1. Load test sources
2. Load saved TF-IDF vectorisers + LightGBM model + threshold
3. Generate / load candidate pairs
4. Build feature matrix
5. Score candidates with the model
6. Apply threshold → predictions dict
7. Return predictions (saving is handled by postprocess.py / pipeline.py)

Public API
----------
predict(split, candidates, return_proba) -> dict[str, set[str]]
score_candidates(candidates, s1, s2s3)   -> (proba, pair_ids)
"""

import logging
import time
from typing import Literal, Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from data_loader import load_source1, load_source2, load_source3
from normalize import add_normalized_columns
from candidate_generation import load_or_generate
from features import build_feature_matrix_batched
from train import load_model
from threshold import load_threshold, apply_threshold_to_scores

logger = logging.getLogger(__name__)

Split = Literal["train", "test"]


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_candidates(
    candidates: dict,
    s1: pd.DataFrame,
    s2s3: pd.DataFrame,
) -> tuple:
    """Build feature matrix and return model probability scores.

    Parameters
    ----------
    candidates : dict  { s1_entity_id: set(candidate_entity_ids) }
    s1         : Source-1 DataFrame (must have norm_name, norm_address, country)
    s2s3       : Concatenated Source-2 + Source-3 DataFrames

    Returns
    -------
    proba    : np.ndarray of shape (n_pairs,)  — match probabilities
    pair_ids : list of (s1_entity_id, candidate_entity_id) tuples
    """
    model = load_model()

    logger.info("Building feature matrix for %d S1 entities …", len(candidates))
    X, pair_ids = build_feature_matrix_batched(
        candidates, s1, s2s3, batch_size=cfg.BATCH_SIZE
    )

    if len(X) == 0:
        logger.warning("Feature matrix is empty — no candidate pairs to score.")
        return np.array([]), []

    logger.info("Scoring %d candidate pairs …", len(pair_ids))
    proba = model.predict_proba(X)[:, 1]   # probability of class 1 (match)
    return proba, pair_ids


# ---------------------------------------------------------------------------
# Main prediction routine
# ---------------------------------------------------------------------------

def predict(
    split: Split = "test",
    candidates: Optional[dict] = None,
    return_proba: bool = False,
) -> dict:
    """Full inference pipeline for a given split.

    Parameters
    ----------
    split        : "train" or "test"
    candidates   : pre-computed candidate dict (optional; generated if None)
    return_proba : if True, also returns raw probability scores as a second
                   element — useful for threshold tuning on the val fold

    Returns
    -------
    predictions  : dict  { s1_entity_id: set(predicted_match_ids) }
    (optionally)  (proba, pair_ids) tuple appended
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("INFERENCE  split=%s", split)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("[1/4] Loading %s data …", split)
    s1 = load_source1(split)
    s2 = load_source2(split)
    s3 = load_source3(split)

    # ------------------------------------------------------------------
    # 2. Normalise
    # ------------------------------------------------------------------
    logger.info("[2/4] Normalising text columns …")
    for df in (s1, s2, s3):
        add_normalized_columns(df)

    s2s3 = pd.concat([s2, s3], ignore_index=True)

    # ------------------------------------------------------------------
    # 3. Candidate pairs
    # ------------------------------------------------------------------
    if candidates is None:
        logger.info("[3/4] Loading / generating candidate pairs …")
        candidates = load_or_generate(s1, s2, s3, split=split)
    else:
        logger.info("[3/4] Using pre-computed candidate pairs (%d entities).", len(candidates))

    # ------------------------------------------------------------------
    # 4. Score & threshold
    # ------------------------------------------------------------------
    logger.info("[4/4] Scoring candidates …")
    proba, pair_ids = score_candidates(candidates, s1, s2s3)

    threshold = load_threshold()
    logger.info("Applying threshold %.4f …", threshold)

    all_s1_ids = s1["entity_id"].tolist()
    predictions = apply_threshold_to_scores(
        proba, pair_ids, threshold, all_s1_ids=all_s1_ids
    )

    # Sanity check: every S1 entity must appear in output
    missing = set(all_s1_ids) - set(predictions.keys())
    if missing:
        logger.warning("%d S1 entities missing from predictions — adding empty rows.", len(missing))
        for eid in missing:
            predictions[eid] = set()

    n_matched  = sum(1 for v in predictions.values() if v)
    n_entities = len(predictions)
    logger.info(
        "Prediction complete in %.1f s.  Entities=%d  Matched=%d  Singletons=%d",
        time.time() - t0, n_entities, n_matched, n_entities - n_matched,
    )

    if return_proba:
        return predictions, (proba, pair_ids)
    return predictions


# ---------------------------------------------------------------------------
# Threshold tuning on training validation fold
# ---------------------------------------------------------------------------

def tune_on_validation() -> float:
    """Run the full pipeline on the training split validation fold and tune threshold.

    Returns the best threshold (also saves it via threshold.save_threshold).
    """
    from data_loader import load_ground_truth
    from labels import parse_ground_truth, train_val_split, sample_training_pairs
    from candidate_generation import load_or_generate
    from threshold import tune_threshold, save_threshold

    logger.info("Tuning threshold on validation fold …")

    s1 = load_source1("train")
    s2 = load_source2("train")
    s3 = load_source3("train")
    gt = load_ground_truth()

    for df in (s1, s2, s3):
        add_normalized_columns(df)

    s2s3    = pd.concat([s2, s3], ignore_index=True)
    gt_dict = parse_ground_truth(gt)

    candidates = load_or_generate(s1, s2, s3, split="train")

    # Use the same seed as training so val fold is identical
    _, val_cands = train_val_split(
        candidates, gt_dict,
        val_fraction=cfg.VAL_SPLIT,
        seed=cfg.RANDOM_SEED,
    )

    proba, pair_ids = score_candidates(val_cands, s1, s2s3)

    best_thr = tune_threshold(proba, pair_ids, gt_dict)
    save_threshold(best_thr)
    return best_thr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--split",  default="test", choices=["train", "test"])
    parser.add_argument("--tune-threshold", action="store_true",
                        help="Tune threshold on val fold before predicting")
    args = parser.parse_args()

    if args.tune_threshold:
        tune_on_validation()

    preds = predict(split=args.split)
    logger.info("Done.  %d predictions generated.", len(preds))
