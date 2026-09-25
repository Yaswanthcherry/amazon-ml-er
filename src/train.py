"""
train.py
--------
Trains the LightGBM pairwise matching model on the candidate pairs produced
by the blocking stage.

Pipeline
--------
1. Load train sources + ground truth
2. Load (or generate) candidate pairs
3. Apply negative sampling  (labels.sample_training_pairs)
4. Build feature matrix     (features.build_feature_matrix_batched)
5. Generate labels          (labels.make_labels)
6. Entity-level train/val split
7. Fit LightGBM with early stopping on the val fold
8. Save model to disk

Public API
----------
train(force_retrain)  -> lgb.LGBMClassifier
load_model()          -> lgb.LGBMClassifier
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from data_loader import load_source1, load_source2, load_source3, load_ground_truth
from normalize import add_normalized_columns
from candidate_generation import load_or_generate
from labels import (
    parse_ground_truth,
    sample_training_pairs,
    train_val_split,
    make_labels,
)
from features import build_feature_matrix_batched, FEATURE_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: LGBMClassifier) -> None:
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", cfg.MODEL_PATH)


def load_model() -> LGBMClassifier:
    if not cfg.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No saved model at {cfg.MODEL_PATH}. Run train() first."
        )
    with open(cfg.MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", cfg.MODEL_PATH)
    return model


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(force_retrain: bool = False) -> LGBMClassifier:
    """End-to-end training.

    If a saved model already exists and force_retrain is False, the saved
    model is returned immediately without re-training.

    Returns the fitted LGBMClassifier.
    """
    if not force_retrain and not cfg.FORCE_RETRAIN_MODEL and cfg.MODEL_PATH.exists():
        logger.info("Saved model found — skipping training. Pass force_retrain=True to override.")
        return load_model()

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("TRAINING START")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("[1/7] Loading training data …")
    s1 = load_source1("train")
    s2 = load_source2("train")
    s3 = load_source3("train")
    gt = load_ground_truth()

    # ------------------------------------------------------------------
    # 2. Normalise text
    # ------------------------------------------------------------------
    logger.info("[2/7] Normalising text columns …")
    for df in (s1, s2, s3):
        add_normalized_columns(df)

    # Merge S2 + S3 for lookup purposes
    s2s3 = pd.concat([s2, s3], ignore_index=True)

    # ------------------------------------------------------------------
    # 3. Candidate pairs
    # ------------------------------------------------------------------
    logger.info("[3/7] Loading / generating candidate pairs …")
    candidates = load_or_generate(s1, s2, s3, split="train")
    logger.info("  Total S1 entities with candidates: %d", len(candidates))

    # ------------------------------------------------------------------
    # 4. Parse ground truth & negative sampling
    # ------------------------------------------------------------------
    logger.info("[4/7] Parsing ground truth and sampling negatives …")
    gt_dict = parse_ground_truth(gt)

    sampled_candidates = sample_training_pairs(
        candidates,
        gt_dict,
        neg_pos_ratio=cfg.NEG_POS_RATIO,
        seed=cfg.RANDOM_SEED,
    )

    # ------------------------------------------------------------------
    # 5. Entity-level train / val split
    # ------------------------------------------------------------------
    logger.info("[5/7] Splitting into train / val (entity-level) …")
    train_cands, val_cands = train_val_split(
        sampled_candidates,
        gt_dict,
        val_fraction=cfg.VAL_SPLIT,
        seed=cfg.RANDOM_SEED,
    )

    # ------------------------------------------------------------------
    # 6. Build feature matrices
    # ------------------------------------------------------------------
    logger.info("[6/7] Building feature matrices …")

    logger.info("  Train set …")
    X_train, train_pairs = build_feature_matrix_batched(
        train_cands, s1, s2s3, batch_size=cfg.BATCH_SIZE
    )
    y_train = make_labels(train_pairs, gt_dict)

    logger.info("  Val set …")
    X_val, val_pairs = build_feature_matrix_batched(
        val_cands, s1, s2s3, batch_size=cfg.BATCH_SIZE
    )
    y_val = make_labels(val_pairs, gt_dict)

    logger.info(
        "  Shapes → X_train=%s  X_val=%s  pos_train=%d  pos_val=%d",
        X_train.shape, X_val.shape, y_train.sum(), y_val.sum(),
    )

    # ------------------------------------------------------------------
    # 7. Fit LightGBM
    # ------------------------------------------------------------------
    logger.info("[7/7] Fitting LightGBM …")

    # Class imbalance weight
    n_pos = max(y_train.sum(), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    scale_pos_weight = n_neg / n_pos
    logger.info("  scale_pos_weight = %.2f", scale_pos_weight)

    params = dict(cfg.LGBM_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    model = LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            _early_stopping_callback(cfg.LGBM_EARLY_STOPPING_ROUNDS),
            _log_callback(),
        ],
    )

    logger.info(
        "Training complete in %.1f s.  Best iteration: %d",
        time.time() - t0, model.best_iteration_,
    )

    # Log feature importances (top 10)
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    logger.info("Top-10 feature importances:")
    for name, imp in importances[:10]:
        logger.info("  %-30s  %d", name, imp)

    save_model(model)
    return model


# ---------------------------------------------------------------------------
# LightGBM callbacks
# ---------------------------------------------------------------------------

def _early_stopping_callback(stopping_rounds: int):
    """LightGBM early-stopping callback compatible with lgbm >= 4.x."""
    try:
        from lightgbm import early_stopping
        return early_stopping(stopping_rounds=stopping_rounds, verbose=False)
    except ImportError:
        # Older API fallback — handled by fit() kwarg
        return None


def _log_callback():
    """LightGBM verbose-logging callback that routes through Python logging."""
    try:
        from lightgbm import log_evaluation
        return log_evaluation(period=50)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train the ER matching model")
    parser.add_argument("--force", action="store_true", help="Force re-training even if model exists")
    args = parser.parse_args()
    train(force_retrain=args.force)
