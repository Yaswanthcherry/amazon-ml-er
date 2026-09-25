"""
pipeline.py
-----------
Top-level orchestrator that wires together all stages of the
Business Entity Resolution pipeline.

Modes
-----
"train"   — Blocking → feature extraction → model training → threshold tuning
"predict" — Inference on the test set and write output files
"full"    — Training then prediction end-to-end
"eval"    — Evaluate on a held-out validation slice of the training data

Usage (programmatic)
--------------------
from src.pipeline import run
run(mode="full")

Usage (CLI)
-----------
python src/pipeline.py --mode full
python src/pipeline.py --mode predict
python src/pipeline.py --mode eval
"""

import argparse
import logging
import time

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — Training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(force_retrain: bool = False) -> None:
    """Blocking → feature extraction → model training → threshold tuning."""
    from data_loader import load_source1, load_source2, load_source3, load_ground_truth
    from normalize import add_normalized_columns
    from candidate_generation import generate_candidates, blocking_stats
    from labels import parse_ground_truth
    from train import train
    from predict import tune_on_validation

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load data
    logger.info("[1/3] Loading training data …")
    s1 = load_source1("train")
    s2 = load_source2("train")
    s3 = load_source3("train")
    gt = load_ground_truth()

    # Normalise
    for df in (s1, s2, s3):
        add_normalized_columns(df)

    # Blocking
    logger.info("[2/3] Blocking — generating candidate pairs …")
    candidates = generate_candidates(s1, s2, s3, split="train", force=force_retrain)

    gt_dict = parse_ground_truth(gt)
    stats = blocking_stats(candidates, gt)
    logger.info(
        "  Recall ceiling: %.4f  |  Avg candidates/entity: %.1f  |  Total pairs: %d",
        stats["recall_ceiling"], stats["avg_candidates"], stats["total_pairs"],
    )

    # Train model
    logger.info("[3/3] Training LightGBM model …")
    train(force_retrain=force_retrain)

    # Threshold tuning
    logger.info("[+] Tuning decision threshold on validation fold …")
    best_thr = tune_on_validation()
    logger.info("  Best threshold: %.4f", best_thr)

    logger.info("Training pipeline complete in %.1f s", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage 2 — Prediction pipeline
# ---------------------------------------------------------------------------

def run_prediction_pipeline(split: str = "test") -> dict:
    """Load saved artefacts → generate candidates → predict → write outputs."""
    from data_loader import load_source1, load_source2, load_source3
    from normalize import add_normalized_columns
    from candidate_generation import generate_candidates
    from predict import predict
    from postprocess import save_outputs

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PREDICTION PIPELINE  split=%s", split)
    logger.info("=" * 60)

    # Load data
    logger.info("[1/3] Loading %s data …", split)
    s1 = load_source1(split)
    s2 = load_source2(split)
    s3 = load_source3(split)

    for df in (s1, s2, s3):
        add_normalized_columns(df)

    # Candidate generation for test set (reuses trained vectorisers)
    logger.info("[2/3] Generating candidate pairs …")
    candidates = generate_candidates(s1, s2, s3, split=split)

    # Score + threshold
    logger.info("[3/3] Running inference …")
    predictions = predict(split=split, candidates=candidates)

    # Write outputs/matching_results.tsv and outputs/candidate_pairs.tsv
    save_outputs(predictions, candidates, s1, s2, s3)

    n_matched = sum(1 for v in predictions.values() if v)
    logger.info(
        "Prediction pipeline complete in %.1f s  |  Matched: %d  |  Singletons: %d",
        time.time() - t0, n_matched, len(predictions) - n_matched,
    )
    return predictions


# ---------------------------------------------------------------------------
# Stage 3 — Evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation_pipeline() -> dict:
    """Evaluate the model on a validation slice of the training data.

    Prints the full F_0.5 metrics report to stdout and returns the metrics dict.
    """
    from data_loader import (
        load_source1, load_source2, load_source3, load_ground_truth,
    )
    from normalize import add_normalized_columns
    from candidate_generation import load_or_generate
    from labels import parse_ground_truth, train_val_split
    from predict import score_candidates
    from threshold import load_threshold, apply_threshold_to_scores
    from evaluate import evaluate_predictions, print_report

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE")
    logger.info("=" * 60)

    # Load
    s1 = load_source1("train")
    s2 = load_source2("train")
    s3 = load_source3("train")
    gt = load_ground_truth()

    for df in (s1, s2, s3):
        add_normalized_columns(df)

    s2s3    = pd.concat([s2, s3], ignore_index=True)
    gt_dict = parse_ground_truth(gt)

    # Use same validation fold as training
    candidates = load_or_generate(s1, s2, s3, split="train")
    _, val_cands = train_val_split(
        candidates, gt_dict,
        val_fraction=cfg.VAL_SPLIT,
        seed=cfg.RANDOM_SEED,
    )

    # Score
    proba, pair_ids = score_candidates(val_cands, s1, s2s3)
    threshold = load_threshold()
    all_val_ids = list(val_cands.keys())
    predictions = apply_threshold_to_scores(proba, pair_ids, threshold, all_s1_ids=all_val_ids)

    # Build a ground-truth DataFrame restricted to the val entities
    import pandas as _pd
    val_id_set = set(val_cands.keys())
    gt_val = gt[gt["source1_entity_id"].isin(val_id_set)].copy()

    metrics = evaluate_predictions(predictions, gt_val)
    print_report(metrics)

    logger.info("Evaluation pipeline complete in %.1f s", time.time() - t0)
    return metrics


# ---------------------------------------------------------------------------
# Full pipeline (train + predict)
# ---------------------------------------------------------------------------

def run(
    mode: str = "full",
    force_retrain: bool = False,
    split: str = "test",
) -> None:
    """Main entry point.

    Parameters
    ----------
    mode          : "train" | "predict" | "full" | "eval"
    force_retrain : if True, re-run training even if saved artefacts exist
    split         : split to run prediction on ("train" or "test")
    """
    mode = mode.lower()

    if mode == "train":
        run_training_pipeline(force_retrain=force_retrain)

    elif mode == "predict":
        run_prediction_pipeline(split=split)

    elif mode == "full":
        run_training_pipeline(force_retrain=force_retrain)
        run_prediction_pipeline(split=split)

    elif mode == "eval":
        run_evaluation_pipeline()

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose from: train | predict | full | eval"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Business Entity Resolution pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  train    Run blocking + model training + threshold tuning
  predict  Run inference on the test set and write output TSVs
  full     Run training then prediction (default)
  eval     Evaluate on the validation fold and print F_0.5 metrics
        """,
    )
    parser.add_argument(
        "--mode", default="full",
        choices=["train", "predict", "full", "eval"],
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Force re-training even if saved model/vectorisers exist",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "test"],
        help="Data split to run prediction on (default: test)",
    )

    args = parser.parse_args()
    run(mode=args.mode, force_retrain=args.force_retrain, split=args.split)
