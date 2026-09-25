#!/usr/bin/env python3
"""
run.py
------
Convenience entry point for the Business Entity Resolution pipeline.

Run from the repository root:

    # Full pipeline (train + predict on test set)
    python run.py

    # Training only
    python run.py --mode train

    # Prediction only (requires saved model / vectorisers / threshold)
    python run.py --mode predict

    # Evaluate on validation fold
    python run.py --mode eval

    # Force re-training even if saved artefacts exist
    python run.py --mode full --force-retrain
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Make sure src/ is on the path regardless of working directory
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import config as cfg
from pipeline import run


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _ensure_dirs() -> None:
    """Create output and model directories if they don't exist."""
    for d in (cfg.OUTPUT_DIR, cfg.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Business Entity Resolution — Amazon ML Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # full pipeline (train + predict)
  python run.py --mode train             # training only
  python run.py --mode predict           # inference only
  python run.py --mode eval              # evaluate on validation fold
  python run.py --mode full --force-retrain   # force re-training
        """,
    )

    parser.add_argument(
        "--mode",
        default="full",
        choices=["train", "predict", "full", "eval"],
        help="Pipeline mode  (default: full)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Re-train even if saved model / vectorisers exist",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Data split to predict on  (default: test)",
    )
    parser.add_argument(
        "--log-level",
        default=cfg.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity  (default: INFO)",
    )

    args = parser.parse_args()

    _setup_logging(args.log_level)
    _ensure_dirs()

    logger = logging.getLogger(__name__)
    logger.info("Starting pipeline  mode=%s  split=%s  force_retrain=%s",
                args.mode, args.split, args.force_retrain)

    try:
        run(
            mode=args.mode,
            force_retrain=args.force_retrain,
            split=args.split,
        )
        logger.info("Pipeline finished successfully.")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
