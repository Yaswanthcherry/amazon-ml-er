"""
data_loader.py
--------------
Handles all I/O for source TSVs and the ground-truth file.

Public API
----------
load_source1(split)         -> pd.DataFrame
load_source2(split)         -> pd.DataFrame
load_source3(split)         -> pd.DataFrame
load_ground_truth()         -> pd.DataFrame
load_all_sources(split)     -> tuple[df1, df2, df3]
load_candidates(path)       -> pd.DataFrame   (reads an existing candidate_pairs.tsv)
save_tsv(df, path)          -> None
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

logger = logging.getLogger(__name__)

Split = Literal["train", "test"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_tsv(path: Path, usecols: Optional[list] = None) -> pd.DataFrame:
    """Read a UTF-8 tab-separated file robustly."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(
        path,
        sep=cfg.TSV_SEP,
        encoding=cfg.FILE_ENCODING,
        usecols=usecols,
        dtype=str,          # keep everything as string; no silent type coercions
        na_values=[""],     # treat empty cells as NaN
        keep_default_na=False,
    )
    logger.info("Loaded %s  →  %d rows", path.name, len(df))
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN in text columns with an empty string."""
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_source1(split: Split = "train") -> pd.DataFrame:
    """Load Source-1 records for the requested split.

    Returns a DataFrame with columns:
        entity_id | business_name | business_address | country
    """
    path = cfg.TRAIN_SOURCE1 if split == "train" else cfg.TEST_SOURCE1
    df = _read_tsv(path)
    df = _fill_missing(df)
    return df


def load_source2(split: Split = "train") -> pd.DataFrame:
    """Load Source-2 records for the requested split."""
    path = cfg.TRAIN_SOURCE2 if split == "train" else cfg.TEST_SOURCE2
    df = _read_tsv(path)
    df = _fill_missing(df)
    return df


def load_source3(split: Split = "train") -> pd.DataFrame:
    """Load Source-3 records for the requested split."""
    path = cfg.TRAIN_SOURCE3 if split == "train" else cfg.TEST_SOURCE3
    df = _read_tsv(path)
    df = _fill_missing(df)
    return df


def load_ground_truth() -> pd.DataFrame:
    """Load the training ground-truth file.

    Returns a DataFrame with columns:
        source1_entity_id | matched_entity_ids

    matched_entity_ids is a raw comma-separated string (or empty string for
    singletons).  Parsing into lists is left to labels.py.
    """
    df = _read_tsv(cfg.TRAIN_GROUND_TRUTH)
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")
    return df


def load_all_sources(split: Split = "train") -> tuple:
    """Convenience wrapper — loads all three source files in one call.

    Returns (source1_df, source2_df, source3_df).
    """
    s1 = load_source1(split)
    s2 = load_source2(split)
    s3 = load_source3(split)
    return s1, s2, s3


def load_candidates(path: Optional[Path] = None) -> pd.DataFrame:
    """Load an existing candidate_pairs.tsv produced by a previous run.

    Returns a DataFrame with columns:
        source1_entity_id | candidate_entity_ids   (raw comma-separated string)
    """
    path = Path(path) if path else cfg.CANDIDATE_PAIRS_PATH
    df = _read_tsv(path)
    df["candidate_entity_ids"] = df["candidate_entity_ids"].fillna("")
    return df


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def save_tsv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write a DataFrame as a UTF-8 tab-separated file.

    Creates parent directories if they do not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        sep=cfg.TSV_SEP,
        index=index,
        encoding=cfg.FILE_ENCODING,
    )
    logger.info("Saved %d rows  →  %s", len(df), path)


# ---------------------------------------------------------------------------
# Tiny self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s1 = load_source1("train")
    s2 = load_source2("train")
    s3 = load_source3("train")
    gt = load_ground_truth()
    print(s1.head(2))
    print(gt.head(2))
    print("Shapes:", s1.shape, s2.shape, s3.shape, gt.shape)
