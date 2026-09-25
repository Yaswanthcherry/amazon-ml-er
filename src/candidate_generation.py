"""
candidate_generation.py
-----------------------
Orchestrates the full candidate-generation stage and writes
`outputs/candidate_pairs.tsv`.

This module is the glue between blocking.py (the retrieval engine) and the
rest of the pipeline.  It handles:
  - Fitting or loading TF-IDF vectorisers
  - Running blocking per split (train / test)
  - Capping per-entity candidate counts (MAX_CANDIDATES_PER_ENTITY)
  - Saving / loading the candidate TSV

Public API
----------
generate_candidates(s1, s2, s3, split, force) -> dict[str, set[str]]
load_or_generate(s1, s2, s3, split)           -> dict[str, set[str]]
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from normalize import add_normalized_columns
from blocking import (
    fit_vectorizers,
    save_vectorizers,
    load_vectorizers,
    retrieve_candidates,
    candidates_to_dataframe,
    dataframe_to_candidates,
)
from data_loader import save_tsv, load_candidates

logger = logging.getLogger(__name__)

Split = Literal["train", "test"]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    split: Split = "train",
    force: bool = False,
) -> dict:
    """Run the full candidate-generation pipeline for the given split.

    Steps:
      1. Normalise text columns on s1, s2, s3 (adds norm_name / norm_address).
      2. Fit TF-IDF vectorisers (train split) or load saved ones (test split).
      3. Retrieve candidates via TF-IDF cosine similarity (country-partitioned).
      4. Cap per-entity candidate counts.
      5. Save outputs/candidate_pairs.tsv.

    Parameters
    ----------
    s1, s2, s3 : DataFrames from data_loader.load_source*
    split       : "train" or "test"
    force       : if True, re-generate even if a cached file exists

    Returns
    -------
    dict  { source1_entity_id : set(candidate_entity_ids) }
    """
    out_path = cfg.CANDIDATE_PAIRS_PATH

    # --- Check cache ---------------------------------------------------------
    if not force and not cfg.FORCE_REGEN_CANDIDATES and out_path.exists() and split == "test":
        logger.info("Cached candidate pairs found at %s — loading.", out_path)
        df_cands = load_candidates(out_path)
        return dataframe_to_candidates(df_cands)

    # --- Normalise -----------------------------------------------------------
    logger.info("Normalising text columns …")
    for df in (s1, s2, s3):
        if "norm_name" not in df.columns:
            add_normalized_columns(df)

    # --- Vectorisers ---------------------------------------------------------
    tfidf_exists = cfg.TFIDF_NAME_PATH.exists() and cfg.TFIDF_ADDR_PATH.exists()

    if split == "train" or cfg.FORCE_RETRAIN_TFIDF or not tfidf_exists:
        logger.info("Fitting TF-IDF vectorisers on %s data …", split)
        name_vec, addr_vec = fit_vectorizers(s1, s2, s3)
        save_vectorizers(name_vec, addr_vec)
    else:
        logger.info("Loading saved TF-IDF vectorisers …")
        name_vec, addr_vec = load_vectorizers()

    # --- Blocking / retrieval ------------------------------------------------
    logger.info("Running TF-IDF blocking …")
    candidates = retrieve_candidates(
        s1, s2, s3,
        name_vec=name_vec,
        addr_vec=addr_vec,
        top_k=cfg.TFIDF_TOP_K,
        min_score=cfg.TFIDF_MIN_SCORE,
    )

    # --- Cap per-entity candidate count -------------------------------------
    candidates = _cap_candidates(candidates, max_k=cfg.MAX_CANDIDATES_PER_ENTITY)

    # --- Save ----------------------------------------------------------------
    df_out = candidates_to_dataframe(candidates)
    save_tsv(df_out, out_path)
    logger.info("Candidate pairs saved to %s", out_path)

    return candidates


def load_or_generate(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    split: Split = "train",
) -> dict:
    """Load candidate pairs from disk if available, otherwise generate them.

    Convenience wrapper for pipeline.py.
    """
    out_path = cfg.CANDIDATE_PAIRS_PATH
    if out_path.exists() and not cfg.FORCE_REGEN_CANDIDATES:
        logger.info("Loading existing candidate pairs from %s …", out_path)
        df_cands = load_candidates(out_path)
        # Verify that the loaded candidates cover the current S1 entities
        loaded_ids = set(df_cands["source1_entity_id"].values)
        s1_ids     = set(s1["entity_id"].values)
        missing    = s1_ids - loaded_ids
        if missing:
            logger.warning(
                "%d S1 entities missing from cached candidates — regenerating.",
                len(missing),
            )
            return generate_candidates(s1, s2, s3, split=split, force=True)
        return dataframe_to_candidates(df_cands)

    return generate_candidates(s1, s2, s3, split=split)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cap_candidates(candidates: dict, max_k: int) -> dict:
    """Keep at most max_k candidates per S1 entity.

    When there are more candidates than max_k we keep the first max_k returned
    by the blocking stage (which are already sorted by decreasing TF-IDF score
    inside retrieve_candidates for each source separately; after merging S2+S3
    the set is unordered, so this is a best-effort cap).
    """
    capped: dict = {}
    over_cap = 0
    for eid, cand_set in candidates.items():
        if len(cand_set) > max_k:
            over_cap += 1
            capped[eid] = set(list(cand_set)[:max_k])
        else:
            capped[eid] = cand_set

    if over_cap:
        logger.debug("Capped %d entities that had >%d candidates.", over_cap, max_k)
    return capped


# ---------------------------------------------------------------------------
# Blocking statistics (useful for EDA / debugging)
# ---------------------------------------------------------------------------

def blocking_stats(candidates: dict, ground_truth: pd.DataFrame) -> dict:
    """Compute recall ceiling and reduction ratio for a candidate set.

    Parameters
    ----------
    candidates    : output of generate_candidates / load_or_generate
    ground_truth  : DataFrame with columns [source1_entity_id, matched_entity_ids]

    Returns
    -------
    dict with keys:
        recall_ceiling   – fraction of true matches that appear in candidates
        reduction_ratio  – fraction of comparisons avoided vs brute-force
        avg_candidates   – average candidates per S1 entity
        total_pairs      – total candidate pairs
    """
    total_true_matches = 0
    matched_in_cands   = 0
    total_cand_pairs   = sum(len(v) for v in candidates.values())

    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        true_matches_raw = row.get("matched_entity_ids", "")
        if not true_matches_raw or not isinstance(true_matches_raw, str):
            continue
        true_set = set(true_matches_raw.split(","))
        cand_set = candidates.get(s1_id, set())
        total_true_matches += len(true_set)
        matched_in_cands   += len(true_set & cand_set)

    recall_ceiling = (
        matched_in_cands / total_true_matches if total_true_matches > 0 else 0.0
    )
    n_s1     = len(candidates)
    n_s2_s3  = sum(
        len(v) for v in candidates.values()
    )  # approximate; actual pool size unknown here

    stats = {
        "recall_ceiling":  round(recall_ceiling, 4),
        "avg_candidates":  round(total_cand_pairs / max(n_s1, 1), 2),
        "total_pairs":     total_cand_pairs,
        "true_matches_covered": matched_in_cands,
        "total_true_matches":   total_true_matches,
    }
    logger.info("Blocking stats: %s", stats)
    return stats
