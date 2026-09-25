"""
features.py
-----------
Builds the feature matrix for a set of candidate pairs.

Each candidate pair (s1_entity_id, candidate_entity_id) becomes one row
with all pairwise similarity features computed by similarity.py.

Public API
----------
build_feature_matrix(candidates, s1, s2s3)  -> (X: np.ndarray, pair_ids: list[tuple])
build_feature_matrix_df(candidates, s1, s2s3) -> pd.DataFrame  (with pair_id cols)
FEATURE_NAMES                                -> list[str]
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from similarity import all_similarity_scores

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature name registry (order must be stable across train and inference)
# ---------------------------------------------------------------------------

# These are derived from the keys returned by all_similarity_scores().
# We define them explicitly here so features.py is self-documenting and so
# the column order is guaranteed even if similarity.py changes key ordering.

FEATURE_NAMES: list = [
    # --- Name features ---
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort",
    "name_token_set",
    "name_wratio",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_token_overlap",
    "name_lev_sim",
    "name_len_diff_ratio",
    # --- Address features ---
    "addr_ratio",
    "addr_partial_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_common_numbers",
    "addr_len_diff_ratio",
    "addr_one_empty",
    # --- Meta features ---
    "country_match",
]


# ---------------------------------------------------------------------------
# Record lookup helpers
# ---------------------------------------------------------------------------

def _build_lookup(df: pd.DataFrame) -> dict:
    """Build a dict  entity_id → (norm_name, norm_address, country)."""
    lookup: dict = {}
    for row in df.itertuples(index=False):
        lookup[row.entity_id] = (
            getattr(row, "norm_name",    "") or "",
            getattr(row, "norm_address", "") or "",
            getattr(row, "country",      "") or "",
        )
    return lookup


def _merge_lookup(s2: pd.DataFrame, s3: pd.DataFrame) -> dict:
    """Merge S2 and S3 into a single lookup dict."""
    lk = _build_lookup(s2)
    lk.update(_build_lookup(s3))
    return lk


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    candidates: dict,
    s1: pd.DataFrame,
    s2s3: pd.DataFrame,
    show_progress: bool = True,
) -> tuple:
    """Build the feature matrix for all candidate pairs.

    Parameters
    ----------
    candidates : dict  { s1_entity_id: set(candidate_entity_ids) }
    s1         : Source-1 DataFrame (must have norm_name, norm_address, country)
    s2s3       : Concatenation of Source-2 and Source-3 DataFrames
                 (must have norm_name, norm_address, country)
    show_progress : whether to display a tqdm progress bar

    Returns
    -------
    X        : np.ndarray of shape (n_pairs, n_features)  dtype=float32
    pair_ids : list of (s1_entity_id, candidate_entity_id) tuples, length n_pairs
    """
    s1_lk    = _build_lookup(s1)
    cand_lk  = _merge_lookup(
        s2s3[s2s3["entity_id"].str.startswith("S2-")],
        s2s3[s2s3["entity_id"].str.startswith("S3-")],
    ) if isinstance(s2s3, pd.DataFrame) else _build_lookup(s2s3)

    # Flatten candidate pairs
    pair_ids: list = []
    for s1_id, cand_set in candidates.items():
        for c_id in cand_set:
            pair_ids.append((s1_id, c_id))

    n_pairs   = len(pair_ids)
    n_feats   = len(FEATURE_NAMES)
    X         = np.zeros((n_pairs, n_feats), dtype=np.float32)

    feat_idx  = {name: i for i, name in enumerate(FEATURE_NAMES)}

    iterator = tqdm(enumerate(pair_ids), total=n_pairs, desc="Feature extraction",
                    disable=not show_progress)

    missing_s1   = 0
    missing_cand = 0

    for row_i, (s1_id, c_id) in iterator:
        s1_rec = s1_lk.get(s1_id)
        c_rec  = cand_lk.get(c_id)

        if s1_rec is None:
            missing_s1 += 1
            continue
        if c_rec is None:
            missing_cand += 1
            continue

        scores = all_similarity_scores(
            norm_name_a=s1_rec[0],
            norm_name_b=c_rec[0],
            norm_addr_a=s1_rec[1],
            norm_addr_b=c_rec[1],
            country_a=s1_rec[2],
            country_b=c_rec[2],
        )

        for feat_name, val in scores.items():
            idx = feat_idx.get(feat_name)
            if idx is not None:
                X[row_i, idx] = val

    if missing_s1 or missing_cand:
        logger.warning(
            "Feature extraction: %d missing S1 records, %d missing candidate records",
            missing_s1, missing_cand,
        )

    logger.info("Feature matrix built: shape=%s", X.shape)
    return X, pair_ids


def build_feature_matrix_df(
    candidates: dict,
    s1: pd.DataFrame,
    s2s3: pd.DataFrame,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Same as build_feature_matrix but returns a DataFrame with ID columns.

    Columns: source1_entity_id | candidate_entity_id | <feature_names...>
    """
    X, pair_ids = build_feature_matrix(candidates, s1, s2s3, show_progress=show_progress)

    ids_df = pd.DataFrame(pair_ids, columns=["source1_entity_id", "candidate_entity_id"])
    feat_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    return pd.concat([ids_df, feat_df], axis=1)


# ---------------------------------------------------------------------------
# Batch builder (memory-efficient for very large candidate sets)
# ---------------------------------------------------------------------------

def build_feature_matrix_batched(
    candidates: dict,
    s1: pd.DataFrame,
    s2s3: pd.DataFrame,
    batch_size: int = cfg.BATCH_SIZE,
    show_progress: bool = True,
) -> tuple:
    """Build features in batches of S1 entities.

    Useful when the full candidate set doesn't fit in memory at once.
    Returns the same (X, pair_ids) as build_feature_matrix.
    """
    s1_ids = list(candidates.keys())
    n = len(s1_ids)

    X_parts:    list = []
    pair_parts: list = []

    batches = range(0, n, batch_size)
    for start in tqdm(batches, desc="Batches", disable=not show_progress):
        batch_ids  = s1_ids[start: start + batch_size]
        batch_cands = {eid: candidates[eid] for eid in batch_ids}
        X_b, pairs_b = build_feature_matrix(
            batch_cands, s1, s2s3, show_progress=False
        )
        X_parts.append(X_b)
        pair_parts.extend(pairs_b)

    X_full = np.vstack(X_parts) if X_parts else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return X_full, pair_parts
