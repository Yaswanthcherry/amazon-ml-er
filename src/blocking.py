"""
blocking.py
-----------
Blocking (coarse candidate filtering) stage.

Goal: reduce the O(N²) comparison space to a manageable set of candidate pairs
      while keeping recall as high as possible.

Strategy — two complementary passes:
  1. TF-IDF cosine similarity on character n-gram representations of the
     combined "name + address" text, retrieved with a sparse matrix multiply.
     Fast for large datasets; robust to word-order variation.
  2. Country equality filter: candidates must share the same country string
     (after lower-casing).  France entities never match US entities, etc.

The two passes together produce `candidate_pairs`: a mapping
    source1_entity_id  →  set of (source2/3) entity_ids

Fitted vectorisers are saved to disk so the test-set blocking reuses the
same vocabulary without re-fitting on test data.

Public API
----------
fit_vectorizers(s1, s2, s3)           -> (name_vec, addr_vec)
build_tfidf_index(df, name_vec, addr_vec) -> sparse matrix
retrieve_candidates(s1, s2, s3,
                    name_vec, addr_vec,
                    top_k, min_score)  -> dict[str, set[str]]
save_vectorizers(name_vec, addr_vec)  -> None
load_vectorizers()                    -> (name_vec, addr_vec)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from normalize import add_normalized_columns

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Combined text key
# ---------------------------------------------------------------------------

def _combined_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate normalised name and address for TF-IDF vectorisation."""
    return df["norm_name"].fillna("") + " " + df["norm_address"].fillna("")


def _ensure_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Add norm_name / norm_address if not already present."""
    if "norm_name" not in df.columns or "norm_address" not in df.columns:
        df = df.copy()
        add_normalized_columns(df)
    return df


# ---------------------------------------------------------------------------
# Vectoriser fit / save / load
# ---------------------------------------------------------------------------

def fit_vectorizers(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> tuple:
    """Fit two TF-IDF vectorisers (name, address) on the union of all three sources.

    Fitting on all sources ensures the vocabulary covers tokens from S2/S3 that
    don't appear in S1 (and vice-versa), which matters for computing cosine
    similarity in a shared feature space.

    Returns (name_vectorizer, addr_vectorizer).
    """
    logger.info("Fitting TF-IDF vectorisers on combined corpus …")

    for df in (s1, s2, s3):
        if "norm_name" not in df.columns:
            add_normalized_columns(df)

    all_names  = pd.concat([s1["norm_name"],    s2["norm_name"],    s3["norm_name"]])
    all_addrs  = pd.concat([s1["norm_address"], s2["norm_address"], s3["norm_address"]])

    name_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.TFIDF_NAME_NGRAM_RANGE,
        max_features=cfg.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
    )
    addr_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.TFIDF_ADDR_NGRAM_RANGE,
        max_features=cfg.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
    )

    name_vec.fit(all_names.fillna(""))
    logger.info("  name vocab size: %d", len(name_vec.vocabulary_))

    addr_vec.fit(all_addrs.fillna(""))
    logger.info("  addr vocab size: %d", len(addr_vec.vocabulary_))

    return name_vec, addr_vec


def save_vectorizers(name_vec: TfidfVectorizer, addr_vec: TfidfVectorizer) -> None:
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.TFIDF_NAME_PATH, "wb") as f:
        pickle.dump(name_vec, f)
    with open(cfg.TFIDF_ADDR_PATH, "wb") as f:
        pickle.dump(addr_vec, f)
    logger.info("Saved TF-IDF vectorisers to %s", cfg.MODELS_DIR)


def load_vectorizers() -> tuple:
    with open(cfg.TFIDF_NAME_PATH, "rb") as f:
        name_vec = pickle.load(f)
    with open(cfg.TFIDF_ADDR_PATH, "rb") as f:
        addr_vec = pickle.load(f)
    logger.info("Loaded TF-IDF vectorisers from %s", cfg.MODELS_DIR)
    return name_vec, addr_vec


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_tfidf_matrix(
    df: pd.DataFrame,
    name_vec: TfidfVectorizer,
    addr_vec: TfidfVectorizer,
    weight_name: float = 0.6,
    weight_addr: float = 0.4,
) -> csr_matrix:
    """Build a combined (L2-normalised) TF-IDF matrix for a source DataFrame.

    Returns a sparse matrix of shape (n_records, total_features) where the
    name and address sub-matrices are horizontally stacked with the given
    weights applied before normalisation.
    """
    df = _ensure_normalized(df)

    name_mat = name_vec.transform(df["norm_name"].fillna(""))
    addr_mat = addr_vec.transform(df["norm_address"].fillna(""))

    # Weight the two sub-spaces then L2-normalise each row so cosine sim
    # is simply a dot product.
    from scipy.sparse import hstack
    combined = hstack([weight_name * name_mat, weight_addr * addr_mat], format="csr")
    combined = sk_normalize(combined, norm="l2", copy=False)
    return combined


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

def _country_index(df: pd.DataFrame) -> dict:
    """Map country string → list of integer row indices in df."""
    index: dict = {}
    countries = df["country"].str.lower().fillna("unknown")
    for i, c in enumerate(countries):
        index.setdefault(c, []).append(i)
    return index


def retrieve_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    name_vec: TfidfVectorizer,
    addr_vec: TfidfVectorizer,
    top_k: int = cfg.TFIDF_TOP_K,
    min_score: float = cfg.TFIDF_MIN_SCORE,
) -> dict:
    """For every S1 entity, retrieve the top-k most similar S2/S3 candidates.

    Algorithm:
      For each country group:
        1. Extract the subset of S1, S2, S3 that share that country.
        2. Compute sparse cosine similarities: S1_matrix @ (S2_matrix | S3_matrix).T
        3. For each S1 row keep the top-k hits above min_score.

    Returns a dict  { s1_entity_id: set(candidate_entity_ids) }
    """
    logger.info("Starting candidate retrieval  (top_k=%d, min_score=%.3f) …", top_k, min_score)

    # Ensure normalised columns exist
    for df in (s1, s2, s3):
        _ensure_normalized(df)

    # Build matrices for the full S2 and S3 (we slice by row index later)
    logger.info("  Building S2 TF-IDF matrix …")
    mat_s2 = build_tfidf_matrix(s2, name_vec, addr_vec)
    logger.info("  Building S3 TF-IDF matrix …")
    mat_s3 = build_tfidf_matrix(s3, name_vec, addr_vec)
    logger.info("  Building S1 TF-IDF matrix …")
    mat_s1 = build_tfidf_matrix(s1, name_vec, addr_vec)

    s1_ids = s1["entity_id"].values
    s2_ids = s2["entity_id"].values
    s3_ids = s3["entity_id"].values

    # Country-partitioned retrieval
    s1_countries = s1["country"].str.lower().fillna("unknown").values
    s2_countries = s2["country"].str.lower().fillna("unknown").values
    s3_countries = s3["country"].str.lower().fillna("unknown").values

    # Build per-country row-index maps
    s2_country_idx: dict = {}
    for i, c in enumerate(s2_countries):
        s2_country_idx.setdefault(c, []).append(i)

    s3_country_idx: dict = {}
    for i, c in enumerate(s3_countries):
        s3_country_idx.setdefault(c, []).append(i)

    candidates: dict = {eid: set() for eid in s1_ids}

    unique_countries = set(s1_countries)
    logger.info("  Processing %d country groups …", len(unique_countries))

    for country in unique_countries:
        s1_rows = np.where(s1_countries == country)[0]
        s2_rows = np.array(s2_country_idx.get(country, []), dtype=np.intp)
        s3_rows = np.array(s3_country_idx.get(country, []), dtype=np.intp)

        logger.debug(
            "    country=%s  s1=%d  s2=%d  s3=%d",
            country, len(s1_rows), len(s2_rows), len(s3_rows),
        )

        # Process each candidate pool (S2 and S3) separately
        for cand_rows, cand_ids, mat_cand in [
            (s2_rows, s2_ids, mat_s2),
            (s3_rows, s3_ids, mat_s3),
        ]:
            if len(cand_rows) == 0:
                continue

            # Slice the matrices to this country
            q_mat = mat_s1[s1_rows]          # (n_s1_country, feat)
            c_mat = mat_cand[cand_rows]       # (n_cand_country, feat)

            # Cosine similarity via dot product (both matrices are L2-normalised)
            # Result shape: (n_s1_country, n_cand_country)
            sims = q_mat.dot(c_mat.T)

            # For large sub-groups, process in chunks to avoid OOM
            _fill_candidates_from_sims(
                sims=sims,
                s1_row_indices=s1_rows,
                cand_row_indices=cand_rows,
                s1_ids=s1_ids,
                cand_ids=cand_ids,
                candidates=candidates,
                top_k=top_k,
                min_score=min_score,
            )

    total_pairs = sum(len(v) for v in candidates.values())
    logger.info(
        "Candidate retrieval complete: %d S1 entities, %d total candidate pairs (avg %.1f/entity)",
        len(s1_ids), total_pairs, total_pairs / max(len(s1_ids), 1),
    )
    return candidates


def _fill_candidates_from_sims(
    sims,
    s1_row_indices: np.ndarray,
    cand_row_indices: np.ndarray,
    s1_ids: np.ndarray,
    cand_ids: np.ndarray,
    candidates: dict,
    top_k: int,
    min_score: float,
) -> None:
    """Fill the `candidates` dict from a similarity sub-matrix."""
    # Convert to dense only if small enough; otherwise iterate row by row
    n_s1, n_cand = sims.shape

    # Dense path (fast)
    if hasattr(sims, "toarray"):
        sims_dense = sims.toarray()
    else:
        sims_dense = np.asarray(sims)

    for local_i in range(n_s1):
        row = sims_dense[local_i]
        # Get top-k indices above threshold
        above = np.where(row >= min_score)[0]
        if len(above) == 0:
            continue
        if len(above) > top_k:
            # Partial sort to find top-k
            top_local = above[np.argpartition(row[above], -top_k)[-top_k:]]
        else:
            top_local = above

        global_s1_row = s1_row_indices[local_i]
        s1_eid = s1_ids[global_s1_row]

        for local_c in top_local:
            global_c_row = cand_row_indices[local_c]
            candidates[s1_eid].add(cand_ids[global_c_row])


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def candidates_to_dataframe(candidates: dict) -> pd.DataFrame:
    """Convert the candidates dict to a DataFrame suitable for saving as TSV."""
    rows = []
    for s1_id, cand_set in candidates.items():
        rows.append({
            "source1_entity_id":   s1_id,
            "candidate_entity_ids": ",".join(sorted(cand_set)) if cand_set else "",
        })
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])


def dataframe_to_candidates(df: pd.DataFrame) -> dict:
    """Reconstruct the candidates dict from a saved candidate_pairs DataFrame."""
    result = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        raw   = row.get("candidate_entity_ids", "")
        if raw and isinstance(raw, str) and raw.strip():
            result[s1_id] = set(raw.split(","))
        else:
            result[s1_id] = set()
    return result
