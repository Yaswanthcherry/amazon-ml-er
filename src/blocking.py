"""
blocking.py
-----------
Memory-efficient blocking / candidate generation.

Strategy:
1. Fit separate TF-IDF vectorizers for name and address.
2. Use float32 to reduce memory.
3. Keep the existing country restriction.
4. Process S1 queries in batches.
5. Process S2/S3 candidate pools in chunks.
6. Retrieve candidates separately using name and address similarity.
7. Combine scores only for the small candidate sets.

This avoids constructing one huge:
    [name_matrix | address_matrix]
and avoids creating a massive dense similarity matrix.
"""

import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from normalize import add_normalized_columns


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUERY_BATCH_SIZE = getattr(cfg, "TFIDF_QUERY_BATCH_SIZE", 5000)
POOL_BATCH_SIZE = getattr(cfg, "TFIDF_POOL_BATCH_SIZE", 50000)

NAME_WEIGHT = 0.6
ADDRESS_WEIGHT = 0.4


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _combined_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate normalised name and address."""
    return (
        df["norm_name"].fillna("")
        + " "
        + df["norm_address"].fillna("")
    )


def _ensure_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure norm_name and norm_address exist."""
    if "norm_name" not in df.columns or "norm_address" not in df.columns:
        df = df.copy()
        add_normalized_columns(df)

    return df


# ---------------------------------------------------------------------------
# Vectorizer fitting
# ---------------------------------------------------------------------------

def fit_vectorizers(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> tuple:
    """
    Fit separate TF-IDF vectorizers for names and addresses.

    float32 is used to reduce memory consumption.
    """

    logger.info("Fitting TF-IDF vectorisers on combined corpus …")

    for df in (s1, s2, s3):
        if "norm_name" not in df.columns or "norm_address" not in df.columns:
            add_normalized_columns(df)

    all_names = pd.concat(
        [
            s1["norm_name"],
            s2["norm_name"],
            s3["norm_name"],
        ],
        ignore_index=True,
    )

    all_addrs = pd.concat(
        [
            s1["norm_address"],
            s2["norm_address"],
            s3["norm_address"],
        ],
        ignore_index=True,
    )

    name_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.TFIDF_NAME_NGRAM_RANGE,
        max_features=cfg.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )

    addr_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.TFIDF_ADDR_NGRAM_RANGE,
        max_features=cfg.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )

    name_vec.fit(all_names.fillna(""))

    logger.info(
        "  name vocab size: %d",
        len(name_vec.vocabulary_),
    )

    addr_vec.fit(all_addrs.fillna(""))

    logger.info(
        "  addr vocab size: %d",
        len(addr_vec.vocabulary_),
    )

    return name_vec, addr_vec


# ---------------------------------------------------------------------------
# Save / load vectorizers
# ---------------------------------------------------------------------------

def save_vectorizers(
    name_vec: TfidfVectorizer,
    addr_vec: TfidfVectorizer,
) -> None:

    cfg.MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(cfg.TFIDF_NAME_PATH, "wb") as f:
        pickle.dump(name_vec, f)

    with open(cfg.TFIDF_ADDR_PATH, "wb") as f:
        pickle.dump(addr_vec, f)

    logger.info(
        "Saved TF-IDF vectorisers to %s",
        cfg.MODELS_DIR,
    )


def load_vectorizers() -> tuple:

    with open(cfg.TFIDF_NAME_PATH, "rb") as f:
        name_vec = pickle.load(f)

    with open(cfg.TFIDF_ADDR_PATH, "rb") as f:
        addr_vec = pickle.load(f)

    logger.info(
        "Loaded TF-IDF vectorisers from %s",
        cfg.MODELS_DIR,
    )

    return name_vec, addr_vec


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------

def build_tfidf_matrix(
    df: pd.DataFrame,
    name_vec: TfidfVectorizer,
    addr_vec: TfidfVectorizer,
    weight_name: float = NAME_WEIGHT,
    weight_addr: float = ADDRESS_WEIGHT,
) -> csr_matrix:
    """
    Build a combined TF-IDF matrix.

    This function is retained for compatibility with the rest of the
    project, but retrieve_candidates() does NOT use it for full datasets.
    """

    df = _ensure_normalized(df)

    name_mat = name_vec.transform(
        df["norm_name"].fillna("")
    ).astype(np.float32)

    addr_mat = addr_vec.transform(
        df["norm_address"].fillna("")
    ).astype(np.float32)

    from scipy.sparse import hstack

    combined = hstack(
        [
            weight_name * name_mat,
            weight_addr * addr_mat,
        ],
        format="csr",
    )

    combined = sk_normalize(
        combined,
        norm="l2",
        copy=False,
    )

    return combined


# ---------------------------------------------------------------------------
# Country index
# ---------------------------------------------------------------------------

def _country_index(df: pd.DataFrame) -> dict:
    """Map country -> row indices."""

    index = {}

    countries = (
        df["country"]
        .astype(str)
        .str.lower()
        .fillna("unknown")
    )

    for i, country in enumerate(countries):
        index.setdefault(country, []).append(i)

    return index


# ---------------------------------------------------------------------------
# Main candidate retrieval
# ---------------------------------------------------------------------------

def retrieve_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    name_vec: TfidfVectorizer,
    addr_vec: TfidfVectorizer,
    top_k: int = cfg.TFIDF_TOP_K,
    min_score: float = cfg.TFIDF_MIN_SCORE,
) -> dict:
    """
    Memory-efficient candidate retrieval.

    IMPORTANT:
    - No full combined S2/S3 matrix.
    - No full S1 x S2/S3 similarity matrix.
    - Country filtering is preserved.
    - S1 is processed in batches.
    - Candidate pools are processed in chunks.
    """

    logger.info(
        "Starting memory-efficient candidate retrieval "
        "(top_k=%d, min_score=%.3f) …",
        top_k,
        min_score,
    )

    # ---------------------------------------------------------
    # Ensure normalization
    # ---------------------------------------------------------

    for df in (s1, s2, s3):
        _ensure_normalized(df)

    # ---------------------------------------------------------
    # IDs
    # ---------------------------------------------------------

    s1_ids = s1["entity_id"].values
    s2_ids = s2["entity_id"].values
    s3_ids = s3["entity_id"].values

    # ---------------------------------------------------------
    # Countries
    # ---------------------------------------------------------

    s1_countries = (
        s1["country"]
        .astype(str)
        .str.lower()
        .fillna("unknown")
        .values
    )

    s2_countries = (
        s2["country"]
        .astype(str)
        .str.lower()
        .fillna("unknown")
        .values
    )

    s3_countries = (
        s3["country"]
        .astype(str)
        .str.lower()
        .fillna("unknown")
        .values
    )

    # ---------------------------------------------------------
    # Country -> row index maps
    # ---------------------------------------------------------

    s2_country_idx = {}

    for i, country in enumerate(s2_countries):
        s2_country_idx.setdefault(
            country,
            [],
        ).append(i)

    s3_country_idx = {}

    for i, country in enumerate(s3_countries):
        s3_country_idx.setdefault(
            country,
            [],
        ).append(i)

    # ---------------------------------------------------------
    # Candidate dictionary
    # ---------------------------------------------------------

    candidates = {
        eid: set()
        for eid in s1_ids
    }

    unique_countries = np.unique(s1_countries)

    logger.info(
        "Processing %d country groups …",
        len(unique_countries),
    )

    # ---------------------------------------------------------
    # Country loop
    # ---------------------------------------------------------

    for country in unique_countries:

        s1_rows = np.where(
            s1_countries == country
        )[0]

        s2_rows = np.asarray(
            s2_country_idx.get(country, []),
            dtype=np.intp,
        )

        s3_rows = np.asarray(
            s3_country_idx.get(country, []),
            dtype=np.intp,
        )

        if len(s1_rows) == 0:
            continue

        logger.info(
            "country=%s | S1=%d | S2=%d | S3=%d",
            country,
            len(s1_rows),
            len(s2_rows),
            len(s3_rows),
        )

        # -----------------------------------------------------
        # Process S2 and S3 independently
        # -----------------------------------------------------

        if len(s2_rows) > 0:

            _process_candidate_pool(
                query_df=s1,
                query_rows=s1_rows,
                candidate_df=s2,
                candidate_rows=s2_rows,
                candidate_ids=s2_ids,
                name_vec=name_vec,
                addr_vec=addr_vec,
                candidates=candidates,
                top_k=top_k,
                min_score=min_score,
                pool_name="S2",
            )

        if len(s3_rows) > 0:

            _process_candidate_pool(
                query_df=s1,
                query_rows=s1_rows,
                candidate_df=s3,
                candidate_rows=s3_rows,
                candidate_ids=s3_ids,
                name_vec=name_vec,
                addr_vec=addr_vec,
                candidates=candidates,
                top_k=top_k,
                min_score=min_score,
                pool_name="S3",
            )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    total_pairs = sum(
        len(v)
        for v in candidates.values()
    )

    logger.info(
        "Candidate retrieval complete: "
        "%d S1 entities, "
        "%d total candidate pairs "
        "(avg %.1f/entity)",
        len(s1_ids),
        total_pairs,
        total_pairs / max(len(s1_ids), 1),
    )

    return candidates


# ---------------------------------------------------------------------------
# Candidate pool processing
# ---------------------------------------------------------------------------

def _process_candidate_pool(
    query_df: pd.DataFrame,
    query_rows: np.ndarray,
    candidate_df: pd.DataFrame,
    candidate_rows: np.ndarray,
    candidate_ids: np.ndarray,
    name_vec: TfidfVectorizer,
    addr_vec: TfidfVectorizer,
    candidates: dict,
    top_k: int,
    min_score: float,
    pool_name: str,
) -> None:
    """
    Memory-efficient candidate retrieval.

    IMPORTANT:
    Never stores all sparse similarity matches in Python dictionaries.

    For every query/pool chunk:
      1. Calculate name similarity.
      2. Calculate address similarity.
      3. Keep only top candidates.
      4. Immediately merge them into candidates.
      5. Free the similarity matrices.
    """

    logger.info(
        "Processing %s candidate pool: %d records",
        pool_name,
        len(candidate_rows),
    )

    for q_start in range(
        0,
        len(query_rows),
        QUERY_BATCH_SIZE,
    ):

        q_end = min(
            q_start + QUERY_BATCH_SIZE,
            len(query_rows),
        )

        q_rows = query_rows[q_start:q_end]

        query_batch = query_df.iloc[q_rows]

        # -----------------------------------------------------
        # Query matrices
        # -----------------------------------------------------

        q_name = name_vec.transform(
            query_batch["norm_name"].fillna("")
        ).astype(np.float32)

        q_addr = addr_vec.transform(
            query_batch["norm_address"].fillna("")
        ).astype(np.float32)

        q_name = sk_normalize(
            q_name,
            norm="l2",
            copy=False,
        )

        q_addr = sk_normalize(
            q_addr,
            norm="l2",
            copy=False,
        )

        # -----------------------------------------------------
        # Process candidate pool in chunks
        # -----------------------------------------------------

        for p_start in range(
            0,
            len(candidate_rows),
            POOL_BATCH_SIZE,
        ):

            p_end = min(
                p_start + POOL_BATCH_SIZE,
                len(candidate_rows),
            )

            p_rows = candidate_rows[p_start:p_end]

            pool = candidate_df.iloc[p_rows]

            # -------------------------------------------------
            # Candidate matrices
            # -------------------------------------------------

            c_name = name_vec.transform(
                pool["norm_name"].fillna("")
            ).astype(np.float32)

            c_addr = addr_vec.transform(
                pool["norm_address"].fillna("")
            ).astype(np.float32)

            c_name = sk_normalize(
                c_name,
                norm="l2",
                copy=False,
            )

            c_addr = sk_normalize(
                c_addr,
                norm="l2",
                copy=False,
            )

            # -------------------------------------------------
            # Sparse similarity
            # -------------------------------------------------

            name_sim = q_name.dot(c_name.T)
            addr_sim = q_addr.dot(c_addr.T)

            # -------------------------------------------------
            # Process every query independently
            # -------------------------------------------------

            for local_i in range(
                len(q_rows)
            ):

                # ---------------------------------------------
                # Name similarities
                # ---------------------------------------------

                name_row = name_sim.getrow(
                    local_i
                )

                # ---------------------------------------------
                # Address similarities
                # ---------------------------------------------

                addr_row = addr_sim.getrow(
                    local_i
                )

                # ---------------------------------------------
                # Collect only top candidates from each field
                # ---------------------------------------------

                name_candidates = []

                if name_row.nnz > 0:

                    name_indices = name_row.indices
                    name_scores = name_row.data

                    valid = (
                        name_scores >= min_score
                    )

                    if valid.any():

                        valid_indices = name_indices[
                            valid
                        ]

                        valid_scores = name_scores[
                            valid
                        ]

                        if len(valid_scores) > top_k:

                            selected = np.argpartition(
                                valid_scores,
                                -top_k
                            )[-top_k:]

                        else:

                            selected = np.arange(
                                len(valid_scores)
                            )

                        for idx in selected:

                            name_candidates.append(
                                (
                                    int(valid_indices[idx]),
                                    float(valid_scores[idx])
                                )
                            )

                # ---------------------------------------------
                # Address candidates
                # ---------------------------------------------

                addr_candidates = []

                if addr_row.nnz > 0:

                    addr_indices = addr_row.indices
                    addr_scores = addr_row.data

                    valid = (
                        addr_scores >= min_score
                    )

                    if valid.any():

                        valid_indices = addr_indices[
                            valid
                        ]

                        valid_scores = addr_scores[
                            valid
                        ]

                        if len(valid_scores) > top_k:

                            selected = np.argpartition(
                                valid_scores,
                                -top_k
                            )[-top_k:]

                        else:

                            selected = np.arange(
                                len(valid_scores)
                            )

                        for idx in selected:

                            addr_candidates.append(
                                (
                                    int(addr_indices[idx]),
                                    float(addr_scores[idx])
                                )
                            )

                # ---------------------------------------------
                # Combine ONLY the small candidate lists
                # ---------------------------------------------

                combined = {}

                for idx, score in name_candidates:

                    combined[idx] = (
                        combined.get(idx, 0.0)
                        + NAME_WEIGHT * score
                    )

                for idx, score in addr_candidates:

                    combined[idx] = (
                        combined.get(idx, 0.0)
                        + ADDRESS_WEIGHT * score
                    )

                if not combined:
                    continue

                # ---------------------------------------------
                # Final top-K
                # ---------------------------------------------

                ranked = sorted(
                    combined.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:top_k]

                # ---------------------------------------------
                # Store immediately
                # ---------------------------------------------

                global_query_row = q_rows[
                    local_i
                ]

                query_id = query_df.iloc[
                    global_query_row
                ]["entity_id"]

                for pool_idx, score in ranked:

                    if score < min_score:
                        continue

                    candidate_id = candidate_ids[
                        p_rows[pool_idx]
                    ]

                    candidates[
                        query_id
                    ].add(candidate_id)

            # -------------------------------------------------
            # Free chunk memory
            # -------------------------------------------------

            del c_name
            del c_addr
            del name_sim
            del addr_sim

        # -----------------------------------------------------
        # Free query memory
        # -----------------------------------------------------

        del q_name
        del q_addr

        logger.info(
            "  %s query batch %d:%d / %d complete",
            pool_name,
            q_start,
            q_end,
            len(query_rows),
        )

# ---------------------------------------------------------------------------
# Compatibility helper
# ---------------------------------------------------------------------------

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
    """
    Compatibility helper retained for other code.
    """

    n_s1, _ = sims.shape

    for local_i in range(n_s1):

        row = sims.getrow(local_i)

        if row.nnz == 0:
            continue

        valid = np.where(
            row.data >= min_score
        )[0]

        if len(valid) == 0:
            continue

        if len(valid) > top_k:

            selected = valid[
                np.argpartition(
                    row.data[valid],
                    -top_k,
                )[-top_k:]
            ]

        else:
            selected = valid

        global_s1_row = s1_row_indices[
            local_i
        ]

        s1_eid = s1_ids[
            global_s1_row
        ]

        for local_index in selected:

            global_c_row = cand_row_indices[
                row.indices[local_index]
            ]

            candidates[
                s1_eid
            ].add(
                cand_ids[global_c_row]
            )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def candidates_to_dataframe(
    candidates: dict,
) -> pd.DataFrame:

    rows = []

    for s1_id, cand_set in candidates.items():

        rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(
                    sorted(cand_set)
                )
                if cand_set
                else "",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_ids",
        ],
    )


def dataframe_to_candidates(
    df: pd.DataFrame,
) -> dict:

    result = {}

    for _, row in df.iterrows():

        s1_id = row[
            "source1_entity_id"
        ]

        raw = row.get(
            "candidate_entity_ids",
            "",
        )

        if (
            raw
            and isinstance(raw, str)
            and raw.strip()
        ):

            result[s1_id] = set(
                raw.split(",")
            )

        else:

            result[s1_id] = set()

    return result