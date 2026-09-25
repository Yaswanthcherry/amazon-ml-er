"""
postprocess.py
--------------
Post-processing of raw model predictions before writing the final output files.

Steps applied
-------------
1. Deduplicate  — remove duplicate IDs within any single matched_entity_ids list.
2. Validate sources — ensure no S1- IDs appear in match lists; only S2-/S3-.
3. Validate existence — remove IDs that don't exist in the test source files.
4. Format to TSV — produce the exact two-column format required by the rules.
5. Write outputs/matching_results.tsv and outputs/candidate_pairs.tsv.

Public API
----------
postprocess(predictions, candidates,
            valid_s2s3_ids, s1_ids)     -> (results_df, candidates_df)
save_outputs(predictions, candidates,
             valid_s2s3_ids, s1_ids)    -> None
format_matching_results(predictions,
                        s1_ids)         -> pd.DataFrame
format_candidate_pairs(candidates,
                       s1_ids)         -> pd.DataFrame
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from data_loader import save_tsv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _clean_id_set(
    id_set: set,
    valid_ids: Optional[set] = None,
    allow_prefixes: tuple = ("S2-", "S3-"),
) -> set:
    """Remove invalid IDs from a set.

    - Drop IDs whose prefix is not in allow_prefixes (catches accidental S1- self-matches).
    - If valid_ids is given, drop IDs not present in that set.
    """
    cleaned = {eid for eid in id_set if any(eid.startswith(p) for p in allow_prefixes)}
    if valid_ids is not None:
        cleaned = cleaned & valid_ids
    return cleaned


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_matching_results(
    predictions: dict,
    s1_ids: list,
    valid_s2s3_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Convert predictions dict to the required matching_results DataFrame.

    Columns: source1_entity_id | matched_entity_ids

    Rules enforced:
    - Every S1 entity in s1_ids has exactly one row.
    - matched_entity_ids is a comma-separated string (no quoting, no spaces).
    - Empty for singletons.
    - No duplicate IDs within a list.
    - Only S2-/S3- IDs; no S1- self-matches; IDs must exist in the test set.
    """
    rows = []
    n_empty = 0

    for s1_id in s1_ids:
        raw_set = predictions.get(s1_id, set())
        cleaned = _clean_id_set(raw_set, valid_ids=valid_s2s3_ids)

        if cleaned:
            matched_str = ",".join(sorted(cleaned))
        else:
            matched_str = ""
            n_empty += 1

        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_str})

    logger.info(
        "Formatted matching results: %d entities  (%d singletons)",
        len(rows), n_empty,
    )
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])


def format_candidate_pairs(
    candidates: dict,
    s1_ids: list,
    valid_s2s3_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Convert candidates dict to the required candidate_pairs DataFrame.

    Columns: source1_entity_id | candidate_entity_ids

    Same rules as matching_results but for the broader candidate set.
    """
    rows = []
    for s1_id in s1_ids:
        raw_set = candidates.get(s1_id, set())
        cleaned = _clean_id_set(raw_set, valid_ids=valid_s2s3_ids)

        cand_str = ",".join(sorted(cleaned)) if cleaned else ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_str})

    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

def check_consistency(predictions: dict, candidates: dict) -> list:
    """Warn about any matched ID that never appeared as a candidate.

    A matched ID that was not in the candidate set indicates a pipeline bug.

    Returns a list of (s1_id, matched_id) offending pairs.
    """
    issues = []
    for s1_id, matched_set in predictions.items():
        cand_set = candidates.get(s1_id, set())
        for mid in matched_set:
            if mid not in cand_set:
                issues.append((s1_id, mid))

    if issues:
        logger.warning(
            "%d matched IDs were NOT in the candidate set — possible pipeline bug.",
            len(issues),
        )
        for s1_id, mid in issues[:10]:
            logger.warning("  %s  →  %s", s1_id, mid)
    else:
        logger.info("Consistency check passed: all matched IDs appeared in candidates.")
    return issues


# ---------------------------------------------------------------------------
# Main postprocess entry point
# ---------------------------------------------------------------------------

def postprocess(
    predictions: dict,
    candidates: dict,
    valid_s2s3_ids: Optional[set] = None,
    s1_ids: Optional[list] = None,
) -> tuple:
    """Apply all post-processing steps and return formatted DataFrames.

    Parameters
    ----------
    predictions    : dict  { s1_id: set(predicted_match_ids) }
    candidates     : dict  { s1_id: set(candidate_ids) }
    valid_s2s3_ids : set of all valid S2-/S3- IDs in the test set
    s1_ids         : ordered list of all S1 entity IDs in the test set

    Returns
    -------
    (matching_results_df, candidate_pairs_df)
    """
    check_consistency(predictions, candidates)

    results_df   = format_matching_results(predictions, s1_ids, valid_s2s3_ids)
    cand_pairs_df = format_candidate_pairs(candidates, s1_ids, valid_s2s3_ids)

    return results_df, cand_pairs_df


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    predictions: dict,
    candidates: dict,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> None:
    """Post-process and write both output TSV files.

    Parameters
    ----------
    predictions : raw predictions dict from predict.predict()
    candidates  : raw candidates dict from candidate_generation
    s1          : Source-1 DataFrame (for ordered s1_ids list)
    s2, s3      : Source-2 and Source-3 DataFrames (for valid ID set)
    """
    s1_ids         = s1["entity_id"].tolist()
    valid_s2s3_ids = set(s2["entity_id"].tolist()) | set(s3["entity_id"].tolist())

    results_df, cand_pairs_df = postprocess(
        predictions, candidates,
        valid_s2s3_ids=valid_s2s3_ids,
        s1_ids=s1_ids,
    )

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_tsv(results_df,    cfg.MATCHING_RESULTS_PATH)
    save_tsv(cand_pairs_df, cfg.CANDIDATE_PAIRS_PATH)

    logger.info("Output files written:")
    logger.info("  %s", cfg.MATCHING_RESULTS_PATH)
    logger.info("  %s", cfg.CANDIDATE_PAIRS_PATH)

    # Quick summary
    n_matched   = results_df["matched_entity_ids"].str.len().gt(0).sum()
    n_singleton = len(results_df) - n_matched
    logger.info(
        "  %d entities matched, %d singletons (%.1f%% match rate)",
        n_matched, n_singleton,
        100 * n_matched / max(len(results_df), 1),
    )
