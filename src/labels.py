"""
labels.py
---------
Constructs binary labels for training by joining the ground-truth file against
the candidate pairs produced by the blocking stage.

A pair (s1_id, cand_id) receives:
  label = 1  if cand_id appears in ground-truth matched_entity_ids for s1_id
  label = 0  otherwise

Also handles negative sampling: for training we don't need all negatives;
we sample a fixed ratio of negatives per positive to keep class balance.

Public API
----------
parse_ground_truth(gt_df)                      -> dict[str, set[str]]
make_labels(pair_ids, gt_dict)                 -> np.ndarray  (0/1)
sample_training_pairs(candidates, gt_dict,
                      neg_pos_ratio, seed)     -> dict[str, set[str]]
"""

import logging
import random
from typing import Optional

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth parsing
# ---------------------------------------------------------------------------

def parse_ground_truth(gt_df: pd.DataFrame) -> dict:
    """Parse the ground-truth DataFrame into a dict.

    Parameters
    ----------
    gt_df : DataFrame with columns [source1_entity_id, matched_entity_ids]
            matched_entity_ids is a comma-separated string or empty/NaN.

    Returns
    -------
    dict  { s1_entity_id : set(matched_entity_ids) }
    """
    gt_dict: dict = {}
    for row in gt_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        raw   = row.matched_entity_ids
        if raw and isinstance(raw, str) and raw.strip():
            gt_dict[s1_id] = set(raw.split(","))
        else:
            gt_dict[s1_id] = set()
    logger.info(
        "Parsed ground truth: %d S1 entities, %d with at least one match",
        len(gt_dict),
        sum(1 for v in gt_dict.values() if v),
    )
    return gt_dict


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def make_labels(pair_ids: list, gt_dict: dict) -> np.ndarray:
    """Generate binary labels for a list of (s1_id, cand_id) pairs.

    Parameters
    ----------
    pair_ids : list of (s1_entity_id, candidate_entity_id) tuples
    gt_dict  : output of parse_ground_truth

    Returns
    -------
    np.ndarray of int8, shape (n_pairs,), values 0 or 1
    """
    labels = np.zeros(len(pair_ids), dtype=np.int8)
    for i, (s1_id, c_id) in enumerate(pair_ids):
        true_matches = gt_dict.get(s1_id, set())
        if c_id in true_matches:
            labels[i] = 1

    pos = labels.sum()
    neg = len(labels) - pos
    logger.info(
        "Labels: %d positives, %d negatives  (ratio %.2f)",
        pos, neg, neg / max(pos, 1),
    )
    return labels


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_training_pairs(
    candidates: dict,
    gt_dict: dict,
    neg_pos_ratio: int = cfg.NEG_POS_RATIO,
    seed: int = cfg.RANDOM_SEED,
) -> dict:
    """Sub-sample negatives from the candidate set for training.

    For each S1 entity:
      - All positive candidates (true matches that appear in the candidate set) are kept.
      - Up to neg_pos_ratio × n_positives negatives are randomly sampled.

    Entities with no positives in the candidate set contribute a small fixed
    number of negatives (to teach the model what a singleton looks like).

    Parameters
    ----------
    candidates    : dict  { s1_id: set(candidate_ids) }
    gt_dict       : dict  { s1_id: set(true_match_ids) }
    neg_pos_ratio : how many negatives to keep per positive
    seed          : random seed for reproducibility

    Returns
    -------
    dict  { s1_id: set(selected_candidate_ids) }
    """
    rng = random.Random(seed)

    sampled: dict = {}
    n_pos_total  = 0
    n_neg_total  = 0
    n_no_pos     = 0

    # Negatives to keep for entities whose true matches are NOT in candidates
    SINGLETON_NEG_SAMPLE = 2

    for s1_id, cand_set in candidates.items():
        true_set = gt_dict.get(s1_id, set())
        positives = cand_set & true_set
        negatives = list(cand_set - true_set)

        n_pos = len(positives)
        n_pos_total += n_pos

        if n_pos == 0:
            # S1 entity is a true singleton OR its matches didn't make it into candidates
            n_neg = min(SINGLETON_NEG_SAMPLE, len(negatives))
            n_no_pos += 1
        else:
            n_neg = min(neg_pos_ratio * n_pos, len(negatives))

        sampled_negs = set(rng.sample(negatives, n_neg)) if n_neg > 0 else set()
        n_neg_total += len(sampled_negs)

        sampled[s1_id] = positives | sampled_negs

    logger.info(
        "After negative sampling: %d positives, %d negatives, "
        "%d entities with no positive in candidates",
        n_pos_total, n_neg_total, n_no_pos,
    )
    return sampled


# ---------------------------------------------------------------------------
# Train / validation split (entity-level stratified)
# ---------------------------------------------------------------------------

def train_val_split(
    candidates: dict,
    gt_dict: dict,
    val_fraction: float = cfg.VAL_SPLIT,
    seed: int = cfg.RANDOM_SEED,
) -> tuple:
    """Split candidate pairs into train and validation sets at the entity level.

    Splitting at the entity level (rather than the pair level) ensures no
    S1 entity appears in both train and validation, preventing data leakage.

    Returns
    -------
    (train_candidates, val_candidates) — two dicts of the same structure
    """
    rng = random.Random(seed)
    all_ids = list(candidates.keys())
    rng.shuffle(all_ids)

    n_val   = max(1, int(len(all_ids) * val_fraction))
    val_ids = set(all_ids[:n_val])

    train_cands = {k: v for k, v in candidates.items() if k not in val_ids}
    val_cands   = {k: v for k, v in candidates.items() if k in val_ids}

    logger.info(
        "Entity-level split: %d train entities, %d val entities",
        len(train_cands), len(val_cands),
    )
    return train_cands, val_cands
