"""
similarity.py
-------------
Pairwise string similarity functions used by features.py.

All scorers return a float in [0, 1].

Uses rapidfuzz for fast Cython-backed string metrics and scikit-learn
for TF-IDF cosine similarity (when USE_TFIDF_COSINE_FEATURE is True).

Public API
----------
name_similarity_scores(a, b)    -> dict[str, float]
address_similarity_scores(a, b) -> dict[str, float]
cosine_sim_from_vecs(v1, v2)    -> float
token_jaccard(a, b)             -> float
common_number_ratio(a, b)       -> float
"""

import re
from typing import Optional

import numpy as np

# rapidfuzz scorers
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz, process
from rapidfuzz.fuzz import (
    ratio,
    partial_ratio,
    token_sort_ratio,
    token_set_ratio,
    WRatio,
)
from rapidfuzz.distance.Jaro import similarity as jaro_sim
from rapidfuzz.distance.JaroWinkler import similarity as jaro_winkler_sim

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_score(fn, a: str, b: str, scale: float = 100.0) -> float:
    """Call a rapidfuzz scorer safely, returning 0.0 on empty inputs."""
    if not a or not b:
        return 0.0
    try:
        return fn(a, b) / scale
    except Exception:
        return 0.0


def _tokenize(text: str) -> set:
    """Split on whitespace and return non-empty tokens."""
    if not text:
        return set()
    return set(text.split())


# ---------------------------------------------------------------------------
# Token Jaccard (no external library needed)
# ---------------------------------------------------------------------------

def token_jaccard(a: Optional[str], b: Optional[str]) -> float:
    """Jaccard similarity on whitespace-tokenized sets."""
    if not a or not b:
        return 0.0
    set_a = _tokenize(a)
    set_b = _tokenize(b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def token_overlap_count(a: Optional[str], b: Optional[str]) -> int:
    """Number of tokens in common (useful as raw count feature)."""
    if not a or not b:
        return 0
    return len(_tokenize(a) & _tokenize(b))


# ---------------------------------------------------------------------------
# Numeric token matching
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\b\d+\b")


def common_number_ratio(a: Optional[str], b: Optional[str]) -> float:
    """Fraction of numeric tokens shared between a and b.

    Rationale: building numbers and zip codes are highly discriminative for
    address matching — even when street names vary.
    """
    if not a or not b:
        return 0.0
    nums_a = set(_NUMBER_RE.findall(a))
    nums_b = set(_NUMBER_RE.findall(b))
    if not nums_a and not nums_b:
        return 1.0   # neither has numbers → no numeric disagreement
    if not nums_a or not nums_b:
        return 0.0
    inter = len(nums_a & nums_b)
    union = len(nums_a | nums_b)
    return inter / union


# ---------------------------------------------------------------------------
# Business name similarity
# ---------------------------------------------------------------------------

def name_similarity_scores(a: Optional[str], b: Optional[str]) -> dict:
    """Compute all configured name similarity features for a pair (a, b).

    Both a and b should already be normalised (output of normalize.normalize_name).

    Returns a flat dict  { feature_name: float_score }.
    """
    a = a or ""
    b = b or ""

    scores: dict = {}

    # rapidfuzz fuzzy scores (all return 0–100, we divide by 100)
    scores["name_ratio"]           = _safe_score(ratio,            a, b)
    scores["name_partial_ratio"]   = _safe_score(partial_ratio,    a, b)
    scores["name_token_sort"]      = _safe_score(token_sort_ratio, a, b)
    scores["name_token_set"]       = _safe_score(token_set_ratio,  a, b)
    scores["name_wratio"]          = _safe_score(WRatio,           a, b)

    # Jaro-Winkler (returns 0–1 already)
    scores["name_jaro_winkler"] = jaro_winkler_sim(a, b) if a and b else 0.0

    # Token-level features
    scores["name_token_jaccard"]   = token_jaccard(a, b)
    scores["name_token_overlap"]   = float(token_overlap_count(a, b))

    # Normalised Levenshtein (0 = identical, 1 = completely different → invert)
    if a and b:
        max_len = max(len(a), len(b))
        lev_dist = Levenshtein.distance(a, b)
        scores["name_lev_sim"] = 1.0 - (lev_dist / max_len) if max_len > 0 else 1.0
    else:
        scores["name_lev_sim"] = 0.0

    # Length difference ratio
    la, lb = len(a), len(b)
    scores["name_len_diff_ratio"] = (
        abs(la - lb) / max(la, lb) if max(la, lb) > 0 else 0.0
    )

    return scores


# ---------------------------------------------------------------------------
# Address similarity
# ---------------------------------------------------------------------------

def address_similarity_scores(a: Optional[str], b: Optional[str]) -> dict:
    """Compute all configured address similarity features for a pair (a, b).

    Both a and b should already be normalised.

    Returns a flat dict  { feature_name: float_score }.
    """
    a = a or ""
    b = b or ""

    scores: dict = {}

    scores["addr_ratio"]         = _safe_score(ratio,            a, b)
    scores["addr_partial_ratio"] = _safe_score(partial_ratio,    a, b)
    scores["addr_token_sort"]    = _safe_score(token_sort_ratio, a, b)
    scores["addr_token_set"]     = _safe_score(token_set_ratio,  a, b)

    scores["addr_token_jaccard"]   = token_jaccard(a, b)
    scores["addr_token_overlap"]   = float(token_overlap_count(a, b))
    scores["addr_common_numbers"]  = common_number_ratio(a, b)

    # Length
    la, lb = len(a), len(b)
    scores["addr_len_diff_ratio"] = (
        abs(la - lb) / max(la, lb) if max(la, lb) > 0 else 0.0
    )

    # Address is empty in one side
    scores["addr_one_empty"] = float(bool(a) != bool(b))

    return scores


# ---------------------------------------------------------------------------
# Country feature
# ---------------------------------------------------------------------------

def country_match(c1: Optional[str], c2: Optional[str]) -> float:
    """1.0 if both country strings (lowercased) are equal, 0.0 otherwise."""
    if not c1 or not c2:
        return 0.0
    return 1.0 if c1.strip().lower() == c2.strip().lower() else 0.0


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity from precomputed sparse row vectors
# ---------------------------------------------------------------------------

def cosine_sim_from_vecs(v1, v2) -> float:
    """Cosine similarity between two sparse (or dense) vectors.

    Assumes both vectors are already L2-normalised (so dot product = cosine).
    """
    if v1 is None or v2 is None:
        return 0.0
    try:
        dot = v1.dot(v2.T)
        if hasattr(dot, "toarray"):
            return float(dot.toarray()[0, 0])
        return float(dot)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# All features combined
# ---------------------------------------------------------------------------

def all_similarity_scores(
    norm_name_a: str,
    norm_name_b: str,
    norm_addr_a: str,
    norm_addr_b: str,
    country_a: str,
    country_b: str,
) -> dict:
    """Convenience wrapper: compute all features for a single pair."""
    feats: dict = {}
    feats.update(name_similarity_scores(norm_name_a, norm_name_b))
    feats.update(address_similarity_scores(norm_addr_a, norm_addr_b))
    if cfg.USE_COUNTRY_FEATURE:
        feats["country_match"] = country_match(country_a, country_b)
    return feats
