"""
normalize.py
------------
Text normalisation utilities for business names and addresses.

All functions are stateless and operate on a single string (or a pandas Series
via the vectorised helpers at the bottom).

Design goals
------------
* Be lossy in a useful direction — collapse surface variations that carry no
  semantic signal (case, punctuation, common abbreviations) while preserving
  tokens that matter (numbers, city names, street tokens).
* Never drop information so aggressively that two different businesses become
  identical after normalisation (that would create false positives at the
  blocking stage).
* Country-agnostic by default; country-specific post-processing is additive.

Public API
----------
normalize_name(text)        -> str
normalize_address(text)     -> str
normalize_series_name(s)    -> pd.Series
normalize_series_address(s) -> pd.Series
get_name_tokens(text)       -> list[str]
get_address_tokens(text)    -> list[str]
"""

import re
import unicodedata
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Abbreviation expansion tables
# ---------------------------------------------------------------------------

# Business-name legal suffixes / common abbreviations
_NAME_ABBREV: dict[str, str] = {
    r"\bcorp\b":        "corporation",
    r"\binc\b":         "incorporated",
    r"\bllc\b":         "limited liability company",
    r"\bllp\b":         "limited liability partnership",
    r"\bltd\b":         "limited",
    r"\bpvt\b":         "private",
    r"\bco\b":          "company",
    r"\bassoc\b":       "associates",
    r"\bbros\b":        "brothers",
    r"\bmfg\b":         "manufacturing",
    r"\bsvcs\b":        "services",
    r"\bsvc\b":         "service",
    r"\bintl\b":        "international",
    r"\bnational\b":    "national",
    r"\bgrp\b":         "group",
    r"\bmgmt\b":        "management",
    r"\bmgr\b":         "manager",
    r"\bentpr\b":       "enterprises",
    r"\benterprise\b":  "enterprises",
    r"\benterpr\b":     "enterprises",
    r"\btech\b":        "technology",
    r"\btechnologies\b": "technology",
    r"\bsol\b":         "solutions",
    r"\bsoln\b":        "solutions",
    r"\bserv\b":        "services",
    r"\bindustries\b":  "industry",
    r"\bindus\b":       "industry",
    r"\bsarl\b":        "societe a responsabilite limitee",
    r"\bsas\b":         "societe par actions simplifiee",
    r"\bsa\b":          "societe anonyme",
    r"\bsci\b":         "societe civile immobiliere",
}

# Address-specific abbreviations
_ADDR_ABBREV: dict[str, str] = {
    r"\bst\b":          "street",
    r"\bstr\b":         "street",
    r"\brd\b":          "road",
    r"\bave\b":         "avenue",
    r"\bav\b":          "avenue",
    r"\bblvd\b":        "boulevard",
    r"\bdr\b":          "drive",
    r"\bln\b":          "lane",
    r"\bct\b":          "court",
    r"\bpl\b":          "place",
    r"\bsq\b":          "square",
    r"\bpkwy\b":        "parkway",
    r"\bpky\b":         "parkway",
    r"\bhwy\b":         "highway",
    r"\bfwy\b":         "freeway",
    r"\bexpy\b":        "expressway",
    r"\btrce\b":        "trace",
    r"\bter\b":         "terrace",
    r"\bterr\b":        "terrace",
    r"\bxing\b":        "crossing",
    r"\bcir\b":         "circle",
    r"\bappt\b":        "apartment",
    r"\bapt\b":         "apartment",
    r"\bste\b":         "suite",
    r"\bfl\b":          "floor",
    r"\bflr\b":         "floor",
    r"\bn\b":           "north",
    r"\bs\b":           "south",
    r"\be\b":           "east",
    r"\bw\b":           "west",
    r"\bne\b":          "northeast",
    r"\bnw\b":          "northwest",
    r"\bse\b":          "southeast",
    r"\bsw\b":          "southwest",
    r"\bno\b":          "number",
    r"\bph\b":          "penthouse",
}

# Tokens that are stopwords in addresses (carry no discriminative signal)
_ADDR_STOPWORDS = frozenset([
    "near", "opp", "opposite", "behind", "beside", "next", "above",
    "below", "floor", "unit", "flat", "room", "shop", "plot", "house",
    "building", "bldg", "complex", "colony", "sector", "block",
    "the", "of", "and", "at", "in", "on", "by", "to",
])

# Punctuation to strip (kept separately so we can tune without touching logic)
_PUNCT_RE = re.compile(r"[^\w\s]")

# Multiple-whitespace collapse
_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _unicode_normalize(text: str) -> str:
    """NFKD decompose then drop combining marks → ASCII-friendly."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _expand_abbreviations(text: str, table: dict[str, str]) -> str:
    for pattern, replacement in table.items():
        text = re.sub(pattern, replacement, text)
    return text


def _strip_punctuation(text: str) -> str:
    return _PUNCT_RE.sub(" ", text)


def _collapse_spaces(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Public normalisation functions
# ---------------------------------------------------------------------------

def normalize_name(text: Optional[str]) -> str:
    """Normalise a business name string.

    Steps:
    1. Handle None / NaN
    2. Unicode → ASCII transliteration
    3. Lowercase
    4. Strip punctuation
    5. Expand common abbreviations
    6. Collapse whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    text = _unicode_normalize(text)
    text = text.lower()
    text = _strip_punctuation(text)
    text = _expand_abbreviations(text, _NAME_ABBREV)
    text = _collapse_spaces(text)
    return text


def normalize_address(text: Optional[str]) -> str:
    """Normalise a business address string.

    Steps:
    1. Handle None / NaN
    2. Unicode → ASCII transliteration
    3. Lowercase
    4. Strip punctuation (but keep numbers)
    5. Expand address abbreviations
    6. Remove pure stopwords
    7. Collapse whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    text = _unicode_normalize(text)
    text = text.lower()
    text = _strip_punctuation(text)
    text = _expand_abbreviations(text, _ADDR_ABBREV)

    # Remove stopword tokens
    tokens = text.split()
    tokens = [t for t in tokens if t not in _ADDR_STOPWORDS]
    text = " ".join(tokens)

    text = _collapse_spaces(text)
    return text


def get_name_tokens(text: Optional[str]) -> list:
    """Return normalised tokens for a business name (used for token-overlap blocking)."""
    normalized = normalize_name(text)
    return [t for t in normalized.split() if len(t) > 1]


def get_address_tokens(text: Optional[str]) -> list:
    """Return normalised tokens for an address (used for token-overlap blocking)."""
    normalized = normalize_address(text)
    return [t for t in normalized.split() if len(t) > 1]


# ---------------------------------------------------------------------------
# Vectorised helpers (operate on pandas Series — fast with .map)
# ---------------------------------------------------------------------------

def normalize_series_name(series: pd.Series) -> pd.Series:
    """Apply normalize_name to every element of a Series."""
    return series.map(normalize_name)


def normalize_series_address(series: pd.Series) -> pd.Series:
    """Apply normalize_address to every element of a Series."""
    return series.map(normalize_address)


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'norm_name' and 'norm_address' columns to a source DataFrame in-place.

    Modifies df directly and also returns it for chaining.
    """
    df["norm_name"]    = normalize_series_name(df["business_name"])
    df["norm_address"] = normalize_series_address(df["business_address"])
    return df


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    examples = [
        ("Davis Family Office",                 "88 Olive Circle, Lebanon, TN"),
        ("Davis Family Offie",                  "88 OLIVE CIR, LEBANON, TN"),
        ("DAVIS FAMILY OFFICE",                 "OLIVE CIR, LEBANON, TN"),
        ("MS Consultancy Corp",                 "Shymala Appts 1St Floor Flat No. 4 Opp Ratna Hospital"),
        ("3520 Main Road Realty Inc",           "TX, 158 Simpson Lane, Somerset"),
        ("à¤°à¤¾à¤® à¤®à¤¾à¤°à¥à¤•à¥‡à¤Ÿà¤¿à¤‚à¤— Private Limited",  "KH NO. -570/13, NEW DELHI"),
    ]
    for name, addr in examples:
        print(f"  name: {name!r:50s}  →  {normalize_name(name)!r}")
        print(f"  addr: {addr!r:50s}  →  {normalize_address(addr)!r}")
        print()
