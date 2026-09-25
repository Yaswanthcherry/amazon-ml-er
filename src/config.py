"""
config.py
---------
Central configuration for the entire pipeline.
All paths, hyperparameters, and feature flags live here.
Downstream modules import from this file — never hard-code values elsewhere.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parent.parent   # repo root
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------
TRAIN_SOURCE1      = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2      = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3      = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# ---------------------------------------------------------------------------
# Output file paths
# ---------------------------------------------------------------------------
CANDIDATE_PAIRS_PATH   = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_RESULTS_PATH  = OUTPUT_DIR / "matching_results.tsv"

# Saved model artefacts
MODEL_PATH             = MODELS_DIR / "lgbm_model.pkl"
THRESHOLD_PATH         = MODELS_DIR / "threshold.pkl"
TFIDF_NAME_PATH        = MODELS_DIR / "tfidf_name.pkl"
TFIDF_ADDR_PATH        = MODELS_DIR / "tfidf_addr.pkl"

# ---------------------------------------------------------------------------
# Encoding / I/O
# ---------------------------------------------------------------------------
FILE_ENCODING = "utf-8"
TSV_SEP       = "\t"

# ---------------------------------------------------------------------------
# Blocking / candidate generation
# ---------------------------------------------------------------------------

# Number of top-k TF-IDF candidates to retrieve per source-1 entity (per source)
TFIDF_TOP_K = 10

# Minimum TF-IDF cosine similarity to keep a candidate (pre-filter before ML)
TFIDF_MIN_SCORE = 0.05

# Maximum candidates per S1 entity fed into the ML model (across S2+S3 combined)
MAX_CANDIDATES_PER_ENTITY = 30

# TF-IDF vectoriser settings
TFIDF_NAME_NGRAM_RANGE  = (2, 3)   # character n-grams for business name
TFIDF_ADDR_NGRAM_RANGE  = (2, 3)   # character n-grams for address
TFIDF_MAX_FEATURES      = 300_000  # vocabulary cap (memory guard)

# Blocking key: minimum token overlap required between normalized name tokens
# (used in the fast exact-token-overlap pre-pass before TF-IDF)
MIN_TOKEN_OVERLAP = 1

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# RapidFuzz scorers used for pairwise string comparison
# Keys are feature names, values are scorer names (resolved in similarity.py)
NAME_SCORERS = [
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "jaro_winkler",
    "ratio",
]

ADDR_SCORERS = [
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "ratio",
]

# Whether to compute TF-IDF cosine similarity as an extra feature
USE_TFIDF_COSINE_FEATURE = True

# Whether to include a country-match binary feature
USE_COUNTRY_FEATURE = True

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Fraction of training data to use as validation (stratified by label)
VAL_SPLIT = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42

# Negative-to-positive ratio when sampling negatives for training
# Each positive pair is matched with this many random negatives from the same
# blocking bucket to keep class imbalance manageable.
NEG_POS_RATIO = 5

# LightGBM hyperparameters
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "n_jobs":           -1,
    "random_state":     RANDOM_SEED,
    "verbose":          -1,
}

LGBM_EARLY_STOPPING_ROUNDS = 30

# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

# Metric to optimise when searching for the best decision threshold
# Options: "f05", "f1", "precision", "recall"
THRESHOLD_METRIC = "f05"

# Grid of thresholds to search over
THRESHOLD_GRID_MIN  = 0.1
THRESHOLD_GRID_MAX  = 0.9
THRESHOLD_GRID_STEP = 0.01

# Default fallback threshold if tuning is skipped
DEFAULT_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Pipeline control flags
# ---------------------------------------------------------------------------

# Set to True to re-fit TF-IDF vectorisers even if saved artefacts exist
FORCE_RETRAIN_TFIDF = False

# Set to True to re-train the LightGBM model even if a saved model exists
FORCE_RETRAIN_MODEL = False

# Set to True to regenerate candidate pairs even if cached file exists
FORCE_REGEN_CANDIDATES = False

# Chunk size for processing S1 entities in batches (keeps memory bounded)
# Reduce if RAM is tight; increase for faster throughput on machines with >32 GB RAM
BATCH_SIZE = 5_000

# Number of parallel jobs for joblib-parallelised stages
N_JOBS = -1   # -1 = use all available cores

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"   # DEBUG | INFO | WARNING | ERROR
