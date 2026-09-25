# Business Entity Resolution — Amazon ML Challenge

An end-to-end ML pipeline that determines which records across three independent
business-data sources refer to the same real-world entity.

---

## Problem Summary

- **Source 1** is the deduplicated reference source (~2.2 M train / ~1.7 M test entities).
- **Sources 2 & 3** are noisy mirrors (~5 M records each).
- For every Source-1 entity, predict which Source-2 / Source-3 records are the same business.
- Evaluated with **macro-averaged F_0.5** (precision-weighted).

---

## Project Structure

```
amazon-ml-er/
├── data/
│   ├── train/                  # training TSVs + ground truth
│   └── test/                   # test TSVs
├── src/
│   ├── config.py               # all paths, hyperparameters, flags
│   ├── data_loader.py          # TSV I/O helpers
│   ├── normalize.py            # text normalisation (unicode, abbreviations)
│   ├── blocking.py             # TF-IDF vectorisers + candidate retrieval
│   ├── candidate_generation.py # orchestrates blocking stage
│   ├── similarity.py           # pairwise string similarity features
│   ├── features.py             # feature matrix builder
│   ├── labels.py               # ground-truth parsing + negative sampling
│   ├── train.py                # LightGBM training
│   ├── evaluate.py             # F_0.5 metrics
│   ├── threshold.py            # threshold search + persistence
│   ├── predict.py              # inference + threshold application
│   ├── postprocess.py          # output formatting + validation
│   └── pipeline.py             # top-level orchestrator
├── models/                     # saved artefacts (vectorisers, model, threshold)
├── outputs/
│   ├── candidate_pairs.tsv     # blocking output
│   └── matching_results.tsv   # final predictions (leaderboard submission)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_blocking_analysis.ipynb
│   └── 03_model_analysis.ipynb
├── requirements.txt
├── run.py                      # CLI entry point
└── README.md
```

---

## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

All commands are run from the repository root.

### Full pipeline (train → predict)

```bash
python run.py
```

This runs:
1. **Blocking** — fits TF-IDF character n-gram vectorisers on the training corpus,
   then retrieves the top-k most similar Source-2 / Source-3 candidates for each
   Source-1 entity using country-partitioned cosine search.
2. **Training** — builds pairwise similarity features (rapidfuzz + Jaccard + number overlap)
   for sampled positive/negative pairs, then trains a LightGBM binary classifier.
3. **Threshold tuning** — searches the threshold that maximises F_0.5 on a held-out
   validation fold.
4. **Prediction** — applies the pipeline to the test set and writes both output files.

### Individual stages

```bash
python run.py --mode train      # blocking + training + threshold tuning only
python run.py --mode predict    # inference only (requires saved artefacts)
python run.py --mode eval       # evaluate on the validation fold, print F_0.5 report
```

### Force re-training

```bash
python run.py --mode full --force-retrain
```

### Logging verbosity

```bash
python run.py --log-level DEBUG
```

---

## Pipeline Architecture

```
Raw TSVs
   │
   ▼
normalize.py          Unicode → ASCII, lowercase, abbreviation expansion
   │
   ▼
blocking.py           TF-IDF char n-gram vectorisers (name + address, weighted)
                      Country-partitioned cosine similarity retrieval (top-k)
   │
   ▼
candidate_generation  Cap at MAX_CANDIDATES_PER_ENTITY, save candidate_pairs.tsv
   │
   ▼
features.py           20 pairwise features per pair:
                        name:  ratio, partial_ratio, token_sort/set, jaro_winkler,
                               jaccard, lev_sim, len_diff, token_overlap
                        addr:  ratio, partial_ratio, token_sort/set, jaccard,
                               common_numbers, len_diff, one_empty, token_overlap
                        meta:  country_match
   │
   ▼
train.py              LightGBM (binary, 500 trees, early stopping on val logloss)
                      Negative sampling 5:1, scale_pos_weight for imbalance
   │
   ▼
threshold.py          Grid search over [0.1, 0.9] to maximise macro-F_0.5 on val fold
   │
   ▼
predict.py            Score test candidates → apply threshold
   │
   ▼
postprocess.py        Deduplicate IDs, validate S2/S3 only, write TSVs
   │
   ▼
outputs/
  matching_results.tsv   ← leaderboard submission
  candidate_pairs.tsv    ← blocking audit
```

---

## Configuration

All tunable parameters are in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `TFIDF_TOP_K` | 10 | Candidates retrieved per source per S1 entity |
| `TFIDF_MIN_SCORE` | 0.05 | Minimum cosine score to keep a candidate |
| `MAX_CANDIDATES_PER_ENTITY` | 30 | Hard cap on candidates fed to the ML model |
| `NEG_POS_RATIO` | 5 | Negatives sampled per positive during training |
| `VAL_SPLIT` | 0.1 | Fraction of S1 entities held out for validation |
| `LGBM_PARAMS` | see config | LightGBM hyperparameters |
| `THRESHOLD_METRIC` | "f05" | Metric optimised during threshold search |
| `BATCH_SIZE` | 5000 | S1 entities processed per feature-building batch |

---

## Output Format

### `outputs/matching_results.tsv`
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003
```

### `outputs/candidate_pairs.tsv`
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003
```

---

## Evaluation Metric

**Macro-averaged F_0.5** (precision-weighted):

```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
```

Computed per Source-1 entity (including singletons), then averaged.
Singletons correctly predicted as empty score **1.0**.

---

## Key Design Decisions

- **Country partitioning in blocking**: retrieval is done within each country group,
  avoiding cross-country false positives and reducing the comparison space ~3×.
- **Char n-gram TF-IDF** (bigrams + trigrams): robust to typos, abbreviations, and
  transliteration variants without needing a language-specific tokeniser.
- **Weighted name + address matrix**: name tokens are upweighted (0.6) over address
  (0.4) because name typos are more informative than address formatting differences.
- **Negative sampling at training time**: 5 negatives per positive keeps the
  feature matrix manageable while teaching the model real hard negatives from the
  same blocking bucket.
- **Threshold tuned for F_0.5**: since F_0.5 penalises false positives 2× more than
  false negatives, the optimal threshold is typically higher than 0.5.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.2.2 | Data loading and manipulation |
| numpy | 1.26.4 | Numerical arrays |
| scikit-learn | 1.5.1 | TF-IDF vectorisers, sparse matrix ops |
| lightgbm | 4.4.0 | Gradient-boosted pairwise classifier |
| rapidfuzz | 3.9.4 | Fast string similarity metrics |
| scipy | 1.13.1 | Sparse matrix support |
| joblib | 1.4.2 | Parallel feature computation |
| tqdm | 4.66.4 | Progress bars |
