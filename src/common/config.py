import os
from pathlib import Path

# project root = repo root (src/common/config.py → src/common → src → PROJECT_ROOT)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Main directories
DATA_DIR_RAW = PROJECT_ROOT / "data" / "raw"
DATA_DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSION_DIR = PROJECT_ROOT / "submission"

# Raw files
TRAIN_RAW_PATH = DATA_DIR_RAW / "train.csv"
TEST_RAW_PATH = DATA_DIR_RAW / "test.csv"

# Resampled 1-minute files
TRAIN_1MIN_PATH = DATA_DIR_PROCESSED / "train_1min.csv"
TEST_1MIN_PATH = DATA_DIR_PROCESSED / "test_1min.csv"

# Feature / target parquet files
TRAIN_FEATURES_PATH = DATA_DIR_PROCESSED / "train_features.parquet"
TRAIN_TARGET_PATH = DATA_DIR_PROCESSED / "train_target.parquet"
TEST_FEATURES_PATH = DATA_DIR_PROCESSED / "test_features.parquet"

# Make sure these dirs exist
for d in [DATA_DIR_RAW, DATA_DIR_PROCESSED, SUBMISSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 1-minute resampling frequency
RESAMPLE_FREQ = "1min"

# Fridge cleaning thresholds
FRIDGE_MAX_W = 450.0
FRIDGE_RATIO_THRESHOLD = 0.7

# Optional: used conceptually for features (even if features.py hardcodes them)
FFT_WINDOW_MINUTES = 60
LAG_MINUTES = [1, 2, 3, 5, 10, 15, 30, 45, 60, 90, 120, 180]

RANDOM_SEED = 42

LGBM_DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "verbose": -1,
}
