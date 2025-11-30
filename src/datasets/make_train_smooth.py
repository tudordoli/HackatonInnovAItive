from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.common.config import (
    TRAIN_1MIN_PATH,
    TEST_1MIN_PATH,
    DATA_DIR_PROCESSED,
)


ROLL_WINDOW = 3  # rolling window size in minutes

OUT_TRAIN_PATH = DATA_DIR_PROCESSED / f"train_1min_roll{ROLL_WINDOW}.csv"
OUT_TEST_PATH = DATA_DIR_PROCESSED / f"test_1min_roll{ROLL_WINDOW}.csv"


def smooth_dataframe(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Apply rolling window smoothing to all numeric columns
    except datetime & home_id.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    drop_cols = [c for c in ["datetime", "home_id"] if c in numeric_cols]
    numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    # Apply rolling smoothing (no future leak)
    df[numeric_cols] = df[numeric_cols].rolling(window=window, min_periods=1).mean()

    return df


def main():
    print("=== LOADING RESAMPLED DATA ===")
    print(f"Train 1-min path: {TRAIN_1MIN_PATH}")
    print(f"Test 1-min path : {TEST_1MIN_PATH}")

    train_df = pd.read_csv(TRAIN_1MIN_PATH)
    test_df = pd.read_csv(TEST_1MIN_PATH)

    # Convert datetime safely
    train_df["datetime"] = pd.to_datetime(train_df["datetime"], utc=True)
    test_df["datetime"] = pd.to_datetime(test_df["datetime"], utc=True)

    print("\n=== APPLYING ROLLING SMOOTH ===")
    print(f"Rolling window: {ROLL_WINDOW} minutes")

    train_smooth = smooth_dataframe(train_df, ROLL_WINDOW)
    test_smooth = smooth_dataframe(test_df, ROLL_WINDOW)

    print("\n=== SAVING NEW SMOOTHED DATASETS ===")
    OUT_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_smooth.to_csv(OUT_TRAIN_PATH, index=False)
    test_smooth.to_csv(OUT_TEST_PATH, index=False)

    print(f"Saved smoothed train → {OUT_TRAIN_PATH}")
    print(f"Saved smoothed test  → {OUT_TEST_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
