from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.common.config import (
    TRAIN_1MIN_PATH,
    TEST_1MIN_PATH,
    DATA_DIR_PROCESSED,
)

TRAIN_SMOOTH_PATH = DATA_DIR_PROCESSED / "train_1min_lightinterp.csv"
TEST_SMOOTH_PATH = DATA_DIR_PROCESSED / "test_1min_lightinterp.csv"


def _light_interp_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    prev = s.shift(1)
    nxt = s.shift(-1)
    smooth = (prev + 2.0 * s + nxt) / 4.0

    # For edges where prev/next are NaN, fall back to original
    return smooth.fillna(s)


def _smooth_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime is parsed and sorted
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    if "home_id" in df.columns:
        df = df.sort_values(["home_id", "datetime"]).reset_index(drop=True)
    else:
        df = df.sort_values("datetime").reset_index(drop=True)

    # Numeric columns to smooth
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Never touch these as numeric
    for col in ["home_id"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    # Apply smoothing
    if "home_id" in df.columns:
        # Per-home smoothing to avoid blending across homes
        for col in numeric_cols:
            df[col] = df.groupby("home_id", observed=True)[col].transform(
                _light_interp_series
            )
    else:
        for col in numeric_cols:
            df[col] = _light_interp_series(df[col])

    return df


def main():
    print(f"[light_smooth] Loading train 1-min: {TRAIN_1MIN_PATH}")
    train_df = pd.read_csv(TRAIN_1MIN_PATH)

    print("[light_smooth] Applying light interpolation to train...")
    train_smooth = _smooth_dataframe(train_df)

    TRAIN_SMOOTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_smooth.to_csv(TRAIN_SMOOTH_PATH, index=False)
    print(f"[light_smooth] Saved smoothed train → {TRAIN_SMOOTH_PATH}")

    print(f"[light_smooth] Loading test 1-min: {TEST_1MIN_PATH}")
    test_df = pd.read_csv(TEST_1MIN_PATH)

    print("[light_smooth] Applying light interpolation to test...")
    test_smooth = _smooth_dataframe(test_df)

    test_smooth.to_csv(TEST_SMOOTH_PATH, index=False)
    print(f"[light_smooth] Saved smoothed test  → {TEST_SMOOTH_PATH}")

    print("[light_smooth] Done.")


if __name__ == "__main__":
    main()
