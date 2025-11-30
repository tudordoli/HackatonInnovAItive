from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common.config import (
    TEST_FEATURES_PATH,
    TEST_1MIN_PATH,
    TEST_RAW_PATH,
    PROJECT_ROOT,
)

EXP_NAME = "lightgbm_loho"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
MODEL_PATH = EXPORT_DIR / "model_all_homes.txt"

SAMPLE_SUB_PATH = PROJECT_ROOT / "submission" / "sample_submission.csv"
OUT_SUB_PATH = PROJECT_ROOT / "submission" / "submission_lightgbm_loho.csv"


def main():
    print(f"[LOHO PRED] Loading test features from: {TEST_FEATURES_PATH}")
    x_test_full = pd.read_parquet(TEST_FEATURES_PATH)
    x_test_full["datetime"] = pd.to_datetime(x_test_full["datetime"], utc=True)

    print(f"[LOHO PRED] Loading LOHO model from: {MODEL_PATH}")
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    # Use the feature names stored inside the model as the source of truth
    model_feature_names = booster.feature_name()
    print(f"[LOHO PRED] Model expects {len(model_feature_names)} features.")

    # Check that all model features exist in test features
    missing = [c for c in model_feature_names if c not in x_test_full.columns]
    if missing:
        raise ValueError(
            "Test features are missing columns required by LOHO model. "
            "You almost certainly need to retrain LOHO.\n"
            f"Missing columns: {missing}"
        )

    # Select columns in the exact order the model was trained with
    X_test = x_test_full[model_feature_names]

    print("[LOHO PRED] Test samples :", X_test.shape[0])
    print("[LOHO PRED] Num features :", X_test.shape[1])

    print(f"[LOHO PRED] Loading resampled test (1min) from: {TEST_1MIN_PATH}")
    test_1min = pd.read_csv(TEST_1MIN_PATH)
    test_1min["datetime"] = pd.to_datetime(test_1min["datetime"], utc=True)
    test_1min = test_1min.sort_values("datetime").reset_index(drop=True)

    print("[LOHO PRED] Predicting on 1-minute grid...")
    y_pred = booster.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, None)

    if len(y_pred) != len(test_1min):
        raise ValueError(
            f"Length mismatch: test_1min has {len(test_1min)} rows, "
            f"predictions have {len(y_pred)} rows."
        )

    test_1min = test_1min.copy()
    test_1min["fridge_pred_raw"] = y_pred

    print(
        "[LOHO PRED] Raw pred range:",
        float(y_pred.min()),
        "→",
        float(y_pred.max()),
    )

    test_1min["fridge_pred"] = (
        test_1min["fridge_pred_raw"].rolling(window=3, min_periods=1).mean()
    )

    print(
        "[LOHO PRED] Smoothed pred range:",
        float(test_1min["fridge_pred"].min()),
        "→",
        float(test_1min["fridge_pred"].max()),
    )

    print(f"[LOHO PRED] Loading raw test from: {TEST_RAW_PATH}")
    raw_test = pd.read_csv(TEST_RAW_PATH)
    raw_test["datetime"] = pd.to_datetime(raw_test["datetime"], utc=True)
    raw_test = raw_test.sort_values("datetime").reset_index(drop=True)

    aligned = pd.merge_asof(
        raw_test[["datetime"]],
        test_1min[["datetime", "fridge_pred"]],
        on="datetime",
        direction="backward",
        allow_exact_matches=True,
    )

    aligned["fridge_pred"] = aligned["fridge_pred"].bfill().ffill().fillna(0.0)

    print(f"[LOHO PRED] Loading sample submission from: {SAMPLE_SUB_PATH}")
    sub = pd.read_csv(SAMPLE_SUB_PATH)

    if len(sub) != len(aligned):
        raise ValueError(
            f"Length mismatch: sample_submission has {len(sub)} rows, "
            f"aligned has {len(aligned)} rows."
        )

    if "fridge" not in sub.columns:
        raise ValueError("sample_submission.csv must contain a 'fridge' column.")

    sub["fridge"] = aligned["fridge_pred"].values

    OUT_SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_SUB_PATH, index=False)

    print("\n[LOHO PRED] SAVED SUBMISSION TO:")
    print(OUT_SUB_PATH)


if __name__ == "__main__":
    main()
