from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common.config import (
    TEST_FEATURES_PATH,
    TEST_1MIN_PATH,
    TEST_RAW_PATH,
    PROJECT_ROOT,
)

EXP_NAME = "lightgbm_simple"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
MODEL_PATH = EXPORT_DIR / "model_final.txt"
ARTIFACT_PATH = EXPORT_DIR / "artifact.json"

SAMPLE_SUB_PATH = PROJECT_ROOT / "submission" / "sample_submission.csv"
OUT_SUB_PATH = PROJECT_ROOT / "submission" / "submission_lightgbm_simple.csv"


def _load_feature_cols(x_test_full: pd.DataFrame) -> list[str]:
    """Load feature_cols from artifact.json if possible, else drop meta cols."""
    if ARTIFACT_PATH.exists():
        with open(ARTIFACT_PATH, "r") as f:
            artifact = json.load(f)
        feat = artifact.get("feature_cols")
        if feat is not None:
            missing = [c for c in feat if c not in x_test_full.columns]
            if missing:
                raise ValueError(
                    f"feature_cols from artifact.json missing in test features: {missing}"
                )
            print("[predict] Using feature_cols from artifact.json")
            return feat

    print(
        "[predict] artifact.json missing or no feature_cols; "
        "using all columns except [datetime, home_id]."
    )
    drop_cols = [c for c in ["datetime", "home_id"] if c in x_test_full.columns]
    return [c for c in x_test_full.columns if c not in drop_cols]


def main():
    print(f"[predict] Loading test features from: {TEST_FEATURES_PATH}")
    x_test_full = pd.read_parquet(TEST_FEATURES_PATH)
    x_test_full["datetime"] = pd.to_datetime(x_test_full["datetime"], utc=True)

    feature_cols = _load_feature_cols(x_test_full)
    X_test = x_test_full[feature_cols]

    print("[predict] Test samples :", X_test.shape[0])
    print("[predict] Num features :", X_test.shape[1])

    print(f"[predict] Loading resampled test (1min) from: {TEST_1MIN_PATH}")
    test_1min = pd.read_csv(TEST_1MIN_PATH)
    test_1min["datetime"] = pd.to_datetime(test_1min["datetime"], utc=True)
    test_1min = test_1min.sort_values("datetime").reset_index(drop=True)
    print(f"[predict] Loading LightGBM model from: {MODEL_PATH}")
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    print("[predict] Predicting on 1-min grid...")
    y_pred = booster.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, None)

    if len(y_pred) != len(test_1min):
        raise ValueError(
            f"Length mismatch: test_1min has {len(test_1min)} rows, "
            f"predictions have {len(y_pred)} rows."
        )

    test_1min = test_1min.copy()
    test_1min["fridge_pred_raw"] = y_pred

    print("[predict] Raw pred range:", float(y_pred.min()), "→", float(y_pred.max()))

    # 3-minute rolling mean (very light)
    test_1min["fridge_pred"] = (
        test_1min["fridge_pred_raw"].rolling(window=3, min_periods=1).mean()
    )

    print(
        "[predict] Smoothed pred range:",
        float(test_1min["fridge_pred"].min()),
        "→",
        float(test_1min["fridge_pred"].max()),
    )

    print(f"[predict] Loading raw test from: {TEST_RAW_PATH}")
    raw_test = pd.read_csv(TEST_RAW_PATH)
    raw_test["datetime"] = pd.to_datetime(raw_test["datetime"], utc=True)
    raw_test = raw_test.sort_values("datetime").reset_index(drop=True)

    print("[predict] Min raw ts :", raw_test["datetime"].min())
    print("[predict] Min 1min ts:", test_1min["datetime"].min())

    aligned = pd.merge_asof(
        raw_test[["datetime"]],
        test_1min[["datetime", "fridge_pred"]],
        on="datetime",
        direction="backward",
        allow_exact_matches=True,
    )

    aligned["fridge_pred"] = aligned["fridge_pred"].bfill().ffill().fillna(0.0)

    print(f"[predict] Loading sample submission from: {SAMPLE_SUB_PATH}")
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
    print("\n[predict] SAVED SUBMISSION TO:")
    print(OUT_SUB_PATH)


if __name__ == "__main__":
    main()
