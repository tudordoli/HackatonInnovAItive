from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.common.config import (
    TEST_FEATURES_PATH,
    TEST_1MIN_PATH,
    TEST_RAW_PATH,
    PROJECT_ROOT,
)

EXP_NAME = "xgboost_simple"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
MODEL_PATH = EXPORT_DIR / "model_final.json"
ARTIFACT_PATH = EXPORT_DIR / "artifact.json"

SAMPLE_SUB_PATH = PROJECT_ROOT / "submission" / "sample_submission.csv"
OUT_SUB_PATH = PROJECT_ROOT / "submission" / "submission_xgboost_simple.csv"


def load_feature_cols(x_test_full: pd.DataFrame) -> list[str]:
    if ARTIFACT_PATH.exists():
        print(f"[XGB PRED] Loading artifact from: {ARTIFACT_PATH}")
        with open(ARTIFACT_PATH, "r") as f:
            artifact = json.load(f)
        feature_cols = artifact.get("feature_cols", None)
        if feature_cols is not None:
            print("[XGB PRED] Using feature_cols from artifact.json")
            missing = [c for c in feature_cols if c not in x_test_full.columns]
            if missing:
                raise ValueError(
                    "Feature columns from artifact.json missing in test features: "
                    + ", ".join(missing)
                )
            return feature_cols

    print(
        "[XGB PRED] artifact.json missing or no feature_cols; "
        "dropping meta columns [datetime, home_id]."
    )
    drop_cols = [c for c in ["datetime", "home_id"] if c in x_test_full.columns]
    return [c for c in x_test_full.columns if c not in drop_cols]


def main():
    print(f"[XGB PRED] Loading test features from: {TEST_FEATURES_PATH}")
    x_test_full = pd.read_parquet(TEST_FEATURES_PATH)

    print(f"[XGB PRED] Loading resampled test (1min) from: {TEST_1MIN_PATH}")
    test_1min = pd.read_csv(TEST_1MIN_PATH)

    # Normalize datetime
    x_test_full["datetime"] = pd.to_datetime(x_test_full["datetime"], utc=True)
    test_1min["datetime"] = pd.to_datetime(test_1min["datetime"], utc=True)

    feature_cols = load_feature_cols(x_test_full)
    x_test = x_test_full[feature_cols]

    print("[XGB PRED] Test (features) samples :", x_test.shape[0])
    print("[XGB PRED] Num features            :", x_test.shape[1])
    print(f"\n[XGB PRED] Loading model from: {MODEL_PATH}")
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    print("[XGB PRED] Predicting on 1-minute grid...")
    y_pred_1min = model.predict(x_test)
    y_pred_1min = np.clip(y_pred_1min, 0.0, None)

    if len(y_pred_1min) != len(test_1min):
        raise ValueError(
            f"Length mismatch: test_1min has {len(test_1min)} rows, "
            f"predictions have {len(y_pred_1min)} rows."
        )

    test_1min = test_1min.copy()
    test_1min["fridge_pred"] = y_pred_1min

    print(
        "[XGB PRED] Pred range on 1-min grid:",
        float(np.min(y_pred_1min)),
        "→",
        float(np.max(y_pred_1min)),
    )

    print(f"\n[XGB PRED] Loading raw test from: {TEST_RAW_PATH}")
    raw_test = pd.read_csv(TEST_RAW_PATH)
    raw_test["datetime"] = pd.to_datetime(raw_test["datetime"], utc=True)

    raw_test = raw_test.sort_values("datetime")
    test_1min = test_1min.sort_values("datetime")

    print("[XGB PRED] Min raw test ts :", raw_test["datetime"].min())
    print("[XGB PRED] Max raw test ts :", raw_test["datetime"].max())
    print("[XGB PRED] Min 1min test ts:", test_1min["datetime"].min())
    print("[XGB PRED] Max 1min test ts:", test_1min["datetime"].max())

    aligned = pd.merge_asof(
        raw_test[["datetime"]],
        test_1min[["datetime", "fridge_pred"]],
        on="datetime",
        direction="backward",
        allow_exact_matches=True,
    )

    # Backfill first (for times before first 1-min ts),
    # then forward-fill, then fill remaining NaNs with 0.
    aligned["fridge_pred"] = aligned["fridge_pred"].bfill().ffill().fillna(0.0)

    print(f"\n[XGB PRED] Loading sample submission from: {SAMPLE_SUB_PATH}")
    sub = pd.read_csv(SAMPLE_SUB_PATH)

    if len(sub) != len(aligned):
        raise ValueError(
            f"Length mismatch: sample_submission has {len(sub)} rows, "
            f"raw_test/aligned has {len(aligned)} rows."
        )

    if "fridge" not in sub.columns:
        raise ValueError("sample_submission.csv must contain a 'fridge' column.")

    sub["fridge"] = aligned["fridge_pred"].values

    OUT_SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_SUB_PATH, index=False)

    print("\n[XGB PRED] SAVED SUBMISSION TO:")
    print(OUT_SUB_PATH)


if __name__ == "__main__":
    main()
