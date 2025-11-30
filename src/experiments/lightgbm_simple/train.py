from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from src.common.config import (
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.common.evaluation import mae, rmse, nde, sae


EXP_NAME = "lightgbm_simple"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = EXPORT_DIR / "model_final.txt"
ARTIFACT_PATH = EXPORT_DIR / "artifact.json"


# LightGBM tuned for stability across homes
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 256,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "n_estimators": 3000,
    "n_jobs": -1,
    "verbosity": -1,
}


def main():
    print(f"[train_simple] Loading train features: {TRAIN_FEATURES_PATH}")
    X_full = pd.read_parquet(TRAIN_FEATURES_PATH)

    print(f"[train_simple] Loading train target: {TRAIN_TARGET_PATH}")
    target_df = pd.read_parquet(TRAIN_TARGET_PATH)

    # Be robust to column naming in the parquet
    if "fridge" in target_df.columns:
        target_col = "fridge"
    else:
        # fall back to the first column if the name is something else / unnamed
        target_col = target_df.columns[0]

    y_full = target_df[target_col].astype(float).values

    drop_cols = [c for c in ["home_id", "datetime"] if c in X_full.columns]
    feature_cols = [c for c in X_full.columns if c not in drop_cols]

    X = X_full[feature_cols]

    print("\n[train_simple] Data overview")
    print("-----------------------------")
    print("Samples :", X.shape[0])
    print("Features:", X.shape[1])
    if "home_id" in X_full.columns:
        print("Homes   :", X_full["home_id"].nunique())

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_full,
        test_size=0.2,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    print("\n[train_simple] Train samples:", X_train.shape[0])
    print("[train_simple] Valid samples :", X_val.shape[0])

    print("\n[train_simple] Training LightGBM…")
    for k, v in LGB_PARAMS.items():
        print(f"  {k}: {v}")

    model = lgb.LGBMRegressor(
        **LGB_PARAMS,
        random_state=RANDOM_SEED,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0.0, None)

    m_mae = mae(y_val, y_pred)
    m_rmse = rmse(y_val, y_pred)
    m_nde = nde(y_val, y_pred)
    m_sae = sae(y_val, y_pred)

    print("\n[train_simple] === Validation metrics ===")
    print(f"MAE : {m_mae:.6f}")
    print(f"RMSE: {m_rmse:.6f}")
    print(f"NDE : {m_nde:.6f}")
    print(f"SAE : {m_sae:.6f}")

    print(f"\n[train_simple] Saving model → {MODEL_PATH}")
    model.booster_.save_model(str(MODEL_PATH))

    artifact = {
        "experiment": EXP_NAME,
        "feature_cols": feature_cols,
        "params": LGB_PARAMS,
        "val_metrics": {
            "mae": float(m_mae),
            "rmse": float(m_rmse),
            "nde": float(m_nde),
            "sae": float(m_sae),
        },
        "data_shape": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        },
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print("[train_simple] Saved artifact →", ARTIFACT_PATH)
    print("[train_simple] Done.")


if __name__ == "__main__":
    main()
