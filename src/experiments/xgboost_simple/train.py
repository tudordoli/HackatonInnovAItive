from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.common.config import (
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.common.evaluation import mae, rmse, nde, sae

EXP_NAME = "xgboost_simple"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = EXPORT_DIR / "model_final.json"
ARTIFACT_PATH = EXPORT_DIR / "artifact.json"


def main():
    print(f"[XGB] Loading train features from: {TRAIN_FEATURES_PATH}")
    X_full = pd.read_parquet(TRAIN_FEATURES_PATH)

    print(f"[XGB] Loading train target from:   {TRAIN_TARGET_PATH}")
    y_full = pd.read_parquet(TRAIN_TARGET_PATH)

    # 1D float target
    y = y_full.values.reshape(-1).astype(float)

    # Drop meta columns
    drop_cols = [c for c in ["home_id", "datetime"] if c in X_full.columns]
    feature_cols = [c for c in X_full.columns if c not in drop_cols]

    X = X_full[feature_cols]

    print("\n[XGB] Data overview")
    print("-------------------")
    print("Total samples :", X.shape[0])
    print("Num features  :", X.shape[1])
    if "home_id" in X_full.columns:
        print("Homes in data :", X_full["home_id"].nunique())

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    print("\n[XGB] Train samples:", X_train.shape[0])
    print("[XGB] Valid samples:", X_valid.shape[0])

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",  # change to "gpu_hist" if you have GPU
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "n_estimators": 3000,
        "n_jobs": -1,
    }

    print("\n[XGB] Training XGBoost with params:")
    for k, v in xgb_params.items():
        print(f"  {k}: {v}")

    model = XGBRegressor(
        **xgb_params,
        random_state=RANDOM_SEED,
    )

    model.fit(X_train, y_train)

    print("\n[XGB] Evaluating on validation split...")
    y_pred = model.predict(X_valid)
    y_pred = np.clip(y_pred, 0.0, None)

    m_mae = mae(y_valid, y_pred)
    m_rmse = rmse(y_valid, y_pred)
    m_nde = nde(y_valid, y_pred)
    m_sae = sae(y_valid, y_pred)

    print("\n[XGB] === Simple split metrics (validation) ===")
    print(f"MAE : {m_mae:.6f}")
    print(f"RMSE: {m_rmse:.6f}")
    print(f"NDE : {m_nde:.6f}")
    print(f"SAE : {m_sae:.6f}")

    print(f"\n[XGB] Saving model to: {MODEL_PATH}")
    model.save_model(str(MODEL_PATH))

    artifact = {
        "experiment": EXP_NAME,
        "model_type": "XGBRegressor",
        "params": xgb_params,
        "random_state": RANDOM_SEED,
        "feature_cols": feature_cols,
        "metrics": {
            "val_mae": float(m_mae),
            "val_rmse": float(m_rmse),
            "val_nde": float(m_nde),
            "val_sae": float(m_sae),
        },
        "data_shape": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        },
        "paths": {
            "model_final": str(MODEL_PATH),
        },
    }

    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[XGB] Saved artifact to: {ARTIFACT_PATH}")
    print("[XGB] Done.")


if __name__ == "__main__":
    main()
