from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common.config import (
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.common.evaluation import mae, rmse, nde, sae, loho_split


EXP_NAME = "lightgbm_loho"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

PER_HOME_METRICS_CSV = EXPORT_DIR / "loho_metrics_per_home.csv"
SUMMARY_JSON = EXPORT_DIR / "loho_summary.json"
FINAL_MODEL_PATH = EXPORT_DIR / "model_all_homes.txt"


def make_model_params() -> dict:
    params = {
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
    return params


def main():
    print(f"[LOHO] Loading train features from: {TRAIN_FEATURES_PATH}")
    X_full = pd.read_parquet(TRAIN_FEATURES_PATH)

    print(f"[LOHO] Loading train target from:   {TRAIN_TARGET_PATH}")
    y_full = pd.read_parquet(TRAIN_TARGET_PATH)["fridge"].astype(float)
    y = y_full.values.reshape(-1)

    if "home_id" not in X_full.columns:
        raise ValueError("LOHO requires 'home_id' column.")

    print("\n[LOHO] Dataset overview")
    print("-----------------------")
    print("Total samples :", X_full.shape[0])
    print("Num features  :", X_full.shape[1])
    print("Num homes     :", X_full["home_id"].nunique())

    drop_cols = [c for c in ["home_id", "datetime"] if c in X_full.columns]
    feature_cols = [c for c in X_full.columns if c not in drop_cols]

    print("\n[LOHO] Using feature columns:", len(feature_cols))

    params = make_model_params()
    print("\n[LOHO] LightGBM params:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    per_home_records = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for home, X_train, X_valid, y_train, y_valid in loho_split(
        X_full, y, home_col="home_id"
    ):
        print(f"\n[LOHO] Home {home}:")
        print("  train samples:", X_train.shape[0])
        print("  valid samples:", X_valid.shape[0])

        X_train_model = X_train[feature_cols]
        X_valid_model = X_valid[feature_cols]

        model = lgb.LGBMRegressor(
            **params,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train_model, y_train)

        y_pred = model.predict(X_valid_model)
        y_pred = np.clip(y_pred, 0.0, None)

        y_true_all.append(np.asarray(y_valid))
        y_pred_all.append(np.asarray(y_pred))

        m_mae = mae(y_valid, y_pred)
        m_rmse = rmse(y_valid, y_pred)
        m_nde = nde(y_valid, y_pred)
        m_sae = sae(y_valid, y_pred)

        print("  MAE :", m_mae)
        print("  RMSE:", m_rmse)
        print("  NDE :", m_nde)
        print("  SAE :", m_sae)

        per_home_records.append(
            {
                "home_id": home,
                "n_train": int(X_train.shape[0]),
                "n_valid": int(X_valid.shape[0]),
                "mae": float(m_mae),
                "rmse": float(m_rmse),
                "nde": float(m_nde),
                "sae": float(m_sae),
            }
        )

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    global_mae = mae(y_true_all, y_pred_all)
    global_rmse = rmse(y_true_all, y_pred_all)
    global_nde = nde(y_true_all, y_pred_all)
    global_sae = sae(y_true_all, y_pred_all)

    print("\n[LOHO] === Global metrics ===")
    print(f"MAE : {global_mae:.6f}")
    print(f"RMSE: {global_rmse:.6f}")
    print(f"NDE : {global_nde:.6f}")
    print(f"SAE : {global_sae:.6f}")

    pd.DataFrame(per_home_records).to_csv(PER_HOME_METRICS_CSV, index=False)
    print(f"[LOHO] Saved per-home metrics → {PER_HOME_METRICS_CSV}")

    summary = {
        "experiment": EXP_NAME,
        "params": params,
        "feature_cols": feature_cols,
        "global_metrics": {
            "mae": float(global_mae),
            "rmse": float(global_rmse),
            "nde": float(global_nde),
            "sae": float(global_sae),
        },
        "model_path": str(FINAL_MODEL_PATH),
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[LOHO] Saved summary JSON → {SUMMARY_JSON}")
