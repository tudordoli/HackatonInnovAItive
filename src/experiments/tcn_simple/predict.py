from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.common.config import (
    PROJECT_ROOT,
    TEST_FEATURES_PATH,
    TEST_1MIN_PATH,
    TEST_RAW_PATH,
)

from src.experiments.tcn_simple.train import SimpleTCN  # reuse model class

EXP_NAME = "tcn_simple"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
MODEL_PATH = EXPORT_DIR / "model.pt"
ARTIFACT_PATH = EXPORT_DIR / "artifact.json"

SAMPLE_SUB_PATH = PROJECT_ROOT / "submission" / "sample_submission.csv"
OUT_SUB_PATH = PROJECT_ROOT / "submission" / "submission_tcn_simple.csv"

BATCH_SIZE = 256


def load_artifact():
    with open(ARTIFACT_PATH, "r") as f:
        art = json.load(f)
    feature_cols = art["feature_cols"]
    window_len = int(art["window_len"])
    mean = np.array(art["mean"], dtype=np.float32)
    std = np.array(art["std"], dtype=np.float32)
    return feature_cols, window_len, mean, std


def build_windows_test(X: np.ndarray, window_len: int):
    T = X.shape[0]
    if T < window_len:
        raise RuntimeError("Test sequence shorter than window_len.")

    starts = list(range(0, T - window_len + 1, 1))
    return starts


def main():
    print(f"[TCN PRED] Loading artifact from: {ARTIFACT_PATH}")
    feature_cols, window_len, mean, std = load_artifact()

    print(f"[TCN PRED] Loading test features from: {TEST_FEATURES_PATH}")
    X_df = pd.read_parquet(TEST_FEATURES_PATH)
    X_df["datetime"] = pd.to_datetime(X_df["datetime"], utc=True)
    X_df = X_df.sort_values("datetime").reset_index(drop=True)

    # Select & normalize features
    X_mat = X_df[feature_cols].astype(np.float32).values
    X_mat = (X_mat - mean) / (std + 1e-6)

    T, D = X_mat.shape
    print(f"[TCN PRED] Test shape (T, D): {T}, {D}")

    starts = build_windows_test(X_mat, window_len)
    print(f"[TCN PRED] Number of windows: {len(starts)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TCN PRED] Using device: {device}")

    model = SimpleTCN(in_channels=D).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preds_window = np.zeros(len(starts), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(starts), BATCH_SIZE):
            batch_starts = starts[i : i + BATCH_SIZE]
            batch_x = []
            for s in batch_starts:
                slice_x = X_mat[s : s + window_len]  # [L, D]
                slice_x = slice_x.T  # [D, L]
                batch_x.append(slice_x)
            xb = torch.from_numpy(np.stack(batch_x, axis=0)).to(device)  # [B, D, L]
            pb = model(xb).cpu().numpy().astype(np.float32)
            preds_window[i : i + len(batch_starts)] = pb

    # Reconstruct per-timestep prediction y[t] from window-wise predictions
    # Window i predicts for timestamp t = start + window_len - 1
    y_full = np.zeros(T, dtype=np.float32)
    first_pred = preds_window[0]
    y_full[: window_len - 1] = first_pred
    for i, s in enumerate(starts):
        t = s + window_len - 1
        y_full[t] = preds_window[i]
    # For any remaining tail (shouldn't happen), ffill
    for t in range(window_len, T):
        if y_full[t] == 0.0 and y_full[t - 1] != 0.0:
            y_full[t] = y_full[t - 1]

    y_full = np.clip(y_full, 0.0, None)

    print(
        "[TCN PRED] Pred range on 1-min grid:",
        float(y_full.min()),
        "→",
        float(y_full.max()),
    )

    # Attach to 1-min timestamps and align to raw test
    print(f"[TCN PRED] Loading 1-min test from: {TEST_1MIN_PATH}")
    test_1min = pd.read_csv(TEST_1MIN_PATH)
    test_1min["datetime"] = pd.to_datetime(test_1min["datetime"], utc=True)
    test_1min = test_1min.sort_values("datetime").reset_index(drop=True)

    if len(test_1min) != T:
        raise ValueError(
            f"Length mismatch: test_1min has {len(test_1min)}, " f"features have {T}"
        )

    test_1min["fridge_pred"] = y_full

    print(f"[TCN PRED] Loading raw test from: {TEST_RAW_PATH}")
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

    print(f"[TCN PRED] Loading sample submission from: {SAMPLE_SUB_PATH}")
    sub = pd.read_csv(SAMPLE_SUB_PATH)

    if len(sub) != len(aligned):
        raise ValueError(
            f"Length mismatch: sample_submission has {len(sub)}, "
            f"aligned has {len(aligned)}"
        )
    if "fridge" not in sub.columns:
        raise ValueError("sample_submission.csv must contain 'fridge' column.")

    sub["fridge"] = aligned["fridge_pred"].values

    OUT_SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_SUB_PATH, index=False)
    print(f"[TCN PRED] Saved submission → {OUT_SUB_PATH}")


if __name__ == "__main__":
    main()
