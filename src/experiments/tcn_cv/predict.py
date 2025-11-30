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


# Reuse same architecture as in train.py
class TCNBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, dilation: int):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            channels_in, channels_out, kernel_size=3, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            channels_out,
            channels_out,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(channels_out)
        if channels_in != channels_out:
            self.downsample = nn.Conv1d(channels_in, channels_out, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class SimpleTCN(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.2):
        super().__init__()
        c = in_channels
        self.block1 = TCNBlock(c, 64, dilation=1)
        self.block2 = TCNBlock(64, 64, dilation=2)
        self.block3 = TCNBlock(64, 64, dilation=4)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        out = self.head(x)
        return out.squeeze(-1)


EXP_NAME = "tcn_cv"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
MODEL_PATH = EXPORT_DIR / "model_final.pt"
ARTIFACT_PATH = EXPORT_DIR / "artifact.json"

SAMPLE_SUB_PATH = PROJECT_ROOT / "submission" / "sample_submission.csv"
OUT_SUB_PATH = PROJECT_ROOT / "submission" / "submission_tcn_cv.csv"

BATCH_SIZE = 256


def load_artifact():
    with open(ARTIFACT_PATH, "r") as f:
        art = json.load(f)
    feature_cols = art["feature_cols"]
    window_len = int(art["window_len"])
    mean = np.array(art["mean"], dtype=np.float32)
    std = np.array(art["std"], dtype=np.float32)
    return feature_cols, window_len, mean, std


def main():
    print(f"[TCN-CV PRED] Loading artifact from: {ARTIFACT_PATH}")
    feature_cols, window_len, mean, std = load_artifact()

    print(f"[TCN-CV PRED] Loading test features from: {TEST_FEATURES_PATH}")
    X_df = pd.read_parquet(TEST_FEATURES_PATH)
    X_df["datetime"] = pd.to_datetime(X_df["datetime"], utc=True)
    X_df = X_df.sort_values("datetime").reset_index(drop=True)

    # Select & normalize features
    X_mat = X_df[feature_cols].astype(np.float32).values
    X_mat = (X_mat - mean) / (std + 1e-6)

    T, D = X_mat.shape
    print(f"[TCN-CV PRED] Test shape (T, D): {T}, {D}")
    if T < window_len:
        raise RuntimeError("Test sequence shorter than window_len.")

    # Sliding windows with stride 1 for prediction
    starts = np.arange(0, T - window_len + 1, 1, dtype=int)
    n_windows = len(starts)
    print(f"[TCN-CV PRED] Number of windows: {n_windows}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TCN-CV PRED] Using device: {device}")

    model = SimpleTCN(in_channels=D, dropout=0.0).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preds_window = np.zeros(n_windows, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_windows, BATCH_SIZE):
            batch_starts = starts[i : i + BATCH_SIZE]
            batch_x = []
            for s in batch_starts:
                slice_x = X_mat[s : s + window_len]  # [L, D]
                slice_x = slice_x.T  # [D, L]
                batch_x.append(slice_x)
            xb = torch.from_numpy(np.stack(batch_x, axis=0)).to(device)  # [B, D, L]
            pb = model(xb).cpu().numpy().astype(np.float32)
            preds_window[i : i + len(batch_starts)] = pb

    # Reconstruct per-timestep prediction:
    # window i predicts for timestamp t = start + window_len - 1
    y_full = np.zeros(T, dtype=np.float32)
    # first window prediction for the warm-up part
    y_full[: window_len - 1] = preds_window[0]
    # from t = window_len-1 onward, each t gets exactly one window prediction
    y_full[window_len - 1 :] = preds_window
    y_full = np.clip(y_full, 0.0, None)

    print(
        "[TCN-CV PRED] Pred range on 1-min grid:",
        float(y_full.min()),
        "→",
        float(y_full.max()),
    )

    # Attach to 1-min timestamps and align to raw test
    print(f"[TCN-CV PRED] Loading 1-min test from: {TEST_1MIN_PATH}")
    test_1min = pd.read_csv(TEST_1MIN_PATH)
    test_1min["datetime"] = pd.to_datetime(test_1min["datetime"], utc=True)
    test_1min = test_1min.sort_values("datetime").reset_index(drop=True)

    if len(test_1min) != T:
        raise ValueError(
            f"Length mismatch: test_1min has {len(test_1min)}, " f"features have {T}"
        )

    test_1min["fridge_pred"] = y_full

    print(f"[TCN-CV PRED] Loading raw test from: {TEST_RAW_PATH}")
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

    print(f"[TCN-CV PRED] Loading sample submission from: {SAMPLE_SUB_PATH}")
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
    print(f"[TCN-CV PRED] Saved submission → {OUT_SUB_PATH}")


if __name__ == "__main__":
    main()
