from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.common.config import (
    PROJECT_ROOT,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
    RANDOM_SEED,
)


EXP_NAME = "tcn_simple"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = EXPORT_DIR / "model.pt"
ARTIFACT_PATH = EXPORT_DIR / "artifact.json"

WINDOW_LEN = 64  # minutes of history per sample
WINDOW_STRIDE = 3  # step in minutes between windows
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.1


class TCNBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, dilation: int):
        super().__init__()
        padding = dilation  # kernel_size=3
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
        # x: [B, C, L]
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
    def __init__(self, in_channels: int):
        super().__init__()
        c = in_channels
        self.block1 = TCNBlock(c, 64, dilation=1)
        self.block2 = TCNBlock(64, 64, dilation=2)
        self.block3 = TCNBlock(64, 64, dilation=4)
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, 64, 1]
            nn.Flatten(),  # [B, 64]
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, C, L]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        out = self.head(x)  # [B, 1]
        return out.squeeze(-1)


class WindowDataset(Dataset):

    def __init__(
        self,
        home_features: List[np.ndarray],
        home_targets: List[np.ndarray],
        windows: List[Tuple[int, int]],
        window_len: int,
    ):
        self.home_features = home_features
        self.home_targets = home_targets
        self.windows = windows
        self.window_len = window_len

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        h, start = self.windows[idx]
        Xh = self.home_features[h]  # [T_h, D]
        yh = self.home_targets[h]  # [T_h]

        end = start + self.window_len  # exclusive
        x_win = Xh[start:end]  # [L, D]
        y_t = yh[end - 1]  # target at window end

        # Convert to [C, L]
        x_win = torch.from_numpy(x_win.T.astype(np.float32))
        y_t = torch.tensor(float(y_t), dtype=torch.float32)
        return x_win, y_t


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_home_sequences(X_df: pd.DataFrame, y: np.ndarray, feature_cols: list[str]):
    if "home_id" not in X_df.columns:
        raise ValueError("tcn_simple expects 'home_id' column in train_features.")

    # Global normalization stats
    X_all = X_df[feature_cols].astype(np.float32)
    mean = X_all.mean(axis=0).values.astype(np.float32)
    std = X_all.std(axis=0).replace(0, 1.0).values.astype(np.float32)

    home_features: List[np.ndarray] = []
    home_targets: List[np.ndarray] = []
    windows: List[Tuple[int, int]] = []

    groups = X_df.groupby("home_id", sort=True)

    home_idx_map: dict[int, int] = {}
    for home_idx, (home_id, g) in enumerate(groups):
        home_idx_map[home_id] = home_idx
        idx = g.index.to_numpy()
        Xi = X_all.loc[idx].values  # [T_h, D]
        yi = y[idx]  # [T_h]

        Xi = (Xi - mean) / (std + 1e-6)

        T_h = Xi.shape[0]
        if T_h < WINDOW_LEN:
            continue

        home_features.append(Xi.astype(np.float32))
        home_targets.append(yi.astype(np.float32))

        # windows within this home
        for start in range(0, T_h - WINDOW_LEN + 1, WINDOW_STRIDE):
            windows.append((home_idx, start))

    return home_features, home_targets, windows, mean, std


def split_windows(windows: List[Tuple[int, int]], val_fraction: float = 0.1):
    n = len(windows)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = int(n * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_windows = [windows[i] for i in train_idx]
    val_windows = [windows[i] for i in val_idx]
    return train_windows, val_windows


def train_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * yb.size(0)
        n += yb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total_loss += float(loss.item()) * yb.size(0)
        n += yb.size(0)
    return total_loss / max(n, 1)


def main():
    set_seed(RANDOM_SEED)

    print(f"[TCN] Loading train features: {TRAIN_FEATURES_PATH}")
    X_df = pd.read_parquet(TRAIN_FEATURES_PATH)
    print(f"[TCN] Loading train target:   {TRAIN_TARGET_PATH}")
    target_df = pd.read_parquet(TRAIN_TARGET_PATH)

    if "fridge" in target_df.columns:
        target_col = "fridge"
    else:
        target_col = target_df.columns[0]
    y = target_df[target_col].astype(float).values

    # Determine feature columns (drop meta)
    drop_cols = [c for c in ["home_id", "datetime"] if c in X_df.columns]
    feature_cols = [c for c in X_df.columns if c not in drop_cols]

    print("\n[TCN] Data overview")
    print("-------------------")
    print("Rows       :", X_df.shape[0])
    print("Features   :", len(feature_cols))
    if "home_id" in X_df.columns:
        print("Homes      :", X_df["home_id"].nunique())

    # Build per-home sequences & windows
    print("\n[TCN] Building per-home sequences and windows...")
    home_feats, home_tgts, windows, mean, std = build_home_sequences(
        X_df, y, feature_cols
    )
    print(f"[TCN] Homes with data: {len(home_feats)}")
    print(f"[TCN] Total windows  : {len(windows)}")

    if len(windows) == 0:
        raise RuntimeError("No windows generated – check WINDOW_LEN / STRIDE.")

    train_windows, val_windows = split_windows(windows, val_fraction=0.1)
    print(f"[TCN] Train windows: {len(train_windows)}")
    print(f"[TCN] Val windows  : {len(val_windows)}")

    train_ds = WindowDataset(home_feats, home_tgts, train_windows, WINDOW_LEN)
    val_ds = WindowDataset(home_feats, home_tgts, val_windows, WINDOW_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TCN] Using device: {device}")

    model = SimpleTCN(in_channels=len(feature_cols)).to(device)
    criterion = nn.L1Loss()  # MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, device, criterion, optimizer)
        val_loss = eval_epoch(model, val_loader, device, criterion)

        print(
            f"[TCN] Epoch {epoch:02d} | train MAE {train_loss:.5f} | val MAE {val_loss:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[TCN] Saved model → {MODEL_PATH}")

    # Save artifact
    artifact = {
        "experiment": EXP_NAME,
        "feature_cols": feature_cols,
        "window_len": WINDOW_LEN,
        "window_stride": WINDOW_STRIDE,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "val_mae": float(best_val),
    }
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[TCN] Saved artifact → {ARTIFACT_PATH}")
    print("[TCN] Done.")


if __name__ == "__main__":
    main()
