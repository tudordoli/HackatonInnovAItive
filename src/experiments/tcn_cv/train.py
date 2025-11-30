from __future__ import annotations

import json
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
from src.common.evaluation import mae


EXP_NAME = "tcn_cv"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = EXPORT_DIR / "model_final.pt"
ARTIFACT_PATH = EXPORT_DIR / "artifact.json"
CV_METRICS_PATH = EXPORT_DIR / "cv_metrics.json"

WINDOW_LEN = 32  # minutes of history per sample (reduced from 64)
WINDOW_STRIDE_TRAIN = 3  # stride for training windows
BATCH_SIZE = 256
EPOCHS = 6  # per-fold training epochs
LR = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2
N_FOLDS = 4


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


def build_per_home_arrays(
    X_df: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], np.ndarray, np.ndarray]:
    if "home_id" not in X_df.columns:
        raise ValueError("tcn_cv requires 'home_id' column in train_features.")

    X_df = X_df.sort_values(["home_id", "datetime"]).reset_index(drop=True)

    X_all = X_df[feature_cols].astype(np.float32)
    # global normalization stats
    mean = X_all.mean(axis=0).values.astype(np.float32)
    std = X_all.std(axis=0).replace(0, 1.0).values.astype(np.float32)

    home_features: List[np.ndarray] = []
    home_targets: List[np.ndarray] = []
    home_ids_sorted: List[str] = []

    for home_id, g in X_df.groupby("home_id", sort=True):
        idx = g.index.to_numpy()
        Xi = X_all.loc[idx].values  # [T_h, D]
        yi = y[idx]  # [T_h]

        Xi = (Xi - mean) / (std + 1e-6)

        home_features.append(Xi.astype(np.float32))
        home_targets.append(yi.astype(np.float32))
        home_ids_sorted.append(str(home_id))

    return home_features, home_targets, home_ids_sorted, mean, std


def build_windows_for_home(
    T_h: int,
    window_len: int,
    stride: int,
) -> list[int]:
    if T_h < window_len:
        return []
    return list(range(0, T_h - window_len + 1, stride))


def build_windows_for_home_set(
    selected_home_indices: list[int],
    home_features: list[np.ndarray],
    window_len: int,
    stride: int,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for h in selected_home_indices:
        T_h = home_features[h].shape[0]
        starts = build_windows_for_home(T_h, window_len, stride)
        for s in starts:
            windows.append((h, s))
    return windows


def make_home_folds(home_ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    ids = np.array(home_ids, dtype=object)
    rng.shuffle(ids)
    folds: list[list[str]] = []
    splits = np.array_split(ids, n_folds)
    for arr in splits:
        folds.append([str(x) for x in arr.tolist()])
    return folds


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.item()) * yb.size(0)
        n += yb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    all_true = []
    all_pred = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total_loss += float(loss.item()) * yb.size(0)
        n += yb.size(0)
        all_true.append(yb.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
    if n == 0:
        return 0.0, np.array([]), np.array([])
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return total_loss / n, y_true, y_pred


def main():
    set_seed(RANDOM_SEED)

    print(f"[TCN-CV] Loading train features: {TRAIN_FEATURES_PATH}")
    X_df = pd.read_parquet(TRAIN_FEATURES_PATH)

    print(f"[TCN-CV] Loading train target:   {TRAIN_TARGET_PATH}")
    target_df = pd.read_parquet(TRAIN_TARGET_PATH)
    if "fridge" in target_df.columns:
        target_col = "fridge"
    else:
        target_col = target_df.columns[0]
    y = target_df[target_col].astype(float).values

    drop_cols = [c for c in ["home_id", "datetime"] if c in X_df.columns]
    feature_cols = [c for c in X_df.columns if c not in drop_cols]

    print("\n[TCN-CV] Data overview")
    print("----------------------")
    print("Rows       :", X_df.shape[0])
    print("Features   :", len(feature_cols))
    if "home_id" in X_df.columns:
        print("Homes      :", X_df["home_id"].nunique())
    else:
        raise ValueError("TCN-CV requires 'home_id' for proper cross-validation.")

    # Build per-home arrays + normalization
    print("\n[TCN-CV] Building per-home arrays and normalization…")
    home_feats, home_tgts, home_ids_sorted, mean, std = build_per_home_arrays(
        X_df, y, feature_cols
    )
    n_homes = len(home_ids_sorted)
    print(f"[TCN-CV] Homes with data: {n_homes}")

    # Map home_id (string) -> index
    home_id_to_idx = {hid: i for i, hid in enumerate(home_ids_sorted)}

    # Make folds on homes
    folds_home_ids = make_home_folds(home_ids_sorted, N_FOLDS, seed=RANDOM_SEED)
    print("\n[TCN-CV] Home folds:")
    for i, f in enumerate(folds_home_ids, 1):
        print(f"  Fold {i}: {len(f)} homes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TCN-CV] Using device: {device}")

    cv_records = []

    for fold_idx, val_home_ids in enumerate(folds_home_ids, 1):
        val_home_indices = [home_id_to_idx[h] for h in val_home_ids]
        train_home_indices = [i for i in range(n_homes) if i not in val_home_indices]

        print(f"\n[TCN-CV] Fold {fold_idx}/{N_FOLDS}")
        print("  Train homes:", len(train_home_indices))
        print("  Val homes  :", len(val_home_indices))

        train_windows = build_windows_for_home_set(
            train_home_indices, home_feats, WINDOW_LEN, WINDOW_STRIDE_TRAIN
        )
        val_windows = build_windows_for_home_set(
            val_home_indices, home_feats, WINDOW_LEN, stride=1  # dense val
        )

        print("  Train windows:", len(train_windows))
        print("  Val windows  :", len(val_windows))

        if len(train_windows) == 0 or len(val_windows) == 0:
            print("  [WARN] Empty windows in this fold, skipping.")
            continue

        train_ds = WindowDataset(home_feats, home_tgts, train_windows, WINDOW_LEN)
        val_ds = WindowDataset(home_feats, home_tgts, val_windows, WINDOW_LEN)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        model = SimpleTCN(in_channels=len(feature_cols)).to(device)
        criterion = nn.L1Loss()  # MAE
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, y_true, y_pred = eval_epoch(model, val_loader, device, criterion)
            val_mae = mae(y_true, y_pred) if y_true.size > 0 else float("inf")

            print(
                f"  [Fold {fold_idx}] Epoch {epoch:02d} "
                f"| train MAE {train_loss:.5f} | val MAE {val_mae:.5f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final val metrics for this fold
        _, y_true, y_pred = eval_epoch(model, val_loader, device, criterion)
        fold_mae = mae(y_true, y_pred) if y_true.size > 0 else float("inf")

        print(f"  [Fold {fold_idx}] Best val MAE: {fold_mae:.5f}")
        cv_records.append(
            {
                "fold": fold_idx,
                "val_mae": float(fold_mae),
                "n_val_samples": int(len(y_true)),
                "val_homes": val_home_ids,  # list of strings
            }
        )

    # Save CV metrics
    with open(CV_METRICS_PATH, "w") as f:
        json.dump(cv_records, f, indent=2)
    print(f"\n[TCN-CV] Saved CV metrics → {CV_METRICS_PATH}")

    print("\n[TCN-CV] Training final model on ALL homes…")

    all_home_indices = list(range(n_homes))
    all_windows = build_windows_for_home_set(
        all_home_indices, home_feats, WINDOW_LEN, WINDOW_STRIDE_TRAIN
    )
    print(f"[TCN-CV] Total windows for final model: {len(all_windows)}")

    all_ds = WindowDataset(home_feats, home_tgts, all_windows, WINDOW_LEN)
    all_loader = DataLoader(all_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = SimpleTCN(in_channels=len(feature_cols)).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, all_loader, device, criterion, optimizer)
        print(f"[TCN-CV] Final model epoch {epoch:02d} | train MAE {train_loss:.5f}")

    # Save model
    state_to_save = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_to_save, MODEL_PATH)
    print(f"[TCN-CV] Saved final model → {MODEL_PATH}")

    # Save artifact with normalization + features
    artifact = {
        "experiment": EXP_NAME,
        "feature_cols": feature_cols,
        "window_len": WINDOW_LEN,
        "window_stride_train": WINDOW_STRIDE_TRAIN,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "cv_metrics_path": str(CV_METRICS_PATH),
    }
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[TCN-CV] Saved artifact → {ARTIFACT_PATH}")
    print("[TCN-CV] Done.")


if __name__ == "__main__":
    main()
