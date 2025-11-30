# src/common/evaluation.py

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_1d_array(y) -> np.ndarray:
    if isinstance(y, (pd.Series, pd.DataFrame)):
        arr = y.values
    else:
        arr = np.asarray(y)
    return arr.reshape(-1)


def mae(y_true, y_pred) -> float:
    yt = _to_1d_array(y_true)
    yp = _to_1d_array(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt = _to_1d_array(y_true)
    yp = _to_1d_array(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def nde(y_true, y_pred) -> float:
    yt = _to_1d_array(y_true)
    yp = _to_1d_array(y_pred)
    num = np.sum((yt - yp) ** 2)
    den = np.sum(yt**2) + 1e-8
    return float(np.sqrt(num / den))


def sae(y_true, y_pred) -> float:
    yt = _to_1d_array(y_true)
    yp = _to_1d_array(y_pred)
    num = np.sum(np.abs(yt - yp))
    den = np.sum(np.abs(yt)) + 1e-8
    return float(num / den)


def loho_split(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    home_col: str = "home_id",
):
    if home_col not in X.columns:
        raise ValueError(f"loho_split: '{home_col}' column missing from X.")

    homes = X[home_col].unique()

    for home in homes:
        valid_mask = X[home_col] == home
        train_mask = ~valid_mask

        X_train = X[train_mask]
        X_valid = X[valid_mask]
        y_train = y[train_mask]
        y_valid = y[valid_mask]

        yield home, X_train, X_valid, y_train, y_valid
