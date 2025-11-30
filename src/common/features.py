# src/common/features.py

from __future__ import annotations

import numpy as np
import pandas as pd

# Lags on 1-minute grid
LAG_MINUTES = [1, 2, 3, 5, 10, 15, 30, 45, 60, 90, 120, 180]


def _ensure_datetime(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    """Ensure datetime column is UTC and dataframe is sorted (by home if present)."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    if "home_id" in df.columns:
        df = df.sort_values(["home_id", time_col]).reset_index(drop=True)
    else:
        df = df.sort_values(time_col).reset_index(drop=True)
    return df


def _group_keys(df: pd.DataFrame):
    return ["home_id"] if "home_id" in df.columns else None


def _gtransform(df: pd.DataFrame, col: str, fn):
    g = _group_keys(df)
    if g:
        return df.groupby(g, observed=True)[col].transform(fn)
    return fn(df[col])


def _gshift(df: pd.DataFrame, col: str, lag: int):
    g = _group_keys(df)
    if g:
        return df.groupby(g, observed=True)[col].shift(lag)
    return df[col].shift(lag)


def _groll(df: pd.DataFrame, col: str, window: int, fn: str, minp: int = 1):
    g = _group_keys(df)
    if g:
        return df.groupby(g, observed=True)[col].transform(
            lambda s: getattr(s.rolling(window, min_periods=minp), fn)()
        )
    return getattr(df[col].rolling(window, min_periods=minp), fn)()


def _add_time_features(df: pd.DataFrame) -> None:
    dt = df["datetime"]
    df["hour"] = dt.dt.hour.astype(np.int16)
    df["dow"] = dt.dt.dayofweek.astype(np.int16)
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)


def _clean_fridge_target(df: pd.DataFrame) -> pd.Series:
    """
    Clean fridge target per home:
    - clip to [0, 350]
    - 5-min rolling median
    - 3-min rolling mean
    - OFF threshold at 10 W
    - forward-fill within home, then 0
    """

    def clean_one(s: pd.Series) -> pd.Series:
        s = s.astype(float).clip(0.0, 350.0)
        s = s.rolling(5, min_periods=1).median()
        s = s.rolling(3, min_periods=1).mean()
        s = s.where(s >= 10.0, 0.0)
        s = s.ffill().fillna(0.0)
        return s

    g = _group_keys(df)
    if g:
        return df.groupby(g, observed=True)["fridge"].transform(clean_one)
    return clean_one(df["fridge"])


def _fft_features(
    df: pd.DataFrame, power_col: str = "power", window: int = 60
) -> pd.DataFrame:
    """
    Simple FFT features on a 60-minute window:
    - energy
    - dominant frequency index
    - high/low magnitude ratio

    Computed per home to avoid leakage.
    """
    df = df.copy()
    g = _group_keys(df)

    energy = np.full(len(df), np.nan, dtype=float)
    domfreq = np.full(len(df), np.nan, dtype=float)
    ratio = np.full(len(df), np.nan, dtype=float)

    if g:
        groups = df.groupby(g, observed=True)
        for _, idx in groups.indices.items():
            vals = df.loc[idx, power_col].to_numpy(dtype=float)
            n = len(vals)
            for i in range(n):
                if i < window:
                    continue
                seg = vals[i - window : i]
                spec = np.fft.rfft(seg)
                mag = np.abs(spec)
                if mag.size == 0:
                    continue
                energy[idx[i]] = float(np.sum(mag))
                # dominant frequency (skip DC)
                if mag.size > 1:
                    dfreq = int(np.argmax(mag[1:]) + 1)
                else:
                    dfreq = 0
                domfreq[idx[i]] = dfreq
                mid = mag.size // 2
                lo = float(np.sum(mag[:mid])) if mid > 0 else 0.0
                hi = float(np.sum(mag[mid:])) if mid < mag.size else 0.0
                ratio[idx[i]] = hi / (lo + 1e-6)
    else:
        vals = df[power_col].to_numpy(dtype=float)
        n = len(vals)
        for i in range(n):
            if i < window:
                continue
            seg = vals[i - window : i]
            spec = np.fft.rfft(seg)
            mag = np.abs(spec)
            if mag.size == 0:
                continue
            energy[i] = float(np.sum(mag))
            if mag.size > 1:
                dfreq = int(np.argmax(mag[1:]) + 1)
            else:
                dfreq = 0
            domfreq[i] = dfreq
            mid = mag.size // 2
            lo = float(np.sum(mag[:mid])) if mid > 0 else 0.0
            hi = float(np.sum(mag[mid:])) if mid < mag.size else 0.0
            ratio[i] = hi / (lo + 1e-6)

    return pd.DataFrame(
        {
            "fft60_energy": energy,
            "fft60_domfreq": domfreq,
            "fft60_lowhigh_ratio": ratio,
        },
        index=df.index,
    )


# ============================
# Main entry
# ============================


def build_features(df: pd.DataFrame, is_train: bool):
    """
    Build 1-minute features.

    Train df columns: datetime, home_id, power, fridge
    Test  df columns: datetime, power (no home_id, no fridge)

    Returns:
        X (features with meta columns),
        y (target Series) or None for test.
    """
    df = _ensure_datetime(df)
    df["power"] = df["power"].astype(float)

    # Base smoothed signals
    df["p_ewm_03"] = _gtransform(
        df, "power", lambda s: s.ewm(span=3, adjust=False, min_periods=1).mean()
    )
    df["p_roll3"] = _groll(df, "power", 3, "mean")
    df["p_roll5"] = _groll(df, "power", 5, "mean")
    df["p_roll15"] = _groll(df, "power", 15, "mean")
    df["p_roll30"] = _groll(df, "power", 30, "mean")
    df["p_roll60"] = _groll(df, "power", 60, "mean")

    # Lags
    for L in LAG_MINUTES:
        df[f"power_lag{L}"] = _gshift(df, "power", L)

    # Rolling stats
    df["roll3_mean"] = df["p_roll3"]
    df["roll5_mean"] = df["p_roll5"]
    df["roll15_mean"] = df["p_roll15"]
    df["roll30_mean"] = df["p_roll30"]

    df["roll5_std"] = _groll(df, "power", 5, "std", minp=2)
    df["roll15_std"] = _groll(df, "power", 15, "std", minp=2)

    df["roll15_min"] = _groll(df, "power", 15, "min")
    df["roll15_max"] = _groll(df, "power", 15, "max")
    df["roll15_range"] = df["roll15_max"] - df["roll15_min"]

    df["roll15_q10"] = _gtransform(
        df, "power", lambda s: s.rolling(15, min_periods=1).quantile(0.10)
    )
    df["roll15_q90"] = _gtransform(
        df, "power", lambda s: s.rolling(15, min_periods=1).quantile(0.90)
    )
    df["roll15_iqr"] = df["roll15_q90"] - df["roll15_q10"]

    # Baselines & high-pass
    df["p_minus_base15"] = df["power"] - df["p_roll15"]
    df["p_minus_base30"] = df["power"] - df["p_roll30"]
    df["p_highpass"] = df["power"] - df["p_roll60"]

    # FFT features (per home, 60-minute window)
    fft_df = _fft_features(df, power_col="power", window=60)
    df = pd.concat([df, fft_df], axis=1)

    # Edge / derivative features
    df["dp1"] = _gtransform(df, "power", lambda s: s.diff(1))
    df["dp3"] = _gtransform(df, "power", lambda s: s.diff(3))

    # Time features
    _add_time_features(df)

    # Target
    y = None
    if is_train:
        if "fridge" not in df.columns:
            raise ValueError("Training data must include 'fridge' column.")
        y = _clean_fridge_target(df)

    # Assemble X
    meta_cols = ["datetime"] + (["home_id"] if "home_id" in df.columns else [])
    drop_cols = {"fridge"}
    feature_cols = [c for c in df.columns if c not in drop_cols.union(meta_cols)]

    X = df[feature_cols].copy()

    # Add meta for LOHO and alignment
    for col in meta_cols:
        X[col] = df[col]

    # Fill NaNs in features causally (ffill within home, then zeros)
    if "home_id" in X.columns:
        feats_only = [c for c in feature_cols]
        X[feats_only] = (
            X.groupby("home_id", observed=True)[feats_only]
            .transform(lambda s: s.ffill())
            .fillna(0.0)
        )
    else:
        X[feature_cols] = X[feature_cols].ffill().fillna(0.0)

    return (X, y if is_train else None)
