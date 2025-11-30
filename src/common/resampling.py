import pandas as pd
from pathlib import Path
from src.common.config import (
    TRAIN_RAW_PATH,
    TEST_RAW_PATH,
    TRAIN_1MIN_PATH,
    TEST_1MIN_PATH,
    RESAMPLE_FREQ,
    FRIDGE_MAX_W,
    FRIDGE_RATIO_THRESHOLD,
)


def resample_chain2(df, time_col="datetime", freq="1min", group_col=None):
    df = df.copy()

    # Normalize timestamps
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # Single test home (no home_id)
    if group_col is None:
        df = df.sort_values(time_col).set_index(time_col)
        return df.resample(freq).ffill().reset_index()

    # Multiple homes → resample separately
    chunks = []
    for gid, g in df.groupby(group_col):
        g = g.sort_values(time_col).set_index(time_col)
        g_resampled = g.resample(freq).ffill()
        g_resampled[group_col] = gid
        chunks.append(g_resampled)

    out = pd.concat(chunks).reset_index()
    return out.sort_values([group_col, time_col])


def clean_fridge_rule(
    df,
    power_col="power",
    fridge_col="fridge",
    max_fridge=FRIDGE_MAX_W,
    ratio_threshold=FRIDGE_RATIO_THRESHOLD,
):

    df = df.copy()
    df[power_col] = df[power_col].clip(lower=0)
    df[fridge_col] = df[fridge_col].clip(lower=0)

    mask = (df[fridge_col] > max_fridge) & (
        df[fridge_col] > ratio_threshold * df[power_col]
    )
    df.loc[mask, fridge_col] = max_fridge

    return df


if __name__ == "__main__":

    print("Resampling train/test...")

    # --- TRAIN ---
    train_df = pd.read_csv(TRAIN_RAW_PATH)

    train_1min = resample_chain2(
        train_df, time_col="datetime", freq=RESAMPLE_FREQ, group_col="home_id"
    )

    train_1min = clean_fridge_rule(train_1min)
    train_1min.to_csv(TRAIN_1MIN_PATH, index=False)

    # --- TEST ---
    test_df = pd.read_csv(TEST_RAW_PATH)

    test_1min = resample_chain2(
        test_df, time_col="datetime", freq=RESAMPLE_FREQ, group_col=None
    )

    test_1min.to_csv(TEST_1MIN_PATH, index=False)

    print("Saved:")
    print(" -", TRAIN_1MIN_PATH)
    print(" -", TEST_1MIN_PATH)
