import pandas as pd
from pathlib import Path


def resample_chain2(
    df: pd.DataFrame,
    time_col: str = "datetime",
    freq: str = "1min",
    group_col: str | None = None,
) -> pd.DataFrame:

    df = df.copy()

    # Normalize timestamps to UTC (handles +00, +01, DST, etc.)
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    if group_col is None:
        df = df.set_index(time_col).sort_index()
        df_resampled = df.resample(freq).ffill().reset_index()
        return df_resampled

    # Multiple homes: resample each home independently
    chunks = []
    for gid, g in df.groupby(group_col):
        g = g.sort_values(time_col).set_index(time_col)
        g_resampled = g.resample(freq).ffill()
        g_resampled[group_col] = gid
        chunks.append(g_resampled)

    df_resampled = pd.concat(chunks).reset_index()
    df_resampled = df_resampled.sort_values([group_col, time_col])
    return df_resampled


def clean_fridge_rule(
    df: pd.DataFrame,
    power_col: str = "power",
    fridge_col: str = "fridge",
    max_fridge: float = 450.0,
    ratio_threshold: float = 0.4,
) -> pd.DataFrame:
    df = df.copy()

    # Basic sanity clip
    df[power_col] = df[power_col].clip(lower=0)
    df[fridge_col] = df[fridge_col].clip(lower=0)

    mask = (df[fridge_col] > max_fridge) & (
        df[fridge_col] > ratio_threshold * df[power_col]
    )
    df.loc[mask, fridge_col] = max_fridge

    return df


if __name__ == "__main__":
    # Paths assuming project root is current working directory
    raw_dir = Path("data/raw")
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    # --- TRAIN ---
    train_path = raw_dir / "train.csv"
    train_df = pd.read_csv(train_path)

    train_1min = resample_chain2(
        train_df,
        time_col="datetime",
        freq="1min",
        group_col="home_id",  # multiple homes
    )
    train_1min = clean_fridge_rule(train_1min)  # apply 0.4 / 450 rule

    train_out = proc_dir / "train_1min.csv"
    train_1min.to_csv(train_out, index=False)

    # --- TEST ---
    test_path = raw_dir / "test.csv"
    test_df = pd.read_csv(test_path)

    # single test home, no home_id
    test_1min = resample_chain2(
        test_df,
        time_col="datetime",
        freq="1min",
        group_col=None,
    )

    test_out = proc_dir / "test_1min.csv"
    test_1min.to_csv(test_out, index=False)

    print(f"Saved resampled files:\n  {train_out}\n  {test_out}")
