import pandas as pd


def resample_chain2(
    df: pd.DataFrame,
    time_col: str = "datetime",
    freq: str = "1min",
    group_col: str | None = None,
) -> pd.DataFrame:
    df = df.copy()

    # Normalize timestamps to UTC to avoid DST / mixed-offset issues
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    if group_col is None:
        # Single time series
        df = df.set_index(time_col).sort_index()
        df_resampled = df.resample(freq).ffill()
        df_resampled = df_resampled.reset_index()  # bring datetime back as a column
        return df_resampled

    # Multiple homes (or other groups): resample independently per group
    resampled_chunks = []

    for gid, g in df.groupby(group_col):
        g = g.sort_values(time_col)
        g = g.set_index(time_col)

        g_resampled = g.resample(freq).ffill()
        g_resampled[group_col] = gid  # keep the home_id on resampled rows

        resampled_chunks.append(g_resampled)

    df_resampled = pd.concat(resampled_chunks)
    df_resampled = df_resampled.reset_index().sort_values([group_col, time_col])

    return df_resampled


if __name__ == "__main__":
    # --- Read raw CSVs ---
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # --- Resample train per home_id (multiple homes) ---
    train_1min = resample_chain2(
        train_df,
        time_col="datetime",
        freq="1min",
        group_col="home_id",  # train has multiple homes
    )

    # --- Resample test as a single home (no home_id column in test.csv) ---
    test_1min = resample_chain2(
        test_df,
        time_col="datetime",
        freq="1min",
        group_col=None,  # single test home
    )

    # --- Save outputs ---
    train_1min.to_csv("train_1min.csv", index=False)
    test_1min.to_csv("test_1min.csv", index=False)

    print("Saved resampled files: train_1min.csv, test_1min.csv")
