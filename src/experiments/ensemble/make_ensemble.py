from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.config import PROJECT_ROOT

SUBMISSION_DIR = PROJECT_ROOT / "submission"

CANDIDATES = [
    SUBMISSION_DIR / "submission_lightgbm_simple.csv",
    SUBMISSION_DIR / "submission_xgboost_simple.csv",
    SUBMISSION_DIR / "submission_lightgbm_loho.csv",
]

OUT_PATH = SUBMISSION_DIR / "submission_ensemble.csv"


def main():
    subs = []
    for path in CANDIDATES:
        if path.exists():
            print(f"[ensemble] Using: {path}")
            subs.append(pd.read_csv(path))
        else:
            print(f"[ensemble] Skipping (not found): {path}")

    if len(subs) == 0:
        raise RuntimeError("No submission files found to ensemble.")
    base = subs[0]
    if "fridge" not in base.columns:
        raise ValueError("Submissions must contain 'fridge' column.")
    if "id" not in base.columns:
        raise ValueError("Submissions must contain 'id' column.")

    for s in subs[1:]:
        if len(s) != len(base):
            raise ValueError("Submissions have different lengths.")
        if not (s["id"].values == base["id"].values).all():
            raise ValueError("Submissions have different 'id' ordering.")

    out = base[["id"]].copy()
    fridge_stack = [s["fridge"].astype(float) for s in subs]
    out["fridge"] = sum(fridge_stack) / len(fridge_stack)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[ensemble] Saved ensemble submission → {OUT_PATH}")


if __name__ == "__main__":
    main()
