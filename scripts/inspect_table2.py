"""
inspect_table2.py — RECON ONLY (Phase C).

Reads Table 2 (teammate's predictions parquet) from HuggingFace and prints a
structural profile. Does NOT write, cache into the repo, or commit the parquet.

Usage:
    python scripts/inspect_table2.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Table 2 is in the same gated dataset repo; the raw resolve URL 401s, so use
# hf_hub_download (resolves credentials from HF_TOKEN or the huggingface-cli cache).
REPO_ID = "SPARC2026Reddit/MessyData-ZT"
FILENAME = "For RAG/rag_predictions_FINAL.parquet"


def load_table(repo_id: str = REPO_ID, filename: str = FILENAME) -> pd.DataFrame:
    """Download Table 2 from HuggingFace (authenticated) and read it with pandas."""
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return pd.read_parquet(local_path)


def looks_like_metric(name: str) -> bool:
    """True if a numeric column name suggests a prediction/score/hour/day."""
    low = name.lower()
    return any(tok in low for tok in ("pred", "score", "hour", "day"))


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 60)

    df = load_table()

    print("=" * 80)
    print("TABLE 2 — RECON PROFILE")
    print("=" * 80)

    # 1. Shape
    print("\n[1] SHAPE (rows, columns):")
    print(f"    {df.shape}")

    # 2. Columns with dtypes
    print("\n[2] COLUMNS + DTYPES:")
    for name, dt in df.dtypes.items():
        print(f"    {name:40} {dt}")

    # 3. head(10)
    print("\n[3] HEAD(10):")
    print(df.head(10).to_string())

    # 4. Null count per column
    print("\n[4] NULL COUNT PER COLUMN:")
    print(df.isna().sum().to_string())

    # 5. Non-numeric columns: unique count + full list if < 30
    print("\n[5] NON-NUMERIC COLUMNS (unique counts; full list if < 30):")
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not non_numeric:
        print("    (none)")
    for c in non_numeric:
        nun = df[c].nunique(dropna=True)
        print(f"    - {c}: {nun} unique")
        if nun < 30:
            print(f"        values: {sorted(map(str, df[c].dropna().unique()))}")

    # 6. Prediction/score/hour/day numeric columns: min/max/mean/median
    print("\n[6] PREDICTION/SCORE/HOUR/DAY NUMERIC COLUMNS:")
    metric_cols = [
        c for c in df.columns
        if looks_like_metric(c) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols:
        print("    (none matched)")
    for c in metric_cols:
        s = df[c].dropna()
        print(f"    - {c}: min={s.min():.6g}  max={s.max():.6g}  "
              f"mean={s.mean():.6g}  median={s.median():.6g}")

    # 7. Hour column: full sorted unique values (confirm 0-23)
    print("\n[7] HOUR COLUMN(S) — sorted unique values:")
    hour_cols = [c for c in df.columns if "hour" in c.lower()]
    if not hour_cols:
        print("    (no hour column found)")
    for c in hour_cols:
        uniq = sorted(df[c].dropna().unique().tolist())
        print(f"    - {c}: {uniq}")
        nums = [u for u in uniq if isinstance(u, (int, float, np.integer, np.floating))]
        if nums:
            print(f"        range check: min={min(nums)} max={max(nums)} "
                  f"(0-23 expected)")

    # 8. Day column: full list of unique values with exact spelling/encoding
    print("\n[8] DAY COLUMN(S) — unique values (exact spelling/encoding):")
    day_cols = [c for c in df.columns if "day" in c.lower()]
    if not day_cols:
        print("    (no day column found)")
    for c in day_cols:
        uniq = df[c].dropna().unique().tolist()
        try:
            uniq_sorted = sorted(uniq)
        except TypeError:
            uniq_sorted = uniq
        print(f"    - {c} (dtype={df[c].dtype}): {uniq_sorted!r}")

    print("\n" + "=" * 80)
    print("END RECON")
    print("=" * 80)


if __name__ == "__main__":
    main()
