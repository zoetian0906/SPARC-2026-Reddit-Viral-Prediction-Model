"""
inspect_table1.py — RECON ONLY (Phase 0).

Reads Table 1 (teammate's RAG model-rules parquet) from HuggingFace and prints a
structural profile so we understand it before designing anything. This script does
NOT write, cache, or commit the parquet — it only reads and prints.

Usage:
    python scripts/inspect_table1.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Table 1 is a gated dataset file, so the raw resolve URL returns 401 without auth.
# We use hf_hub_download (same pattern as src/build_model_ready.py), which resolves
# credentials from HF_TOKEN or the local huggingface-cli login cache.
# Reference URL:
#   https://huggingface.co/datasets/SPARC2026Reddit/MessyData-ZT/resolve/main/For%20RAG/rag_model_rules_FINAL.parquet
REPO_ID = "SPARC2026Reddit/MessyData-ZT"
FILENAME = "For RAG/rag_model_rules_FINAL.parquet"


def load_table(repo_id: str = REPO_ID, filename: str = FILENAME) -> pd.DataFrame:
    """Download Table 1 from HuggingFace (authenticated) and read it with pandas.

    hf_hub_download caches into the HF cache dir (not the repo); nothing is written
    into the working tree and nothing is committed.
    """
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return pd.read_parquet(local_path)


def looks_like_metric(name: str) -> bool:
    """True if a column name suggests a metric: r2, rmse, sample_size, or n."""
    low = name.lower()
    if low in {"n"} or low.endswith("_n") or low.startswith("n_"):
        return True
    return any(tok in low for tok in ("r2", "rmse", "sample_size"))


def looks_like_shap(name: str) -> bool:
    """True if a column name suggests a SHAP value."""
    return "shap" in name.lower()


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 60)

    df = load_table()

    print("=" * 80)
    print("TABLE 1 — RECON PROFILE")
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

    # 5. Non-numeric columns: nunique + full value list if < 30 unique
    print("\n[5] NON-NUMERIC COLUMNS (unique counts; full list if < 30):")
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not non_numeric:
        print("    (none)")
    for c in non_numeric:
        nun = df[c].nunique(dropna=True)
        print(f"    - {c}: {nun} unique")
        if nun < 30:
            print(f"        values: {sorted(map(str, df[c].dropna().unique()))}")

    # 6. Metric-like columns: min/max/mean/median/deciles
    print("\n[6] METRIC-LIKE COLUMNS (name suggests r2 / rmse / sample_size / n):")
    metric_cols = [
        c for c in df.columns
        if looks_like_metric(c) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metric_cols:
        print("    (none matched)")
    deciles = [i / 10 for i in range(1, 10)]
    for c in metric_cols:
        s = df[c].dropna()
        print(f"    - {c}:")
        print(f"        min={s.min():.6g}  max={s.max():.6g}  "
              f"mean={s.mean():.6g}  median={s.median():.6g}")
        dq = s.quantile(deciles)
        print("        deciles (10%..90%): "
              + ", ".join(f"{int(q*100)}%={v:.6g}" for q, v in dq.items()))

    # 7. r2 buckets
    print("\n[7] R2 BUCKET COUNTS:")
    r2_cols = [
        c for c in df.columns
        if "r2" in c.lower() and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not r2_cols:
        print("    (no numeric r2 column found)")
    bins = [-np.inf, 0.0, 0.05, 0.10, 0.20, np.inf]
    labels = ["below 0", "0 to 0.05", "0.05 to 0.10", "0.10 to 0.20", "above 0.20"]
    for c in r2_cols:
        buckets = pd.cut(df[c], bins=bins, labels=labels, right=False)
        counts = buckets.value_counts().reindex(labels).fillna(0).astype(int)
        print(f"    - {c}:")
        for label in labels:
            print(f"        {label:14} {counts[label]}")
        na = int(df[c].isna().sum())
        if na:
            print(f"        {'(null)':14} {na}")

    # 8. SHAP-looking columns
    print("\n[8] SHAP-LOOKING COLUMNS + NAMING:")
    shap_cols = [c for c in df.columns if looks_like_shap(c)]
    if not shap_cols:
        print("    (none matched 'shap')")
    else:
        print(f"    count: {len(shap_cols)}")
        for c in shap_cols:
            print(f"    - {c}  ({df[c].dtype})")

    print("\n" + "=" * 80)
    print("END RECON")
    print("=" * 80)


if __name__ == "__main__":
    main()
