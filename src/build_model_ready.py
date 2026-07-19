"""
build_model_ready.py — Assemble the model-ready training table.

Purpose
-------
Create the `model_ready_dataset` table inside reddit_warehouse.db by LEFT JOINing,
on post_id, the target label with all engineered features:
    - post_labels       -> post_id, virality_index   (virality_index is the TARGET y)
    - post_features     -> engineered features (temporal, media, length), minus post_id
    - zoe_nlp_features  -> NLP features (VADER sentiment, NRC emotions, readability), minus post_id

Ownership
---------
Zoe mapped the initial join architecture. Kristin optimized the ingestion of the remote 
NLP feature parquet via huggingface_hub to handle repo authentication safely, maintaining
high-performance database-level joins.
"""

import os
import duckdb
from huggingface_hub import hf_hub_download

DB_PATH = "reddit_warehouse.db"

def build_model_ready(db_path: str = DB_PATH):
    print("STATUS: Connecting to local DuckDB warehouse...")
    con = duckdb.connect(db_path)

    # ── 1. Securely Fetch NLP Features from Hugging Face ─────────────────────
    repo_id = "SPARC2026Reddit/MessyData-ZT"
    
    # UPDATED: Pointing to the new 96k row file in the data folder
    filename = "data/train-00000-of-00001.parquet"
    
    print(f"STATUS: Authenticating and downloading {filename} from Hugging Face...")
    try:
        # Resolves credentials from HF_TOKEN env var or local huggingface-cli login cache
        local_parquet_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        print(f"SUCCESS: Feature file cached locally at: {local_parquet_path}")
    except Exception as e:
        print(f"ERROR: Authentication or download failed. Ensure you are logged into HF. Reason: {e}")
        con.close()
        return

    # ── 2. Build the Unified Dataset ─────────────────────────────────────────
    print("STATUS: Building [model_ready_dataset] via optimized DuckDB LEFT JOIN...")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE model_ready_dataset AS
        SELECT
            l.post_id,
            l.virality_index,              -- UPDATED: matches your previous schema change
            f.* EXCLUDE (post_id),         -- post_features
            z.* EXCLUDE (post_id)          -- zoe_nlp_features
        FROM post_labels l
        LEFT JOIN post_features f ON l.post_id = f.post_id
        LEFT JOIN read_parquet('{local_parquet_path}') z ON l.post_id = z.post_id
        """
    )

    # ── 3. Export for ML Handoff ─────────────────────────────────────────────
    print("STATUS: Exporting final dataset to local Parquet file for ML handoff...")
    con.execute("""
        COPY model_ready_dataset TO 'model_ready_dataset.parquet' (FORMAT PARQUET)
    """)

    # ── 4. Validation & Reporting ────────────────────────────────────────────
    df = con.execute("SELECT * FROM model_ready_dataset").df()

    print("=" * 70)
    print("model_ready_dataset — build report")
    print("=" * 70)

    print(f"\nfinal shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\ncolumns with dtypes:")
    for name, dt in [(c[1], c[2]) for c in con.execute(
        "PRAGMA table_info('model_ready_dataset')"
    ).fetchall()]:
        print(f"   {name:22} {dt}")

    # Trainable set validation
    # UPDATED: Exclude the new virality_index column
    feature_cols = [c for c in df.columns if c not in ("post_id", "virality_index")]
    
    # UPDATED: Check virality_index for nulls
    trainable_mask = df["virality_index"].notna() & df[feature_cols].notna().all(axis=1)
    trainable = int(trainable_mask.sum())
    print(
        f"\ntrainable rows (non-null virality_index AND complete features): "
        f"{trainable} / {len(df)}"
    )

    con.close()
    return df

if __name__ == "__main__":
    build_model_ready()