"""
build_model_ready.py — Assemble the model-ready training table.

Purpose
-------
Create the `model_ready_dataset` table inside reddit_warehouse.db by LEFT JOINing,
on post_id, the target label with all engineered features:
    - post_labels      -> post_id, viral_score   (viral_score is the TARGET y)
    - post_features     -> engineered features (temporal, media, length), minus post_id
    - zoe_nlp_features  -> NLP features (VADER sentiment, NRC emotions, readability), minus post_id

`viral_score` is the prediction TARGET (y); every other non-key column is a model feature (X).

Ownership
---------
This join was planned by Kristin for the data pipeline but had not been written yet.
Zoe built this first pass to unblock the team. Kristin will OWN this going forward and
fold it into the production labeling/feature pipeline in src/.
"""

import duckdb

DB_PATH = "reddit_warehouse.db"


def build_model_ready(db_path: str = DB_PATH):
    con = duckdb.connect(db_path)

    # LEFT JOIN keeps every labeled post (post_labels is the spine) and attaches
    # features where post_id matches. EXCLUDE drops the duplicate join keys.
    con.execute(
        """
        CREATE OR REPLACE TABLE model_ready_dataset AS
        SELECT
            l.post_id,
            l.viral_score,                 -- TARGET y
            f.* EXCLUDE (post_id),         -- post_features
            z.* EXCLUDE (post_id)          -- zoe_nlp_features
        FROM post_labels l
        LEFT JOIN post_features f   ON l.post_id = f.post_id
        LEFT JOIN zoe_nlp_features z ON l.post_id = z.post_id
        """
    )

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

    print("\nhead(10):")
    print(df.head(10).to_string())

    print("\nnull count per column:")
    print(df.isna().sum().to_string())

    # Trainable set: non-null target AND every feature column present.
    feature_cols = [c for c in df.columns if c not in ("post_id", "viral_score")]
    trainable_mask = df["viral_score"].notna() & df[feature_cols].notna().all(axis=1)
    trainable = int(trainable_mask.sum())
    print(
        f"\ntrainable rows (non-null viral_score AND complete features): "
        f"{trainable} / {len(df)}"
    )

    con.close()
    return df


if __name__ == "__main__":
    build_model_ready()
