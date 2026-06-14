"""
Reddit Processing Pipeline — Data Engineering
Source: Local DuckDB Staging (raw_posts)
Goal: Apply text normalization, text-splitting, and generate schema skeletons
"""

import duckdb
import pandas as pd

def run_processing_pipeline():
    print("STATUS: Connecting to local DuckDB warehouse...")
    conn = duckdb.connect("reddit_warehouse.db")

    # ── Basic Cleaning & Text Splitting────────────────────────────────
    print("STATUS: Extracting raw posts for processing...")
    df = conn.execute("SELECT * FROM raw_posts WHERE language = 'en'").df()

    print("STATUS: Applying title/body split logic...")
    SEP = ": "
    split = df["text"].str.split(SEP, n=1, expand=True)
    df["title"] = split[0]
    df["body"]  = split[1].fillna("")

    # ── Table 1: Build Processed Posts Schema ────────────────────────────────
    print("STATUS: Building [processed_posts] schema...")
    conn.execute("DROP TABLE IF EXISTS processed_posts")
    conn.execute("""
        CREATE TABLE processed_posts AS
        SELECT 
            post_id,
            subreddit,
            title,
            body,
            url,
            score,
            author,
            CAST(date AS TIMESTAMP) AS posted_at_local,
            token_count
        FROM df
    """)

    # ── Table 2: Build Post Features Schema ──────────────────────────────────
    print("STATUS: Building [post_features] schema...")
    conn.execute("DROP TABLE IF EXISTS post_features")
    conn.execute("""
        CREATE TABLE post_features AS
        SELECT 
            post_id,
            EXTRACT(HOUR FROM posted_at_local) AS hour_of_day,
            DAYOFWEEK(posted_at_local) AS day_of_week,
            CASE 
                WHEN url LIKE '%i.redd.it%' OR url LIKE '%imgur%' THEN true 
                ELSE false 
            END AS has_media,
            token_count AS post_length_proxy
        FROM processed_posts
    """)

    # ── Table 3: Build Post Labels Skeleton ──────────────────────────────────
    print("STATUS: Building [post_labels] skeleton...")
    conn.execute("DROP TABLE IF EXISTS post_labels")
    conn.execute("""
        CREATE TABLE post_labels AS
        SELECT 
            post_id,
            subreddit,
            0.0 AS subreddit_baseline, 
            0.0 AS viral_score,        
            false AS is_top_quartile   
        FROM processed_posts
    """)

    final_count = conn.execute("SELECT COUNT(*) FROM processed_posts").fetchone()[0]
    print(f"SUCCESS: Pipeline complete. Processed {final_count} English posts.")
    
    conn.close()

if __name__ == "__main__":
    run_processing_pipeline()