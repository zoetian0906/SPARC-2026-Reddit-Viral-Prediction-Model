"""
Reddit Ingestion Pipeline — Data Engineering
Source: SPARC2026Reddit/MessyData-ZT (HuggingFace)
Goal: Pull processed cloud records into local DuckDB staging storage
"""

import os
import duckdb
from datasets import load_dataset

def run_data_ingestion():
    # ── Fetch raw data from Hugging Face ─────────────────────────────────────
    print("STATUS: Accessing Hugging Face Hub...")
    repo_id = "SPARC2026Reddit/MessyData-ZT"
    
    try:
        dataset = load_dataset(repo_id)
        df_raw = dataset['train'].to_pandas()
        print(f"SUCCESS: Retrieved {len(df_raw)} records from remote repository.")
    except Exception as e:
        print(f"ERROR: Failed to pull data from Hugging Face. Reason: {e}")
        return

    # ── Initialize local DuckDB warehouse ────────────────────────────────────
    print("\nSTATUS: Initializing persistent storage file [reddit_warehouse.db]...")
    conn = duckdb.connect("reddit_warehouse.db")

    # ── Populate staging table (writes seed data into structural tables)──────
    print("STATUS: Loading records into staging database environment...")
    conn.execute("""
        CREATE OR REPLACE TABLE raw_posts AS 
        SELECT * FROM df_raw
    """)
    
    db_count = conn.execute("SELECT COUNT(*) FROM raw_posts").fetchone()[0]
    print(f"SUCCESS: Relational database populated. Table [raw_posts] total rows: {db_count}")
    
    conn.close()
    print("STATUS: Database system execution closed safely. Ingestion process complete.")

if __name__ == "__main__":
    run_data_ingestion()