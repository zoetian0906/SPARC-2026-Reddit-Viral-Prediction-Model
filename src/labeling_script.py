"""
Reddit Labeling Pipeline — Data Engineering
Source: Local DuckDB Staging (processed_posts)
Goal: Apply Virality Metric math and populate post_labels table
"""

import duckdb

def run_labeling_pipeline():
    print("STATUS: Connecting to local DuckDB warehouse...")
    conn = duckdb.connect("reddit_warehouse.db")

    print("STATUS: Applying Virality Metric transformations...")
    
    conn.execute("""
        CREATE OR REPLACE TABLE post_labels AS
        
        -- Step 1: Calculate the subreddit baseline average
        WITH base_stats AS (
            SELECT 
                post_id,
                subreddit,
                score,
                AVG(score) OVER(PARTITION BY subreddit) AS subreddit_baseline
            FROM processed_posts
        ),
        
        -- Step 2: Calculate the score-to-average ratio
        ratios AS (
            SELECT 
                post_id,
                subreddit,
                subreddit_baseline,
                score,
                score / NULLIF(subreddit_baseline, 0) AS score_to_avg_ratio
            FROM base_stats
        ),
        
        -- Step 3: Apply the log1p transformation (ln(x + 1) in SQL)
        log_transformed AS (
            SELECT 
                *,
                ln(score + 1) AS log_score,
                ln(score_to_avg_ratio + 1) AS log_ratio
            FROM ratios
        ),
        
        -- Step 4: Apply Min-Max Normalization over the entire dataset
        scaled AS (
            SELECT 
                *,
                (log_score - MIN(log_score) OVER()) / 
                 NULLIF((MAX(log_score) OVER() - MIN(log_score) OVER()), 0) AS normalized_score,
                 
                (log_ratio - MIN(log_ratio) OVER()) / 
                 NULLIF((MAX(log_ratio) OVER() - MIN(log_ratio) OVER()), 0) AS normalized_ratio
            FROM log_transformed
        )
        
        -- Step 5: Calculate final 50/50 Virality Index (0-100)
        -- REMOVED is_top_quartile permanently from this selection
        SELECT 
            post_id,
            subreddit,
            subreddit_baseline,
            ((0.5 * COALESCE(normalized_score, 0)) + (0.5 * COALESCE(normalized_ratio, 0))) * 100 AS viral_score
        FROM scaled
    """)

    final_count = conn.execute("SELECT COUNT(*) FROM post_labels").fetchone()[0]
    print(f"SUCCESS: post_labels table populated. Generated virality scores for {final_count} posts.")
    
    conn.close()

if __name__ == "__main__":
    run_labeling_pipeline()