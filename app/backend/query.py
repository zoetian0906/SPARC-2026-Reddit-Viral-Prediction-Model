"""
query.py — real lookups against the DuckDB tables built by loader.build_db.

ALL SQL lives in this file. Pure functions (take a duckdb connection, return
plain Python data); no Streamlit, no HuggingFace.

Segment keys use the "ALL" aggregate convention on category / has_media /
engagement_mechanism, matching both Table 1 (model_metadata) and Table 2
(predictions). has_media is stored as the strings "True"/"False"/"ALL".

Day-of-week uses the DuckDB DAYOFWEEK convention discovered in recon: Sunday = 0.
"""

from __future__ import annotations

import duckdb

# Sunday=0 .. Saturday=6 (DuckDB DAYOFWEEK / Postgres dow convention). This MUST
# match how src/processing_script.py derived day_of_week, or every "best day"
# recommendation shifts by a day.
DAY_NAMES: dict[int, str] = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}


def _normalize(post_type: str | None, mechanism: str | None) -> tuple[str, str]:
    """Map optional inputs to the segment keys ('ALL' for None)."""
    return (post_type or "ALL", mechanism or "ALL")


def lookup_segment(
    conn: duckdb.DuckDBPyConnection,
    category: str,
    post_type: str | None,
    mechanism: str | None,
) -> dict | None:
    """Return the best-matching model_metadata row as a dict, or None.

    Lookup order (first match wins):
      1. exact:      category + post_type + mechanism
      2. partial:    category + "ALL"     + mechanism
      3. partial:    category + post_type + "ALL"
      4. category:   category + "ALL"     + "ALL"
      5. global:     "ALL"    + "ALL"     + "ALL"
    """
    pt, mech = _normalize(post_type, mechanism)
    candidates = [
        (category, pt, mech),
        (category, "ALL", mech),
        (category, pt, "ALL"),
        (category, "ALL", "ALL"),
        ("ALL", "ALL", "ALL"),
    ]

    seen: set[tuple[str, str, str]] = set()
    for cat, has_media, eng in candidates:
        if (cat, has_media, eng) in seen:
            continue
        seen.add((cat, has_media, eng))
        df = conn.execute(
            """
            SELECT *
            FROM model_metadata
            WHERE category = ? AND has_media = ? AND engagement_mechanism = ?
            LIMIT 1
            """,
            [cat, has_media, eng],
        ).df()
        if len(df):
            return df.iloc[0].to_dict()
    return None


def lookup_predictions(
    conn: duckdb.DuckDBPyConnection,
    category: str,
    post_type: str | None,
    mechanism: str | None,
    top_n: int = 5,
) -> list[dict]:
    """Top subreddits for a segment by mean predicted_viral_score.

    For each of the top-N subreddits, also return its single best posting slot
    (hour_of_day + day_of_week with the highest predicted score). predicted_score
    is the mean predicted_viral_score for that subreddit across the segment's
    hour/day grid; sample_size is the number of prediction-grid rows for it.

    Returns [] if no rows match the segment.
    """
    cat = category
    pt, mech = _normalize(post_type, mechanism)

    top = conn.execute(
        """
        SELECT subreddit,
               AVG(predicted_viral_score) AS avg_score,
               COUNT(*)                   AS sample_size
        FROM predictions
        WHERE category = ? AND has_media = ? AND engagement_mechanism = ?
        GROUP BY subreddit
        ORDER BY avg_score DESC, subreddit ASC
        LIMIT ?
        """,
        [cat, pt, mech, top_n],
    ).df()

    results: list[dict] = []
    for _, row in top.iterrows():
        subreddit = str(row["subreddit"])
        best = conn.execute(
            """
            SELECT hour_of_day, day_of_week, predicted_viral_score
            FROM predictions
            WHERE category = ? AND has_media = ? AND engagement_mechanism = ?
                  AND subreddit = ?
            ORDER BY predicted_viral_score DESC, day_of_week ASC, hour_of_day ASC
            LIMIT 1
            """,
            [cat, pt, mech, subreddit],
        ).df()
        best_row = best.iloc[0]
        day_int = int(best_row["day_of_week"])
        results.append(
            {
                "subreddit": subreddit,
                "predicted_score": float(row["avg_score"]),
                "best_hour": int(best_row["hour_of_day"]),
                "best_day": DAY_NAMES.get(day_int, str(day_int)),
                "sample_size": int(row["sample_size"]),
            }
        )
    return results
