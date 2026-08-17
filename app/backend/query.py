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


# Tables _lookup_by_segment may read. A table name cannot be a bound parameter,
# so this allowlist is the guard — never interpolate a caller-supplied value.
_SEGMENT_TABLES = {"model_metadata", "optimal_ranges"}

def _lookup_by_segment(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    category: str,
    post_type: str | None,
    mechanism: str | None,
) -> dict | None:
    """Shared 5-step fallback ladder over a segment-keyed table."""
    if table not in _SEGMENT_TABLES:
        raise ValueError(f"unknown segment table: {table}")

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
            f"""
            SELECT *
            FROM {table}
            WHERE category = ? AND has_media = ? AND engagement_mechanism = ?
            LIMIT 1
            """,
            [cat, has_media, eng],
        ).df()
        if len(df):
            return df.iloc[0].to_dict()
    return None


def lookup_optimal_ranges(
    conn: duckdb.DuckDBPyConnection,
    category: str,
    post_type: str | None,
    mechanism: str | None,
) -> dict | None:
    """Optimal-range row for a segment, same ladder as lookup_segment.

    optimal_ranges shares model_metadata's key space exactly (verified: 82
    segments each, no orphans), so looking up with an already-matched segment's
    own keys is guaranteed to hit its exact row.
    """
    return _lookup_by_segment(conn, "optimal_ranges", category, post_type, mechanism)


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
    return _lookup_by_segment(conn, "model_metadata", category, post_type, mechanism)


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

def segment_summary(
    conn: duckdb.DuckDBPyConnection,
    category: str,
    post_type: str | None,
    mechanism: str | None,
) -> dict:
    """Aggregate facts for one segment, for the LLM advice line in advice.py.

    Every segment key is BOUND as a query parameter, never string-formatted, so
    a value chosen upstream by an LLM cannot inject SQL. Returns {} when the
    segment has no prediction rows — callers treat that as "no advice".

    Beyond what lookup_predictions already returns, this adds the two
    comparisons that make advice actionable: how media vs no-media scores in
    this category, and which engagement mechanism scores best.
    """
    pt, mech = _normalize(post_type, mechanism)

    overall = conn.execute(
        """
        SELECT COUNT(*)                   AS n_rows,
               COUNT(DISTINCT subreddit)  AS n_subreddits,
               AVG(predicted_viral_score) AS avg_score
        FROM predictions
        WHERE category = ? AND has_media = ? AND engagement_mechanism = ?
        """,
        [category, pt, mech],
    ).df()
    if not len(overall) or int(overall.iloc[0]["n_rows"] or 0) == 0:
        return {}

    slot = conn.execute(
        """
        SELECT day_of_week, hour_of_day, AVG(predicted_viral_score) AS avg_score
        FROM predictions
        WHERE category = ? AND has_media = ? AND engagement_mechanism = ?
        GROUP BY day_of_week, hour_of_day
        ORDER BY avg_score DESC, day_of_week ASC, hour_of_day ASC
        LIMIT 1
        """,
        [category, pt, mech],
    ).df()

    media = conn.execute(
        """
        SELECT has_media, AVG(predicted_viral_score) AS avg_score
        FROM predictions
        WHERE category = ? AND engagement_mechanism = ?
              AND has_media IN ('True', 'False')
        GROUP BY has_media
        """,
        [category, mech],
    ).df()

    mechanisms = conn.execute(
        """
        SELECT engagement_mechanism, AVG(predicted_viral_score) AS avg_score
        FROM predictions
        WHERE category = ? AND has_media = ? AND engagement_mechanism <> 'ALL'
        GROUP BY engagement_mechanism
        ORDER BY avg_score DESC
        LIMIT 1
        """,
        [category, pt],
    ).df()

    media_scores = {
        str(row["has_media"]): float(row["avg_score"]) for _, row in media.iterrows()
    }

    return {
        "category": category,
        "has_media": pt,
        "engagement_mechanism": mech,
        "n_subreddits": int(overall.iloc[0]["n_subreddits"]),
        "avg_score": float(overall.iloc[0]["avg_score"]),
        "best_day": DAY_NAMES.get(int(slot.iloc[0]["day_of_week"])) if len(slot) else None,
        "best_hour": int(slot.iloc[0]["hour_of_day"]) if len(slot) else None,
        "avg_score_with_media": media_scores.get("True"),
        "avg_score_without_media": media_scores.get("False"),
        "best_mechanism": (
            str(mechanisms.iloc[0]["engagement_mechanism"]) if len(mechanisms) else None
        ),
    }
