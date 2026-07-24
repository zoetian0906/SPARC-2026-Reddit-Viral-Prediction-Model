"""
contract.py — the response contract for the recommendations backend.

`get_recommendations` returns the SAME dict shape for every input (stub or real,
error or empty). The assembly rules (confidence tiering, hiding predicted_score
unless high, expert-only drivers, none -> empty recs) are shared across both
paths so behavior is identical apart from where the data comes from.

Pure logic apart from get_db() (Streamlit-cached), which is only called on the
real path and is monkeypatchable in tests.
"""

from __future__ import annotations

from app.backend.confidence import assign_confidence
from app.backend.loader import get_db
from app.backend.query import lookup_predictions, lookup_segment
from app.backend.stub import get_stub_segment

# Canonical model feature names = Table 1 SHAP columns minus the "_shap" suffix.
FEATURE_NAMES: list[str] = [
    "hour_of_day", "day_of_week", "has_media", "post_length_proxy",
    "vader_compound", "nrc_joy", "nrc_trust", "nrc_fear", "nrc_surprise",
    "nrc_sadness", "nrc_disgust", "nrc_anger", "nrc_anticipation",
    "readability", "title_length", "body_length",
    "subreddit", "category", "engagement_mechanism",
]


def _drivers_from_segment(seg: dict) -> list[dict]:
    """Extract SHAP drivers from a model_metadata row: read the 19 *_shap columns,
    strip the suffix, sort by absolute value descending."""
    drivers = []
    for feature in FEATURE_NAMES:
        col = f"{feature}_shap"
        if col in seg and seg[col] is not None:
            drivers.append({"feature": feature, "shap_value": float(seg[col])})
    drivers.sort(key=lambda d: abs(d["shap_value"]), reverse=True)
    return drivers


def get_recommendations(
    category: str,
    post_type: str | None = None,   # maps to Table 1/2 has_media
    mechanism: str | None = None,    # maps to Table 1/2 engagement_mechanism
    mode: str = "newbie",            # newbie | experienced | expert
    stub: bool = True,
) -> dict:
    """Return recommendations for a query segment in the fixed contract shape.

    Never raises on unknown input — unrecognized segments come back as
    confidence "none" with an explanatory reason/note.
    """
    query = {"category": category, "post_type": post_type, "mechanism": mechanism}

    if stub:
        seg = get_stub_segment(category, post_type, mechanism)
        model_quality = {
            "test_r2": seg["test_r2"],
            "test_rmse": seg["test_rmse"],
            "sample_size": seg["sample_size"],
        }
        raw_recs = seg["recommendations"]
        drivers_all = seg["drivers"]
        notes = list(seg["notes"])
        r2, n = seg["test_r2"], seg["sample_size"]
    else:
        conn = get_db()
        seg = lookup_segment(conn, category, post_type, mechanism)
        if seg is None:
            model_quality = {"test_r2": None, "test_rmse": None, "sample_size": None}
            raw_recs = []
            drivers_all = []
            notes = ["No model metadata found for this query."]
            r2, n = None, None
        else:
            r2 = None if seg["test_r2"] is None else float(seg["test_r2"])
            rmse = None if seg["test_rmse"] is None else float(seg["test_rmse"])
            n = None if seg["sample_size"] is None else int(seg["sample_size"])
            model_quality = {"test_r2": r2, "test_rmse": rmse, "sample_size": n}
            raw_recs = lookup_predictions(conn, category, post_type, mechanism, top_n=5)
            drivers_all = _drivers_from_segment(seg)
            notes = [
                f"Matched segment: {seg['category']} / has_media={seg['has_media']} "
                f"/ {seg['engagement_mechanism']}."
            ]

    tier, reason = assign_confidence(r2, n)

    # Invariant: "none" never emits recommendations. predicted_score only at "high".
    raw_recs = [] if tier == "none" else raw_recs
    recommendations = [
        {
            "subreddit": r["subreddit"],
            "best_hour": r.get("best_hour"),
            "best_day": r.get("best_day"),
            "predicted_score": r.get("predicted_score") if tier == "high" else None,
            "sample_size": r["sample_size"],
        }
        for r in raw_recs
    ]

    # drivers only in expert mode
    drivers = list(drivers_all) if mode == "expert" else []

    return {
        "query": query,
        "confidence": tier,
        "confidence_reason": reason,
        "recommendations": recommendations,
        "model_quality": model_quality,
        "drivers": drivers,
        "notes": notes,
    }
