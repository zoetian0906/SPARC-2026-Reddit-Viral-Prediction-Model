"""
contract.py — the response contract for the recommendations backend.

`get_recommendations` returns the SAME dict shape for every input (stub or real,
error or empty). The assembly rules (confidence tiering, hiding predicted_score
unless high, technical-only drivers, none -> empty recs) are shared across both
paths so behavior is identical apart from where the data comes from.

Pure logic apart from get_db() (Streamlit-cached), which is only called on the
real path and is monkeypatchable in tests.
"""

from __future__ import annotations

from app.backend.confidence import assign_confidence
from app.backend.guidance import generate_guidance_batch
from app.backend.loader import get_db
from app.backend.query import lookup_predictions, lookup_segment, segment_summary
from app.backend.stub import get_stub_segment
from app.backend.advice import generate_advice

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
    mode: str = "experienced",       # experienced | technical
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

    # Top features for guidance come from the segment's SHAP drivers (if any),
    # independent of whether drivers are surfaced in the response (technical only).
    top_features = [d["feature"] for d in drivers_all[:3]]

    # Invariant: "none" never emits recommendations. predicted_score only at "high".
    # Rank subreddits by predicted engagement, highest first (Item 6). Real data is
    # already sorted by query.py; this makes ordering explicit and source-agnostic.
    raw_recs = (
        []
        if tier == "none"
        else sorted(raw_recs, key=lambda r: (r.get("predicted_score") or 0.0), reverse=True)
    )
    recommendations = []
    for r in raw_recs:
        recommendations.append(
            {
                "subreddit": r["subreddit"],
                "best_hour": r.get("best_hour"),
                "best_day": r.get("best_day"),
                "predicted_score": r.get("predicted_score") if tier == "high" else None,
                "sample_size": r["sample_size"],
            }
        )

    # Guidance for ALL recs in a SINGLE Gemini call (falls back to templates with
    # no key / on error), keeping total LLM calls per query to at most 2. The UI's
    # "technical" mode maps to guidance.py's analytical "expert" persona (guidance.py
    # is Sarah's zone and still keys personas as newbie/experienced/expert).
    guidance_mode = "expert" if mode == "technical" else mode
    guidances = generate_guidance_batch(
        recommendations, top_features=top_features, mode=guidance_mode, category=category
    )
    for rec, guidance in zip(recommendations, guidances):
        rec["guidance"] = guidance

    # Segment-level advice: one extra parameterized query + at most ONE Groq
    # call, both optional. Uses the segment we ACTUALLY matched (lookup_segment
    # falls back through ALL), not the raw request, so the numbers always
    # describe the rows we read. Real-data path only.
    advice = ""
    if not stub and seg is not None and tier != "none":
        try:
            facts = segment_summary(
                conn, seg["category"], seg["has_media"], seg["engagement_mechanism"]
            )
            advice = generate_advice(facts, level=mode)
        except Exception:
            advice = ""

    # drivers (SHAP feature importances) only surface in technical mode
    drivers = list(drivers_all) if mode == "technical" else []

    return {
        "query": query,
        "confidence": tier,
        "confidence_reason": reason,
        "recommendations": recommendations,
        "model_quality": model_quality,
        "drivers": drivers,
        "advice": advice,
        "notes": notes,
    }
