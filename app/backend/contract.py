"""
contract.py — the response contract for the recommendations backend.

`get_recommendations` returns the SAME dict shape for every input, including
error and empty cases. The assembly logic here (confidence tiering, hiding
predicted_score at low confidence, expert-only drivers, Table 2 timing = None)
is the real logic and will carry over unchanged when real data replaces the stub.

Pure logic, no Streamlit — testable without a Streamlit runtime.
"""

from __future__ import annotations

from app.backend.confidence import assign_confidence
from app.backend.stub import get_stub_segment

# Canonical model feature names = Table 1 SHAP columns minus the "_shap" suffix.
FEATURE_NAMES: list[str] = [
    "hour_of_day", "day_of_week", "has_media", "post_length_proxy",
    "vader_compound", "nrc_joy", "nrc_trust", "nrc_fear", "nrc_surprise",
    "nrc_sadness", "nrc_disgust", "nrc_anger", "nrc_anticipation",
    "readability", "title_length", "body_length",
    "subreddit", "category", "engagement_mechanism",
]

VALID_MODES = {"newbie", "experienced", "expert"}


def _empty_segment(note: str) -> dict:
    """A segment record with no data (used for the real-data path in Phase A)."""
    return {
        "test_r2": None,
        "test_rmse": None,
        "sample_size": None,
        "recommendations": [],
        "drivers": [],
        "notes": [note],
    }


def get_recommendations(
    category: str,
    post_type: str | None = None,   # maps to Table 1 has_media
    mechanism: str | None = None,    # maps to Table 1 engagement_mechanism
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
    else:
        # Real Table 1/2 integration is not wired yet in Phase A.
        seg = _empty_segment(
            "Live data source not wired yet (stub=False); Table 1/2 integration pending."
        )

    tier, reason = assign_confidence(seg["test_r2"], seg["sample_size"])

    # Invariant: "none" never emits recommendations. Otherwise, map raw recs into
    # the contract shape.
    raw_recs = [] if tier == "none" else seg["recommendations"]
    recommendations = [
        {
            "subreddit": r["subreddit"],
            "best_hour_utc": None,   # from Table 2 (not available yet)
            "best_day_utc": None,    # from Table 2 (not available yet)
            # predicted_score only surfaces at high confidence
            "predicted_score": r.get("predicted_score") if tier == "high" else None,
            "sample_size": r["sample_size"],
        }
        for r in raw_recs
    ]

    # drivers only in expert mode
    drivers = list(seg["drivers"]) if mode == "expert" else []

    return {
        "query": query,
        "confidence": tier,
        "confidence_reason": reason,
        "recommendations": recommendations,
        "model_quality": {
            "test_r2": seg["test_r2"],
            "test_rmse": seg["test_rmse"],
            "sample_size": seg["sample_size"],
        },
        "drivers": drivers,
        "notes": list(seg["notes"]),
    }
