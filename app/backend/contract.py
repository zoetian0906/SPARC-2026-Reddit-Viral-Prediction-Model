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


def _fmt_days(lo: float, hi: float) -> str:
    """Render a day_of_week range as names, not the raw Sunday=0 integers."""
    lo_name = DAY_NAMES.get(int(lo), str(int(lo)))
    hi_name = DAY_NAMES.get(int(hi), str(int(hi)))
    return lo_name if lo_name == hi_name else f"{lo_name} to {hi_name}"


# Features whose optimal range is actionable for a human, with a display
# formatter each. The NRC emotion columns and vader_compound are deliberately
# absent: they are populated in only 6-15% of segments AND "keep nrc_disgust
# between 0.08 and 0.14" is not advice anyone can act on.
RANGE_LABELS: dict[str, tuple[str, object]] = {
    "title_length":      ("title length", lambda lo, hi: f"{lo:.0f}-{hi:.0f} words"),
    "body_length":       ("body length", lambda lo, hi: f"{lo:.0f}-{hi:.0f} words"),
    "post_length_proxy": ("total length", lambda lo, hi: f"{lo:.0f}-{hi:.0f} tokens"),
    "hour_of_day":       ("posting hour", lambda lo, hi: f"{lo:.0f}:00-{hi:.0f}:00 UTC"),
    "day_of_week":       ("posting day", _fmt_days),
}


def _is_missing(value) -> bool:
    """True for None, NaN, or non-numeric — i.e. no usable bound.

    NaN is caught with the value != value identity so contract.py stays
    pandas-free; parquet nulls arrive as float('nan') through .to_dict().
    """
    if value is None:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    return number != number


def _optimal_ranges(range_row: dict | None, drivers: list[dict], limit: int = 3) -> list[dict]:
    """Showable optimal ranges for a segment, most important first.

    A range survives only if both bounds are present, numeric, and NOT equal —
    4 body_length rows in the table have min == max, and "0-0 words" is worse
    than silence. Order follows the segment's own |SHAP| ranking; features with
    no SHAP column sort last rather than being dropped.
    """
    if not range_row:
        return []

    rank = {d["feature"]: i for i, d in enumerate(drivers)}
    found: list[dict] = []

    for feature, (label, fmt) in RANGE_LABELS.items():
        lo = range_row.get(f"{feature}_opt_min")
        hi = range_row.get(f"{feature}_opt_max")
        if _is_missing(lo) or _is_missing(hi):
            continue
        lo, hi = float(lo), float(hi)
        if hi < lo:
            lo, hi = hi, lo
        if lo == hi:
            continue
        found.append({
            "feature": feature,
            "label": label,
            "text": fmt(lo, hi),
            "rank": rank.get(feature, len(rank)),
        })

    found.sort(key=lambda r: r["rank"])
    return found[:limit]
    

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
    ranges: list[dict] = []
    if not stub and seg is not None and tier != "none":
        try:
            ranges = _optimal_ranges(
                lookup_optimal_ranges(
                    conn, seg["category"], seg["has_media"], seg["engagement_mechanism"]
                ),
                drivers_all,
            )
            facts = segment_summary(
                conn, seg["category"], seg["has_media"], seg["engagement_mechanism"]
            )
            if facts:
                facts["optimal_ranges"] = ranges
            advice = generate_advice(facts, level=mode)
        except Exception:
            advice, ranges = "", []

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
        "optimal_ranges": ranges,
        "notes": notes,
    }
