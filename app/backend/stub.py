"""
stub.py — hardcoded fake segment lookups for Phase A.

No real data, no HuggingFace, no DuckDB. Returns raw "segment records" that
contract.py shapes into the response contract. This lets us lock the response
shape and the assembly/confidence logic before Table 1/2 are wired in.

A segment record has the shape:
    {
      "test_r2": float | None,
      "test_rmse": float | None,
      "sample_size": int | None,
      "recommendations": [
          {"subreddit": str, "predicted_score": float | None, "sample_size": int}, ...
      ],
      "drivers": [{"feature": str, "shap_value": float}, ...],   # names WITHOUT _shap
      "notes": [str, ...],
    }

Timing (best_hour/best_day) is intentionally absent here — it comes from Table 2,
which does not exist yet; contract.py fills those with None.
"""

from __future__ import annotations

# Known categories from Phase 0 recon (excluding the "ALL" aggregate row).
KNOWN_CATEGORIES = [
    "Career & Work", "Fitness & Health", "Food & Cooking", "Gaming",
    "Home & Interior", "Mental Health", "Personal Finance",
    "Relationships & Advice", "Skincare & Beauty", "Tech & Gadgets",
]


def _food_cooking_question() -> dict:
    """High confidence: strong R², large sample, 5 recommendations."""
    return {
        "test_r2": 0.19,
        "test_rmse": 11.2,
        "sample_size": 4245,
        "recommendations": [
            {"subreddit": "Cooking", "predicted_score": 62.4, "sample_size": 8123},
            {"subreddit": "Baking", "predicted_score": 59.8, "sample_size": 6011},
            {"subreddit": "MealPrepSunday", "predicted_score": 57.1, "sample_size": 4230},
            {"subreddit": "food", "predicted_score": 55.3, "sample_size": 3902},
            {"subreddit": "EatCheapAndHealthy", "predicted_score": 52.0, "sample_size": 1438},
        ],
        "drivers": [
            {"feature": "subreddit", "shap_value": 1.97},
            {"feature": "title_length", "shap_value": 1.29},
            {"feature": "body_length", "shap_value": 0.67},
            {"feature": "engagement_mechanism", "shap_value": 0.61},
            {"feature": "hour_of_day", "shap_value": 0.60},
            {"feature": "post_length_proxy", "shap_value": 0.54},
            {"feature": "readability", "shap_value": 0.50},
            {"feature": "vader_compound", "shap_value": 0.42},
        ],
        "notes": ["Segment-specific model: Food & Cooking + question posts."],
    }


def _home_interior_all() -> dict:
    """Low confidence: positive but weak R²; predicted_score is suppressed by
    contract.py (must be None at low confidence)."""
    return {
        "test_r2": 0.04,
        "test_rmse": 12.3,
        "sample_size": 9060,
        "recommendations": [
            {"subreddit": "InteriorDesign", "predicted_score": 41.2, "sample_size": 5120},
            {"subreddit": "DIY", "predicted_score": 38.7, "sample_size": 3940},
        ],
        "drivers": [
            {"feature": "subreddit", "shap_value": 1.80},
            {"feature": "title_length", "shap_value": 1.10},
            {"feature": "body_length", "shap_value": 0.72},
            {"feature": "hour_of_day", "shap_value": 0.58},
            {"feature": "post_length_proxy", "shap_value": 0.49},
        ],
        "notes": [
            "Weak signal: R² is positive but below the global baseline; treat "
            "recommendations as directional, not precise."
        ],
    }


def _mental_health_showcase() -> dict:
    """None: real segment exists but sample is too small to trust; no recs. Note
    points at the closest covered slice."""
    return {
        "test_r2": 0.018,
        "test_rmse": 10.9,
        "sample_size": 48,
        "recommendations": [],
        "drivers": [],
        "notes": [
            "Mental Health + showcase has only 48 posts — too few for "
            "segment-specific advice. Closest covered slice: Mental Health "
            "across all post types."
        ],
    }


def _unknown(category: str) -> dict:
    """None: category not in Table 1 at all. Never raises."""
    return {
        "test_r2": None,
        "test_rmse": None,
        "sample_size": None,
        "recommendations": [],
        "drivers": [],
        "notes": [
            f"Unknown category {category!r}. Known categories: "
            + ", ".join(KNOWN_CATEGORIES)
            + "."
        ],
    }


def get_stub_segment(
    category: str,
    post_type: str | None = None,
    mechanism: str | None = None,
) -> dict:
    """Return a hardcoded segment record for the four Phase A cases.

    Anything not explicitly covered falls through to the unknown/none case, so
    this never raises on unexpected input.
    """
    if category == "Food & Cooking" and mechanism == "question":
        return _food_cooking_question()
    if category == "Home & Interior":
        return _home_interior_all()
    if category == "Mental Health" and mechanism == "showcase":
        return _mental_health_showcase()
    return _unknown(category)
