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
          {"subreddit": str, "predicted_score": float | None, "sample_size": int,
           "guidance": str}, ...
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
            {"subreddit": "Cooking", "predicted_score": 62.4, "sample_size": 8123,
             "guidance": "Strong fit for recipe questions."},
            {"subreddit": "Baking", "predicted_score": 59.8, "sample_size": 6011,
             "guidance": "Good for baking-specific questions."},
            {"subreddit": "MealPrepSunday", "predicted_score": 57.1, "sample_size": 4230,
             "guidance": "Great for meal-prep and planning posts."},
            {"subreddit": "food", "predicted_score": 55.3, "sample_size": 3902,
             "guidance": "Broad reach; keep the title specific."},
            {"subreddit": "EatCheapAndHealthy", "predicted_score": 52.0, "sample_size": 1438,
             "guidance": "Best for budget-friendly angles."},
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


def _skincare_beauty() -> dict:
    """High confidence: backs the default 'sunscreen' example so the tutorial
    shows a full result even when HuggingFace is unreachable (stub fallback)."""
    return {
        "test_r2": 0.17,
        "test_rmse": 11.6,
        "sample_size": 5120,
        "recommendations": [
            {"subreddit": "SkincareAddiction", "predicted_score": 64.1, "sample_size": 9032,
             "guidance": "Strong fit for routines and product questions."},
            {"subreddit": "30PlusSkinCare", "predicted_score": 58.9, "sample_size": 4211,
             "guidance": "Great for age-specific skincare advice."},
            {"subreddit": "AsianBeauty", "predicted_score": 56.7, "sample_size": 3877,
             "guidance": "Best for ingredient-focused discussion."},
            {"subreddit": "Skincare_Addiction", "predicted_score": 54.2, "sample_size": 2600,
             "guidance": "Broad reach; keep the title specific."},
            {"subreddit": "beauty", "predicted_score": 51.5, "sample_size": 1710,
             "guidance": "Good for general beauty crossovers."},
        ],
        "drivers": [
            {"feature": "subreddit", "shap_value": 1.88},
            {"feature": "title_length", "shap_value": 1.21},
            {"feature": "body_length", "shap_value": 0.70},
            {"feature": "engagement_mechanism", "shap_value": 0.58},
            {"feature": "hour_of_day", "shap_value": 0.55},
            {"feature": "vader_compound", "shap_value": 0.44},
        ],
        "notes": ["Segment-specific model: Skincare & Beauty."],
    }


def _home_interior_all() -> dict:
    """Low confidence: positive but weak R²; predicted_score is suppressed by
    contract.py (must be None at low confidence)."""
    return {
        "test_r2": 0.04,
        "test_rmse": 12.3,
        "sample_size": 9060,
        "recommendations": [
            {"subreddit": "InteriorDesign", "predicted_score": 41.2, "sample_size": 5120,
             "guidance": "Directional pick; data is limited here."},
            {"subreddit": "DIY", "predicted_score": 38.7, "sample_size": 3940,
             "guidance": "Directional pick; data is limited here."},
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
    if category == "Skincare & Beauty":
        return _skincare_beauty()
    if category == "Home & Interior":
        return _home_interior_all()
    if category == "Mental Health" and mechanism == "showcase":
        return _mental_health_showcase()
    return _unknown(category)
