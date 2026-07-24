"""
test_query.py — Phase C real-lookup tests.

Builds small dataframes mimicking Table 1 (model_metadata) and Table 2
(predictions), registers them in a test-local in-memory DuckDB via build_db, and
exercises the fallback lookup order + prediction ranking. No network / HF /
Streamlit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.backend.contract import FEATURE_NAMES
from app.backend.loader import build_db
from app.backend.query import DAY_NAMES, lookup_predictions, lookup_segment


def _meta_row(category, has_media, mechanism, sample_size, test_r2, test_rmse) -> dict:
    row = {
        "category": category,
        "has_media": has_media,
        "engagement_mechanism": mechanism,
        "sample_size": sample_size,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
    }
    # 19 SHAP columns so the schema matches Table 1.
    for i, feature in enumerate(FEATURE_NAMES):
        row[f"{feature}_shap"] = float(i) / 10.0
    return row


@pytest.fixture
def conn():
    meta = pd.DataFrame(
        [
            _meta_row("Food & Cooking", "True", "question", 1000, 0.15, 11.0),
            _meta_row("Food & Cooking", "True", "ALL", 1500, 0.12, 11.5),
            _meta_row("Food & Cooking", "ALL", "question", 1200, 0.11, 11.8),
            _meta_row("Food & Cooking", "ALL", "ALL", 3000, 0.09, 12.0),
            _meta_row("ALL", "ALL", "ALL", 94859, 0.05, 12.5),
        ]
    )

    def _pred(sub, hour, day, score, hm="True", em="question", cat="Food & Cooking"):
        return {
            "category": cat, "has_media": hm, "engagement_mechanism": em,
            "subreddit": sub, "hour_of_day": hour, "day_of_week": day,
            "predicted_viral_score": score,
        }

    preds = pd.DataFrame(
        [
            _pred("Cooking", 9, 0, 90.0),   # peak on day_of_week=0 -> Sunday
            _pred("Cooking", 10, 1, 50.0),  # Cooking avg = 70
            _pred("Baking", 18, 6, 80.0),
            _pred("Baking", 2, 3, 40.0),    # Baking avg = 60
            _pred("food", 12, 2, 30.0),     # food avg = 30
        ]
    )
    return build_db({"model_metadata": meta, "predictions": preds})


def test_exact_match_returns_correct_row(conn) -> None:
    seg = lookup_segment(conn, "Food & Cooking", "True", "question")
    assert seg is not None
    assert seg["engagement_mechanism"] == "question"
    assert seg["has_media"] == "True"
    assert seg["test_r2"] == pytest.approx(0.15)


def test_missing_mechanism_falls_back_to_all_mechanism(conn) -> None:
    # mechanism None -> "ALL"; first candidate is (category, post_type, "ALL").
    seg = lookup_segment(conn, "Food & Cooking", "True", None)
    assert seg is not None
    assert seg["engagement_mechanism"] == "ALL"
    assert seg["test_r2"] == pytest.approx(0.12)


def test_missing_everything_falls_back_to_global(conn) -> None:
    seg = lookup_segment(conn, "Nonexistent Category", None, None)
    assert seg is not None
    assert seg["category"] == "ALL"
    assert seg["test_r2"] == pytest.approx(0.05)


def test_category_with_no_data_returns_global(conn) -> None:
    # A real-looking category that simply has no rows -> global "ALL/ALL/ALL".
    seg = lookup_segment(conn, "Gaming", "True", "showcase")
    assert seg is not None
    assert (seg["category"], seg["has_media"], seg["engagement_mechanism"]) == (
        "ALL", "ALL", "ALL",
    )


def test_category_only_fallback(conn) -> None:
    # category present but media+mechanism missing -> (category, ALL, ALL).
    seg = lookup_segment(conn, "Food & Cooking", None, None)
    assert seg is not None
    assert (seg["has_media"], seg["engagement_mechanism"]) == ("ALL", "ALL")
    assert seg["test_r2"] == pytest.approx(0.09)


def test_lookup_predictions_keys(conn) -> None:
    recs = lookup_predictions(conn, "Food & Cooking", "True", "question")
    assert recs
    for rec in recs:
        assert set(rec.keys()) == {
            "subreddit", "predicted_score", "best_hour", "best_day", "sample_size",
        }
        assert isinstance(rec["subreddit"], str)
        assert isinstance(rec["predicted_score"], float)
        assert isinstance(rec["best_hour"], int)
        assert isinstance(rec["best_day"], str)
        assert isinstance(rec["sample_size"], int)


def test_predictions_ranked_by_avg_and_values(conn) -> None:
    recs = lookup_predictions(conn, "Food & Cooking", "True", "question")
    # Cooking avg 70 > Baking avg 60 > food avg 30
    assert [r["subreddit"] for r in recs] == ["Cooking", "Baking", "food"]
    cooking = recs[0]
    assert cooking["predicted_score"] == pytest.approx(70.0)
    assert cooking["best_hour"] == 9
    assert cooking["sample_size"] == 2


def test_best_day_is_string_day_name(conn) -> None:
    recs = lookup_predictions(conn, "Food & Cooking", "True", "question")
    for rec in recs:
        assert isinstance(rec["best_day"], str)
        assert rec["best_day"] in DAY_NAMES.values()


def test_day_mapping_uses_sunday_zero(conn) -> None:
    # Cooking's peak score is at day_of_week=0, which must map to "Sunday".
    recs = lookup_predictions(conn, "Food & Cooking", "True", "question")
    cooking = next(r for r in recs if r["subreddit"] == "Cooking")
    assert cooking["best_day"] == "Sunday"
    assert DAY_NAMES[0] == "Sunday"


def test_top_n_limits_results(conn) -> None:
    recs = lookup_predictions(conn, "Food & Cooking", "True", "question", top_n=2)
    assert len(recs) == 2
    assert [r["subreddit"] for r in recs] == ["Cooking", "Baking"]


def test_lookup_predictions_no_match_returns_empty(conn) -> None:
    recs = lookup_predictions(conn, "Nonexistent", "True", "question")
    assert recs == []
