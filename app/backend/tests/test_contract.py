"""
test_contract.py — Phase A contract SHAPE tests.

These assert on the SHAPE and invariants of the response, not on specific data
values, so they keep passing when the stub is replaced by real Table 1/2 data.
No network, no DuckDB, no fixtures on disk.
"""

from __future__ import annotations

import pytest

from app.backend.contract import FEATURE_NAMES, get_recommendations

ALLOWED_CONFIDENCE = {"high", "low", "none"}

TOP_LEVEL_KEYS = {
    "query", "confidence", "confidence_reason",
    "recommendations", "model_quality", "drivers", "notes",
}
REC_KEYS = {"subreddit", "best_hour_utc", "best_day_utc", "predicted_score", "sample_size"}
MODEL_QUALITY_KEYS = {"test_r2", "test_rmse", "sample_size"}

# (kwargs, expected_confidence) — the four Phase A stub cases.
STUB_CASES = [
    ({"category": "Food & Cooking", "mechanism": "question"}, "high"),
    ({"category": "Home & Interior"}, "low"),
    ({"category": "Mental Health", "mechanism": "showcase"}, "none"),
    ({"category": "Pets"}, "none"),
]


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_top_level_shape(kwargs: dict, expected: str) -> None:
    res = get_recommendations(**kwargs)

    assert isinstance(res, dict)
    assert set(res.keys()) == TOP_LEVEL_KEYS

    assert isinstance(res["query"], dict)
    assert set(res["query"].keys()) == {"category", "post_type", "mechanism"}

    assert res["confidence"] in ALLOWED_CONFIDENCE
    assert res["confidence"] == expected
    assert isinstance(res["confidence_reason"], str) and res["confidence_reason"]

    assert isinstance(res["recommendations"], list)

    mq = res["model_quality"]
    assert set(mq.keys()) == MODEL_QUALITY_KEYS
    assert mq["test_r2"] is None or isinstance(mq["test_r2"], float)
    assert mq["test_rmse"] is None or isinstance(mq["test_rmse"], float)
    assert mq["sample_size"] is None or isinstance(mq["sample_size"], int)

    assert isinstance(res["drivers"], list)
    assert isinstance(res["notes"], list)
    assert all(isinstance(n, str) for n in res["notes"])


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_recommendation_item_shape(kwargs: dict, expected: str) -> None:
    res = get_recommendations(**kwargs, mode="expert")
    for rec in res["recommendations"]:
        assert set(rec.keys()) == REC_KEYS
        assert isinstance(rec["subreddit"], str)

        assert rec["best_hour_utc"] is None or (
            isinstance(rec["best_hour_utc"], int) and 0 <= rec["best_hour_utc"] <= 23
        )
        assert rec["best_day_utc"] is None or isinstance(rec["best_day_utc"], str)
        assert rec["predicted_score"] is None or (
            isinstance(rec["predicted_score"], (int, float))
            and 0 <= rec["predicted_score"] <= 100
        )
        assert isinstance(rec["sample_size"], int)


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_confidence_string_is_allowed(kwargs: dict, expected: str) -> None:
    res = get_recommendations(**kwargs)
    assert res["confidence"] in ALLOWED_CONFIDENCE


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_none_implies_empty_recommendations(kwargs: dict, expected: str) -> None:
    res = get_recommendations(**kwargs)
    if res["confidence"] == "none":
        assert res["recommendations"] == []


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_low_implies_predicted_score_none(kwargs: dict, expected: str) -> None:
    res = get_recommendations(**kwargs)
    if res["confidence"] == "low":
        assert res["recommendations"], "low-confidence case should still have recs"
        assert all(r["predicted_score"] is None for r in res["recommendations"])


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_best_time_fields_none_in_stub(kwargs: dict, expected: str) -> None:
    # Table 2 not available yet -> timing is always None in stubs.
    res = get_recommendations(**kwargs, mode="expert")
    for rec in res["recommendations"]:
        assert rec["best_hour_utc"] is None
        assert rec["best_day_utc"] is None


def test_high_confidence_shows_predicted_score() -> None:
    res = get_recommendations("Food & Cooking", mechanism="question")
    assert res["confidence"] == "high"
    assert res["recommendations"]
    assert all(r["predicted_score"] is not None for r in res["recommendations"])


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_newbie_has_no_drivers(kwargs: dict, expected: str) -> None:
    res = get_recommendations(**kwargs, mode="newbie")
    assert res["drivers"] == []


def test_expert_has_drivers() -> None:
    res = get_recommendations("Food & Cooking", mechanism="question", mode="expert")
    assert len(res["drivers"]) > 0
    for d in res["drivers"]:
        assert set(d.keys()) == {"feature", "shap_value"}
        assert isinstance(d["feature"], str)
        assert isinstance(d["shap_value"], float)


def test_expert_driver_names_have_no_shap_suffix() -> None:
    res = get_recommendations("Food & Cooking", mechanism="question", mode="expert")
    assert res["drivers"]
    for d in res["drivers"]:
        assert not d["feature"].endswith("_shap")
        assert d["feature"] in FEATURE_NAMES


def test_unknown_category_returns_none_and_does_not_raise() -> None:
    res = get_recommendations("Pets")  # must not raise
    assert res["confidence"] == "none"
    assert res["recommendations"] == []


def test_unknown_mode_does_not_raise_and_has_no_drivers() -> None:
    # Only mode == "expert" gets drivers; anything else (incl. junk) -> empty.
    res = get_recommendations("Food & Cooking", mechanism="question", mode="banana")
    assert res["drivers"] == []
