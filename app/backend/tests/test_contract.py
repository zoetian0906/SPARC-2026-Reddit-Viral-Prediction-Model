"""
test_contract.py — contract SHAPE tests (Phase A shape, extended in Phase C).

Every test asserts on SHAPE / invariants, not data values, and now runs against
BOTH backends:
  - stub  : get_recommendations(stub=True)  -> hardcoded stub data
  - real  : get_recommendations(stub=False) -> a test-local DuckDB fixture built
            from small in-test dataframes (contract.get_db is monkeypatched).

No network, no HuggingFace, no Streamlit runtime.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.backend.contract import FEATURE_NAMES, get_recommendations
from app.backend.loader import build_db

ALLOWED_CONFIDENCE = {"high", "low", "none"}

TOP_LEVEL_KEYS = {
    "query", "confidence", "confidence_reason",
    "recommendations", "model_quality", "drivers", "notes",
}
REC_KEYS = {"subreddit", "best_hour", "best_day", "predicted_score", "sample_size"}
MODEL_QUALITY_KEYS = {"test_r2", "test_rmse", "sample_size"}

# (kwargs, expected_confidence) — the four canonical cases.
STUB_CASES = [
    ({"category": "Food & Cooking", "mechanism": "question"}, "high"),
    ({"category": "Home & Interior"}, "low"),
    ({"category": "Mental Health", "mechanism": "showcase"}, "none"),
    ({"category": "Pets"}, "none"),
]


# ── real-backend fixture DB ────────────────────────────────────────────────
def _meta_row(category, has_media, mechanism, sample_size, test_r2, test_rmse) -> dict:
    row = {
        "category": category,
        "has_media": has_media,
        "engagement_mechanism": mechanism,
        "sample_size": sample_size,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
    }
    for i, feature in enumerate(FEATURE_NAMES):
        row[f"{feature}_shap"] = float(i + 1) / 10.0
    return row


def _pred(category, has_media, mechanism, subreddit, hour, day, score) -> dict:
    return {
        "category": category, "has_media": has_media,
        "engagement_mechanism": mechanism, "subreddit": subreddit,
        "hour_of_day": hour, "day_of_week": day, "predicted_viral_score": score,
    }


def _build_real_db():
    # NOTE: intentionally NO global ("ALL","ALL","ALL") row, so an unknown
    # category ("Pets") resolves to None -> confidence "none" (matches the
    # unknown-category invariant). Real production data would include a global
    # fallback row; that behavior is covered in test_query.py.
    meta = pd.DataFrame(
        [
            _meta_row("Food & Cooking", "ALL", "question", 4245, 0.19, 11.2),  # high
            _meta_row("Home & Interior", "ALL", "ALL", 9060, 0.04, 12.3),      # low
            _meta_row("Mental Health", "ALL", "showcase", 48, 0.018, 10.9),    # none (small n)
        ]
    )
    preds = pd.DataFrame(
        [
            _pred("Food & Cooking", "ALL", "question", "Cooking", 9, 0, 62.4),
            _pred("Food & Cooking", "ALL", "question", "Baking", 18, 6, 59.8),
            _pred("Food & Cooking", "ALL", "question", "food", 12, 2, 55.3),
            _pred("Home & Interior", "ALL", "ALL", "InteriorDesign", 8, 1, 41.2),
            _pred("Home & Interior", "ALL", "ALL", "DIY", 20, 5, 38.7),
            # Mental Health / showcase deliberately has no prediction rows.
        ]
    )
    return build_db({"model_metadata": meta, "predictions": preds})


@pytest.fixture(params=["stub", "real"])
def backend(request, monkeypatch):
    """Return a callable that runs get_recommendations against one backend."""
    if request.param == "stub":
        def run(**kwargs):
            return get_recommendations(stub=True, **kwargs)
        return run

    conn = _build_real_db()
    monkeypatch.setattr("app.backend.contract.get_db", lambda: conn)

    def run(**kwargs):
        return get_recommendations(stub=False, **kwargs)
    return run


# ── shape tests (run against BOTH backends) ────────────────────────────────
@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_top_level_shape(backend, kwargs, expected) -> None:
    res = backend(**kwargs)

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
def test_recommendation_item_shape(backend, kwargs, expected) -> None:
    res = backend(**kwargs, mode="expert")
    for rec in res["recommendations"]:
        assert set(rec.keys()) == REC_KEYS
        assert isinstance(rec["subreddit"], str)

        # best_hour: None or int 0-23; best_day: None or a string day name.
        assert rec["best_hour"] is None or (
            isinstance(rec["best_hour"], int) and 0 <= rec["best_hour"] <= 23
        )
        assert rec["best_day"] is None or isinstance(rec["best_day"], str)
        assert rec["predicted_score"] is None or (
            isinstance(rec["predicted_score"], (int, float))
            and 0 <= rec["predicted_score"] <= 100
        )
        assert isinstance(rec["sample_size"], int)


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_confidence_string_is_allowed(backend, kwargs, expected) -> None:
    res = backend(**kwargs)
    assert res["confidence"] in ALLOWED_CONFIDENCE


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_none_implies_empty_recommendations(backend, kwargs, expected) -> None:
    res = backend(**kwargs)
    if res["confidence"] == "none":
        assert res["recommendations"] == []


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_low_implies_predicted_score_none(backend, kwargs, expected) -> None:
    res = backend(**kwargs)
    if res["confidence"] == "low":
        assert res["recommendations"], "low-confidence case should still have recs"
        assert all(r["predicted_score"] is None for r in res["recommendations"])


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_present_best_fields_types(backend, kwargs, expected) -> None:
    # When timing fields are present (real backend), best_day is a string day name
    # and best_hour is an int 0-23.
    res = backend(**kwargs, mode="expert")
    for rec in res["recommendations"]:
        if rec["best_day"] is not None:
            assert isinstance(rec["best_day"], str)
            assert not rec["best_day"].isdigit()
        if rec["best_hour"] is not None:
            assert isinstance(rec["best_hour"], int)
            assert 0 <= rec["best_hour"] <= 23


def test_high_confidence_shows_predicted_score(backend) -> None:
    res = backend(category="Food & Cooking", mechanism="question")
    assert res["confidence"] == "high"
    assert res["recommendations"]
    assert all(r["predicted_score"] is not None for r in res["recommendations"])


@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_newbie_has_no_drivers(backend, kwargs, expected) -> None:
    res = backend(**kwargs, mode="newbie")
    assert res["drivers"] == []


def test_expert_has_drivers(backend) -> None:
    res = backend(category="Food & Cooking", mechanism="question", mode="expert")
    assert len(res["drivers"]) > 0
    for d in res["drivers"]:
        assert set(d.keys()) == {"feature", "shap_value"}
        assert isinstance(d["feature"], str)
        assert isinstance(d["shap_value"], float)


def test_expert_driver_names_have_no_shap_suffix(backend) -> None:
    res = backend(category="Food & Cooking", mechanism="question", mode="expert")
    assert res["drivers"]
    for d in res["drivers"]:
        assert not d["feature"].endswith("_shap")
        assert d["feature"] in FEATURE_NAMES


def test_unknown_category_returns_none_and_does_not_raise(backend) -> None:
    res = backend(category="Pets")  # must not raise
    assert res["confidence"] == "none"
    assert res["recommendations"] == []


def test_unknown_mode_does_not_raise_and_has_no_drivers(backend) -> None:
    res = backend(category="Food & Cooking", mechanism="question", mode="banana")
    assert res["drivers"] == []


# ── stub-only test (Table-2 timing not wired in the stub) ──────────────────
@pytest.mark.parametrize("kwargs,expected", STUB_CASES)
def test_best_time_fields_none_in_stub(kwargs, expected) -> None:
    res = get_recommendations(stub=True, **kwargs, mode="expert")
    for rec in res["recommendations"]:
        assert rec["best_hour"] is None
        assert rec["best_day"] is None
