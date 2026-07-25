"""
test_parse.py — Phase E keyword parser tests. Pure logic; no network/Streamlit.
"""

from __future__ import annotations

from app.backend.parse import location_note, parse_query


def test_sourdough_is_food_statement() -> None:
    res = parse_query("best recipe for sourdough bread")
    assert res["category"] == "Food & Cooking"
    assert res["mechanism"] == "statement"
    assert res["post_type"] is None


def test_resume_question_is_career_question() -> None:
    res = parse_query("how do I fix my resume?")
    assert res["category"] == "Career & Work"
    assert res["mechanism"] == "question"


def test_gaming_pc_build_is_gaming_showcase() -> None:
    res = parse_query("just finished my gaming PC build")
    assert res["category"] == "Gaming"
    assert res["mechanism"] == "showcase"


def test_anxiety_tips_is_mental_health_statement() -> None:
    res = parse_query("anxiety tips for college students")
    assert res["category"] == "Mental Health"
    assert res["mechanism"] == "statement"


def test_knitting_in_chicago_no_category_with_location() -> None:
    res = parse_query("knitting workshop in Chicago")
    assert res["category"] is None
    assert res["location_mentioned"] is not None
    assert "Chicago" in res["location_mentioned"]


def test_restaurant_in_austin_food_with_location() -> None:
    res = parse_query("promote my restaurant in Austin")
    assert res["category"] == "Food & Cooking"
    assert res["location_mentioned"] is not None
    assert "Austin" in res["location_mentioned"]


def test_empty_string_all_none_no_crash() -> None:
    res = parse_query("")
    assert res["category"] is None
    assert res["mechanism"] is None
    assert res["post_type"] is None
    assert res["location_mentioned"] is None
    assert res["raw_text"] == ""


def test_gibberish_no_category_no_mechanism() -> None:
    res = parse_query("asdfghjkl")
    assert res["category"] is None
    assert res["mechanism"] is None


def test_return_shape_keys() -> None:
    res = parse_query("anything")
    assert set(res.keys()) == {
        "category", "post_type", "mechanism", "location_mentioned", "raw_text",
    }


def test_multiple_category_matches_picks_most_hits() -> None:
    # Two food keywords ("recipe", "cook") vs one gaming ("game").
    res = parse_query("easy recipe to cook while playing a game")
    assert res["category"] == "Food & Cooking"


def test_location_note_mentions_location_and_disclaimer() -> None:
    note = location_note("Austin")
    assert "Austin" in note
    assert "general guidance" in note
