"""
test_guidance.py — guidance fallback tests. No network/LLM (conftest forces the
no-key path), so generate_guidance must return the deterministic template.
"""

from __future__ import annotations

import pytest

from app.backend.guidance import generate_guidance, llm_generate_guidance


@pytest.mark.parametrize("mode", ["newbie", "experienced", "expert"])
def test_generate_guidance_non_empty_without_key(mode) -> None:
    text = generate_guidance(
        subreddit="Cooking",
        best_hour=9,
        best_day="Sunday",
        top_features=["subreddit", "title_length", "body_length"],
        mode=mode,
    )
    assert isinstance(text, str)
    assert text.strip() != ""


def test_generate_guidance_differs_by_mode_without_key() -> None:
    texts = [
        generate_guidance("Cooking", 9, "Sunday", ["subreddit"], mode)
        for mode in ("newbie", "experienced", "expert")
    ]
    assert len(set(texts)) == 3


def test_llm_generate_guidance_raises_without_key() -> None:
    # The raw LLM function raises with no key; the wrapper (above) catches it.
    with pytest.raises(Exception):
        llm_generate_guidance("Cooking", 9, "Sunday", ["subreddit"], "newbie", "Food & Cooking")
