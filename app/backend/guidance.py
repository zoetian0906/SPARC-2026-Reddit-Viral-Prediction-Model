"""
guidance.py — per-recommendation guidance text.

THIS IS A PLACEHOLDER STUB. Sarah will replace `generate_guidance` with a real
LLM call later; for now it returns simple mode-dependent template text so the
contract's "guidance" field exists and the frontend has something to render.

The function signature below is the integration point — keep it stable so the
LLM swap is drop-in (same pattern as parse.py). No network, no Streamlit, no
DuckDB.
"""

from __future__ import annotations


def _timing_clause(best_day: str | None, best_hour: int | None) -> str | None:
    """Return a 'best_day around best_hour:00' fragment, or None if incomplete."""
    if best_day is None or best_hour is None:
        return None
    return f"{best_day} around {best_hour}:00"


def generate_guidance(
    subreddit: str,
    best_hour: int | None,
    best_day: str | None,
    top_features: list[str],
    mode: str,
) -> str:
    """Return short guidance text for one recommendation, varying by mode.

    Placeholder templates only (no LLM). Always returns a non-empty string.
    """
    timing = _timing_clause(best_day, best_hour)

    if mode == "expert":
        parts: list[str] = []
        if top_features:
            feats = ", ".join(top_features[:3])
            parts.append(f"For this segment the model weights {feats} most heavily.")
        if timing:
            parts.append(f"Post on {timing}.")
        parts.append("Structure your post with those factors in mind.")
        return " ".join(parts)

    if mode == "experienced":
        parts = [f"r/{subreddit} works well for this topic."]
        if timing:
            parts.append(f"Aim for {timing} for better visibility.")
        parts.append("Keep your post specific and genuine rather than promotional.")
        return " ".join(parts)

    # newbie (default): one short sentence.
    if timing:
        return f"r/{subreddit} is a good fit. Try posting on {timing}."
    return f"r/{subreddit} is a good fit."
