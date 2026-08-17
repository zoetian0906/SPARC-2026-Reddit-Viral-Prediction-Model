"""
advice.py — one segment-level advice line per query, via Groq.

Sits alongside guidance.py: guidance.py writes per-subreddit copy, advice.py
writes ONE sentence or two about the whole
(category / has_media / engagement_mechanism) slice, tuned to the user's
experience level. The numbers come from query.segment_summary — the model never
sees the tables and never writes SQL, it only phrases facts it is handed.

Falls back to a deterministic template when no key is configured or the call
fails, so the app still works offline and in tests.
"""

from __future__ import annotations

from app.backend.llm import GROQ_ADVICE_MODEL, get_llm, message_text, perf_timer

# Experience level -> length and register. Keys match contract.py's `mode`.
_LEVEL_STYLE = {
    "newbie": "Write ONE short sentence in plain language. No jargon, no numbers.",
    "experienced": (
        "Write TWO sentences: what to do, and the timing or format detail that "
        "supports it. Cite at most one number."
    ),
    "technical": (
        "Write TWO analytical sentences citing the relevant averages and how "
        "many subreddits they cover. Precise, no filler."
    ),
}


def _pct_lift(a: float | None, b: float | None) -> float | None:
    """Percent difference of a over b, or None if either side is missing."""
    if a is None or b is None or not b:
        return None
    return (a - b) / abs(b) * 100.0


def _fact_lines(facts: dict) -> str:
    """Render the summary dict as the fact block the model must stay inside."""
    lines = [
        f"- Segment: {facts['category']} / has_media={facts['has_media']} / "
        f"{facts['engagement_mechanism']}",
        f"- Subreddits covered: {facts['n_subreddits']}",
        f"- Mean predicted virality: {facts['avg_score']:.1f}",
    ]
    if facts.get("best_day") and facts.get("best_hour") is not None:
        lines.append(
            f"- Strongest slot: {facts['best_day']} at {facts['best_hour']}:00 UTC"
        )
    lift = _pct_lift(
        facts.get("avg_score_with_media"), facts.get("avg_score_without_media")
    )
    if lift is not None:
        direction = "higher" if lift >= 0 else "lower"
        lines.append(
            f"- Posts WITH media score {abs(lift):.0f}% {direction} in this category"
        )
    if facts.get("best_mechanism"):
        lines.append(f"- Strongest post type here: {facts['best_mechanism']}")
    for r in facts.get("optimal_ranges", []):
        lines.append(f"- Optimal {r['label']}: {r['text']}")
    return "\n".join(lines)


def _template_advice(facts: dict, level: str) -> str:
    """Deterministic fallback (no LLM). Always non-empty when facts exist."""
    category = facts.get("category") or "this topic"
    ranges = facts.get("optimal_ranges") or []
    if ranges:
        parts.append(f"Aim for a {ranges[0]['label']} of {ranges[0]['text']}.")  
    when = (
        f"{facts['best_day']} around {facts['best_hour']}:00 UTC"
        if facts.get("best_day") and facts.get("best_hour") is not None
        else None
    )
    if level == "technical":
        parts = [
            f"Across {facts.get('n_subreddits', 0)} subreddits in {category}, this "
            f"segment averages a predicted virality of {facts.get('avg_score', 0):.1f}."
        ]
        if when:
            parts.append(f"The strongest slot in the grid is {when}.")
        return " ".join(parts)
    if when:
        return f"For {category} posts like yours, the data favors posting {when}."
    return f"For {category} posts like yours, start with the ranked subreddits above."


def generate_advice(facts: dict, level: str) -> str:
    """One or two sentences of advice for the segment. Never raises.

    Returns "" when there are no facts to ground it in, so the caller renders
    nothing rather than something invented.
    """
    if not facts:
        return ""

    style = _LEVEL_STYLE.get(level, _LEVEL_STYLE["experienced"])
    try:
        llm = get_llm(temperature=0.3, model=GROQ_ADVICE_MODEL)
        if llm is None:
            raise RuntimeError("No LLM key configured")

        from langchain_core.messages import HumanMessage, SystemMessage

        system = (
            "You advise a Reddit creator on how to post. Use ONLY the facts "
            "given; never invent numbers, subreddits, or claims. Some segments "
            "list no optimal ranges — if none are listed, do not mention ranges "
            "at all. Plain prose only — no preamble, no bullets, no markdown. " + style
        )
        user = f"Facts:\n{_fact_lines(facts)}\n\nWrite the advice now."

        with perf_timer("groq:advice"):
            text = message_text(
                llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            ).strip()
        if not text:
            raise RuntimeError("Empty response")
        return text
    except Exception:
        return _template_advice(facts, level)
