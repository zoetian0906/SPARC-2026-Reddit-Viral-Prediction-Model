"""
guidance.py — per-recommendation guidance text.

generate_guidance_batch generates guidance for ALL recommendations in a SINGLE
Gemini call (contract.py uses this so a query costs at most one guidance call,
never one-per-rec). generate_guidance keeps the single-rec API for callers/tests.
Both use Gemini when a GOOGLE_API_KEY is configured and fall back to a
deterministic template (_template_guidance) otherwise — so the contract's
"guidance" field is always a non-empty string, works offline, and stays fast in
tests (which run with no key).

Streamlit/langchain are imported lazily inside app.backend.llm only when a key
is present.
"""

from __future__ import annotations

import json

from app.backend.llm import get_gemini, message_text, perf_timer

# Persona guidance per mode, mirroring Sarah's chatbot personas.
_MODE_PERSONA = {
    "newbie": (
        "Audience: a beginner. Use simple, encouraging, plain language and avoid "
        "jargon. Write 1-2 sentences."
    ),
    "experienced": (
        "Audience: a marketing-savvy creator. Be actionable and specific; mention "
        "the best timing and how to structure the post. Write 3-4 sentences."
    ),
    "expert": (
        "Audience: a data-literate expert. Be analytical and reference the listed "
        "top model features by name. Mention the best timing. Write 3-4 sentences."
    ),
}


def _timing_clause(best_day: str | None, best_hour: int | None) -> str | None:
    """Return a 'best_day around best_hour:00' fragment, or None if incomplete."""
    if best_day is None or best_hour is None:
        return None
    return f"{best_day} around {best_hour}:00"


def _template_guidance(
    subreddit: str,
    best_hour: int | None,
    best_day: str | None,
    top_features: list[str],
    mode: str,
) -> str:
    """Deterministic fallback guidance (no LLM). Always non-empty."""
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


def llm_generate_guidance(
    subreddit: str,
    best_hour: int | None,
    best_day: str | None,
    top_features: list[str],
    mode: str,
    category: str | None = None,
) -> str:
    """Generate guidance with Gemini, grounded in the provided facts.

    Raises if no key is configured or the call fails, so the generate_guidance
    wrapper can fall back to the template. Temperature 0.4 for light variation.
    """
    llm = get_gemini(temperature=0.4)
    if llm is None:
        raise RuntimeError("No GOOGLE_API_KEY configured")

    from langchain_core.prompts import ChatPromptTemplate

    persona = _MODE_PERSONA.get(mode, _MODE_PERSONA["newbie"])
    timing = _timing_clause(best_day, best_hour) or "an unspecified time"
    features = ", ".join(top_features[:3]) if top_features else "none provided"

    system = (
        "You write short posting guidance for a Reddit creator. "
        "Ground everything strictly in the facts provided; do NOT invent "
        "statistics, numbers, or subreddits. " + persona
    )
    user = (
        "Facts:\n"
        f"- Subreddit: r/{{subreddit}}\n"
        f"- Category: {{category}}\n"
        f"- Best time to post: {{timing}}\n"
        f"- Top model features: {{features}}\n\n"
        "Write the guidance now."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system), ("user", user)])
    chain = prompt | llm
    text = message_text(
        chain.invoke(
            {
                "subreddit": subreddit,
                "category": category or "general",
                "timing": timing,
                "features": features,
            }
        )
    ).strip()
    if not text:
        raise RuntimeError("Empty LLM response")
    return text


def generate_guidance(
    subreddit: str,
    best_hour: int | None,
    best_day: str | None,
    top_features: list[str],
    mode: str,
    category: str | None = None,
) -> str:
    """Return guidance text for one recommendation.

    Tries the Gemini path first; falls back to the deterministic template on any
    error or when no API key is present. Always returns a non-empty string.
    """
    try:
        text = llm_generate_guidance(
            subreddit, best_hour, best_day, top_features, mode, category
        )
        if text and text.strip():
            return text.strip()
    except Exception:
        pass
    return _template_guidance(subreddit, best_hour, best_day, top_features, mode)


def _strip_code_fences(text: str) -> str:
    """Strip a leading ```json / ``` fence and trailing ``` if Gemini adds one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def _llm_generate_guidance_batch(
    recs: list[dict],
    top_features: list[str],
    mode: str,
    category: str | None = None,
) -> list[str]:
    """One Gemini call for ALL recs; returns a guidance string per rec, in order.

    Raises if no key is configured, the call fails, or the reply doesn't parse
    into exactly len(recs) strings — so generate_guidance_batch can fall back to
    per-rec templates. This keeps guidance to a SINGLE network call per query.
    """
    llm = get_gemini(temperature=0.4)
    if llm is None:
        raise RuntimeError("No GOOGLE_API_KEY configured")

    from langchain_core.messages import HumanMessage, SystemMessage

    persona = _MODE_PERSONA.get(mode, _MODE_PERSONA["newbie"])
    features = ", ".join(top_features[:3]) if top_features else "none provided"

    lines = []
    for i, r in enumerate(recs, start=1):
        timing = _timing_clause(r.get("best_day"), r.get("best_hour")) or "an unspecified time"
        lines.append(f"{i}. r/{r['subreddit']} (best time to post: {timing})")
    subreddit_block = "\n".join(lines)

    system = (
        "You write short posting guidance for a Reddit creator. Ground everything "
        "strictly in the facts provided; do NOT invent statistics, numbers, or "
        "subreddits. " + persona + " Return ONLY a JSON array of exactly "
        f"{len(recs)} strings — one guidance string per subreddit, in the same "
        "order given. No markdown, no keys, no extra text."
    )
    user = (
        f"Category: {category or 'general'}\n"
        f"Top model features: {features}\n"
        f"Subreddits (in order):\n{subreddit_block}\n\n"
        "Write the guidance array now."
    )

    with perf_timer("gemini:guidance_batch"):
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = _strip_code_fences(message_text(response))
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) != len(recs):
        raise ValueError("Batch reply did not match rec count")
    return [str(item).strip() for item in data]


def generate_guidance_batch(
    recs: list[dict],
    top_features: list[str],
    mode: str,
    category: str | None = None,
) -> list[str]:
    """Guidance for every rec in a SINGLE Gemini call, aligned to recs order.

    Tries the batched LLM path first (one call); on any error or when no API key
    is present, falls back to the deterministic per-rec template (zero calls).
    Always returns len(recs) non-empty strings.
    """
    if not recs:
        return []

    try:
        texts = _llm_generate_guidance_batch(recs, top_features, mode, category)
        cleaned = [
            t.strip()
            if t and t.strip()
            else _template_guidance(
                r["subreddit"], r.get("best_hour"), r.get("best_day"), top_features, mode
            )
            for t, r in zip(texts, recs)
        ]
        return cleaned
    except Exception:
        return [
            _template_guidance(
                r["subreddit"], r.get("best_hour"), r.get("best_day"), top_features, mode
            )
            for r in recs
        ]
