"""
guidance.py — per-recommendation guidance text.

generate_guidance uses Gemini (llm_generate_guidance) when a GOOGLE_API_KEY is
configured, and falls back to a deterministic template (_template_guidance)
otherwise — so the contract's "guidance" field is always a non-empty string,
works offline, and stays fast in tests (which run with no key). The public
signature is unchanged, so contract.py's call is untouched: the LLM swap is
invisible to it.

Streamlit/langchain are imported lazily inside app.backend.llm only when a key
is present.
"""

from __future__ import annotations

from app.backend.llm import get_gemini, message_text

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
