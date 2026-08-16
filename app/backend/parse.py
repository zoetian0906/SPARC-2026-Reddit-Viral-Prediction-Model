"""
parse.py — text-to-params.

Category detection uses Gemini (llm_detect_category) when a GOOGLE_API_KEY is
configured, and falls back to the keyword heuristic (_detect_category) otherwise
— so parse_query never raises, works fully offline, and stays fast in tests
(which run with no key). Mechanism and location detection remain keyword/regex
based. The signature parse_query(text) -> dict is stable.

Pure logic apart from the optional LLM call: no DuckDB; Streamlit/langchain are
imported lazily inside app.backend.llm only when a key is present.
"""

from __future__ import annotations

import os
import re
import sys

from app.backend.llm import get_gemini, message_text, perf_timer

# The fixed set of categories the LLM must map to (or NONE).
CATEGORIES: list[str] = [
    "Career & Work", "Fitness & Health", "Food & Cooking", "Gaming",
    "Home & Interior", "Mental Health", "Personal Finance",
    "Relationships & Advice", "Skincare & Beauty", "Tech & Gadgets",
]


def _normalize_category(text: str) -> str:
    """Normalize a category string for tolerant matching.

    Lowercases, strips surrounding quotes/markdown/whitespace and leading/trailing
    punctuation, unifies "&" with "and", and collapses internal whitespace. So
    "Skincare & Beauty.", " skincare and beauty ", and "**Skincare&Beauty**" all
    normalize to the same "skincare and beauty".
    """
    s = text.strip().lower()
    s = s.strip("`*\"' \t\r\n")          # markdown / quotes
    s = s.strip(".,!?:;")                  # trailing/leading sentence punctuation
    s = s.replace("&", " and ")           # unify ampersand vs the word "and"
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Normalized form -> canonical category, so any formatting variance in the LLM
# reply resolves back to the exact canonical string.
_CATEGORY_LOOKUP: dict[str, str] = {_normalize_category(c): c for c in CATEGORIES}


def _match_category(reply: str) -> str | None:
    """Resolve a raw LLM reply to a canonical category, or None.

    Tries an exact normalized match first, then a containment check so replies
    with preamble (e.g. "Category: Skincare & Beauty") still resolve.
    """
    norm = _normalize_category(reply)
    if not norm or norm == "none":
        return None
    if norm in _CATEGORY_LOOKUP:
        return _CATEGORY_LOOKUP[norm]
    # Preamble/extra words: find a canonical name embedded in the reply.
    for canon_norm, canonical in _CATEGORY_LOOKUP.items():
        if canon_norm in norm:
            return canonical
    return None


def _dbg(msg: str) -> None:
    """Print a debug line to stderr only when DEBUG_LLM is set (off by default)."""
    if os.environ.get("DEBUG_LLM"):
        print(msg, file=sys.stderr, flush=True)

# Lowercase keyword -> category name. Substring match against lowercased text.
# "Cover obvious mappings" — extend freely; this is a heuristic stub.
CATEGORY_KEYWORDS: dict[str, str] = {
    "cook": "Food & Cooking",
    "recipe": "Food & Cooking",
    "food": "Food & Cooking",
    "baking": "Food & Cooking",
    "meal": "Food & Cooking",
    "restaurant": "Food & Cooking",
    "game": "Gaming",
    "gaming": "Gaming",
    "xbox": "Gaming",
    "playstation": "Gaming",
    "nintendo": "Gaming",
    "skin": "Skincare & Beauty",
    "skincare": "Skincare & Beauty",
    "acne": "Skincare & Beauty",
    "moisturizer": "Skincare & Beauty",
    "lipstick": "Skincare & Beauty",
    "makeup": "Skincare & Beauty",
    "beauty": "Skincare & Beauty",
    "cosmetic": "Skincare & Beauty",
    "foundation": "Skincare & Beauty",
    "mascara": "Skincare & Beauty",
    "serum": "Skincare & Beauty",
    "sunscreen": "Skincare & Beauty",
    "perfume": "Skincare & Beauty",
    "fragrance": "Skincare & Beauty",
    "cologne": "Skincare & Beauty",
    "nail": "Skincare & Beauty",
    "haircare": "Skincare & Beauty",
    "invest": "Personal Finance",
    "budget": "Personal Finance",
    "savings": "Personal Finance",
    "retirement": "Personal Finance",
    "money": "Personal Finance",
    "rent": "Personal Finance",
    "mortgage": "Personal Finance",
    "apartment": "Personal Finance",
    "lease": "Personal Finance",
    "career": "Career & Work",
    "job": "Career & Work",
    "resume": "Career & Work",
    "interview": "Career & Work",
    "salary": "Career & Work",
    "side hustle": "Career & Work",
    "freelance": "Career & Work",
    "remote work": "Career & Work",
    "workout": "Fitness & Health",
    "gym": "Fitness & Health",
    "fitness": "Fitness & Health",
    "exercise": "Fitness & Health",
    "diet": "Fitness & Health",
    "yoga": "Fitness & Health",
    "pilates": "Fitness & Health",
    "stretching": "Fitness & Health",
    "running": "Fitness & Health",
    "cardio": "Fitness & Health",
    "mental": "Mental Health",
    "anxiety": "Mental Health",
    "therapy": "Mental Health",
    "depression": "Mental Health",
    "stress": "Mental Health",
    "burnout": "Mental Health",
    "overwhelmed": "Mental Health",
    "meditation": "Mental Health",
    "relationship": "Relationships & Advice",
    "dating": "Relationships & Advice",
    "breakup": "Relationships & Advice",
    "marriage": "Relationships & Advice",
    "tech": "Tech & Gadgets",
    "laptop": "Tech & Gadgets",
    "phone": "Tech & Gadgets",
    "software": "Tech & Gadgets",
    "coding": "Tech & Gadgets",
    "home": "Home & Interior",
    "interior": "Home & Interior",
    "furniture": "Home & Interior",
    "renovation": "Home & Interior",
}

# Phrases that signal a "showcase" post.
SHOWCASE_SIGNALS: list[str] = [
    "check out", "built", "made", "my project", "just finished",
]

# Simple location signals: "in <City>" / "near <Place>" (capitalized word).
# Not comprehensive by design — a teammate/LLM will improve this later.
LOCATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:in|near)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)"),
]


def llm_detect_category(text: str) -> str | None:
    """Map free text to one of CATEGORIES via Gemini, or None.

    The reply is resolved tolerantly (_match_category): case-insensitive, "&"/"and"
    unified, punctuation/markdown stripped, with a containment fallback for
    preamble. Returns None on ANY failure (no key, network error, or a reply that
    resolves to nothing) so parse_query can fall back to the keyword heuristic.
    Temperature 0 for deterministic mapping. Set DEBUG_LLM=1 to log raw replies.
    """
    try:
        llm = get_gemini(temperature=0.0)
        if llm is None:
            _dbg(f"[dbg] llm_detect_category({text!r}): get_gemini returned None (no key?)")
            return None

        from langchain_core.prompts import ChatPromptTemplate

        system = (
            "You classify a user's Reddit post topic into EXACTLY ONE of these "
            "categories, or NONE if none is a reasonable fit.\n"
            "Categories: " + ", ".join(CATEGORIES) + ".\n"
            "Rules:\n"
            "- Reply with ONLY the category name copied verbatim from the list, "
            "or the single word NONE. No punctuation, quotes, or explanation.\n"
            "- Handle single keywords, full sentences, and likely typos by "
            "interpreting the user's intended meaning.\n"
            "Examples:\n"
            "perfume -> Skincare & Beauty\n"
            "yoja -> Fitness & Health\n"
            "I want to share my baking recipe -> Food & Cooking\n"
            "a spinning hat with lights -> NONE"
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("user", "{text}")]
        )
        chain = prompt | llm
        with perf_timer("gemini:category"):
            reply = message_text(chain.invoke({"text": text}))
        matched = _match_category(reply)
        _dbg(f"[dbg] llm_detect_category({text!r}): raw={reply!r} matched={matched!r}")
        return matched
    except Exception as e:
        _dbg(f"[dbg] llm_detect_category({text!r}): EXCEPTION {type(e).__name__}: {e}")
        return None


def _detect_category(text_lower: str) -> str | None:
    """Return the category with the most keyword hits, or None."""
    counts: dict[str, int] = {}
    for keyword, category in CATEGORY_KEYWORDS.items():
        if keyword in text_lower:
            counts[category] = counts.get(category, 0) + 1
    if not counts:
        return None
    # Most hits wins; ties resolved by first-seen order (stable max).
    return max(counts, key=lambda c: counts[c])


def _detect_mechanism(text: str, text_lower: str, category: str | None) -> str | None:
    """Question if it asks, showcase if it announces, else statement.

    Returns None only when there is no usable signal at all (no category and no
    explicit question/showcase marker), e.g. empty or gibberish input.
    """
    if "?" in text:
        return "question"
    if any(sig in text_lower for sig in SHOWCASE_SIGNALS):
        return "showcase"
    if category is not None:
        return "statement"
    return None


def _detect_location(text: str) -> str | None:
    for pattern in LOCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def parse_query(text: str) -> dict:
    """Parse free text into recommendation params (keyword stub).

    Never raises. post_type is always None because plain text carries no media
    signal. Returns the fixed dict shape every time.
    """
    text = text or ""
    stripped = text.strip()

    if not stripped:
        return {
            "category": None,
            "post_type": None,
            "mechanism": None,
            "location_mentioned": None,
            "raw_text": text,
        }

    text_lower = stripped.lower()
    # Prefer the LLM mapping; fall back to keywords when it returns None/errors
    # (including the no-API-key case, which keeps tests offline and fast).
    category = llm_detect_category(stripped)
    if category is None:
        category = _detect_category(text_lower)
    mechanism = _detect_mechanism(stripped, text_lower, category)
    location = _detect_location(stripped)

    return {
        "category": category,
        "post_type": None,
        "mechanism": mechanism,
        "location_mentioned": location,
        "raw_text": text,
    }


def location_note(location: str) -> str:
    """Human-readable disclaimer that the model is not geographic."""
    return (
        f"You mentioned {location}. Our data isn't geographic, so this is "
        "general guidance, not specific to that area."
    )
