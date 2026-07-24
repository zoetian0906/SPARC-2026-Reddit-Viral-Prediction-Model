"""
parse.py — keyword-based text-to-params (Phase E).

This is a STUB. A teammate will later replace parse_query with an LLM call; the
signature (parse_query(text) -> dict) must stay stable so the swap is a drop-in.

Pure logic: no network, no Streamlit, no DuckDB.
"""

from __future__ import annotations

import re

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
    "invest": "Personal Finance",
    "budget": "Personal Finance",
    "savings": "Personal Finance",
    "retirement": "Personal Finance",
    "money": "Personal Finance",
    "career": "Career & Work",
    "job": "Career & Work",
    "resume": "Career & Work",
    "interview": "Career & Work",
    "salary": "Career & Work",
    "workout": "Fitness & Health",
    "gym": "Fitness & Health",
    "fitness": "Fitness & Health",
    "exercise": "Fitness & Health",
    "diet": "Fitness & Health",
    "mental": "Mental Health",
    "anxiety": "Mental Health",
    "therapy": "Mental Health",
    "depression": "Mental Health",
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
