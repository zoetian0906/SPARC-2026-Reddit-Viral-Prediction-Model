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

import re
from typing import Literal, Optional
from pydantic import BaseModel, Field

from app.backend.llm import get_gemini, message_text

# The fixed set of categories the LLM must map to (or NONE).
CATEGORIES: list[str] = [
    "Career & Work", "Fitness & Health", "Food & Cooking", "Gaming",
    "Home & Interior", "Mental Health", "Personal Finance",
    "Relationships & Advice", "Skincare & Beauty", "Tech & Gadgets",
]

# Lowercase keyword -> category name. Substring match against lowercased text.
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
    "mental": "Mental Health",
    "anxiety": "Mental Health",
    "therapy": "Mental Health",
    "depression": "Mental Health",
    "stress": "Mental Health",
    "burnout": "Mental Health",
    "overwhelmed": "Mental Health",
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
LOCATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(?:in|near)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)"),
]


class QueryData(BaseModel):
    category: Optional[Literal[
        "Career & Work", "Fitness & Health", "Food & Cooking", "Gaming",
        "Home & Interior", "Mental Health", "Personal Finance",
        "Relationships & Advice", "Skincare & Beauty", "Tech & Gadgets"
    ]] = Field(default=None, description="The closest matching category. Null if the text doesn't fit any.")
    mechanism: Optional[Literal["question", "showcase", "statement"]] = Field(
        default=None, description="'question' if asking something, 'showcase' if showing off a project/item, 'statement' if general commentary. Null if indeterminate."
    )
    location_mentioned: Optional[str] = Field(
        default=None, description="Any specific city, state, or geographic location mentioned. Null if none."
    )


def llm_parse_all(text: str) -> dict | None:
    """Parse all fields using Gemini structured output. Returns None on failure (like missing API key)."""
    try:
        llm = get_gemini(temperature=0.0)
        if llm is None:
            return None

        from langchain_core.prompts import ChatPromptTemplate
        structured_llm = llm.with_structured_output(QueryData)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent natural language parser. Extract the requested fields from the user's text based strictly on the provided schema."),
            ("user", "{text}")
        ])
        chain = prompt | structured_llm
        parsed = chain.invoke({"text": text})
        
        return {
            "category": parsed.category,
            "mechanism": parsed.mechanism,
            "location_mentioned": parsed.location_mentioned
        }
    except Exception:
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
    """Question if it asks, showcase if it announces, else statement."""
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
    """Parse free text into recommendation params using structured LLM (with heuristic fallback).

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
    
    # Prefer the LLM structured parser; fall back to keywords when it returns None/errors
    # (including the no-API-key case, which keeps tests offline and fast).
    llm_result = llm_parse_all(stripped)
    
    if llm_result is not None:
        category = llm_result["category"]
        mechanism = llm_result["mechanism"]
        location = llm_result["location_mentioned"]
    else:
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
