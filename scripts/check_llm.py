"""
check_llm.py — does the category LLM actually run, and what does it say?

parse.py swallows every LLM failure (parse.py:208) and returns None, which is
indistinguishable from "no category". This prints the step that actually failed.

Usage:
    python scripts/check_llm.py
    python scripts/check_llm.py yogga sunscren "moisturiser for oily skin"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DEBUG_LLM", "1")  # turn on parse.py's _dbg lines

from app.backend import llm as llm_mod
from app.backend.parse import llm_detect_category, parse_query

DEFAULT_PROBES = ["yogga", "sunscren", "yoga", "sunscreen"]


def main() -> None:
    print("=" * 70, "\n1. KEYS VISIBLE TO THE APP\n", "=" * 70, sep="")
    for name, getter in [
        ("GROQ_API_KEY", getattr(llm_mod, "get_groq_key", None)),
        ("GOOGLE_API_KEY", llm_mod.get_google_key),
    ]:
        if getter is None:
            print(f"  {name:16} n/a — get_groq_key missing (Groq edits not applied)")
            continue
        key = getter()
        print(f"  {name:16} {'FOUND (' + key[:6] + '...)' if key else 'MISSING'}")

    print("\n" + "=" * 70, "\n2. CLIENT CONSTRUCTION\n", "=" * 70, sep="")
    getter = getattr(llm_mod, "get_llm", llm_mod.get_gemini)
    print(f"  using {getter.__name__}()")
    try:
        client = getter()
        print(f"  -> {type(client).__name__ if client else 'None (keyword fallback only)'}")
    except Exception as e:
        print(f"  -> RAISED {type(e).__name__}: {e}")

    print("\n" + "=" * 70, "\n3. ROUND TRIP: prompt -> reply -> category\n", "=" * 70, sep="")
    for text in sys.argv[1:] or DEFAULT_PROBES:
        matched = llm_detect_category(text)  # DEBUG_LLM prints the raw reply
        parsed = parse_query(text)
        route = parsed.get("category_source", "?")
        print(f"  {text!r:22} llm={matched!r:24} final={parsed['category']!r:24} via {route}")


if __name__ == "__main__":
    main()
