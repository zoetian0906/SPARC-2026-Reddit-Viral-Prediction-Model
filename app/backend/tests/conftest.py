"""
Shared test fixtures.

Force the "no API key" path for the whole test suite so tests never make real
Gemini calls and stay fast/offline/deterministic — regardless of whether a
GOOGLE_API_KEY happens to be set in the developer's environment. With no key,
get_gemini() returns None and parse.py / guidance.py hit their stub fallbacks.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_google_key(monkeypatch):
    monkeypatch.setattr("app.backend.llm.get_google_key", lambda: None)
