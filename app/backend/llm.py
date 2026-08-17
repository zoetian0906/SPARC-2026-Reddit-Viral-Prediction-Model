"""
llm.py — shared Gemini (Google Generative AI) plumbing.

Centralizes API-key resolution and client creation so parse.py and guidance.py
can call the LLM with a single, testable integration point. All heavy imports
(streamlit, langchain) are LAZY so importing this module — and therefore the
pure stubs that depend on it — never requires those packages or a running
Streamlit at import time. If no key is configured, get_gemini() returns None and
callers fall back to their existing stub/template behavior.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

MODEL_NAME = "gemini-3.5-flash"

# Hard caps so a single request can never hang the app. langchain-google-genai
# defaults to NO timeout and max_retries=6 (a retry storm under transient errors);
# these bounds keep each call fast and bounded.
REQUEST_TIMEOUT_S = 8.0
MAX_RETRIES = 1


@contextmanager
def perf_timer(label: str):
    """Print '[perf] <label>: N.NNs' to stderr (visible in Streamlit Cloud logs)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        print(
            f"[perf] {label}: {time.perf_counter() - start:.2f}s",
            file=sys.stderr,
            flush=True,
        )


def get_google_key() -> str | None:
    """Return the Google API key from st.secrets, then env, else None.

    Never hardcoded. st.secrets is read lazily and defensively (accessing it
    without a Streamlit runtime, or with no secrets file, must not raise).
    """
    key = None
    try:
        import streamlit as st

        key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        key = None
    if not key:
        key = os.environ.get("GOOGLE_API_KEY")
    return key or None


def get_gemini(temperature: float = 0.0):
    """Return a ChatGoogleGenerativeAI client, or None if no key is present.

    langchain is imported lazily so this module stays importable without it.
    """
    key = get_google_key()
    if not key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
        google_api_key=key,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=MAX_RETRIES,
    )


def message_text(response) -> str:
    """Flatten a LangChain message .content into a plain string.

    Gemini can return content either as a string or as a list of content-part
    dicts (e.g. [{"type": "text", "text": "..."}]); handle both.
    """
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)

# ── Groq ──────────────────────────────────────────────────────────────────────
# Groq retired the llama-3.x ids on 2026-06-17; gpt-oss is the named successor
# pair. 120b for prose, 20b for classification (cheaper and much faster).
GROQ_ADVICE_MODEL = "openai/gpt-oss-120b"
GROQ_FAST_MODEL = "openai/gpt-oss-20b"


def get_groq_key() -> str | None:
    """Return the Groq API key from st.secrets, then env, else None.

    Same defensive pattern as get_google_key: st.secrets must not raise outside
    a Streamlit runtime.
    """
    key = None
    try:
        import streamlit as st

        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = None
    if not key:
        key = os.environ.get("GROQ_API_KEY")
    return key or None


def get_groq(temperature: float = 0.0, model: str = GROQ_ADVICE_MODEL):
    """Return a ChatGroq client, or None if no key is present."""
    key = get_groq_key()
    if not key:
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=key,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=MAX_RETRIES,
    )


def get_llm(temperature: float = 0.0, model: str = GROQ_ADVICE_MODEL):
    """Prefer Groq, fall back to Gemini, else None.

    This is the spread-the-load switch: call sites use get_llm and get Groq
    whenever GROQ_API_KEY is set, without losing the Gemini path if it isn't.
    Both clients are LangChain chat models, so message_text and the existing
    prompt|llm chains work unchanged against either.
    """
    return get_groq(temperature, model) or get_gemini(temperature)
