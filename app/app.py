"""
app.py — minimal Streamlit front end (Phase D + Phase E parse layer).

A free-text box (keyword parse stub) sits above the manual dropdowns. Text input
drives the query when present; otherwise the selectboxes do (fallback). The
backend runs on real data (stub=False), falling back to stub data if HuggingFace
can't be reached. Intentionally ugly and functional; Kristin replaces the
rendering later.
"""

import os
import sys

# Ensure the repo root is importable so `app.backend...` resolves when launched
# via `streamlit run app/app.py` (Streamlit puts app/ on sys.path, not the root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from app.backend.contract import get_recommendations
from app.backend.parse import location_note, parse_query

CATEGORIES = [
    "Career & Work", "Fitness & Health", "Food & Cooking", "Gaming",
    "Home & Interior", "Mental Health", "Personal Finance",
    "Relationships & Advice", "Skincare & Beauty", "Tech & Gadgets",
]

st.set_page_config(page_title="Reddit Viral Prediction")
st.title("Reddit Viral Prediction")

# ── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("Query")

# Free-text box (Phase E parse stub). When filled, it drives the query and the
# dropdowns act as a fallback.
query_text = st.sidebar.text_input(
    "What do you want to post about?",
    placeholder="e.g. best sourdough recipe tips",
)

category = st.sidebar.selectbox("Category", CATEGORIES)

post_type = st.sidebar.selectbox(
    "Has Media",
    [None, "True", "False"],
    format_func=lambda x: "Any" if x is None else x,
)

mechanism = st.sidebar.selectbox(
    "Engagement mechanism",
    [None, "question", "showcase", "statement"],
    format_func=lambda x: "Any" if x is None else x,
)

mode = st.sidebar.selectbox("Mode", ["newbie", "experienced", "expert"])

# ── Resolve query params: text input (parsed) takes priority over dropdowns ──
extra_notes: list[str] = []
q_category, q_post_type, q_mechanism = category, post_type, mechanism

if query_text.strip():
    parsed = parse_query(query_text)
    if parsed["location_mentioned"]:
        extra_notes.append(location_note(parsed["location_mentioned"]))
    if parsed["category"] is None:
        st.warning(
            "Couldn't match your topic to a category. "
            "Try the dropdowns below or rephrase."
        )
        # Fall through to the manual selectbox values (already the defaults).
    else:
        q_category = parsed["category"]
        q_post_type = parsed["post_type"]      # always None from text
        q_mechanism = parsed["mechanism"]

# ── Query the backend (real data, stub fallback) ───────────────────────────
try:
    result = get_recommendations(
        category=q_category,
        post_type=q_post_type,
        mechanism=q_mechanism,
        mode=mode,
        stub=False,
    )
except Exception as e:
    st.warning(f"Running on sample data: {e}")
    result = get_recommendations(
        category=q_category,
        post_type=q_post_type,
        mechanism=q_mechanism,
        mode=mode,
        stub=True,
    )

recommendations = result["recommendations"]
notes = list(result["notes"])  # extra_notes (location) rendered separately below

# ── Confidence banner ──────────────────────────────────────────────────────
confidence = result["confidence"]
reason = result["confidence_reason"]
if confidence == "high":
    st.success(reason)
elif confidence == "low":
    st.warning(reason)
else:
    st.error(reason)

# Location disclaimer (from the parse layer) — shown regardless of results.
for note in extra_notes:
    st.caption(note)

# ── Recommendations ────────────────────────────────────────────────────────
if recommendations:
    # Hide best_hour / best_day columns while they are None.
    optional_cols = [
        c for c in ("best_hour", "best_day")
        if any(r.get(c) is not None for r in recommendations)
    ]
    display_cols = ["subreddit"] + optional_cols + ["predicted_score", "sample_size"]
    df = pd.DataFrame(recommendations)[display_cols]
    st.dataframe(df, hide_index=True)
else:
    st.info(notes[0] if notes else "No recommendations for this query.")

# ── Model quality (experienced + expert) ───────────────────────────────────
if mode in ("experienced", "expert"):
    with st.expander("Model Quality"):
        mq = result["model_quality"]
        st.write("test_r2:", mq["test_r2"])
        st.write("test_rmse:", mq["test_rmse"])
        st.write("sample_size:", mq["sample_size"])

# ── Feature importance / SHAP (expert only) ────────────────────────────────
drivers = result["drivers"]
if mode == "expert" and drivers:
    with st.expander("Feature Importance (SHAP)"):
        ddf = (
            pd.DataFrame(drivers)
            .sort_values("shap_value", ascending=False)
            .set_index("feature")
        )
        st.bar_chart(ddf["shap_value"])

# ── Notes (only when not already shown in the empty state) ──────────────────
if notes and recommendations:
    with st.expander("Notes"):
        for note in notes:
            st.write("-", note)
