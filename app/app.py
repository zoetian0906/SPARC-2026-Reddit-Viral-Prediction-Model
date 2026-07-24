"""
app.py — minimal Streamlit front end (Phase D).

Stub data only: the single backend import is get_recommendations(..., stub=True),
so this boots fully offline with no HuggingFace or DuckDB. Intentionally ugly and
functional; Kristin replaces the rendering later.
"""

import os
import sys

# Ensure the repo root is importable so `app.backend...` resolves when launched
# via `streamlit run app/app.py` (Streamlit puts app/ on sys.path, not the root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from app.backend.contract import get_recommendations

CATEGORIES = [
    "Career & Work", "Fitness & Health", "Food & Cooking", "Gaming",
    "Home & Interior", "Mental Health", "Personal Finance",
    "Relationships & Advice", "Skincare & Beauty", "Tech & Gadgets",
]

st.set_page_config(page_title="Reddit Viral Prediction")
st.title("Reddit Viral Prediction")

# ── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("Query")

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

# ── Query the backend (stub) ───────────────────────────────────────────────
result = get_recommendations(
    category=category,
    post_type=post_type,
    mechanism=mechanism,
    mode=mode,
    stub=True,
)

recommendations = result["recommendations"]
notes = result["notes"]

# ── Confidence banner ──────────────────────────────────────────────────────
confidence = result["confidence"]
reason = result["confidence_reason"]
if confidence == "high":
    st.success(reason)
elif confidence == "low":
    st.warning(reason)
else:
    st.error(reason)

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
