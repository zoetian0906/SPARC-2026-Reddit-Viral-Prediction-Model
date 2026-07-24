"""
app.py — Streamlit front end (Phase D/E + UI pass).

Main-area layout (no sidebar): a free-text box drives the query via the keyword
parse stub; three dropdowns (mode / has media / engagement) refine it. Results
render only after "Get Recommendations" is clicked, and the level of detail
depends on the selected mode. Backend runs on real data (stub=False) with a
fallback to stub data if HuggingFace can't be reached.
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

MODE_LABELS = {"Newbie": "newbie", "Experienced": "experienced", "Expert": "expert"}
MEDIA_LABELS = {"Any": None, "Yes": "True", "No": "False"}
ENGAGEMENT_LABELS = {
    "Any": None, "Question": "question", "Showcase": "showcase", "Statement": "statement",
}

st.set_page_config(page_title="Reddit Viral Prediction")

# ── Header ──────────────────────────────────────────────────────────────────
st.title("Reddit Viral Prediction")
st.write("Find the best subreddit, timing, and strategy for your post.")

# ── Inputs (all in the main area) ───────────────────────────────────────────
query_text = st.text_area(
    "What do you want to post about?",
    height=100,
    placeholder=(
        "Describe what you want to post about...\n"
        "e.g. I want to share my sourdough recipe with beginners"
    ),
)

col1, col2, col3 = st.columns(3)
with col1:
    mode_label = st.selectbox("Mode", list(MODE_LABELS.keys()))
with col2:
    media_label = st.selectbox("Has Media", list(MEDIA_LABELS.keys()))
with col3:
    engagement_label = st.selectbox("Engagement type", list(ENGAGEMENT_LABELS.keys()))

st.caption(
    "Newbie: plain English advice | Experienced: data tables | "
    "Expert: full model details and feature importance"
)

get_clicked = st.button("Get Recommendations")


# ── Helpers ─────────────────────────────────────────────────────────────────
def _banner(confidence: str, sample_size) -> None:
    """Plain-English confidence banner shared by all modes."""
    if confidence == "high":
        n = sample_size if sample_size is not None else "many"
        st.success(
            f"Great match! Based on {n} similar posts, here are the best "
            "subreddits for this topic."
        )
    else:  # low
        st.warning(
            "We have some data on this, but not a lot. Take these as rough "
            "suggestions."
        )


def _fetch(category, post_type, mechanism, mode):
    """Call the backend on real data, falling back to stub data on failure."""
    try:
        return get_recommendations(
            category=category, post_type=post_type, mechanism=mechanism,
            mode=mode, stub=False,
        )
    except Exception:
        st.warning("Running on sample data (could not connect to HuggingFace).")
        return get_recommendations(
            category=category, post_type=post_type, mechanism=mechanism,
            mode=mode, stub=True,
        )


# ── Run + render (only after the button is clicked) ─────────────────────────
if get_clicked:
    mode = MODE_LABELS[mode_label]
    post_type = MEDIA_LABELS[media_label]
    mechanism = ENGAGEMENT_LABELS[engagement_label]
    category = None
    location = None

    if query_text.strip():
        parsed = parse_query(query_text)
        location = parsed["location_mentioned"]
        if parsed["category"] is None:
            st.warning(
                "Couldn't match your topic to a category. "
                "Try rephrasing, or use the dropdowns to refine."
            )
        else:
            category = parsed["category"]
            mechanism = parsed["mechanism"] or mechanism

    # Location disclaimer sits ABOVE the results.
    if location:
        st.caption(location_note(location))

    result = _fetch(category, post_type, mechanism, mode)

    confidence = result["confidence"]
    recommendations = result["recommendations"]
    notes = result["notes"]
    model_quality = result["model_quality"]
    drivers = result["drivers"]

    st.divider()

    if confidence == "none":
        st.error(result["confidence_reason"])
        if notes:
            st.info(notes[0])
    else:
        sample_size = model_quality.get("sample_size")
        _banner(confidence, sample_size)

        if mode == "newbie":
            if recommendations:
                lines = [
                    f"{i}. **{r['subreddit']}**"
                    for i, r in enumerate(recommendations, start=1)
                ]
                st.markdown("\n".join(lines))
                top = recommendations[0]
                if top.get("best_hour") is not None and top.get("best_day") is not None:
                    st.markdown(
                        f"**Best time to post:** {top['best_day']} around "
                        f"{top['best_hour']}:00"
                    )
            else:
                st.info(notes[0] if notes else "No specific subreddits to suggest.")

        else:  # experienced or expert
            if recommendations:
                cols = ["subreddit", "best_hour", "best_day"]
                if confidence == "high":
                    cols.append("predicted_score")
                cols.append("sample_size")
                df = pd.DataFrame(recommendations)[cols]
                st.dataframe(df, hide_index=True)
            else:
                st.info(notes[0] if notes else "No specific subreddits to suggest.")

            with st.expander("Model Quality"):
                st.write("test_r2:", model_quality["test_r2"])
                st.write("test_rmse:", model_quality["test_rmse"])
                st.write("sample_size:", model_quality["sample_size"])

            if mode == "expert":
                if drivers:
                    with st.expander("Feature Importance (SHAP)"):
                        ddf = (
                            pd.DataFrame(drivers)
                            .sort_values("shap_value", ascending=False)
                            .set_index("feature")
                        )
                        st.bar_chart(ddf["shap_value"])
                with st.expander("Notes"):
                    for note in notes:
                        st.write("-", note)
