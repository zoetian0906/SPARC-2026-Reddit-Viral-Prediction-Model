"""
app.py — Streamlit front end (Phase D/E + UI pass).

Main-area layout (no sidebar for inputs): a free-text box drives the query via the keyword
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

MODE_LABELS = {"Experienced": "experienced", "Technical Professional": "technical"}
MEDIA_LABELS = {"Any": None, "Yes": "True", "No": "False"}
ENGAGEMENT_LABELS = {
    "Question": "question", "Showcase": "showcase", "Statement": "statement",
}

st.set_page_config(page_title="Reddit Viral Prediction", layout="wide", page_icon="🔴")

# --- CUSTOM REDDIT CSS INJECTION ---
st.markdown("""
<style>
    /* 1. Page Background (Grey) and Default Font */
    .stApp {
        background-color: #DAE0E6; 
        font-family: 'IBM Plex Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* 2. Main Content Area (White Card) */
    .block-container {
        background-color: #FFFFFF;
        padding: 2rem 3rem !important;
        border-radius: 5px;
        border: 1px solid #CCCCCC;
        margin-top: 2rem;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 3. Red/Orange Outlines when clicking inputs/dropdowns */
    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {
        border-color: #FF4500 !important;
        box-shadow: 0 0 0 1px #FF4500 !important;
    }

    /* 4. Primary Button Styling */
    .stButton>button {
        background-color: #FF4500 !important;
        color: white !important;
        border-radius: 9999px !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    .stButton>button:hover {
        background-color: #E03D00 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["Viral Predictor", "About this Tool"])

if page == "Viral Predictor":
    # ── Header with Reddit Logo ──────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 0px;">
            <img src="https://www.redditstatic.com/shreddit/assets/favicon/192x192.png" width="42" height="42" style="border-radius: 50%;">
            <h1 style="margin: 0; padding: 0; font-size: 2.2rem;">Reddit Viral Prediction</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("Find the best subreddit, timing, and strategy for your post.")

    # ── Inputs ──────────────────────────────────────────────────────────────────
    # First load is pre-filled with a working example so a complete result shows
    # immediately as a tutorial (Item 7). Everything is obviously editable.
    query_text = st.text_area(
        "What do you want to post about?",
        value="sunscreen",
        height=100,
        help="Describe what you want to post about or paste your draft concept here to analyze virality potential."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        mode_label = st.selectbox(
            "Mode",
            list(MODE_LABELS.keys()),
            index=0,  # Experienced is the default (required, no blank option)
            help=(
                "**Choose your detail level (required):**\n\n"
                "• **Experienced**: Ranked subreddit table with best posting times and "
                "actionable guidance. Best for standard content strategy.\n\n"
                "• **Technical Professional**: Everything in Experienced, plus the ML "
                "model quality metrics (R², RMSE, sample size) and ranked feature "
                "importances. Best for data scientists and marketers."
            )
        )
    with col2:
        media_label = st.selectbox(
            "Has Media", 
            list(MEDIA_LABELS.keys()),
            help="Filter recommendations based on whether your post contains media (images/videos)."
        )
    with col3:
        engagement_label = st.selectbox(
            "Engagement type",
            list(ENGAGEMENT_LABELS.keys()),
            index=0,  # Question is the default (required, no "Any" option)
            help=(
                "**Select the intent of your post (required):**\n\n"
                "• **Question**: asking the community something "
                "(\"What sunscreen works for oily skin?\").\n\n"
                "• **Showcase**: sharing something you made "
                "(\"My 6-month skincare progress\").\n\n"
                "• **Statement**: sharing an opinion or info "
                "(\"SPF is the most underrated skincare step\")."
            )
        )

    st.caption(
        "Experienced: ranked subreddits, timing, and guidance | "
        "Technical Professional: adds model metrics and feature importance"
    )

    get_clicked = st.button("Get Recommendations")

    # Show the tutorial result on first load without a click (Item 7); after that,
    # the user changes inputs and clicks "Get Recommendations" (Item 4).
    first_load = "has_run" not in st.session_state
    if first_load:
        st.session_state["has_run"] = True
    should_run = get_clicked or first_load

    # ── Helpers ─────────────────────────────────────────────────────────────────
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

    # ── Run + render ────────────────────────────────────────────────────────────
    if should_run:
        mode = MODE_LABELS[mode_label]
        post_type = MEDIA_LABELS[media_label]
        # Engagement type is a required, explicit choice, so it drives the query.
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

        if location:
            st.caption(location_note(location))

        result = _fetch(category, post_type, mechanism, mode)

        confidence = result.get("confidence", "none")
        recommendations = result.get("recommendations", [])
        model_quality = result.get("model_quality", {})
        drivers = result.get("drivers", [])

        st.divider()

        # --- NO ERROR SCREENS FOR LOW/NONE CONFIDENCE ---
        if confidence == "none" or not recommendations:
            st.warning("Unfortunately we do not have meangingful data to issue a recommendation in that category.")
        else:
            if confidence == "low":
                st.info("Note: Recommendations are based on limited historical data for this category.")

            # Subreddits are already ranked by predicted engagement (highest first).
            df = pd.DataFrame(recommendations)
            df.insert(0, "Rank", range(1, len(df) + 1))

            cols = ["Rank", "subreddit", "best_hour", "best_day"]
            display_cols = [c for c in cols if c in df.columns]

            st.dataframe(
                df[display_cols].rename(columns={
                    "Rank": "Rank",
                    "subreddit": "Subreddit",
                    "best_hour": "Best Hour (UTC)",
                    "best_day": "Best Day"
                }),
                hide_index=True,
                use_container_width=True
            )

            # --- EXPERIENCED MODE ---
            if mode == "experienced":
                st.subheader("Actionable Guidance")
                for r in recommendations:
                    if r.get("guidance"):
                        st.write(f"**r/{r['subreddit']}**: {r['guidance']}")

            # --- TECHNICAL PROFESSIONAL MODE ---
            elif mode == "technical":
                st.subheader("Model Quality & Reliability")
                st.caption(
                    "**How to read this:** Test R² shows the proportion of variance in virality explained by our features (higher is better). "
                    "RMSE measures the average prediction error. Sample size indicates the volume of historical posts driving this output."
                )
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Test R²", round(model_quality.get("test_r2", 0), 3))
                col_b.metric("Test RMSE", round(model_quality.get("test_rmse", 0), 3))
                col_c.metric("Sample Size", model_quality.get("sample_size", 0))

                if drivers:
                    st.subheader("Top 3 Feature Importances (SHAP)")
                    # Rank by importance = absolute SHAP magnitude, highest first.
                    drivers_df = pd.DataFrame(drivers)
                    drivers_df["importance"] = drivers_df["shap_value"].abs()
                    drivers_df = drivers_df.sort_values("importance", ascending=False).head(3)
                    st.bar_chart(drivers_df.set_index("feature")["importance"])

                st.subheader("Strategic Guidance")
                for r in recommendations:
                    if r.get("guidance"):
                        st.write(f"**r/{r['subreddit']}**: {r['guidance']}")

elif page == "About this Tool":
    st.title("About the Reddit Virality Predictor")

    with st.expander("Product Motivation", expanded=True):
        st.write("Navigating Reddit can be challenging. This tool exists to help users identify the optimal subreddit, timing, and strategy to maximize the reach of their posts based on historical data.")

    with st.expander("Target User"):
        st.write("From casual posters wanting to share a recipe to digital marketers aiming for maximum engagement, this tool scales to your technical comfort level.")

    with st.expander("How to Use It"):
        st.write("- **Experienced Mode**: A ranked subreddit table with best posting times and actionable guidance. Best for standard content strategy.\n- **Technical Professional Mode**: Everything in Experienced, plus the underlying ML model metrics (R², RMSE, sample size) and ranked feature importances. Best for data scientists and marketers requiring high precision.")

    with st.expander("Project Background"):
        st.write("Built as part of a collaborative data engineering and machine learning pipeline. It provides directional guidance based on historical trends, rather than absolute guarantees of virality.")

    with st.expander("Built by"):
        st.markdown(
            "**Zoe Tian**  \n"
            "Backend, feature engineering, modeling, deployment\n\n"
            "**Sarah Gillis**  \n"
            "Machine learning, LLM integration\n\n"
            "**Kristin [last name]**  \n"
            "Data pipeline, frontend design\n\n"
            "---\n\n"
            "SPARC 2026 Summer Research Program  \n"
            "University of Pennsylvania  \n"
            "August 2026"
        )

    st.subheader("Questions or feedback?")
    st.markdown("[zoetian@engineering.upenn.edu](mailto:zoetian@engineering.upenn.edu)")

# ── Disclaimer (for Reddit Logo Use) ─────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style="text-align: center; margin-top: 1rem;">
        <p style="color: #787C7E; font-size: 0.9rem; font-weight: bold;">
            This is a SPARC 2026 student research project. Not for profit, not for commercial use.<br>
            <span style="font-size: 0.7rem; font-weight: normal;">Disclaimer: All logos and trademarks are property of their respective owners and are used for educational purposes only.</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)