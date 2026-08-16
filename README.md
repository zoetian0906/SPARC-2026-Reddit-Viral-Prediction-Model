# Reddit Virality Predictor

**Data-backed guidance on where, when, and how to post on Reddit.**

A web platform that turns a topic into concrete, evidence-based posting guidance:
which community to post in, the best time to post, and how to frame it — all derived
from models trained on historical Reddit data, not opinions generated on the spot.

Built for the **SPARC 2026 Summer Research Program** at the University of Pennsylvania.
**Status: complete.** Final presentation: **August 19, 2026**.

---

## What it does

A user enters a topic (free text or dropdowns) and picks a display mode and an
engagement type. The app returns a ranked set of recommended subreddits, the best
hour and day to post, a predicted virality score, and framing guidance — each answer
carrying an explicit **confidence tier** so the tool is honest about how much to trust it.

A defining design principle is **honesty about uncertainty**: post-level virality is
genuinely hard to predict from content alone, so where the data is thin or the signal is
weak, the product says so instead of inventing a precise number.

### Highlights
- **Real models, not a wrapper.** Predictions come from trained, benchmarked gradient
  boosting models (XGBoost), selected and tuned per segment. The optional LLM layer only
  interprets input and explains results — it never invents the underlying numbers.
- **Honest about uncertainty.** Every response carries a `high` / `low` / `none`
  confidence tier and withholds a precise score when the evidence doesn't support one.
- **Decoupled and fast.** Models run offline in batch; predictions are precomputed and
  stored, and the live app only ever reads from those tables — no model inference at
  request time.
- **Tested.** ~129 automated tests run entirely on in-memory fixtures with no network
  dependency, verifying the contract between layers on every change.

---

## How it works

The system has five layers. The first four run **offline in batch**; only the
application layer runs **live** for the user, and it only ever *reads* precomputed tables.

```
OFFLINE (batch)                                  LIVE (per request, read-only)
────────────────────────────────                 ───────────────────────────────
Data    → collect + clean (DuckDB)               User input (text / dropdowns,
Feature → 16 features per post                                 mode, engagement)
Model   → segmented XGBoost bank                         │
Serving → batch inference → 2 parquet tables ──▶ parse.py     → {category, mechanism, location}
                                     (HuggingFace)  query.py     → segment + predictions lookup
                                                    confidence.py→ high / low / none
                                                    contract.py  → fixed-shape response
                                                    app.py       → renders by mode
```

### Confidence tiers

| Tier   | Condition                              | What the user sees                                  |
|--------|----------------------------------------|-----------------------------------------------------|
| `high` | R² ≥ 0.10 and sample ≥ 500             | Predicted score shown, encouraging copy             |
| `low`  | R² ≥ 0.00 and sample ≥ 100             | Subreddits + timing, no numeric score, hedged copy  |
| `none` | Worse than guessing, or too little data| Empty recommendations with an honest explanation    |

A large share of segments intentionally return `low` or `none` — a direct consequence of
honest thresholds applied to a hard problem. These are treated as real product moments,
not error screens.

---

## The backend contract

The seam between backend and frontend is a single function that always returns the same
shaped dictionary — this locked contract let the frontend and backend be built in parallel.

```python
from app.backend.contract import get_recommendations

result = get_recommendations(
    category="Food & Cooking",
    post_type=None,        # None | "True" | "False"  (has media)
    mechanism="question",  # None | "question" | "showcase" | "statement"
    mode="experienced",    # "experienced" | "technical"
    stub=False,
)
```

Free text is turned into structured fields by a separate parser (`app.backend.parse`),
which never raises and falls back to keyword matching when the LLM is unavailable.

---

## Repository structure

```
app/
├── app.py              # Streamlit front end (renders by display mode)
└── backend/            # pure, testable backend logic
    ├── contract.py     # get_recommendations — the fixed-shape response
    ├── parse.py        # free text -> {category, mechanism, location} (LLM + keyword fallback)
    ├── query.py        # DuckDB lookups: segment metadata + predictions (5-level fallback)
    ├── confidence.py   # high / low / none tiering
    ├── loader.py       # HuggingFace -> DuckDB, cached
    ├── guidance.py     # framing guidance (LLM with template fallback)
    ├── llm.py          # shared Gemini plumbing (timeouts, key resolution)
    ├── stub.py         # hardcoded fixtures for offline / fallback
    └── tests/          # ~129 pytest tests, no network required
src/                    # offline data + modeling pipeline
├── collection_script.py
├── processing_script.py
├── labeling_script.py
├── virality_label.py
└── build_model_ready.py
notebooks/              # exploration (zoe / Sarah / kristin)
scripts/                # data inspection + HuggingFace upload utilities
requirements.txt
pytest.ini
```

Code lives on GitHub; **data lives on HuggingFace**. Data files (`.parquet`, `.db`,
`.csv`, model binaries) are gitignored and never committed.

---

## Tech stack

- **Data & modeling:** Python, DuckDB, XGBoost, scikit-learn, Optuna, SHAP, MLflow
- **NLP features:** VADER (sentiment), NRC Emotion Lexicon (8 emotions), textstat (readability)
- **LLM layer:** LangChain + Google Gemini (input interpretation + framing guidance)
- **App & serving:** Streamlit, HuggingFace Datasets (parquet), pandas
- **Quality & ops:** pytest, Streamlit Cloud (continuous deploy from `main`)

---

## Data

- **Source:** the UpVoteWeb 2024 Reddit dataset on HuggingFace, chosen as the most recent
  large, cross-subreddit dataset with the needed engagement signals and a research license.
- **Scale:** 96,170 matched posts across 48 subreddits.
- **Features:** 16 per post — VADER sentiment, 8 normalized NRC emotions, Flesch readability,
  and structural/contextual fields (title/body length, hour, day, subreddit, category,
  engagement mechanism). The final feature table has zero nulls.
- **Label:** `viral_score`, a continuous 0–100 target — a 50/50 blend of the post's
  normalized log score and its score relative to its subreddit average, so a strong post in
  a small community isn't buried beneath a mediocre post in a huge one.
- **Serving tables:** Table 1 (per-segment model quality + SHAP importances) and Table 2
  (recommended subreddits with best hour/day + predicted score), stored as parquet.

---

## Running locally

```bash
pip install -r requirements.txt

# Backend tests (no network / no secrets needed — runs on in-memory fixtures)
pytest -v

# Launch the app
streamlit run app/app.py
```

The live app reads private data from HuggingFace and (optionally) calls Gemini, so a full
run needs two secrets. Set them as environment variables or in `.streamlit/secrets.toml`:

- `HF_TOKEN` — HuggingFace access token for the prediction tables
- `GOOGLE_API_KEY` — enables the LLM parsing/guidance layer (falls back gracefully if absent)

If HuggingFace is unreachable, the app falls back to bundled sample data so the UI still works.

---

## Team

| Name | Role | Focus |
|------|------|-------|
| **Zoe Tian** | ML Tech Lead & Project Manager | Data acquisition, feature engineering, backend, deployment |
| **Kristin Lai** | Data Engineering & Frontend / UI/UX | DuckDB warehouse & pipeline, Streamlit app |
| **Sarah Gillis** | Data Science & Modeling | Virality metric, segmented model bank, LLM integration |

---

## Deployment

Deployed on **Streamlit Cloud**, which builds directly from the `main` branch (continuous
deploy). Because the app only reads precomputed prediction tables, the frontend can move
hosts without touching the modeling or data layers, and the prediction tables can be
regenerated and re-uploaded independently of any app deployment.

---

## Disclaimer

This is a SPARC 2026 student research project — not for profit, not for commercial use.
It provides **directional guidance** based on historical trends, not guarantees of
virality. All logos and trademarks are the property of their respective owners and are
used for educational purposes only.
