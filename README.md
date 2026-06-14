# SPARC 2026 — Reddit Viral Prediction Model

A web-based Reddit viralness prediction and strategy analytics platform.  
Built by UPenn MAS CS students, Summer 2026.

## What We're Building

A tool that analyzes historical Reddit data to predict what makes posts go viral 
and recommends data-backed posting strategies. Users can input a topic or sector 
and receive recommendations on where to post, when to post, what format to use, 
and what signals drive engagement.

## North Star Metric

Can our model identify which posts are likely to perform in the top quartile of 
engagement within their subreddit or sector?

## Team

| Name | Role | Focus |
|------|------|-------|
| Zoe Tian | PM + ML Tech Lead | Data collection, model architecture |
| Kristin Lai | Data Engineer | Schema design, pipeline |
| Sarah Gillis | ML/DS Research | Research questions, model target |

## Stack

Python, PRAW, DuckDB, XGBoost, scikit-learn, MLflow, LangChain, FastAPI, Streamlit, GitHub Actions

## Project Status

| Week | Focus | Status |
|------|-------|--------|
| W1 | Kickoff, scope, roles | ✅ Done |
| W2 | Data discovery, schema draft | 🔄 In progress |

## Repo Structure

├── data/
│   ├── raw/          # collected Reddit data
│   └── processed/    # cleaned, feature-engineered data
├── notebooks/
│   └── zoe/          # data collection
├── src/              # production scripts
├── docs/             # technical documentation
└── requirements.txt

## Final Deliverable

Working web demo + Reddit data pipeline + viralness prediction model  
Final SPARC presentation: August 19, 2026
