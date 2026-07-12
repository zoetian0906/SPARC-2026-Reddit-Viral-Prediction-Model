"""
Reddit Data Collection — Zoe
Source: OpenCo7/UpVoteWeb (HuggingFace)
Goal: Stream and filter posts from 50 target subreddits across 10 categories
"""

from datasets import load_dataset
import pandas as pd
import os
from datetime import datetime

# ── Target subreddits by category ──────────────────────────────────────────
SUBREDDITS = {
    "skincare_beauty": ["SkincareAddiction","AsianBeauty","30PlusSkinCare","MakeupAddiction","Makeup"],
    "personal_finance": ["personalfinance","financialindependence","frugal","investing","Debt"],
    "food_cooking":     ["Cooking","MealPrepSunday","EatCheapAndHealthy","Baking","food"],
    "fitness_health":   ["loseit","xxfitness","running","weightlifting","flexibility"],
    "tech_gadgets":     ["gadgets","homelab","buildapc","apple","Android"],
    "mental_health":    ["mentalhealth","anxiety","depression","therapy","selfimprovement"],
    "relationships":    ["relationship_advice","AmItheAsshole","dating_advice","Marriage","Divorce"],
    "career_work":      ["cscareerquestions","careerguidance","jobs","WorkReform","remotework"],
    "gaming":           ["gaming","pcgaming","indiegaming","patientgamers","gamedev"],
    "home_interior":    ["malelivingspace","femalelivingspace","InteriorDesign","DIY","HomeImprovement"],
}

# Flat set for fast lookup
TARGET_SUBS = {sub.lower() for subs in SUBREDDITS.values() for sub in subs}

# ── Output path ─────────────────────────────────────────────────────────────
os.makedirs("data/raw", exist_ok=True)
OUTPUT_PATH = "data/raw/reddit_raw.parquet"

# ── Stream and filter ────────────────────────────────────────────────────────
print(f"Starting stream at {datetime.now().strftime('%H:%M:%S')}")
print(f"Filtering for {len(TARGET_SUBS)} subreddits across 10 categories")

collected = []
checked = 0
CHECKPOINT_EVERY = 100_000

dataset = load_dataset(
    "BEE-spoke-data/upvoteweb-posts",
    split="train",
    streaming=True,
)

for row in dataset:
    checked += 1

    sub = str(row.get("subreddit", "")).lower().strip()
    if sub in TARGET_SUBS:
        collected.append({
            "post_id":      row.get("post_id"),
            "subreddit":    row.get("subreddit"),
            "text":         row.get("text"),
            "url":          row.get("url"),
            "score":        row.get("score"),
            "date":         row.get("date"),
            "author":       row.get("author"),
            "token_count":  row.get("token_count"),
            "language":     row.get("language"),
        })

    if checked % CHECKPOINT_EVERY == 0:
        print(f"  Checked {checked:,} rows — collected {len(collected):,} matches so far")

# ── Save ─────────────────────────────────────────────────────────────────────
df = pd.DataFrame(collected)
df.to_parquet(OUTPUT_PATH, index=False)

print(f"\nDone. {len(collected):,} posts collected from {df['subreddit'].nunique()} subreddits")
print(f"Saved to {OUTPUT_PATH}")
print(f"\nSubreddit breakdown:")
print(df['subreddit'].value_counts().to_string())
