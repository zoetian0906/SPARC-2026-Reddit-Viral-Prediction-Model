import pandas as pd
from datasets import Dataset, DatasetDict

# Load raw data from a local CSV file
print("Loading raw data...")
df_raw = pd.read_csv("raw_reddit_data.csv")

# ---------------------------------------------------------
# TABLE 1: raw_posts 
# ---------------------------------------------------------
# Retain the original data structure as a backup and source of truth

# ---------------------------------------------------------
# TABLE 2: processed_posts 
# ---------------------------------------------------------
print("Cleaning data...")
df_processed = pd.DataFrame()
df_processed["post_id"] = df_raw["post_id"]

# Standardize text to lowercase for easier natural language processing
df_processed["clean_text"] = df_raw["text"].astype(str).str.lower()

# Flag posts where the content has been removed or deleted by moderators
df_processed["is_deleted"] = df_raw["text"].str.contains(r'\[deleted\]|\[removed\]', na=False, regex=True)

# Convert ISO date strings directly to pandas datetime objects
df_processed["posted_at_local"] = pd.to_datetime(df_raw["date"])

# ---------------------------------------------------------
# TABLE 3: post_features 
# ---------------------------------------------------------
print("Extracting features...")
df_features = pd.DataFrame()
df_features["post_id"] = df_raw["post_id"]

# Calculate word count for the primary text body
df_features["text_length"] = df_processed["clean_text"].apply(lambda x: len(str(x).split()))

# Extract temporal features for posting time analysis
df_features["hour_of_day"] = df_processed["posted_at_local"].dt.hour
df_features["day_of_week"] = df_processed["posted_at_local"].dt.dayofweek

# ---------------------------------------------------------
# TABLE 4: post_labels 
# ---------------------------------------------------------
print("Calculating benchmarks...")
df_labels = pd.DataFrame()
df_labels["post_id"] = df_raw["post_id"]

# Calculate the median engagement score grouped by subreddit
subreddit_baselines = df_raw.groupby("subreddit")["score"].transform("median")
df_labels["subreddit_baseline"] = subreddit_baselines

# Calculate viral score: ratio of post score to subreddit median
# (Adding +1 to the denominator to prevent division by zero errors)
df_labels["viral_score"] = df_raw["score"] / (subreddit_baselines + 1)

# Flag posts that fall into the top performance tier (1.5x the baseline)
df_labels["is_top_quartile"] = df_labels["viral_score"] > 1.5

# ---------------------------------------------------------
# PUSH TO HUGGING FACE
# ---------------------------------------------------------
print("Pushing tables to Hugging Face...")

# Package dataframes into a multi-table Dataset Dictionary
repo_tables = DatasetDict({
    "raw_posts": Dataset.from_pandas(df_raw),
    "processed_posts": Dataset.from_pandas(df_processed),
    "post_features": Dataset.from_pandas(df_features),
    "post_labels": Dataset.from_pandas(df_labels)
})

# Push to the Hugging Face hub
repo_id = "your-hf-username/reddit-viralness-tracker"
repo_tables.push_to_hub(repo_id)

print(f"Success! Pipeline complete. Data live at: huggingface.co/datasets/{repo_id}")