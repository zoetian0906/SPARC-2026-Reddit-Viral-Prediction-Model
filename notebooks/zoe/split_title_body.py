import pandas as pd

INPUT  = "data/raw/reddit_raw.parquet"
OUTPUT = "data/raw/reddit_with_title_body.parquet"
SEP    = ": "

df = pd.read_parquet(INPUT)

split = df["text"].str.split(SEP, n=1, expand=True)
df["title"] = split[0]
df["body"]  = split[1].fillna("")

df.to_parquet(OUTPUT, index=False)

has_body = (df["body"] != "").sum()
print(f"Total rows     : {len(df):,}")
print(f"Posts with body: {has_body:,}")
print(f"Title-only     : {len(df) - has_body:,}")

print("\n--- 3 sample rows ---")
for i, row in df.head(3).iterrows():
    print(f"\nRow {i} | subreddit: {row['subreddit']} | score: {row['score']}")
    print(f"  TITLE: {row['title']}")
    print(f"  BODY : {row['body'][:300] if row['body'] else '(none)'}")
