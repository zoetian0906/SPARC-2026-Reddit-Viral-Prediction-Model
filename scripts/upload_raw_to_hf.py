from huggingface_hub import HfApi

LOCAL_FILE = "data/raw/reddit_raw.parquet"
REPO_ID    = "SPARC2026Reddit/MessyData-ZT"
REPO_PATH  = "data/train-00000-of-00001.parquet"

api = HfApi()

api.upload_file(
    path_or_fileobj=LOCAL_FILE,
    path_in_repo=REPO_PATH,
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Full-scale collection: 96,170 posts across 48 subreddits (was 5,832)",
)

print(f"Uploaded {LOCAL_FILE}")
print(f"  -> {REPO_ID}/{REPO_PATH}")
