from huggingface_hub import HfApi

LOCAL_FILE = "notebooks/zoe/zoe_nlp_features.parquet"
REPO_ID    = "SPARC2026Reddit/MessyData-ZT"
REPO_PATH  = "features/zoe_nlp_features.parquet"

api = HfApi()

api.upload_file(
    path_or_fileobj=LOCAL_FILE,
    path_in_repo=REPO_PATH,
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add Zoe NLP feature table (VADER + NRC emotions + readability)",
)

print(f"Uploaded {LOCAL_FILE}")
print(f"  -> {REPO_ID}/{REPO_PATH}")