"""
loader.py — data loading for the recommendations backend.

Table 1 (model metadata / SHAP rules) is a GATED HuggingFace dataset: the direct
resolve URL returns 401. Phase 0 confirmed that `huggingface_hub.hf_hub_download`
works with cached credentials, so that is the primary (and only) fetch path here.

Design for Table 2 (predictions, lands later): add ONE entry to HF_TABLES. Because
get_db() iterates HF_TABLES, no function changes are needed when it arrives.

Purity / testability:
  - load_table, read_parquet_file, build_db are PURE (no Streamlit). They are what
    the unit tests exercise.
  - get_db() is the only thing that touches Streamlit (st.cache_resource + secrets),
    and it imports streamlit lazily so the pure functions import cleanly without a
    Streamlit runtime installed.
"""

from __future__ import annotations

import os

import duckdb
import pandas as pd
from huggingface_hub import hf_hub_download

# HuggingFace coordinates ----------------------------------------------------
REPO_ID = "SPARC2026Reddit/MessyData-ZT"

# logical table name -> filename inside the HF dataset repo.
# Table 2 (predictions) slots in here as a second entry when it lands; get_db()
# needs no changes because it iterates this dict.
HF_TABLES: dict[str, str] = {
    "model_metadata": "For RAG/rag_model_rules_FINAL.parquet",
    # "predictions": "<Table 2 filename TBD in Phase C>",
}


def read_parquet_file(path: str) -> pd.DataFrame:
    """Read a local parquet file with pandas. Pure; the unit-testable read step."""
    return pd.read_parquet(path)


def load_table(repo_id: str, filename: str, token: str | None = None) -> pd.DataFrame:
    """Download a parquet from a gated HF dataset and read it with pandas.

    Uses hf_hub_download (not a raw URL fetch, which 401s on gated repos). The
    network download is an I/O boundary and is intentionally not unit tested;
    the pandas read is covered via read_parquet_file.
    """
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
    )
    return read_parquet_file(local_path)


def build_db(tables: dict[str, pd.DataFrame]) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB and materialize each dataframe as a table.

    Keys of `tables` become table names. Pure; no Streamlit.
    """
    con = duckdb.connect(database=":memory:")
    for name, df in tables.items():
        src = f"__src_{name}"
        con.register(src, df)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "{src}"')
        con.unregister(src)
    return con


def _resolve_token(st) -> str | None:
    """Resolve the HF token from st.secrets, falling back to the environment.

    Never reads a token file committed in the repo. `st` is passed in so this
    helper itself does not import streamlit.
    """
    token: str | None = None
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        token = None
    if not token:
        token = os.environ.get("HF_TOKEN")
    return token


# Memoized cached wrapper (created once, on first get_db call) so that
# st.cache_resource is applied a single time and streamlit stays out of the
# module import path.
_cached_get_db = None


def get_db() -> duckdb.DuckDBPyConnection:
    """Load all HF_TABLES and return a cached in-memory DuckDB connection.

    Thin Streamlit wrapper around the pure functions above: it resolves the HF
    token, calls load_table for each entry in HF_TABLES, and hands the dict to
    build_db. Cached with st.cache_resource so the download happens once.
    """
    global _cached_get_db
    import streamlit as st  # lazy: keep streamlit out of the pure import path

    if _cached_get_db is None:

        @st.cache_resource
        def _load_and_build() -> duckdb.DuckDBPyConnection:
            token = _resolve_token(st)
            tables = {
                name: load_table(REPO_ID, filename, token=token)
                for name, filename in HF_TABLES.items()
            }
            return build_db(tables)

        _cached_get_db = _load_and_build

    return _cached_get_db()
