"""
test_loader.py — Phase B data-loader tests.

No network, no HuggingFace, no Streamlit runtime. Fixtures are built in-test with
pandas and tmp_path. These exercise the PURE functions (read_parquet_file,
build_db); the hf_hub_download network step is an I/O boundary and is not tested.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from app.backend.loader import build_db, read_parquet_file


def test_load_roundtrip(tmp_path) -> None:
    # Tests the pandas read path used by load_table (not hf_hub_download).
    df = pd.DataFrame(
        {
            "category": ["Food & Cooking", "Gaming", "ALL"],
            "test_r2": [0.19, 0.04, 0.10],
            "sample_size": [4245, 9763, 94859],
        }
    )
    path = tmp_path / "table1.parquet"
    df.to_parquet(path, index=False)

    loaded = read_parquet_file(str(path))

    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == len(df)


def test_build_db_registers_tables() -> None:
    t1 = pd.DataFrame({"a": [1, 2, 3]})
    t2 = pd.DataFrame({"b": ["x", "y"]})

    con = build_db({"model_metadata": t1, "predictions": t2})

    n1 = con.execute("SELECT count(*) FROM model_metadata").fetchone()[0]
    n2 = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert n1 == 3
    assert n2 == 2


def test_build_db_query_missing_table() -> None:
    con = build_db({"model_metadata": pd.DataFrame({"a": [1]})})

    # Querying a table that was never registered must raise a clear duckdb error,
    # not silently return an empty result.
    with pytest.raises(duckdb.Error):
        con.execute("SELECT count(*) FROM does_not_exist").fetchone()


def test_build_db_column_access() -> None:
    df = pd.DataFrame(
        {
            "category": ["ALL"],
            "test_r2": [0.10],
            "subreddit_shap": [1.97],
        }
    )
    con = build_db({"model_metadata": df})

    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'model_metadata'"
    ).fetchall()
    cols = {r[0] for r in rows}

    assert {"category", "test_r2", "subreddit_shap"} <= cols
