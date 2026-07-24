"""
test_confidence.py — Phase C confidence tiering tests.

Covers None, negative r2, small-sample, and exact-boundary cases. Pure logic; no
network / DuckDB / Streamlit.
"""

from __future__ import annotations

import pytest

from app.backend.confidence import assign_confidence


@pytest.mark.parametrize(
    "test_r2,sample_size,expected",
    [
        (0.15, 1000, "high"),
        (0.05, 200, "low"),
        (-0.1, 5000, "none"),
        (None, 1000, "none"),
        (0.15, 50, "none"),      # good r2 but tiny sample
        (0.10, 500, "high"),     # exact boundary
        (0.00, 100, "low"),      # exact boundary
    ],
)
def test_assign_confidence_tier(test_r2, sample_size, expected) -> None:
    tier, reason = assign_confidence(test_r2, sample_size)
    assert tier == expected
    assert isinstance(reason, str) and reason


def test_none_sample_size_is_none_tier() -> None:
    tier, _ = assign_confidence(0.15, None)
    assert tier == "none"
