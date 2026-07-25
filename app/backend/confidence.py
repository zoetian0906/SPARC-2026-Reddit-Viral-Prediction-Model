"""
confidence.py — segment confidence tiering.

Maps a segment's model quality (test R² + sample size) to one of three tiers:
high / low / none, with a specific human-readable reason. Pure logic, no I/O,
no Streamlit — safe to unit test.

Thresholds (revised after Phase 0 recon of Table 1):
  - R2_HIGH = 0.10  beats the global baseline (~0.11)
  - R2_LOW  = 0.00  positive signal but weak; below 0 the model is worse than
              predicting the average, so it collapses to "none"
  - N_HIGH / N_LOW   minimum sample sizes for each tier
"""

from __future__ import annotations

R2_HIGH = 0.10
R2_LOW = 0.00
N_HIGH = 500
N_LOW = 100


def _summary(test_r2: float, sample_size: int) -> str:
    """Human-readable metric summary, e.g. 'R² of 0.18 on 4,245 posts'."""
    return f"R\u00b2 of {test_r2:.2f} on {sample_size:,} posts"


def assign_confidence(
    test_r2: float | None,
    sample_size: int | None,
) -> tuple[str, str]:
    """Return (tier, reason) for a segment.

    tier is one of "high" | "low" | "none". reason is specific, e.g.
    "R² of 0.18 on 4,245 posts (above baseline)".
    """
    if test_r2 is None or sample_size is None:
        return "none", "no model metrics available for this segment"

    if test_r2 >= R2_HIGH and sample_size >= N_HIGH:
        return "high", f"{_summary(test_r2, sample_size)} (above baseline)"

    if test_r2 >= R2_LOW and sample_size >= N_LOW:
        return "low", f"{_summary(test_r2, sample_size)} (positive but weak signal)"

    # Everything else is "none" — be specific about why.
    if test_r2 < R2_LOW:
        return "none", f"{_summary(test_r2, sample_size)} (worse than predicting the average)"
    return "none", f"{_summary(test_r2, sample_size)} (sample too small for a reliable estimate)"
