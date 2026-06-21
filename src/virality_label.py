# virality_label.py
# Source: Sarah's locked virality_index formula (W4 decision).
# This computes the LABEL (target y), not a feature.
# REQUIRES a 'score_to_avg_ratio' column to exist before running.
# To be integrated into the labeling pipeline by Kristin.

import pandas as pd
import numpy as np

def calculate_marketing_virality(df):
    """
    Computes a balanced 50/50 Virality Index by normalizing
    raw scores and subreddit ratios to a shared 0-1 scale.
    """
    df['log_score'] = np.log1p(df['score'])
    df['log_ratio'] = np.log1p(df['score_to_avg_ratio'])

    def min_max_scale(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return 0.0
        return (series - min_val) / (max_val - min_val)

    df['normalized_score'] = min_max_scale(df['log_score'])
    df['normalized_ratio'] = min_max_scale(df['log_ratio'])

    df['virality_index'] = (0.5 * df['normalized_score']) + (0.5 * df['normalized_ratio'])
    df['virality_index'] = df['virality_index'] * 100

    return df