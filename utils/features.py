"""Composite feature helpers shared between model/train.py and model/inference.py."""

import pandas as pd

from config import LEAGUE_AVG

# The 3 composite features are derived at train/inference time and never fetched.
# Used by features/assembler.py to adjust the missing-feature gate denominator.
COMPOSITE_FEATURES = frozenset({"sum_sp_era_l3", "sum_ops_14d", "avg_sp_k_pct"})


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add run-environment composite columns derived from existing features."""
    df = df.copy()
    df["sum_sp_era_l3"] = (
        df.get("home_sp_era_l3", LEAGUE_AVG["sp_era_l3"]) +
        df.get("away_sp_era_l3", LEAGUE_AVG["sp_era_l3"])
    )
    df["sum_ops_14d"] = (
        df.get("home_ops_14d", LEAGUE_AVG["ops_14d"]) +
        df.get("away_ops_14d", LEAGUE_AVG["ops_14d"])
    )
    df["avg_sp_k_pct"] = (
        df.get("home_sp_k_pct", LEAGUE_AVG["sp_k_pct"]) +
        df.get("away_sp_k_pct", LEAGUE_AVG["sp_k_pct"])
    ) / 2
    return df
