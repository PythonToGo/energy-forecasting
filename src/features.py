"""Shared feature engineering for energy-consumption forecasting.

Single source of truth used by BOTH training (``src/train_model.py``) and
serving (``api/main.py``), which guarantees train/serve feature parity. Every
lag and rolling feature looks strictly into the past (via ``shift``), so the
target at time *t* never leaks into the features for time *t*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Global_active_power"

# Lag features: how many hours back from the current step.
LAGS: dict[str, int] = {"lag_1h": 1, "lag_24h": 24, "lag_168h": 168}

# Rolling-window sizes (hours) for mean/std of recent consumption.
ROLLING_WINDOWS: tuple[int, ...] = (24, 168)

CALENDAR_FEATURES: list[str] = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
]
LAG_FEATURES: list[str] = list(LAGS)
ROLLING_FEATURES: list[str] = [
    f"roll_{stat}_{w}" for w in ROLLING_WINDOWS for stat in ("mean", "std")
]
FEATURE_COLUMNS: list[str] = CALENDAR_FEATURES + LAG_FEATURES + ROLLING_FEATURES

# Leading rows that cannot have complete lag/rolling features and must be dropped.
WARMUP_HOURS: int = max(max(LAGS.values()), max(ROLLING_WINDOWS))


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour/dayofweek/month, a weekend flag, and cyclical encodings."""
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cyclical encoding keeps hour 23 and hour 0 adjacent (and similarly for
    # day-of-week / month boundaries).
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add past-consumption lag features (strictly t-k, so no leakage)."""
    for name, k in LAGS.items():
        df[name] = df[TARGET].shift(k)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean/std over past windows (shifted by 1 to exclude t)."""
    past = df[TARGET].shift(1)
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}"] = past.rolling(w).mean()
        df[f"roll_std_{w}"] = past.rolling(w).std()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every column in ``FEATURE_COLUMNS`` added.

    ``df`` must be indexed by an hourly ``DatetimeIndex`` and contain ``TARGET``.
    """
    df = df.copy()
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df


def make_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add features and drop warmup/invalid rows (no NaN/inf in features)."""
    out = add_features(df)
    out = out[~np.isinf(out[TARGET])]
    return out.dropna(subset=[*FEATURE_COLUMNS, TARGET])
