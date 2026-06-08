"""Unit tests for the shared feature-engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import features


def _make_df(n: int = 400) -> pd.DataFrame:
    """Hourly frame with a strictly increasing, deterministic target."""
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    vals = np.arange(n, dtype=float) + 0.5
    return pd.DataFrame({features.TARGET: vals}, index=idx)


def test_feature_columns_composition() -> None:
    assert len(features.FEATURE_COLUMNS) == 17
    assert features.FEATURE_COLUMNS[:10] == features.CALENDAR_FEATURES
    assert features.LAG_FEATURES == ["lag_1h", "lag_24h", "lag_168h"]
    assert features.ROLLING_FEATURES == [
        "roll_mean_24",
        "roll_std_24",
        "roll_mean_168",
        "roll_std_168",
    ]
    assert features.WARMUP_HOURS == 168


def test_calendar_values() -> None:
    out = features.add_features(_make_df(50))
    # 2020-01-01 00:00 is a Wednesday (dayofweek=2), January, hour 0.
    assert out["hour"].iloc[0] == 0
    assert out["month"].iloc[0] == 1
    assert out["dayofweek"].iloc[0] == pd.Timestamp("2020-01-01").dayofweek
    assert out["is_weekend"].iloc[0] == 0
    assert out["hour_sin"].between(-1, 1).all()
    assert out["hour_cos"].between(-1, 1).all()


def test_lag_alignment() -> None:
    df = _make_df(300)
    out = features.add_features(df)
    # lag_24h at row t equals the target 24 rows earlier.
    assert out["lag_24h"].iloc[100] == df[features.TARGET].iloc[76]
    assert out["lag_1h"].iloc[100] == df[features.TARGET].iloc[99]
    # Warmup rows are NaN for the longest lag.
    assert np.isnan(out["lag_168h"].iloc[10])


def test_rolling_excludes_current_step() -> None:
    df = _make_df(300)
    out = features.add_features(df)
    t = 200
    expected_mean = df[features.TARGET].iloc[t - 24 : t].mean()
    assert out["roll_mean_24"].iloc[t] == pytest.approx(expected_mean)


def test_no_future_leakage() -> None:
    """Mutating the target at row t must not change the features at row t."""
    df = _make_df(300)
    out_a = features.add_features(df)
    df_b = df.copy()
    df_b.iloc[200, 0] = 9999.0
    out_b = features.add_features(df_b)
    pd.testing.assert_series_equal(
        out_a[features.FEATURE_COLUMNS].iloc[200],
        out_b[features.FEATURE_COLUMNS].iloc[200],
    )


def test_make_training_frame_drops_warmup() -> None:
    df = _make_df(400)
    tf = features.make_training_frame(df)
    assert not tf[features.FEATURE_COLUMNS].isna().any().any()
    assert len(tf) == len(df) - features.WARMUP_HOURS
