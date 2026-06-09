"""Tests for the recursive serving logic (train/serve parity in action)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import api.main as api_main
from api.main import ForecastPoint, _recursive_forecast


class _NaiveModel:
    """Stand-in quantile model: returns (p10, p50, p90) around the lag_1h value."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        v = float(x["lag_1h"].iloc[0])
        return np.asarray([[v * 0.9, v, v * 1.1]])  # shape (1, 3)


def _history(n: int = 200, offset: float = 0.0) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.Series(np.arange(n, dtype=float) + offset, index=idx)


def test_forecast_shape_and_timestamps() -> None:
    hist = _history()
    points = _recursive_forecast(_NaiveModel(), hist, horizon=5)
    assert len(points) == 5
    assert all(isinstance(p, ForecastPoint) for p in points)
    assert all(p.p10 <= p.p50 <= p.p90 for p in points)  # monotone interval
    assert points[0].timestamp == (hist.index[-1] + pd.Timedelta(hours=1)).to_pydatetime()
    assert points[1].timestamp == (hist.index[-1] + pd.Timedelta(hours=2)).to_pydatetime()


def test_forecast_depends_on_recent_history() -> None:
    """The core Phase 1 property: a different recent series → different forecast."""
    a = _recursive_forecast(_NaiveModel(), _history(offset=0.0), horizon=3)
    b = _recursive_forecast(_NaiveModel(), _history(offset=1000.0), horizon=3)
    assert a[0].p50 != b[0].p50


def test_load_history_is_contiguous_hourly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: serving must rebuild a contiguous hourly grid so lags stay
    hour-aligned with training (the old dropna() compressed gaps and misaligned)."""
    idx = pd.date_range("2020-01-01", periods=400, freq="h")
    series = pd.Series(np.arange(400, dtype=float), index=idx)
    # Gaps both outside and inside the recent window (last 216h = index 184..399).
    series = series.drop(series.index[[50, 51, 300, 301]])
    csv = tmp_path / "energy_clean.csv"
    series.to_frame(name="Global_active_power").to_csv(csv, index_label="datetime")

    monkeypatch.setattr(api_main, "DATA_PATH", str(csv))
    api_main.load_history.cache_clear()
    try:
        hist = api_main.load_history()
    finally:
        api_main.load_history.cache_clear()

    diffs = hist.index.to_series().diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()  # contiguous hourly
    assert not bool(hist.isna().any())  # gaps filled → features always defined
