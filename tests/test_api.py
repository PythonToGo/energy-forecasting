"""Tests for the recursive serving logic (train/serve parity in action)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.main import ForecastPoint, _recursive_forecast


class _NaiveModel:
    """Stand-in model: predicts the most recent value (the lag_1h feature)."""

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray([float(x["lag_1h"].iloc[0])])


def _history(n: int = 200, offset: float = 0.0) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.Series(np.arange(n, dtype=float) + offset, index=idx)


def test_forecast_shape_and_timestamps() -> None:
    hist = _history()
    points = _recursive_forecast(_NaiveModel(), hist, horizon=5)
    assert len(points) == 5
    assert all(isinstance(p, ForecastPoint) for p in points)
    assert points[0].timestamp == (hist.index[-1] + pd.Timedelta(hours=1)).to_pydatetime()
    assert points[1].timestamp == (hist.index[-1] + pd.Timedelta(hours=2)).to_pydatetime()


def test_forecast_depends_on_recent_history() -> None:
    """The core Phase 1 property: a different recent series → different forecast."""
    a = _recursive_forecast(_NaiveModel(), _history(offset=0.0), horizon=3)
    b = _recursive_forecast(_NaiveModel(), _history(offset=1000.0), horizon=3)
    assert a[0].predicted_energy_kW != b[0].predicted_energy_kW
