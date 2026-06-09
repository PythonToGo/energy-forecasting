"""Unit tests for walk-forward backtesting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest


def _synthetic(n: int = 600) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    vals = 2.0 + np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 0.1, n)
    return pd.DataFrame({"Global_active_power": vals}, index=idx)


def test_expanding_window_is_contiguous_and_future() -> None:
    res = backtest.walk_forward_backtest(_synthetic(), n_splits=3, min_train_frac=0.5)
    assert res["n_splits"] == 3
    folds = res["folds"]
    assert all(f["test_size"] > 0 and f["train_size"] > 0 for f in folds)
    # Expanding window: each fold trains on everything up to the previous test end.
    for prev, cur in zip(folds, folds[1:]):
        assert cur["train_size"] == prev["train_size"] + prev["test_size"]
    assert "mean_mae" in res and "mean_skill_vs_daily" in res


def test_run_backtest_writes_report(tmp_path: Path) -> None:
    data_path = tmp_path / "energy_clean.csv"
    _synthetic().to_csv(data_path, index_label="datetime")
    report = tmp_path / "reports" / "backtest_report.json"

    result = backtest.run_backtest(data_path=str(data_path), report_path=str(report))

    assert report.exists()
    saved = json.loads(report.read_text())
    assert saved["n_splits"] >= 1
    assert "mean_mae" in result
