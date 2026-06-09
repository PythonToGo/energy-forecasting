"""Smoke test for the training pipeline on small synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_model


def _write_synthetic_csv(path: Path, n: int = 500) -> None:
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    # Daily seasonality + mild noise so the model has signal to learn.
    vals = 2.0 + np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 0.1, n)
    pd.DataFrame({"Global_active_power": vals}, index=idx).to_csv(path, index_label="datetime")


def test_train_xgb_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "energy_clean.csv"
    _write_synthetic_csv(data_path)
    model_dir = tmp_path / "models"

    train_model.train_xgb(data_path=str(data_path), model_dir=str(model_dir), track=False)

    assert (model_dir / "latest_model_path.txt").exists()
    meta = json.loads((model_dir / "latest_model_metadata.json").read_text())
    assert meta["features"] == train_model.FEATURE_COLUMNS
    assert len(meta["features"]) == 17
    assert meta["quantiles"] == [0.1, 0.5, 0.9]
    assert meta["test_mae"] >= 0.0
