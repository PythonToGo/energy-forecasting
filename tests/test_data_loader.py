"""Unit test for the raw-data preprocessing step."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import data_loader


def test_load_data_resamples_hourly(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text(
        "Date;Time;Global_active_power\n"
        "16/12/2006;17:24:00;4.216\n"
        "16/12/2006;17:44:00;5.360\n"
        "16/12/2006;18:10:00;3.666\n"
    )
    out = tmp_path / "clean.csv"

    data_loader.load_data(raw_path=str(raw), save_path=str(out))

    assert out.exists()
    df = pd.read_csv(out, parse_dates=["datetime"])
    assert list(df.columns) == ["datetime", "Global_active_power"]
    assert len(df) == 2  # two hourly buckets (17:00 and 18:00)
    assert df["Global_active_power"].iloc[0] == pytest.approx(4.788, abs=1e-3)
