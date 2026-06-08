"""FastAPI inference service — stateful, recursive multi-step forecasting.

The model now consumes recent-consumption features (lags + rolling stats), so
serving is **stateful**: we seed each forecast from the tail of the historical
series and feed every prediction back in to build the next step's features.
Feature construction is shared with training via ``features.py``, guaranteeing
train/serve parity.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# The shared feature pipeline lives in src/; the package restructure (a later
# phase) will replace this path shim with a proper import.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH_FILE = os.environ.get("MODEL_PATH_FILE", "models/latest_model_path.txt")
MODEL_METADATA_FILE = os.environ.get("MODEL_METADATA_FILE", "models/latest_model_metadata.json")
DATA_PATH = os.environ.get("PROCESSED_DATA_PATH", "data/processed/energy_clean.csv")
HISTORY_HOURS = features.WARMUP_HOURS + 48  # context window to seed lags + rolling
MAX_HORIZON = 168

app = FastAPI(
    title="Energy Consumption Forecasting API",
    description="Recursive multi-step forecasting from recent household consumption.",
    version="3.0.0",
)


class ForecastPoint(BaseModel):
    timestamp: datetime
    predicted_energy_kW: float


class ForecastResponse(BaseModel):
    from_timestamp: datetime
    horizon_hours: int
    forecast: list[ForecastPoint]


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not os.path.exists(MODEL_PATH_FILE):
        raise HTTPException(status_code=503, detail="Model not available — train first")
    with open(MODEL_PATH_FILE) as f:
        model_path = f.read().strip()
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail=f"Model file missing: {model_path}")
    logger.info("Loading model from %s", model_path)
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, Any]:
    if not os.path.exists(MODEL_METADATA_FILE):
        return {"note": "metadata not found — retrain the model"}
    with open(MODEL_METADATA_FILE) as f:
        data: dict[str, Any] = json.load(f)
    return data


@lru_cache(maxsize=1)
def load_history() -> pd.Series:
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=503, detail="Historical data not available")
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"], index_col="datetime")
    # Reindex onto a contiguous hourly grid so lag/rolling features carry the
    # SAME hour-aligned meaning as during training (which builds features on the
    # gap-preserving series from data_loader's resample). Dropping NaNs first
    # would compress gaps and silently misalign the lags. Short gaps in the
    # recent window are then filled so every feature is defined at forecast time.
    series = df[features.TARGET].asfreq("h").tail(HISTORY_HOURS)
    series = series.interpolate().ffill().bfill()
    if len(series) <= features.WARMUP_HOURS or bool(series.isna().any()):
        raise HTTPException(status_code=503, detail="Not enough clean history for a forecast")
    return series


def _recursive_forecast(model: Any, history: pd.Series, horizon: int) -> list[ForecastPoint]:
    """Roll the model forward `horizon` hours, feeding predictions back as lags."""
    series = history.copy()
    points: list[ForecastPoint] = []
    for _ in range(horizon):
        next_ts = series.index[-1] + pd.Timedelta(hours=1)
        extended = pd.concat([series, pd.Series([np.nan], index=[next_ts])])
        feat = features.add_features(extended.to_frame(name=features.TARGET))
        x = feat.loc[[next_ts], features.FEATURE_COLUMNS]
        y_pred = float(model.predict(x)[0])
        series.loc[next_ts] = y_pred
        points.append(
            ForecastPoint(
                timestamp=next_ts.to_pydatetime(),
                predicted_energy_kW=round(y_pred, 3),
            )
        )
    return points


@app.get("/", tags=["General"])
def read_root() -> dict[str, str]:
    return {"message": "Energy Consumption Forecasting API v3"}


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/model/info", tags=["Model"])
def model_info() -> dict[str, Any]:
    """Return metadata for the currently loaded model."""
    return load_metadata()


@app.get("/forecast", response_model=ForecastResponse, tags=["Prediction"])
def forecast(horizon: int = 24) -> ForecastResponse:
    """Forecast the next `horizon` hours recursively from the latest data."""
    if not 1 <= horizon <= MAX_HORIZON:
        raise HTTPException(status_code=422, detail=f"horizon must be 1..{MAX_HORIZON}")
    history = load_history()
    points = _recursive_forecast(load_model(), history, horizon)
    return ForecastResponse(
        from_timestamp=history.index[-1].to_pydatetime(),
        horizon_hours=horizon,
        forecast=points,
    )


@app.get("/predict", response_model=ForecastPoint, tags=["Prediction"])
def predict_next_hour() -> ForecastPoint:
    """Predict the single next hour after the latest observed data."""
    history = load_history()
    return _recursive_forecast(load_model(), history, 1)[0]
