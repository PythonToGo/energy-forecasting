from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
import json
import os
import logging
from functools import lru_cache
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH_FILE = 'models/latest_model_path.txt'
MODEL_METADATA_FILE = 'models/latest_model_metadata.json'

app = FastAPI(
    title="Energy Consumption Prediction API",
    description="API for predicting household energy consumption based on time features",
    version="2.0.0"
)


# ── Input / output schemas ───────────────────────────────────────────────────

class EnergyInput(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0–23)")
    dayofweek: int = Field(..., ge=0, le=6, description="Day of the week (0=Mon … 6=Sun)")
    month: int = Field(..., ge=1, le=12, description="Month of the year (1–12)")

    @validator('hour')
    def validate_hour(cls, v):
        if not 0 <= v <= 23:
            raise ValueError("Hour must be between 0 and 23")
        return v

    @validator('dayofweek')
    def validate_dayofweek(cls, v):
        if not 0 <= v <= 6:
            raise ValueError("Day of week must be between 0 and 6")
        return v

    @validator('month')
    def validate_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError("Month must be between 1 and 12")
        return v


class PredictionResponse(BaseModel):
    predicted_energy_kW: float
    input_data: Dict[str, Any]


class ForecastPoint(BaseModel):
    hour: int
    dayofweek: int
    month: int
    predicted_energy_kW: float


class ForecastResponse(BaseModel):
    horizon_hours: int
    forecast: List[ForecastPoint]


# ── Feature engineering (must match train_model.py) ─────────────────────────

def build_feature_vector(hour: int, dayofweek: int, month: int) -> np.ndarray:
    is_weekend = int(dayofweek >= 5)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7)
    month_sin = np.sin(2 * np.pi * (month - 1) / 12)
    month_cos = np.cos(2 * np.pi * (month - 1) / 12)
    # Order must match FEATURES in train_model.py
    return np.array([[
        hour, dayofweek, month,
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        month_sin, month_cos,
        is_weekend,
    ]])


# ── Model / metadata loading ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_model():
    if not os.path.exists(MODEL_PATH_FILE):
        logger.error(f"Model path file not found: {MODEL_PATH_FILE}")
        raise HTTPException(status_code=500, detail="Model configuration not found")
    with open(MODEL_PATH_FILE, 'r') as f:
        model_path = f.read().strip()
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise HTTPException(status_code=500, detail=f"Model file not found at {model_path}")
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    if not os.path.exists(MODEL_METADATA_FILE):
        return {"note": "metadata file not found — retrain the model to generate it"}
    with open(MODEL_METADATA_FILE, 'r') as f:
        return json.load(f)


def get_model():
    return load_model()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Energy Consumption Prediction API v2"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}


@app.get("/model/info", tags=["Model"])
def model_info():
    """
    Return metadata for the currently loaded model: features, split sizes,
    evaluation metrics, and baseline comparisons.
    """
    return load_metadata()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_energy(data: EnergyInput, model=Depends(get_model)):
    """
    Predict energy consumption for a single time point using time-based features.
    """
    try:
        X = build_feature_vector(data.hour, data.dayofweek, data.month)
        y_pred = float(model.predict(X)[0])
        return PredictionResponse(
            predicted_energy_kW=round(y_pred, 3),
            input_data=data.dict(),
        )
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/forecast", response_model=ForecastResponse, tags=["Prediction"])
def forecast_energy(data: EnergyInput, horizon: int = 24, model=Depends(get_model)):
    """
    Predict energy consumption for the next `horizon` hours (default 24) starting
    from the given hour / dayofweek / month.

    The sequence rolls forward hour by hour, advancing dayofweek and month
    when crossing midnight / month boundaries.
    """
    if not 1 <= horizon <= 168:
        raise HTTPException(status_code=422, detail="horizon must be between 1 and 168")
    try:
        # Build a reference datetime using the provided fields; year/day are arbitrary
        # since the model only uses hour, dayofweek, month.
        # We pick a known Monday in the given month to anchor day-of-week arithmetic.
        base = datetime(2000, data.month, 1)
        # Advance to the target dayofweek
        days_ahead = (data.dayofweek - base.weekday()) % 7
        base = base + timedelta(days=days_ahead)
        base = base.replace(hour=data.hour)

        results: List[ForecastPoint] = []
        for step in range(horizon):
            ts = base + timedelta(hours=step)
            h, dow, m = ts.hour, ts.weekday(), ts.month
            X = build_feature_vector(h, dow, m)
            y_pred = round(float(model.predict(X)[0]), 3)
            results.append(ForecastPoint(
                hour=h, dayofweek=dow, month=m, predicted_energy_kW=y_pred
            ))

        return ForecastResponse(horizon_hours=horizon, forecast=results)
    except Exception as e:
        logger.exception("Forecast error")
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")
