"""Train the XGBoost energy-consumption forecaster.

Uses the shared feature pipeline in ``features.py`` (calendar + lag + rolling),
so the model now learns from *recent consumption*, not just the calendar slot.
"""

import datetime as dt
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import FEATURE_COLUMNS, TARGET, make_training_frame

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "tree_method": "hist",
}

# Quantiles the model predicts. P50 (the median) is the point forecast — the 0.5
# quantile minimizes MAE — and P10/P90 form an 80% prediction interval.
QUANTILES = [0.1, 0.5, 0.9]
MEDIAN_INDEX = QUANTILES.index(0.5)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered split: 70% train / 15% val / 15% test."""
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def compute_naive_baselines(df: pd.DataFrame) -> dict[str, float]:
    """MAE of naive 'repeat the value k hours ago' predictors on `df`."""
    target = df[TARGET].dropna()

    def _mae_safe(actual: pd.Series, pred: pd.Series) -> float:
        mask = pred.notna() & actual.notna()
        if mask.sum() == 0:
            return float("nan")
        return float(mean_absolute_error(actual[mask], pred[mask]))

    return {
        "baseline_naive_last_mae": _mae_safe(target, df["lag_1h"]),
        "baseline_naive_daily_mae": _mae_safe(target, df["lag_24h"]),
        "baseline_naive_weekly_mae": _mae_safe(target, df["lag_168h"]),
    }


def _persist_model(
    model: xgb.XGBRegressor,
    model_dir: str,
    model_path: str,
    timestamp: str,
    sizes: tuple[int, int, int],
    metrics: dict[str, float],
    baselines: dict[str, float],
) -> None:
    """Save the model, the latest-path pointer, and a metadata sidecar."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    latest_path_file = os.path.join(model_dir, "latest_model_path.txt")
    with open(latest_path_file, "w") as f:
        f.write(model_path)

    train_size, val_size, test_size = sizes
    metadata = {
        "model_path": model_path,
        "features": FEATURE_COLUMNS,
        "quantiles": QUANTILES,
        "trained_at": timestamp,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "val_mae": round(metrics["val_mae"], 4),
        "test_mae": round(metrics["test_mae"], 4),
        "test_rmse": round(metrics["test_rmse"], 4),
        "skill_vs_daily_baseline": round(metrics["skill_vs_daily"], 4),
        **{k: round(v, 4) for k, v in baselines.items()},
        **XGB_PARAMS,
    }
    metadata_path = os.path.join(model_dir, "latest_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")


def _log_to_mlflow(  # pragma: no cover
    model: xgb.XGBRegressor,
    timestamp: str,
    sizes: tuple[int, int, int],
    metrics: dict[str, float],
    baselines: dict[str, float],
    input_example: pd.DataFrame,
) -> None:
    """Log params, metrics, baselines, and the model artifact to MLflow."""
    train_size, val_size, test_size = sizes
    mlflow.set_experiment("energy_xgb_experiment")
    with mlflow.start_run(run_name=f"xgb_model_{timestamp}"):
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("features", FEATURE_COLUMNS)
        mlflow.log_param("quantiles", QUANTILES)
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("test_size", test_size)

        mlflow.log_metric("val_mae", metrics["val_mae"])
        mlflow.log_metric("test_mae", metrics["test_mae"])
        mlflow.log_metric("test_rmse", metrics["test_rmse"])
        mlflow.log_metric("skill_vs_daily_baseline", metrics["skill_vs_daily"])
        for k, v in baselines.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, "xgb_model", input_example=input_example)
        print(f"MLflow run logged (val_mae={metrics['val_mae']:.4f})")


def train_xgb(
    data_path: str = "data/processed/energy_clean.csv",
    model_dir: str = "models",
    track: bool = True,
) -> None:
    timestamp = dt.datetime.now().strftime("%m%d_%H%M")
    model_path = os.path.join(model_dir, f"model_{timestamp}.pkl")

    df_raw = pd.read_csv(data_path, parse_dates=["datetime"], index_col="datetime")
    df = make_training_frame(df_raw)

    train_df, val_df, test_df = time_split(df)
    sizes = (len(train_df), len(val_df), len(test_df))
    print(f"Split sizes — train: {sizes[0]}, val: {sizes[1]}, test: {sizes[2]}")

    x_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET]
    x_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET]
    x_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET]

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=np.array(QUANTILES),
        **XGB_PARAMS,
    )
    model.fit(x_train, y_train)

    # Multi-quantile predictions are (n, len(QUANTILES)); the median column is the
    # point forecast used for the headline error metrics.
    val_median = model.predict(x_val)[:, MEDIAN_INDEX]
    test_median = model.predict(x_test)[:, MEDIAN_INDEX]
    baselines = compute_naive_baselines(test_df)
    metrics = {
        "val_mae": float(mean_absolute_error(y_val, val_median)),
        "test_mae": float(mean_absolute_error(y_test, test_median)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_median))),
    }
    metrics["skill_vs_daily"] = 1 - metrics["test_mae"] / baselines["baseline_naive_daily_mae"]
    print(
        f"val_mae={metrics['val_mae']:.4f}  test_mae={metrics['test_mae']:.4f}  "
        f"test_rmse={metrics['test_rmse']:.4f}  skill_vs_daily={metrics['skill_vs_daily']:.3f}"
    )

    _persist_model(model, model_dir, model_path, timestamp, sizes, metrics, baselines)
    if track:
        _log_to_mlflow(model, timestamp, sizes, metrics, baselines, x_test.head(1))


if __name__ == "__main__":
    train_xgb()
