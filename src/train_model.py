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

FEATURES = [
    "hour",
    "dayofweek",
    "month",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]

XGB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 4,
    "random_state": 42,
    "tree_method": "hist",
}


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cyclical encoding — prevents hour 23 and hour 0 from being far apart
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered split: 70% train / 15% val / 15% test."""
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def compute_naive_baselines(df: pd.DataFrame) -> dict[str, float]:
    """
    Evaluate three naive baselines on a dataset that has temporal order.
    These require lag columns derived from the full series before splitting.
    """
    target = df["Global_active_power"].dropna()

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
        "features": FEATURES,
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


def _log_to_mlflow(
    model: xgb.XGBRegressor,
    timestamp: str,
    sizes: tuple[int, int, int],
    metrics: dict[str, float],
    baselines: dict[str, float],
) -> None:
    """Log params, metrics, baselines, and the model artifact to MLflow."""
    train_size, val_size, test_size = sizes
    mlflow.set_experiment("energy_xgb_experiment")
    with mlflow.start_run(run_name=f"xgb_model_{timestamp}"):
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("val_size", val_size)
        mlflow.log_param("test_size", test_size)

        mlflow.log_metric("val_mae", metrics["val_mae"])
        mlflow.log_metric("test_mae", metrics["test_mae"])
        mlflow.log_metric("test_rmse", metrics["test_rmse"])
        mlflow.log_metric("skill_vs_daily_baseline", metrics["skill_vs_daily"])
        for k, v in baselines.items():
            mlflow.log_metric(k, v)

        input_example = pd.DataFrame(
            [[12, 2, 4, 0.0, -1.0, 0.78, 0.62, 0.5, 0.87, 0]], columns=FEATURES
        )
        mlflow.sklearn.log_model(model, "xgb_model", input_example=input_example)
        print(f"MLflow run logged (val_mae={metrics['val_mae']:.4f})")


def train_xgb(
    data_path: str = "data/processed/energy_clean.csv", model_dir: str = "models"
) -> None:
    timestamp = dt.datetime.now().strftime("%m%d_%H%M")
    model_path = os.path.join(model_dir, f"model_{timestamp}.pkl")

    # ── Load & feature engineer ──────────────────────────────────────────────
    df = pd.read_csv(data_path, parse_dates=["datetime"], index_col="datetime")
    df = df.dropna(subset=["Global_active_power"])
    df = df[~np.isinf(df["Global_active_power"])]
    df = create_features(df)

    # Lag columns are computed on the full series so they're correct at boundaries
    df["lag_1h"] = df["Global_active_power"].shift(1)
    df["lag_24h"] = df["Global_active_power"].shift(24)
    df["lag_168h"] = df["Global_active_power"].shift(168)

    # ── Time-based split ─────────────────────────────────────────────────────
    train_df, val_df, test_df = time_split(df)
    sizes = (len(train_df), len(val_df), len(test_df))
    print(f"Split sizes — train: {sizes[0]}, val: {sizes[1]}, test: {sizes[2]}")

    X_train, y_train = train_df[FEATURES].fillna(0), train_df["Global_active_power"]
    X_val, y_val = val_df[FEATURES].fillna(0), val_df["Global_active_power"]
    X_test, y_test = test_df[FEATURES].fillna(0), test_df["Global_active_power"]

    # ── Train & evaluate ─────────────────────────────────────────────────────
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    baselines = compute_naive_baselines(test_df)
    metrics = {
        "val_mae": float(mean_absolute_error(y_val, val_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
    }
    metrics["skill_vs_daily"] = 1 - metrics["test_mae"] / baselines["baseline_naive_daily_mae"]
    print(
        f"val_mae={metrics['val_mae']:.4f}  test_mae={metrics['test_mae']:.4f}  "
        f"test_rmse={metrics['test_rmse']:.4f}  "
        f"skill_vs_daily={metrics['skill_vs_daily']:.3f}"
    )

    # ── Persist & track ──────────────────────────────────────────────────────
    _persist_model(model, model_dir, model_path, timestamp, sizes, metrics, baselines)
    _log_to_mlflow(model, timestamp, sizes, metrics, baselines)


if __name__ == "__main__":
    train_xgb()
