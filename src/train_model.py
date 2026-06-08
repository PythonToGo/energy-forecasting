import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import datetime as dt
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = [
    'hour', 'dayofweek', 'month',
    'hour_sin', 'hour_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'month_sin', 'month_cos',
    'is_weekend',
]

XGB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 4,
    'random_state': 42,
    'tree_method': 'hist',
}


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Cyclical encoding — prevents hour 23 and hour 0 from being far apart
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    return df


def time_split(df: pd.DataFrame):
    """Time-ordered split: 70% train / 15% val / 15% test."""
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def compute_naive_baselines(df: pd.DataFrame) -> dict:
    """
    Evaluate three naive baselines on a dataset that has temporal order.
    These require lag columns derived from the full series before splitting.
    """
    target = df['Global_active_power'].dropna()

    def _mae_safe(actual, pred):
        mask = pred.notna() & actual.notna()
        if mask.sum() == 0:
            return float('nan')
        return mean_absolute_error(actual[mask], pred[mask])

    return {
        'baseline_naive_last_mae': _mae_safe(target, df['lag_1h']),
        'baseline_naive_daily_mae': _mae_safe(target, df['lag_24h']),
        'baseline_naive_weekly_mae': _mae_safe(target, df['lag_168h']),
    }


def train_xgb(data_path="data/processed/energy_clean.csv", model_dir='models'):
    timestamp = dt.datetime.now().strftime('%m%d_%H%M')
    model_filename = f"model_{timestamp}.pkl"
    model_path = os.path.join(model_dir, model_filename)

    # ── Load & feature engineer ──────────────────────────────────────────────
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    df = df.dropna(subset=['Global_active_power'])
    df = df[~np.isinf(df['Global_active_power'])]
    df = create_features(df)

    # Lag columns are computed on the full series so they're correct at split boundaries
    df['lag_1h'] = df['Global_active_power'].shift(1)
    df['lag_24h'] = df['Global_active_power'].shift(24)
    df['lag_168h'] = df['Global_active_power'].shift(168)

    # ── Time-based split ─────────────────────────────────────────────────────
    train_df, val_df, test_df = time_split(df)
    print(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    X_train = train_df[FEATURES].fillna(0)
    y_train = train_df['Global_active_power']
    X_val = val_df[FEATURES].fillna(0)
    y_val = val_df['Global_active_power']
    X_test = test_df[FEATURES].fillna(0)
    y_test = test_df['Global_active_power']

    # ── Train ────────────────────────────────────────────────────────────────
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)

    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    # Baseline MAEs computed on the test slice
    baselines = compute_naive_baselines(test_df)
    skill_vs_daily = 1 - test_mae / baselines['baseline_naive_daily_mae']

    print(f"val_mae={val_mae:.4f}  test_mae={test_mae:.4f}  test_rmse={test_rmse:.4f}")
    print(f"Baselines — last_1h: {baselines['baseline_naive_last_mae']:.4f}, "
          f"daily: {baselines['baseline_naive_daily_mae']:.4f}, "
          f"weekly: {baselines['baseline_naive_weekly_mae']:.4f}")
    print(f"Skill score vs daily baseline: {skill_vs_daily:.3f}")

    # ── Save model ───────────────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    latest_path_file = os.path.join(model_dir, "latest_model_path.txt")
    with open(latest_path_file, "w") as f:
        f.write(model_path)

    metadata = {
        "model_path": model_path,
        "features": FEATURES,
        "trained_at": timestamp,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "val_mae": round(val_mae, 4),
        "test_mae": round(test_mae, 4),
        "test_rmse": round(test_rmse, 4),
        "skill_vs_daily_baseline": round(skill_vs_daily, 4),
        **{k: round(v, 4) for k, v in baselines.items()},
        **XGB_PARAMS,
    }
    metadata_path = os.path.join(model_dir, "latest_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow.set_experiment("energy_xgb_experiment")
    with mlflow.start_run(run_name=f"xgb_model_{timestamp}"):
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("val_size", len(val_df))
        mlflow.log_param("test_size", len(test_df))

        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("skill_vs_daily_baseline", skill_vs_daily)
        for k, v in baselines.items():
            mlflow.log_metric(k, v)

        input_example = pd.DataFrame(
            [[12, 2, 4, 0.0, -1.0, 0.78, 0.62, 0.5, 0.87, 0]],
            columns=FEATURES
        )
        mlflow.sklearn.log_model(model, "xgb_model", input_example=input_example)
        print(f"MLflow run logged (val_mae={val_mae:.4f}, test_mae={test_mae:.4f})")


if __name__ == "__main__":
    train_xgb()
