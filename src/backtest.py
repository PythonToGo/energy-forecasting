"""Walk-forward (expanding-window) backtesting for the forecaster.

A single train/test split can flatter or punish a time-series model depending on
where the cut happens to fall. Walk-forward evaluation trains on an expanding
history and always tests on the *next* contiguous block, repeated across several
folds — a more honest estimate of how the model generalizes over time.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import FEATURE_COLUMNS, TARGET, make_training_frame

# A lighter model than final training keeps backtests quick; the deliverable here
# is the evaluation methodology, not squeezing out the last bit of accuracy.
BACKTEST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": 42,
    "tree_method": "hist",
}


def _skill_vs_daily(test_df: pd.DataFrame, pred: np.ndarray) -> float:
    """1 - model_MAE / naive_daily_MAE on a test slice (higher is better)."""
    actual = test_df[TARGET]
    naive = test_df["lag_24h"]
    model_mae = float(mean_absolute_error(actual, pred))
    naive_mae = float(mean_absolute_error(actual, naive))
    return float("nan") if naive_mae == 0 else 1 - model_mae / naive_mae


def walk_forward_backtest(
    df: pd.DataFrame, n_splits: int = 5, min_train_frac: float = 0.5
) -> dict[str, Any]:
    """Expanding-window CV; returns per-fold metrics and aggregates.

    Features are built once up front (lags/rolling only look back, so this is
    leakage-free), then folds are carved by row position so every test block is
    strictly in the future relative to its training data.
    """
    data = make_training_frame(df)
    n = len(data)
    start = int(n * min_train_frac)
    if n_splits < 1 or start <= 0 or start >= n:
        raise ValueError("not enough data for the requested backtest")
    bounds = np.linspace(start, n, n_splits + 1, dtype=int)

    folds: list[dict[str, Any]] = []
    for i in range(n_splits):
        train_end, test_end = int(bounds[i]), int(bounds[i + 1])
        if test_end <= train_end:
            continue
        train, test = data.iloc[:train_end], data.iloc[train_end:test_end]
        model = xgb.XGBRegressor(**BACKTEST_PARAMS)
        model.fit(train[FEATURE_COLUMNS], train[TARGET])
        pred = model.predict(test[FEATURE_COLUMNS])
        folds.append(
            {
                "fold": i,
                "train_size": train_end,
                "test_size": test_end - train_end,
                "test_start": str(test.index[0]),
                "test_end": str(test.index[-1]),
                "mae": round(float(mean_absolute_error(test[TARGET], pred)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(test[TARGET], pred))), 4),
                "skill_vs_daily": round(_skill_vs_daily(test, pred), 4),
            }
        )

    maes = [f["mae"] for f in folds]
    skills = [f["skill_vs_daily"] for f in folds]
    return {
        "n_splits": len(folds),
        "folds": folds,
        "mean_mae": round(float(np.mean(maes)), 4),
        "std_mae": round(float(np.std(maes)), 4),
        "mean_skill_vs_daily": round(float(np.nanmean(skills)), 4),
    }


def run_backtest(
    data_path: str = "data/processed/energy_clean.csv",
    report_path: str = "reports/backtest_report.json",
) -> dict[str, Any]:
    df = pd.read_csv(data_path, parse_dates=["datetime"], index_col="datetime")
    result = walk_forward_backtest(df)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    for fold in result["folds"]:
        print(
            f"fold {fold['fold']}: mae={fold['mae']} skill={fold['skill_vs_daily']} "
            f"({fold['test_start'][:10]}..{fold['test_end'][:10]})"
        )
    print(
        f"mean_mae={result['mean_mae']} ± {result['std_mae']}  "
        f"mean_skill_vs_daily={result['mean_skill_vs_daily']}"
    )
    return result


if __name__ == "__main__":
    run_backtest()
