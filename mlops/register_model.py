import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

# Feature order must match FEATURES in src/train_model.py
FEATURES = [
    "hour", "dayofweek", "month",
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "month_sin", "month_cos",
    "is_weekend",
]


def register_latest_model(latest_path_file: str = "models/latest_model_path.txt") -> None:
    if not os.path.exists(latest_path_file):
        print("latest_model_path.txt not found.")
        return

    with open(latest_path_file, "r") as f:
        latest_model_path = f.read().strip()

    if not os.path.exists(latest_model_path):
        print(f"Model file does not exist: {latest_model_path}")
        return

    print(f"Found model: {latest_model_path}")

    mlflow.set_experiment("energy_xgb_experiment")

    with mlflow.start_run(run_name="register"):
        # 모델 로딩 후 등록
        model = joblib.load(latest_model_path)
        # input_example must match the 10-feature vector the model was trained on
        # (previously a stale 3-feature example that did not match the schema).
        input_example = pd.DataFrame(
            [[12, 2, 4, 0.0, -1.0, 0.78, 0.62, 0.5, 0.87, 0]],
            columns=FEATURES,
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgb_model",
            input_example=input_example,
        )

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/xgb_model"
        result = mlflow.register_model(model_uri, "xgb_energy_forecast")

        print(f"📦 Model registered: {result.name}, version: {result.version}")


if __name__ == "__main__":
    register_latest_model()
