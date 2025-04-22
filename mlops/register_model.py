import mlflow
import os
import joblib

def register_latest_model(latest_path_file="models/latest_model_path.txt"):
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
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgb_model",
            input_example=[[12.0, 2.0, 4.0]]
        )

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/xgb_model"
        result = mlflow.register_model(model_uri, "xgb_energy_forecast")

        print(f"📦 Model registered: {result.name}, version: {result.version}")


if __name__ == "__main__":
    register_latest_model()
