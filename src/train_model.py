import pandas as pd
import numpy as np
import xgboost as xgb 
import joblib
import datetime as dt
import os 
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error


def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df


def train_xgb (data_path="data/processed/energy_clean.csv", model_dir='models'):
    # model path
    timestamp = dt.datetime.now().strftime('%m%d_%H%M')
    model_filename = f"model_{timestamp}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    
    # load data
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    df = create_features(df)
    df = df.dropna(subset=['Global_active_power'])  # Drop NaN
    df = df[~np.isinf(df['Global_active_power'])]   # Remove infinite value
    
    X = df[['hour', 'dayofweek', 'month']]
    y = df['Global_active_power']
    
    # train model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        # tree_method='gpu_hist'
        tree_method='hist'
    )
    model.fit(X, y)
    
    # save model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # save model path
    latest_path_file = os.path.join(model_dir, "latest_model_path.txt")
    with open(latest_path_file, "w") as f:
        f.write(model_path)
    print(f"Saved model path to {latest_path_file}")
    
    ### MLflow ###
    # set experiment name
    mlflow.set_experiment("energy_xgb_experiment")
    with mlflow.start_run(run_name=f"xgb_model_{timestamp}"):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("tree_method", "hist")
        mlflow.log_param("random_state", 42)
        
        input_example = pd.DataFrame([[12.0, 2.0, 4.0]], columns=['hour', 'dayofweek', 'month'])
        mlflow.sklearn.log_model(model, "xgb_model", input_example=input_example)
        
        # MAE and log
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        mlflow.log_metric("mae", mae)
        
        print(f"MLflow run logged (MAE: {mae:.3f})")


if __name__ == "__main__":
    train_xgb()


