import pandas as pd
import numpy as np
import xgboost as xgb 
import joblib
import datetime as dt
import os 

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df


def train_xgb (data_path="data/processed/energy_clean.csv", model_dir='mlops'):
    # model path
    timestamp = dt.datetime.now().strftime('%m%d_%H%M')
    model_filename = f"model_{timestamp}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    
    # load data
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    
    # create features
    df = create_features(df)
    
    # Check for NaN or infinite values in the label column
    if df['Global_active_power'].isnull().any() or np.isinf(df['Global_active_power']).any():
        # Handle NaN or infinite values
        df = df.dropna(subset=['Global_active_power'])  # Drop rows with NaN in the label column
        df = df[~np.isinf(df['Global_active_power'])]   # Remove rows with infinite values
    
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
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_xgb()