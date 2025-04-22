from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# init app
app = FastAPI()

# load model
# MODEL_DIR = 'models'
MODEL_PATH_FILE = 'models/latest_model_path.txt'

# def get_latest_model():
#     model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
#     latest_model = sorted(model_files)[-1]
#     return joblib.load(os.path.join(MODEL_DIR, latest_model))

def get_latest_model():
    if not os.path.exists(MODEL_PATH_FILE):
        raise FileNotFoundError("latest_model_path.txt not found")

    with open(MODEL_PATH_FILE, 'r') as f:
        model_path = f.read().strip()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    return joblib.load(model_path)

model = get_latest_model()


# define input data
class EnergyInput(BaseModel):
    hour: int
    dayofweek: int
    month: int

@app.get("/")
def read_root():
    return {"message": "Welcome to Taey's Energy Prediction API"}

@app.post("/predict")
def predict_energy(data: EnergyInput):
    X = np.array([[data.hour, data.dayofweek, data.month]])
    y_pred = model.predict(X)[0]
    return {"predicted_energy_in_kW": round(float(y_pred), 3)}
