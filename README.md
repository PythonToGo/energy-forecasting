# 🔋 Energy Forecasting Dashboard

An end-to-end MLOps pipeline for predicting household energy consumption using time-based features, built with **FastAPI**, **Streamlit**, **MLflow**, **DVC**, and **Prefect**.  
This project is fully containerized and reproducible via Docker.


> 🚀 Current setup: Local deployment via Docker on `0.0.0.0`.

---

## 📌 Project Overview

This project forecasts **household hourly electricity usage** based on:
- Hour of the day
- Day of the week
- Month

It features:
- **XGBoost regression model**
- A real-time **FastAPI** inference service
- A user-friendly **Streamlit dashboard**
- End-to-end experiment tracking with **MLflow**
- **DVC** for data/model versioning
- **Prefect** for pipeline automation
- Packaged & deployable with **Docker**

---

## 🧠 Model

- Model type: `XGBoostRegressor`
- **17 features** built by the shared pipeline in `src/features.py`
  (guarantees train/serve parity):
  - **Calendar (10):** hour, dayofweek, month, is_weekend + cyclical (sin/cos) encodings
  - **Lags (3):** consumption 1h / 24h / 168h ago
  - **Rolling (4):** mean & std over the past 24h and 168h
- Target: `Global_active_power` (kW)
- Because the model uses *recent consumption*, the API forecasts **recursively**:
  each predicted hour is fed back in to build the next hour's features.

---

## 🗂️ Project Structure

```
energy-forecasting/
├── api/                    # FastAPI inference server
│   └── main.py
├── dashboard/              # Streamlit dashboard
│   └── app.py
├── data/
│   ├── raw/                # Original dataset (from UCI)
│   └── processed/          # Cleaned + resampled data
├── models/                 # Trained model files (via DVC)
│   └── latest_model_path.txt
├── mlops/
│   ├── mlflow_config.yaml
│   └── register_model.py
├── pipelines/              # Prefect automation
│   └── prefect_flow.py
├── src/
│   ├── features.py         # Shared feature pipeline (train/serve parity)
│   ├── data_loader.py      # Preprocessing script
│   └── train_model.py      # Model training & logging
├── tests/                  # pytest unit tests
├── pyproject.toml          # Deps + ruff / mypy / pytest config
├── Dockerfile              # Container setup
├── docker-compose.yaml     # Service orchestration
├── dvc.yaml                # Pipeline stages
├── dvc.lock
├── requirements.txt
└── README.md
```

---

## ⚙️ MLOps Pipeline Overview

| Stage               | Tool         | Description                                      |
|---------------------|--------------|--------------------------------------------------|
| Data versioning     | `DVC`        | Tracks data and models (e.g. energy_clean.csv)  |
| Training            | `XGBoost`    | 17 features: calendar + lags + rolling stats    |
| Experiment tracking | `MLflow`     | Logs parameters, metrics, model artifacts |
| Automation          | `Prefect`    | Defines retraining pipeline (data → train)      |
| Serving             | `FastAPI`    | Recursive multi-step forecast on `/forecast`    |
| Monitoring UI       | `Streamlit`  | Frontend to submit inputs & visualize results |
| Packaging           | `Docker`     | Full stack in one container                     |

---

## 📦 How to Run Locally (Docker)

```bash
# 1. Build and start
docker-compose up --build

# 2. Access:
FastAPI     → http://localhost:8000
Streamlit   → http://localhost:8502
MLflow UI   → http://localhost:5050
```

---

## 🚀 Forecasting API (FastAPI)

The model is **stateful**: it forecasts forward from the latest observed data.

**`GET /forecast?horizon=24`** — recursive multi-step forecast (1–168 hours):

```json
{
  "from_timestamp": "2010-11-26T20:00:00",
  "horizon_hours": 24,
  "forecast": [
    {"timestamp": "2010-11-26T21:00:00", "predicted_energy_kW": 1.234}
  ]
}
```

Also: **`GET /predict`** (single next hour), **`GET /model/info`** (features,
metrics, baseline skill), **`GET /health`** (liveness).

---

## 🖥️ Dashboard (Streamlit)

Access: `http://localhost:8502`

- Pick a forecast horizon (1–168 h) and run a recursive forward forecast
- View the predicted consumption curve
- See model metrics & baseline skill in the sidebar
- Compare against recent actual consumption


![image](https://github.com/user-attachments/assets/716d1da6-d36b-48f6-967f-89fcdc56391a)
---

## 🔄 Reproducible Training (DVC)

```bash
# Run full pipeline
dvc repro

# Push data + model versions to remote (optional)
dvc push
```

---

## 🔁 Model Retraining (Prefect)

```bash
python pipelines/prefect_flow.py
```

Runs:
- `data_loader.py` → preprocessing
- `train_model.py` → model training + MLflow logging

---

## 🧪 Track Experiments (MLflow)
![image](https://github.com/user-attachments/assets/934038ae-f9d6-4fd1-9f6b-58d7542e7871)

Visit: [http://localhost:5050](http://localhost:5050)  
Browse runs, parameters, metrics, models.

---

## 📄 Dataset Info

- Source: UCI - Individual household electric power consumption
- Resampled to hourly intervals
- Target: `Global_active_power`

---

## 🛠️ Tech Stack

- `Python 3.10`
- `FastAPI`, `Streamlit`
- `xgboost`, `scikit-learn`, `pandas`
- `MLflow`, `Prefect`, `DVC`
- `Docker`, `docker-compose`

---

## 🧠 Author

👨‍💻 Taey Kim  
📫 [GitHub](https://github.com/PythonToGo)
💡 Passionate about MLOps, system automation, and real-time inference!

---

## 📌 To-Do

- [ ] Add CI/CD via GitHub Actions
- [ ] Deploy to Heroku / Fly.io
- [ ] Batch forecasting + scheduling
- [ ] User login for dashboard

---

> MIT License | 2025  
