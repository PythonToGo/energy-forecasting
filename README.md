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
- Features used:
  - `hour`: Hour of the day (0–23)
  - `dayofweek`: Day of the week (0=Monday)
  - `month`: Month number (1–12)
- Target: `Global_active_power` (kW)

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
│   ├── data_loader.py      # Preprocessing script
│   └── train_model.py      # Model training & logging
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
| Training            | `XGBoost`    | Model trained on 3 time features                |
| Experiment tracking | `MLflow`     | Logs parameters, metrics, model artifacts |
| Automation          | `Prefect`    | Defines retraining pipeline (data → train)      |
| Serving             | `FastAPI`    | Real-time prediction API on `/predict`          |
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

## 🚀 Prediction API (FastAPI)
![image](https://github.com/user-attachments/assets/5956dd43-8acb-4ee6-b432-220d179daf5a)

**Endpoint:** `POST /predict`  
**Input JSON:**

```json
{
  "hour": 13,
  "dayofweek": 2,
  "month": 4
}
```

**Response:**

```json
{
  "predicted_energy_in_kW": 3.123
}
```

---

## 🖥️ Dashboard (Streamlit)

Access: `http://localhost:8502`

- Input desired time (hour, weekday, month)
- View predicted energy in kW
- Visualize prediction history
- Check real historical energy usage


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
