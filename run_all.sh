#!/bin/bash

# run_all.sh

echo "Starting FastAPI..."
uvicorn api.main:app --reload &

echo "Starting Streamlit dashboard..."
streamlit run dashboard/app.py &

echo "Starting MLflow UI..."
mlflow ui &

echo "All services started. Press [CTRL+C] to stop."
wait
