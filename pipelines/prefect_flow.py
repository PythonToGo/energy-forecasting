from prefect import flow
import subprocess
import pandas as pd
import os

@flow
def retrain_pipeline():
    subprocess.run(["python", "src/data_loader.py"])
    subprocess.run(["python", "src/train_model.py"])

if __name__ == "__main__":
    retrain_pipeline()