import pandas as pd
import os

def load_data(save_path="data/processed/energy_clean.csv"):
    raw_path="data/raw/household_power_consumption.txt"
    
    df = pd.read_csv(
        raw_path,
        sep=';',
        na_values='?',
        low_memory=False
    )
    # Combine Date and Time columns into a single datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    # Select only the relevant columns
    df = df[['datetime', 'Global_active_power']]
    
    # Drop rows with missing values
    df = df.dropna()
    
    # set datetime as index
    df = df.set_index('datetime')
    df = df.resample('1h').mean()
    
    df = df.round(3)
    
    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print(f"Saved cleaned data to {save_path}")

if __name__ == "__main__":
    load_data()

