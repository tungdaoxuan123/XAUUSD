import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
from datetime import datetime

# Configuration matching plan.txt
REPO_ID = "Pcitycrypto/xauusd"
FILENAME = "XAUUSD_1m.csv"
OUTPUT_DIR = Path("train_pipeline/data")
OUTPUT_FILE = OUTPUT_DIR / "xauusd_m1.csv"

# Labeling parameters for preview
HORIZON = 5
BUY_THRESHOLD = 0.0005
SELL_THRESHOLD = 0.0005

def main():
    print(f"--- HuggingFace XAUUSD Data Fetcher & Cleaner ---")
    
    # 1. Download
    print(f"Downloading {FILENAME} from {REPO_ID}...")
    try:
        file_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
        print(f"Downloaded to cache: {file_path}")
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        sys.exit(1)

    # 2. Load
    print(f"Loading CSV into memory...")
    df = pd.read_csv(file_path)
    
    # 3. Clean and Rename
    print(f"Cleaning data (Weekends, 0s, NaNs, duplicates)...")
    
    # Clean column names (strip whitespace and remove brackets if present)
    df.columns = [c.strip().replace('<', '').replace('>', '') for c in df.columns]

    # Rename columns to match pipeline: "datetime" -> "time", "tickvol" -> "tick_volume"
    df = df.rename(columns={
        "datetime": "time",
        "tickvol": "tick_volume"
    })
    
    # Parse time
    df["time"] = pd.to_datetime(df["time"])
    
    # Drop rows where any required column is NaN or 0 (especially volume/close)
    required = ["time", "open", "high", "low", "close", "tick_volume"]
    before_count = len(df)
    
    # Drop NaNs
    df = df.dropna(subset=required)
    
    # Drop where price or volume <= 0
    df = df[
        (df["open"] > 0) & 
        (df["high"] > 0) & 
        (df["low"] > 0) & 
        (df["close"] > 0) & 
        (df["tick_volume"] > 0)
    ]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["time"])
    
    # Sort ascending
    df = df.sort_values("time")
    
    # Remove Weekends (Saturday=5, Sunday=6)
    df = df[df["time"].dt.dayofweek < 5]
    
    after_count = len(df)
    print(f"Cleaned: {after_count:,} rows (Dropped {before_count - after_count:,} invalid/weekend rows)")

    # 4. Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df = df[required] # Ensure column order
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Saved cleaned data to: {OUTPUT_FILE}")

    # 5. Label Distribution Preview
    print(f"\n--- Label Distribution Preview (Horizon={HORIZON}, Thr={BUY_THRESHOLD}) ---")
    # Simple vectorized label calc
    close = final_df["close"].values
    n = len(close)
    
    # We need i+horizon
    labels = np.zeros(n)
    # Fast vectorized return calculation for preview
    # ret = (close[i+horizon] - close[i]) / close[i]
    if n > HORIZON:
        rets = (close[HORIZON:] - close[:-HORIZON]) / close[:-HORIZON]
        # Align rets with indices i
        labels[:-HORIZON][rets > BUY_THRESHOLD] = 1
        labels[:-HORIZON][rets < -SELL_THRESHOLD] = -1
        
    counts = pd.Series(labels[:-HORIZON]).value_counts().sort_index()
    print(f"SELL (-1): {counts.get(-1.0, 0):,}")
    print(f"HOLD (0):  {counts.get(0.0, 0):,}")
    print(f"BUY  (+1): {counts.get(1.0, 0):,}")
    
    print(f"\nDate Range: {final_df['time'].iloc[0]} -> {final_df['time'].iloc[-1]}")
    print(f"Total rows: {len(final_df):,}")

if __name__ == "__main__":
    main()
