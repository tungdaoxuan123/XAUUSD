#!/usr/bin/env python3
"""
generate_backtest_vis_csv.py

Evaluates the XAUUSD ensemble on historical micro-enriched M1 data 
and exports a CSV formatted for MT5 chart visualization.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from train_pipeline.ensemble_gpu import EnsembleGPU, _add_indicators, get_feature_names
except ImportError:
    from ensemble_gpu import EnsembleGPU, _add_indicators, get_feature_names

# Configuration
DATA_PATH = "train_pipeline/data/xauusd_m1_micro.csv"
MODEL_DIR = "train_pipeline/models_gpu"
OUTPUT_PATH = "train_pipeline/reports/backtest_signals_xauusd.csv"
CONFIDENCE_THRESHOLD = 0.7
MAX_HORIZON = 240  # 4 hours max for simulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VisGenerator")

def simulate_outcome(df, i, side, entry_price, sl_price, tp_price, max_horizon):
    """
    Scan forward in time to determine if the trade hits TP, SL, or Timeout.
    """
    n = len(df)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    
    # Start looking from the very next bar
    for j in range(i + 1, min(i + max_horizon + 1, n)):
        hit_tp = False
        hit_sl = False
        
        if side == 1: # LONG
            if high[j] >= tp_price: hit_tp = True
            if low[j] <= sl_price: hit_sl = True
        else: # SHORT
            if low[j] <= tp_price: hit_tp = True
            if high[j] >= sl_price: hit_sl = True
            
        # Handle same-bar tie
        if hit_tp and hit_sl:
            return "TIE"
        if hit_tp:
            return "WIN"
        if hit_sl:
            return "LOSS"
            
    return "TIMEOUT"

def main():
    parser = argparse.ArgumentParser(description="Generate MT5 Signal Visualization CSV")
    parser.add_argument("--data", default="train_pipeline/data/xauusd_m1_micro.csv", help="Input M1 micro CSV")
    parser.add_argument("--model-dir", default="train_pipeline/models_gpu", help="Ensemble model directory")
    parser.add_argument("--output", default="train_pipeline/reports/backtest_signals_xauusd.csv", help="Output signal CSV")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--horizon", type=int, default=120, help="Max trade horizon in bars")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return

    # 1. Load Data
    logger.info(f"Loading data: {args.data}")
    df_raw = pd.read_csv(args.data)
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    
    # Ensure indicators are calculated
    df = _add_indicators(df_raw)
    n = len(df)
    logger.info(f"Loaded {n:,} bars. Computing feature matrix...")
    
    # 2. Build Feature Matrix
    # 3. Load Model
    logger.info(f"Loading ensemble: {args.model_dir}")
    try:
        ens = EnsembleGPU.load(args.model_dir)
    except Exception as e:
        logger.error(f"Failed to load ensemble: {e}")
        return

    lookback = ens.lookback
    expanded = ens.metadata.get("expanded_features", False)
    micro = ens.metadata.get("microstructure_features", False) or ens.metadata.get("micro", False)
    
    feature_names = get_feature_names(lookback, expanded=expanded, micro=micro)
    
    rows = []
    for i in range(lookback - 1, n):
        # Determine observation vector based on model type
        price_lags = df["close"].iloc[i - lookback + 1 : i + 1].values
        latest = df.iloc[i]
        
        # Base observation
        row = list(price_lags) + [
            float(latest["RSI"]),
            float(latest["MACD"]),
            float(latest["Signal_Line"]),
            0.0,      # current_position
            10000.0,  # current_balance
        ]
        
        if expanded:
            row += [
                float(latest["MACD_Hist"]),
                float(latest["VWAP"]),
                float(latest.get("close_minus_vwap", 0.0)),
                float(latest["ATR"]),
                float(latest["BB_width"]),
            ]
            
        if micro:
            row += [
                float(latest.get("tick_imbalance", 0.0)),
                float(latest.get("bid_ask_vol_imbalance", 0.0)),
                float(latest.get("spread_mean", 0.0)),
                float(latest.get("ofi_window", 0.0)),
                float(latest.get("of_pressure_flag", 0.0)),
                float(latest.get("vprof_poc_dist", 0.0)),
                float(latest.get("vprof_in_value_area", 0.0)),
                float(latest.get("vprof_hvn_flag", 0.0)),
                float(latest.get("vprof_lvn_flag", 0.0)),
            ]
            
        rows.append(row)
        
    X = pd.DataFrame(rows, columns=feature_names)
    # Align primary dataframe with X
    df_eval = df.iloc[lookback-1:].reset_index(drop=True)
        
    # 4. Generate Predictions
    logger.info(f"Generating model predictions (vectorized) for {len(X)} rows...")
    actions, confs = ens.predict_batch(X)
    
    # 5. Filter Signals and Simulate Outcomes
    logger.info(f"Processing signals (Confidence >= {args.confidence})...")
    signals_data = []
    
    for i in range(len(df_eval)):
        action = actions[i]
        conf = confs[i]
        
        # Confidence threshold
        if abs(action) < 0.4 or conf < args.confidence:
            continue
            
        # Confluence check (matches live bot)
        row = df_eval.iloc[i]
        bull_confirm = row["MACD"] > row["Signal_Line"] and row["close"] > row["VWAP"]
        bear_confirm = row["MACD"] < row["Signal_Line"] and row["close"] < row["VWAP"]
        
        is_buy = action > 0 and bull_confirm
        is_sell = action < 0 and bear_confirm
        
        if not (is_buy or is_sell):
            continue
            
        # Success!
        side = 1 if is_buy else -1
        entry_price = row["close"]
        atr = row["ATR"]
        
        if side == 1:
            sl = entry_price - (2 * atr)
            tp = entry_price + (3 * atr)
        else:
            sl = entry_price + (2 * atr)
            tp = entry_price - (3 * atr)
            
        outcome = simulate_outcome(df_eval, i, side, entry_price, sl, tp, args.horizon)
        t_str = row["time"].strftime("%Y.%m.%d %H:%M")
        
        signals_data.append({
            "time": t_str,
            "signal": side,
            "entry": round(entry_price, 2),
            "tp": round(tp, 2),
            "sl": round(sl, 2),
            "outcome": outcome
        })
        
    # 6. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df = pd.DataFrame(signals_data)
    
    if out_df.empty:
        logger.warning("No signals generated.")
    else:
        out_df.to_csv(args.output, index=False)
        logger.info(f"SUCCESS: Generated {len(out_df)} signals -> {args.output}")
        win_rate = (out_df["outcome"] == "WIN").mean()
        logger.info(f"Win Rate: {win_rate:.1%}")
        logger.info(f"Outcomes: \n{out_df['outcome'].value_counts().to_string()}")

if __name__ == "__main__":
    main()
