#!/usr/bin/env python3
"""
backtest_comparison.py - Comparative Backtester

Compares the old Random Forest model against the new LightGBM Ensemble
using a walk-forward trade simulation with Stop-Loss, Take-Profit, and Timeout.
"""

import argparse
import joblib
import logging
import os
import sys
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Backtester")

# ---------------------------------------------------------------------------
# Simulation Logic
# ---------------------------------------------------------------------------

def run_simulation(df, signals, tp_pct, sl_pct, max_horizon):
    """
    Simulate trades bar-by-bar.
    signals: Series of (-1, 0, 1) aligned with df
    """
    trades = []
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    time = df["time"].values
    n = len(df)
    
    active_trade = None
    
    for i in range(n):
        # 1. Check if we hit exit on existing trade
        if active_trade:
            entry_price = active_trade["entry_price"]
            side = active_trade["side"] # 1 for BUY, -1 for SELL
            bars_held = i - active_trade["entry_idx"]
            
            # Barriers
            tp_price = active_trade["tp_price"]
            sl_price = active_trade["sl_price"]
            
            hit_tp = False
            hit_sl = False
            
            if side == 1: # LONG
                if high[i] >= tp_price: hit_tp = True
                if low[i] <= sl_price: hit_sl = True
            else: # SHORT
                if low[i] <= tp_price: hit_tp = True
                if high[i] >= sl_price: hit_sl = True
                
            exit_reason = None
            exit_price = None
            
            if hit_tp and hit_sl:
                exit_reason = "TIE (0 PnL)"
                exit_price = entry_price
            elif hit_tp:
                exit_reason = "TP"
                exit_price = tp_price
            elif hit_sl:
                exit_reason = "SL"
                exit_price = sl_price
            elif bars_held >= max_horizon:
                exit_reason = "TIMEOUT"
                exit_price = close[i]
                
            if exit_reason:
                # Calculate PnL
                if side == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                
                trades.append({
                    "entry_time": active_trade["entry_time"],
                    "exit_time": time[i],
                    "side": "BUY" if side == 1 else "SELL",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "reason": exit_reason,
                    "duration": bars_held
                })
                active_trade = None
                
        # 2. Check for new entry signal (if not already in trade)
        if not active_trade and signals.iloc[i] != 0:
            side = int(signals.iloc[i])
            p0 = close[i]
            
            if side == 1: # BUY
                tp = p0 * (1 + tp_pct)
                sl = p0 * (1 - sl_pct)
            else: # SELL
                tp = p0 * (1 - tp_pct)
                sl = p0 * (1 + sl_pct)
                
            active_trade = {
                "entry_idx": i,
                "entry_time": time[i],
                "entry_price": p0,
                "side": side,
                "tp_price": tp,
                "sl_price": sl
            }
            
    return pd.DataFrame(trades)

def compute_metrics(trades):
    if trades.empty:
        return {
            "Total Trades": 0,
            "Win Rate": 0.0,
            "Net PnL %": 0.0,
            "Profit Factor": 0.0,
            "Avg Trade %": 0.0,
            "Max Drawdown %": 0.0
        }
    
    win_rate = (trades["pnl_pct"] > 0).mean()
    net_pnl = trades["pnl_pct"].sum()
    
    gross_profit = trades[trades["pnl_pct"] > 0]["pnl_pct"].sum()
    gross_loss = abs(trades[trades["pnl_pct"] < 0]["pnl_pct"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Drawdown
    equity_curve = (1 + trades["pnl_pct"]).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        "Total Trades": len(trades),
        "Win Rate": f"{win_rate:.1%}",
        "Net PnL %": f"{net_pnl:.2%}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Avg Trade %": f"{trades['pnl_pct'].mean():.4%}",
        "Max Drawdown %": f"{abs(max_dd):.2%}"
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RF vs. LightGBM Backtest Comparison")
    parser.add_argument("--data", required=True, help="Path to M1 CSV data")
    parser.add_argument("--rf-model", default="train_pipeline/models/sklearn_random_forest_default.joblib")
    parser.add_argument("--ensemble-dir", default="train_pipeline/models_gpu")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--sl-pct", type=float, default=0.001)
    parser.add_argument("--tp-pct", type=float, default=0.001)
    parser.add_argument("--max-horizon", type=int, default=60)
    args = parser.parse_args()

    # 1. Load Data
    logger.info(f"Loading data: {args.data}")
    df_raw = pd.read_csv(args.data)
    if "time" in df_raw.columns:
        df_raw["time"] = pd.to_datetime(df_raw["time"])
    
    df = _add_indicators(df_raw)
    n = len(df)
    logger.info(f"Loaded {n:,} rows after indicators.")

    # 2. Build Feature Matrix (15 standard features)
    lookback = 10
    feature_names = get_feature_names(lookback, expanded=False, micro=False)
    
    rows = []
    for i in range(lookback - 1, n):
        price_lags = df["close"].iloc[i - lookback + 1: i + 1].values
        latest = df.iloc[i]
        # Same logic as train_ensemble.py
        row = list(price_lags) + [
            float(latest["RSI"]),
            float(latest["MACD"]),
            float(latest["Signal_Line"]),
            0.0, # current_position
            10000.0, # current_balance
        ]
        rows.append(row)
    
    X = pd.DataFrame(rows, columns=feature_names)
    df_eval = df.iloc[lookback-1:].reset_index(drop=True)
    logger.info(f"Evaluation Matrix: {X.shape}")

    # 3. Load Models
    logger.info(f"Loading RF model: {args.rf_model}")
    rf_model = joblib.load(args.rf_model)
    
    logger.info(f"Loading LightGBM Ensemble: {args.ensemble_dir}")
    ens_gpu = EnsembleGPU.load(args.ensemble_dir)

    # 4. Generate Predictions
    logger.info("Generating signals...")
    # RF
    rf_probs = rf_model.predict_proba(X)
    rf_class = np.argmax(rf_probs, axis=1) # 0=-1, 1=0, 2=1
    rf_conf = np.max(rf_probs, axis=1)

    # Ensemble
    # Use the new vectorized predict_batch method
    ens_action, ens_conf = ens_gpu.predict_batch(X)
    
    # Use numpy arrays for signal building to avoid pandas indexing pitfalls
    rf_signals_arr = np.zeros(len(df_eval), dtype=int)
    rf_signals_arr[(rf_class == 0) & (rf_conf >= args.confidence)] = -1
    rf_signals_arr[(rf_class == 2) & (rf_conf >= args.confidence)] = 1
    rf_signals = pd.Series(rf_signals_arr)
    
    ens_signals_arr = np.zeros(len(df_eval), dtype=int)
    # Ensemble actions: -0.5 = SELL, +0.5 = BUY
    ens_signals_arr[(ens_action < -0.4) & (ens_conf >= args.confidence)] = -1
    ens_signals_arr[(ens_action > 0.4) & (ens_conf >= args.confidence)] = 1
    ens_signals = pd.Series(ens_signals_arr)
    
    # 5. Run Simulations
    logger.info("Running RF Simulation...")
    rf_trades = run_simulation(df_eval, rf_signals, args.tp_pct, args.sl_pct, args.max_horizon)
    
    logger.info("Running Ensemble Simulation...")
    ens_trades = run_simulation(df_eval, ens_signals, args.tp_pct, args.sl_pct, args.max_horizon)

    # 6. Report
    rf_metrics = compute_metrics(rf_trades)
    ens_metrics = compute_metrics(ens_trades)
    
    results = pd.DataFrame({
        "Metric": rf_metrics.keys(),
        "Old RF Model": rf_metrics.values(),
        "New LGBM Ensemble": ens_metrics.values()
    })
    
    print("\n" + "="*50)
    print("      BACKTEST COMPARISON RESULTS")
    print("="*50)
    print(results.to_string(index=False))
    print("="*50)
    
    if not ens_trades.empty:
        logger.info(f"Ensemble Best Trade: {ens_trades['pnl_pct'].max():.2%} | Worst: {ens_trades['pnl_pct'].min():.2%}")

if __name__ == "__main__":
    main()
