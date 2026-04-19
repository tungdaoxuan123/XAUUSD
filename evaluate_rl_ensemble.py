#!/usr/bin/env python3
"""
evaluate_rl_ensemble.py - Evaluate PPO/TD3/SAC Ensemble on Historical Data

Loads the RL models from ensemble_models/ and runs a full walk-through
backtest on any OHLCV CSV, reporting:
  - Overall accuracy (SELL / HOLD / BUY direction vs actual next-bar returns)
  - Simulated P&L (no real trading, just signal tracking)
  - Per-bar signal distribution

The RL models expect a 15-feature observation:
    [close_lag_9, ..., close_lag_0, RSI, MACD, MACD_signal, position, balance]

Usage:
    python evaluate_rl_ensemble.py
    python evaluate_rl_ensemble.py --data train_pipeline/data/xauusd_m1_2m.csv
    python evaluate_rl_ensemble.py --data train_pipeline/data/xauusd_m1_2m.csv --out-dir reports/rl_eval
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, TD3, SAC

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RLEnsembleEval")

# ---------------------------------------------------------------------------
# Observation Builder (mirrors TradingEnv._get_observation)
# ---------------------------------------------------------------------------
LOOKBACK = 10


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI and MACD on a lowercase-column OHLCV DataFrame."""
    df = df.copy()
    close = df["close"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD (12-26-9 to match TradingEnv)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df.dropna().reset_index(drop=True)


def build_observation(df: pd.DataFrame, idx: int, position: float, balance: float) -> np.ndarray:
    """
    Build a 15-feature observation at row `idx` (must be >= LOOKBACK).
    Features: [10 close prices] + [RSI, MACD, MACD_signal, position, balance]
    """
    prices = df["close"].iloc[idx - LOOKBACK: idx].values.astype(np.float32)
    rsi = float(df["RSI"].iloc[idx - 1])
    macd = float(df["MACD"].iloc[idx - 1])
    macd_signal = float(df["MACD_signal"].iloc[idx - 1])
    return np.array([*prices, rsi, macd, macd_signal, position, balance], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------
ALGO_MAP = {"PPO": PPO, "TD3": TD3, "SAC": SAC}


def load_ensemble(model_dir: str) -> tuple:
    """Load RL models and config. Returns (models_dict, weights_dict)."""
    config_path = os.path.join(model_dir, "ensemble_config.json")
    if not os.path.exists(config_path):
        logger.error(f"ensemble_config.json not found in {model_dir}")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    models = {}
    weights = config.get("weights", {})
    for model_name in config["models"]:
        algo_name = config["config"][model_name]["algorithm"]
        model_path = os.path.join(model_dir, f"{model_name}_model.zip")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)
        algo_cls = ALGO_MAP[algo_name]
        models[model_name] = algo_cls.load(model_path)
        logger.info(f"  Loaded [{algo_name}] from {model_path}")

    logger.info(f"  {len(models)} models loaded: {list(models.keys())}")
    return models, weights


# ---------------------------------------------------------------------------
# Ensemble Prediction (mirrors EnsembleTrader._weighted_vote_with_confidence)
# ---------------------------------------------------------------------------

def predict_ensemble(models: dict, weights: dict, obs: np.ndarray) -> tuple:
    """
    Returns (action_label, confidence)
    action_label: +1 (BUY) | -1 (SELL) | 0 (HOLD)
    """
    buy_votes, sell_votes, hold_votes = 0.0, 0.0, 0.0
    total_weight = 0.0

    for name, model in models.items():
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        raw = float(action[0])
        conf = abs(raw)
        w = weights.get(name, 1.0) * conf
        total_weight += w

        if raw > 0.1:
            buy_votes += w
        elif raw < -0.1:
            sell_votes += w
        else:
            hold_votes += w

    max_votes = max(buy_votes, sell_votes, hold_votes)
    confidence = max_votes / total_weight if total_weight > 0 else 0.0

    if buy_votes > sell_votes and buy_votes > hold_votes:
        return 1, confidence
    elif sell_votes > buy_votes and sell_votes > hold_votes:
        return -1, confidence
    else:
        return 0, confidence


# ---------------------------------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------------------------------

def run_evaluation(data_path: str, model_dir: str, horizon: int, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    logger.info(f"Loading data: {data_path}")
    df_raw = pd.read_csv(data_path)
    df_raw.columns = [c.lower().strip() for c in df_raw.columns]
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    df_raw = df_raw.sort_values("time").reset_index(drop=True)
    logger.info(f"  {len(df_raw):,} rows | {df_raw['time'].iloc[0]} -> {df_raw['time'].iloc[-1]}")

    # ---- Indicators ----
    logger.info("Computing indicators (RSI, MACD 12-26-9)...")
    df = compute_indicators(df_raw)
    logger.info(f"  {len(df):,} rows after dropna")

    # ---- Load models ----
    logger.info(f"Loading ensemble from: {model_dir}")
    models, weights = load_ensemble(model_dir)

    # ---- Walk-forward evaluation ----
    logger.info(f"Running walk-forward evaluation (horizon={horizon} bars)...")

    results = []
    position = 0.0
    balance = 10000.0

    n = len(df)
    eval_range = range(LOOKBACK, n - horizon)
    total = len(eval_range)

    for i, idx in enumerate(eval_range):
        if i % 5000 == 0:
            logger.info(f"  Progress: {i:,}/{total:,} ({100*i/total:.1f}%)")

        obs = build_observation(df, idx, position, balance)
        signal, conf = predict_ensemble(models, weights, obs)

        # Actual outcome: forward return over 'horizon' bars
        current_price = df["close"].iloc[idx - 1]
        future_price  = df["close"].iloc[idx - 1 + horizon]
        fwd_ret = (future_price - current_price) / current_price

        # True direction label (same thresholds as RL env's action)
        if fwd_ret > 0.001:
            true_label = 1
        elif fwd_ret < -0.001:
            true_label = -1
        else:
            true_label = 0

        # Simple P&L simulation (long on BUY, short on SELL, 0.03 lots)
        lot = 0.03
        pip_value = 100  # XAUUSD: ~$1 per pip per 0.01 lot
        pnl = 0.0
        if signal == 1:
            pnl = fwd_ret * current_price * lot * pip_value
        elif signal == -1:
            pnl = -fwd_ret * current_price * lot * pip_value

        results.append({
            "time": df["time"].iloc[idx - 1],
            "signal": signal,
            "confidence": conf,
            "true_label": true_label,
            "fwd_return": fwd_ret,
            "pnl": pnl,
        })

    # ---- Metrics ----
    results_df = pd.DataFrame(results)

    active = results_df[results_df["signal"] != 0]
    correct = (active["signal"] == active["true_label"]).sum()
    total_active = len(active)
    accuracy = correct / total_active if total_active > 0 else 0

    cumulative_pnl = results_df["pnl"].sum()

    buy_pct  = (results_df["signal"] == 1).mean() * 100
    sell_pct = (results_df["signal"] == -1).mean() * 100
    hold_pct = (results_df["signal"] == 0).mean() * 100

    avg_conf = results_df["confidence"].mean()

    logger.info("\n" + "=" * 65)
    logger.info("  RL ENSEMBLE — FTMO DATA EVALUATION RESULTS")
    logger.info("=" * 65)
    logger.info(f"  Models         : PPO + TD3 + SAC (ensemble_models/)")
    logger.info(f"  Data           : {data_path}")
    logger.info(f"  Total Bars     : {len(results_df):,}")
    logger.info(f"  Horizon        : {horizon} bars")
    logger.info("-" * 65)
    logger.info(f"  Signal Direction Accuracy : {accuracy:.2%}")
    logger.info(f"  (Only on active BUY/SELL signals, excl. HOLD)")
    logger.info(f"  Simulated P&L  : ${cumulative_pnl:+,.2f}")
    logger.info(f"  Avg Confidence : {avg_conf:.4f}")
    logger.info("-" * 65)
    logger.info(f"  Signal Mix     : BUY {buy_pct:.1f}%  |  SELL {sell_pct:.1f}%  |  HOLD {hold_pct:.1f}%")
    logger.info("=" * 65)

    # ---- Save results ----
    out_csv = os.path.join(out_dir, "rl_ensemble_eval_results.csv")
    results_df.to_csv(out_csv, index=False)
    logger.info(f"\nDetailed results saved to: {out_csv}")

    # Per-model stats
    logger.info("\nPer-model raw summary (spot check last 100 signals):")
    for name, model in models.items():
        sample_obs = build_observation(df, LOOKBACK + 5, 0.0, 10000.0)
        action, _ = model.predict(sample_obs.reshape(1, -1), deterministic=True)
        logger.info(f"  [{name}] Sample action: {float(action[0]):.4f}")

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RL Ensemble (PPO/TD3/SAC) on FTMO data")
    parser.add_argument("--data",      default="train_pipeline/data/xauusd_m1_2m.csv",
                        help="Path to OHLCV CSV (default: FTMO 2M bars dataset)")
    parser.add_argument("--models",    default="ensemble_models",
                        help="Path to ensemble_models directory (default: ensemble_models)")
    parser.add_argument("--horizon",   type=int, default=10,
                        help="Forward bar horizon for measuring accuracy (default: 10)")
    parser.add_argument("--out-dir",   default="train_pipeline/reports/rl_eval",
                        help="Output directory for results CSV")
    args = parser.parse_args()

    run_evaluation(args.data, args.models, args.horizon, args.out_dir)


if __name__ == "__main__":
    main()
