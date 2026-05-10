#!/usr/bin/env python3
"""
walk_forward_backtest.py — Genuine Walk-Forward Backtest
=========================================================

Trains on prior 12 months, tests on the next month. Repeats 36 steps.
Only evaluates trades the model has NEVER seen during training.

Usage:
    python train_pipeline/walk_forward_backtest.py \
      --data train_pipeline/data/events_long_events_long_labeled.csv \
      --model-dir train_pipeline/models_gpu_long_lb15_momentum \
      --side long --lookback 15 --threshold 0.55 --out walk_forward_results.csv

For both sides, run separately and concatenate outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("WFBacktest")

SNAPSHOT_FEATURES = ["atr_norm", "bb_position", "candle_body", "upper_wick", "lower_wick", "range_vs_atr"]
VELOCITY_FEATURES = ["pullback_speed", "vwap_slope_5", "volume_ratio"]
MICRO_COLS = ["tick_imbalance", "ofi_window", "cs_spread", "kyle_lambda", "vprof_poc_dist"]

TRAIN_MONTHS = 12
TEST_MONTHS = 1
MIN_TRAIN_EVENTS = 500


def compute_features(df: pd.DataFrame, lookback: int = 15) -> pd.DataFrame:
    close = df["close"].values.astype("float64")
    high = df["high"].values.astype("float64")
    low = df["low"].values.astype("float64")
    open_v = df["open"].values.astype("float64")
    vol = df["tick_volume"].values.astype("float64") if "tick_volume" in df.columns else np.ones(len(df))
    atr = df["ATR"].values.astype("float64") if "ATR" in df.columns else np.ones(len(df))
    vwap = df["VWAP"].values.astype("float64") if "VWAP" in df.columns else close
    ema20 = df["EMA20"].values.astype("float64") if "EMA20" in df.columns else close

    if all(c in df.columns for c in ["Upper_Band", "Lower_Band"]):
        bb_pos = np.clip((close - df["Lower_Band"].values) / (df["BB_width"].values + 1e-9), 0, 1)
    else:
        sma20 = pd.Series(close).rolling(20).mean().values
        std20 = pd.Series(close).rolling(20).std().values
        bb_pos = np.clip((close - (sma20 - std20 * 2)) / (std20 * 4 + 1e-9), 0, 1)

    df_out = pd.DataFrame(index=df.index)
    df_out["atr_norm"] = atr / (close + 1e-9)
    df_out["bb_position"] = bb_pos
    df_out["candle_body"] = (close - open_v) / (atr + 1e-9)
    df_out["upper_wick"] = (high - np.maximum(open_v, close)) / (atr + 1e-9)
    df_out["lower_wick"] = (np.minimum(open_v, close) - low) / (atr + 1e-9)
    df_out["range_vs_atr"] = (high - low) / (atr + 1e-9)
    df_out["above_ema200"] = (close > pd.Series(close).ewm(span=200, adjust=False).mean().values).astype("float64")

    n = len(df)
    for i in range(n):
        if i < 15:
            df_out.loc[i, "pullback_speed"] = 0.0
            df_out.loc[i, "vwap_slope_5"] = 0.0
            df_out.loc[i, "volume_ratio"] = 1.0
            continue
        df_out.loc[i, "pullback_speed"] = np.clip((close[i] - close[i - 5]) / (atr[i] * 5 + 1e-9), -10, 10)
        df_out.loc[i, "vwap_slope_5"] = (vwap[i] - vwap[i - 5]) / (atr[i] + 1e-9)
        df_out.loc[i, "volume_ratio"] = vol[i] / (np.mean(vol[max(0, i - 5):i]) + 1e-9)

    df_out = df_out.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in MICRO_COLS:
        df_out[col] = df[col].values.astype("float64") if col in df.columns else 0.0

    if lookback > 0:
        for k in range(lookback):
            col = f"return_lag_{k}"
            df_out[col] = np.clip((close - np.roll(close, k + 1)) / (np.roll(atr, k + 1) + 1e-9), -10, 10)
            df_out[col] = np.nan_to_num(df_out[col], nan=0.0)

    return df_out


def train_monthly_model(train_df: pd.DataFrame, lookback: int):
    """Train LightGBM on 12-month window. Returns (model, feature_names)."""
    import lightgbm as lgb
    X = compute_features(train_df, lookback=lookback)
    feats = [c for c in X.columns if c not in ["tb_label"]]
    X_in = X[feats].values.astype(np.float32)
    y = train_df["tb_label"].values.astype(int)

    n_neg = (y == 0).sum()
    n_pos = max((y == 1).sum(), 1)
    scale_pos_weight = min(n_neg / n_pos, 3.0)

    params = {
        "objective": "binary", "verbose": -1, "random_state": 42, "max_bin": 63,
        "num_leaves": 31, "max_depth": 6, "min_child_samples": 30,
        "n_estimators": 500, "subsample": 0.8, "colsample_bytree": 0.8,
        "lambda_l1": 0.5, "lambda_l2": 0.5,
        "scale_pos_weight": scale_pos_weight, "learning_rate": 0.02,
    }
    try:
        params["device_type"] = "gpu"
        train_ds = lgb.Dataset(X_in, label=y)
        model = lgb.train(params, train_ds, num_boost_round=100)
    except Exception:
        params.pop("device_type", None)
        train_ds = lgb.Dataset(X_in, label=y)
        model = lgb.train(params, train_ds, num_boost_round=100)

    return model, feats


def main():
    ap = argparse.ArgumentParser(description="Genuine Walk-Forward Backtest")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--side", type=str, default="long")
    ap.add_argument("--lookback", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--out", type=str, default="walk_forward_results.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df.columns = [c.lower().strip() for c in df.columns]

    if "time" not in df.columns:
        logger.error("Data must have 'time' column"); sys.exit(1)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Drop timeouts, remap labels
    df = df[df["tb_label"] != 1].copy().reset_index(drop=True)
    df["tb_label"] = df["tb_label"].map({2: 1, 0: 0})
    logger.info(f"{args.side.upper()} | {len(df):,} clean events | {df['time'].min()} -> {df['time'].max()}")

    min_date = df["time"].min()
    max_date = df["time"].max()

    all_results = []

    test_start = min_date + pd.DateOffset(months=TRAIN_MONTHS)
    while test_start < max_date:
        train_start = test_start - pd.DateOffset(months=TRAIN_MONTHS)
        train_end = test_start
        test_end = min(test_start + pd.DateOffset(months=TEST_MONTHS), max_date)

        train_mask = (df["time"] >= train_start) & (df["time"] < train_end)
        test_mask = (df["time"] >= test_start) & (df["time"] < test_end)

        train_df = df[train_mask].copy().reset_index(drop=True)
        test_df = df[test_mask].copy().reset_index(drop=True)

        if len(train_df) < MIN_TRAIN_EVENTS:
            test_start = test_end
            continue
        if len(test_df) < 10:
            test_start = test_end
            continue

        logger.info(f"Window: train {train_start.date()}..{train_end.date()} ({len(train_df):,}) | "
                    f"test {test_start.date()}..{test_end.date()} ({len(test_df):,})")

        try:
            model, feats = train_monthly_model(train_df, args.lookback)

            X_test = compute_features(test_df, lookback=args.lookback)
            available = [f for f in feats if f in X_test.columns]
            X_in = X_test[available].values.astype(np.float32)

            p_raw = model.predict(X_in)

            for i in range(len(test_df)):
                p = float(p_raw[i])
                if p < args.threshold:
                    continue
                row = test_df.iloc[i]
                label = int(row["tb_label"])
                entry = float(row["close"])
                atr_val = float(row.get("ATR", entry * 0.01))
                sl_dist = atr_val * 1.0
                tp_dist = 2.0 * sl_dist

                if label == 1:
                    r_multiple = 2.0
                    outcome = "TP"
                else:
                    r_multiple = -1.0
                    outcome = "SL"

                all_results.append({
                    "side": args.side,
                    "test_month": str(test_start.date()),
                    "time": row.get("time", i),
                    "entry": entry, "p_raw": p,
                    "sl_dist": sl_dist, "tp_dist": tp_dist,
                    "outcome": outcome, "r_multiple": r_multiple,
                })

        except Exception as e:
            logger.error(f"Window failed: {e}")

        test_start = test_end

    if not all_results:
        logger.error("No trades generated")
        sys.exit(1)

    final = pd.DataFrame(all_results)
    final["cumulative_r"] = final["r_multiple"].cumsum()
    final.to_csv(args.out, index=False)

    n_trades = len(final)
    n_wins = (final["outcome"] == "TP").sum()
    wr = n_wins / max(n_trades, 1) * 100
    total_r = final["r_multiple"].sum()
    n_months = final["test_month"].nunique()
    logger.info(f"\n{'='*50}")
    logger.info(f"  WALK-FORWARD RESULTS ({args.side.upper()}, T={args.threshold:.2f})")
    logger.info(f"{'='*50}")
    logger.info(f"  Windows       : {n_months}")
    logger.info(f"  Total trades  : {n_trades}")
    logger.info(f"  Wins (TP)     : {n_wins} ({wr:.1f}%)")
    logger.info(f"  Total R       : {total_r:+.2f}")
    logger.info(f"  Avg R/trade   : {total_r/max(n_trades,1):+.3f}")
    logger.info(f"  Avg trades/mo : {n_trades/max(n_months,1):.0f}")
    logger.info(f"  Results saved : {args.out}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
