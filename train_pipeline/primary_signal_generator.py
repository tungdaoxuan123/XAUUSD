#!/usr/bin/env python3
"""
primary_signal_generator.py — Trend Pullback Event Filter
=========================================================

Takes raw OHLCV CSV and outputs only bars where a trend pullback event fires.
Two setups:

  Setup A (Long):  close > VWAP, EMA5 > EMA20, close reclaims EMA20 from below,
                   ATR14 > 0.8 * ATR50 mean

  Setup B (Short): close < VWAP, EMA5 < EMA20, close loses EMA20 from above,
                   ATR14 > 0.8 * ATR50 mean

Mutual exclusion: if both fire at the same bar, skip the bar.

Output: events_raw.csv  (OHLCV + ATR + EMA5 + EMA20 + VWAP + RSI + MACD_Hist + side)

Usage:
    python train_pipeline/primary_signal_generator.py --data train_pipeline/data/xauusd_m1_4y_dukas.csv --out train_pipeline/data/events_raw.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PrimarySignal")

OUTPUT_COLUMNS = [
    "time", "open", "high", "low", "close", "tick_volume",
    "ATR", "EMA5", "EMA20", "VWAP", "RSI", "MACD_Hist", "side",
]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators needed for event detection."""
    df = df.sort_values("time").reset_index(drop=True).copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["tick_volume"].replace(0, np.nan).ffill().fillna(1)

    # EMA
    df["EMA5"] = close.ewm(span=5, adjust=False).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()

    # VWAP
    tp = (high + low + close) / 3
    df["TPV"] = tp * vol
    df["Cum_TPV"] = df["TPV"].cumsum()
    df["Cum_Vol"] = vol.cumsum()
    df["VWAP"] = df["Cum_TPV"] / df["Cum_Vol"].replace(0, np.nan)

    # ATR
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD Histogram
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema24 = close.ewm(span=24, adjust=False).mean()
    macd_line = ema8 - ema24
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = macd_line - signal_line

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def filter_events(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Setup A (Long) and Setup B (Short) filters with mutual exclusion."""
    close = df["close"]
    ema5 = df["EMA5"]
    ema20 = df["EMA20"]
    vwap = df["VWAP"]
    atr14 = df["ATR"]
    atr50 = atr14.rolling(50).mean()

    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema20_slope = df["EMA20"].diff(3)

    # Volatility filter (must have enough data)
    vol_ok = atr14 > atr50 * 0.5

    # Setup A — LONG: Momentum Continuation
    long_signal = (
        (close > df["EMA20"]) &
        (df["EMA20"] > ema50) &
        (ema50 > ema200) &
        vol_ok &
        (ema20_slope > 0)
    )
    setup_a = long_signal

    # Setup B — Short Pullback
    trend_dn = ema5 < ema20
    below_vwap = close < vwap
    below_ema20 = close < ema20   # relaxed: more short events, used when --side short
    # EMA loss: close[i-1] > EMA20[i-1] AND close[i] < EMA20[i]
    lose = (close.shift(1) > ema20.shift(1)) & (close < ema20)
    setup_b_vwap = trend_dn & below_vwap & lose & vol_ok      # strict (VWAP)
    setup_b_ema20 = trend_dn & below_ema20 & lose & vol_ok    # relaxed (EMA20)

    # Diagnostic: per-condition pass rates
    n_total = int((~df["ATR"].isna()).sum())
    logger.info(f"  Condition breakdown (n={n_total:,} valid bars):")
    logger.info(f"    close>EMA20                 : {(close > df['EMA20']).sum():>8,}")
    logger.info(f"    EMA20>EMA50                 : {(df['EMA20'] > ema50).sum():>8,}")
    logger.info(f"    EMA50>EMA200                : {(ema50 > ema200).sum():>8,}")
    logger.info(f"    EMA20_slope>0               : {(ema20_slope > 0).sum():>8,}")
    logger.info(f"    vol_ok                      : {vol_ok.sum():>8,}")
    logger.info(f"    LONG momentum (all)         : {setup_a.sum():>8,}")
    logger.info(f"    trend_dn  (EMA5 < EMA20)    : {trend_dn.sum():>8,}  ({trend_dn.sum()/max(n_total,1)*100:.1f}%)")
    logger.info(f"    below_vwap (close < VWAP)   : {below_vwap.sum():>8,}  ({below_vwap.sum()/max(n_total,1)*100:.1f}%)")
    logger.info(f"    lose      (EMA cross down)   : {lose.sum():>8,}  ({lose.sum()/max(n_total,1)*100:.2f}%)")
    logger.info(f"    vol_ok    (ATR > 0.5*ATR50)  : {vol_ok.sum():>8,}  ({vol_ok.sum()/max(n_total,1)*100:.1f}%)")

    # Mutual exclusion
    both = setup_a & setup_b_ema20
    if both.any():
        logger.info(f"Skipping {both.sum()} ambiguous bars (both setups fire simultaneously)")
    setup_a = setup_a & ~both
    setup_b_ema20 = setup_b_ema20 & ~both
    setup_b_vwap = setup_b_vwap & ~both

    # Mark events (filter by --side in main())
    df["side"] = 0

    # Filter to event rows only
    drop_cols = ["TPV", "Cum_TPV", "Cum_Vol"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df, setup_a, setup_b_vwap, setup_b_ema20, len(df)


def main():
    ap = argparse.ArgumentParser(description="Trend Pullback Event Filter")
    ap.add_argument("--data", required=True, help="Raw OHLCV CSV")
    ap.add_argument("--out", default=None, help="Output CSV path (default: events_raw.csv)")
    ap.add_argument("--side", type=str, default="both", choices=["long", "short", "both"],
                    help="Which side to generate (long uses VWAP, short uses EMA20)")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df.columns = [c.lower().strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])

    logger.info(f"Loaded {len(df):,} rows | {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    logger.info("Computing indicators...")
    df = add_indicators(df)
    # Drop NaN rows from rolling windows
    df = df.dropna(subset=["ATR", "EMA5", "EMA20", "VWAP", "RSI"]).reset_index(drop=True)
    logger.info(f"After indicator dropna: {len(df):,} rows")

    df, setup_a, setup_b_vwap, setup_b_ema20, n_total = filter_events(df)

    if args.side == "long":
        df.loc[setup_a.fillna(False), "side"] = 1
        events = df[df["side"] != 0].copy()
        logger.info(f"Setup A (Long) events:     {len(events):>8,}")
    elif args.side == "short":
        df.loc[setup_b_ema20.fillna(False), "side"] = -1
        events = df[df["side"] != 0].copy()
        logger.info(f"Setup B (Short, EMA20) events: {len(events):>8,}")
    else:
        df.loc[setup_a.fillna(False), "side"] = 1
        df.loc[setup_b_ema20.fillna(False), "side"] = -1
        events = df[df["side"] != 0].copy()
        logger.info(f"Setup A (Long) events:     {setup_a.sum():>8,}")
        logger.info(f"Setup B (Short, EMA20) events: {setup_b_ema20.sum():>8,}")
        logger.info(f"Total events:              {len(events):>8,} out of {n_total:,} bars ({len(events)/max(n_total,1)*100:.2f}%)")

    if len(events) == 0:
        logger.error("No events generated. Check your data or relax filter conditions.")
        sys.exit(1)

    out_path = args.out or args.data.replace(".csv", "_events_raw.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    events[OUTPUT_COLUMNS].to_csv(out_path, index=False)
    logger.info(f"Saved {len(events):,} events -> {out_path}")
    logger.info(f"columns: {OUTPUT_COLUMNS}")


if __name__ == "__main__":
    main()
