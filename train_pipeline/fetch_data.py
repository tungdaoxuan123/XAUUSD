#!/usr/bin/env python3
"""
fetch_data.py - Export historical XAUUSD candle data from MT5 to CSV

This script connects to your MT5 account (using config.py credentials),
fetches historical OHLCV bars for a given timeframe and bar count,
and saves them to train_pipeline/data/ ready for train_ensemble.py.

Usage:
    python train_pipeline/fetch_data.py
    python train_pipeline/fetch_data.py --bars 50000 --timeframe M1
    python train_pipeline/fetch_data.py --bars 10000 --timeframe M5 --out data/xauusd_m5.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
from config import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DataFetcher")

# ---------------------------------------------------------------------------
# Timeframe map
# ---------------------------------------------------------------------------
TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


# ---------------------------------------------------------------------------
# MT5 connection
# ---------------------------------------------------------------------------

def connect_mt5() -> str:
    """Initialise MT5 and return the resolved XAUUSD symbol name."""
    if not mt5.initialize():
        logger.error(f"mt5.initialize() failed: {mt5.last_error()}")
        sys.exit(1)

    login_ok = mt5.login(
        login=Settings.MT5_LOGIN,
        password=Settings.MT5_PASSWORD,
        server=Settings.MT5_SERVER,
    )
    if not login_ok:
        logger.error(f"mt5.login() failed: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    account = mt5.account_info()
    logger.info(f"Connected - Account: {account.login} | Server: {account.server} | Balance: ${account.balance:,.2f}")

    # Resolve symbol
    symbol = Settings.SYMBOL
    if mt5.symbol_info(symbol) is None:
        logger.warning(f"Symbol '{symbol}' not found. Searching...")
        symbols = mt5.symbols_get()
        match = next((s.name for s in symbols if "XAUUSD" in s.name.upper()), None)
        if not match:
            logger.error("No XAUUSD symbol found. Exiting.")
            mt5.shutdown()
            sys.exit(1)
        symbol = match
        logger.info(f"Using symbol: {symbol}")

    mt5.symbol_select(symbol, True)
    return symbol


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_rates(symbol: str, timeframe_str: str, bars: int) -> pd.DataFrame:
    """
    Fetch `bars` candles from MT5 using batched date-range requests.

    MT5 has a hard per-request limit (~99,999 bars). We bypass this by
    iterating backwards in time in chunks, then stitching the results.
    """
    tf = TIMEFRAME_MAP.get(timeframe_str.upper())
    if tf is None:
        logger.error(f"Unknown timeframe: {timeframe_str}. Choose from: {list(TIMEFRAME_MAP.keys())}")
        sys.exit(1)

    CHUNK_SIZE = 50_000  # well under MT5's per-request limit
    all_dfs = []
    total_fetched = 0
    
    # Minutes per bar (for stepping back in time)
    tf_minutes = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440
    }
    mins_per_bar = tf_minutes.get(timeframe_str.upper(), 1)

    logger.info(f"Fetching {bars:,} bars of {symbol} {timeframe_str} in chunks of {CHUNK_SIZE:,}...")

    # Start from current time, walk backwards
    end_date = datetime.utcnow()

    while total_fetched < bars:
        chunk_bars = min(CHUNK_SIZE, bars - total_fetched)
        start_date = end_date - __import__('datetime').timedelta(minutes=mins_per_bar * chunk_bars)

        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)

        if rates is None or len(rates) == 0:
            logger.warning(f"No data returned for range {start_date} -> {end_date}. Stopping early.")
            break

        chunk_df = pd.DataFrame(rates)
        chunk_df["time"] = pd.to_datetime(chunk_df["time"], unit="s")
        all_dfs.append(chunk_df)
        total_fetched += len(rates)

        # Next chunk ends where this one started (move window back)
        end_date = start_date

        logger.info(f"  Fetched {total_fetched:,} / {bars:,} bars...")

        # If the chunk returned fewer bars than requested, we've hit the broker's history limit
        if len(rates) < chunk_bars * 0.5:
            logger.warning("Fewer bars than expected — likely at broker history limit. Stopping.")
            break

    if not all_dfs:
        logger.error("No data fetched at all. Exiting.")
        mt5.shutdown()
        sys.exit(1)

    # Combine, deduplicate, sort
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df = df[["time", "open", "high", "low", "close", "tick_volume"]].copy()

    logger.info(
        f"Total fetched: {len(df):,} bars | "
        f"From: {df['time'].iloc[0]} | "
        f"To:   {df['time'].iloc[-1]}"
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch MT5 XAUUSD data for the training pipeline")
    parser.add_argument(
        "--bars", type=int, default=50000,
        help="Number of historical bars to fetch (default: 50000 ≈ ~35 days of M1)"
    )
    parser.add_argument(
        "--timeframe", type=str, default="M1",
        choices=list(TIMEFRAME_MAP.keys()),
        help="Candle timeframe (default: M1)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output CSV path (default: train_pipeline/data/xauusd_<TF>_<timestamp>.csv)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Default output path
    if args.out is None:
        out_dir = Path(__file__).parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = out_dir / f"xauusd_{args.timeframe.lower()}_{ts}.csv"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("  MT5 Data Fetcher - XAUUSD Training Pipeline")
    logger.info("=" * 55)
    logger.info(f"  Bars       : {args.bars:,}")
    logger.info(f"  Timeframe  : {args.timeframe}")
    logger.info(f"  Output     : {out_path}")
    logger.info("=" * 55)

    # Connect and fetch
    symbol = connect_mt5()
    df = fetch_rates(symbol, args.timeframe, args.bars)

    # Save
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df):,} rows to: {out_path}")

    # Quick summary
    logger.info(f"\nColumn summary:\n{df.dtypes.to_string()}")
    logger.info(f"\nSample (last 3 rows):\n{df.tail(3).to_string(index=False)}")

    mt5.shutdown()
    logger.info("MT5 disconnected. Done.")

    print(f"\nData saved to: {out_path}")
    print(f"   Next step: python train_pipeline/train_ensemble.py --data \"{out_path}\" --model random_forest")


if __name__ == "__main__":
    main()
