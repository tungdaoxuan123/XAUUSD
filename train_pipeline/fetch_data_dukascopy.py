#!/usr/bin/env python3
"""
fetch_data_dukascopy.py - Bulk XAUUSD M1 data fetcher from Dukascopy.

Downloads LZMA-compressed binary (.bi5) M1 candle files, decompresses,
and parses them into a CSV format compatible with the MT5 training pipeline.

This allows getting years of history (2015-2026) instead of the 100k-bar
limit imposed by most MT5 MetaTrader brokers.

Usage:
    python train_pipeline/fetch_data_dukascopy.py --from 2015-01-01 --to 2026-04-10
"""

import argparse
import datetime as dt
import logging
import lzma
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DukasFetch")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DUKAS_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Row format: Offset(I), Open(I), Close(I), Low(I), High(I), Volume(f)
STRUCT_FMT = ">5If"
ROW_SIZE = struct.calcsize(STRUCT_FMT)

# CSV Columns for MT5 training pipeline compatibility
OUTPUT_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]

# XAUUSD scale: Dukascopy uses 1000 multiplier (3 decimals)
SYMBOL_DIVISORS = {
    "XAUUSD": 1000.0,
}


# ---------------------------------------------------------------------------
# Core Fetcher
# ---------------------------------------------------------------------------

class DukascopyFetcher:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol.upper()
        self.start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        self.end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        self.divisor = SYMBOL_DIVISORS.get(self.symbol, 100000.0)
        self.dukas_symbol = self.symbol
        
    def fetch_day(self, current_dt: dt.datetime) -> pd.DataFrame:
        """Fetch and parse one day of M1 bars with retries."""
        url = DUKAS_URL.format(
            symbol=self.dukas_symbol,
            year=current_dt.year,
            month=current_dt.month - 1,
            day=current_dt.day
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                if resp.status_code == 404:
                    return None  # Weekend/Holidays
                resp.raise_for_status()
                
                # Decompress LZMA
                data = lzma.decompress(resp.content)
                
                rows = []
                day_start = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                
                for i in range(0, len(data), ROW_SIZE):
                    chunk = data[i:i+ROW_SIZE]
                    if len(chunk) < ROW_SIZE:
                        continue
                    
                    # Unpack: Offset(s), Open, Close, Low, High, Volume
                    offset, o, c, l, h, vol = struct.unpack(STRUCT_FMT, chunk)
                    
                    # Convert offset (seconds) to absolute time
                    bar_time = day_start + dt.timedelta(seconds=offset)
                    
                    rows.append({
                        "time": bar_time,
                        "open": o / self.divisor,
                        "high": h / self.divisor,
                        "low": l / self.divisor,
                        "close": c / self.divisor,
                        "tick_volume": vol,
                        "spread": 0,
                        "real_volume": 0
                    })
                
                return pd.DataFrame(rows)
                
            except Exception as e:
                wait_sec = (attempt + 1) * 2
                if attempt < max_retries - 1:
                    logger.debug(f"Retrying {url} in {wait_sec}s... Error: {e}")
                    time.sleep(wait_sec)
                else:
                    logger.error(f"Final failure for {url}: {e}")
        
        return None

    def download_range(self) -> pd.DataFrame:
        """Iterate through the date range day-by-day."""
        current_dt = self.start_dt
        all_frames = []
        cumulative = 0
        
        total_days = (self.end_dt - self.start_dt).days + 1
        processed_days = 0
        
        logger.info(f"Starting download for {self.symbol} | {self.start_dt.date()} -> {self.end_dt.date()}")

        while current_dt <= self.end_dt:
            day_df = self.fetch_day(current_dt)
            if day_df is not None:
                all_frames.append(day_df)
                cumulative += len(day_df)
            
            processed_days += 1
            if processed_days % 10 == 0 or processed_days == total_days:
                percent = (processed_days / total_days) * 100
                logger.info(f"  Progress: {percent:6.1f}% | Day: {current_dt.date()} | Total Rows: {cumulative:,}")

            current_dt += dt.timedelta(days=1)
            
            # Simple rate limiting
            time.sleep(0.1)

        if not all_frames:
            return pd.DataFrame()

        logger.info("Finalizing dataset (deduplicating and sorting)...")
        full_df = pd.concat(all_frames, ignore_index=True)
        full_df = full_df.drop_duplicates(subset=["time"])
        full_df = full_df.sort_values("time").reset_index(drop=True)
        
        # Ensure correct column order
        full_df = full_df[OUTPUT_COLUMNS]
        
        return full_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Dukascopy Historical XAUUSD Fetcher")
    parser.add_argument("--symbol",    type=str, default="XAUUSD", help="Symbol (default: XAUUSD)")
    parser.add_argument("--from",      dest="start_date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to",        dest="end_date",   required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--out",       type=str, default=None, help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.out is None:
        out_dir = Path(__file__).parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.symbol.lower()}_m1_dukascopy.csv"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fetcher = DukascopyFetcher(args.symbol, args.start_date, args.end_date)
    
    start_time = time.time()
    df = fetcher.download_range()
    duration = time.time() - start_time

    if df.empty:
        logger.error("No data fetched. Check your date range or internet connection.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"  Download Complete")
    logger.info("=" * 60)
    logger.info(f"  Symbol      : {args.symbol}")
    logger.info(f"  Total Rows  : {len(df):,}")
    logger.info(f"  Date Range  : {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    logger.info(f"  Duration    : {duration/60:.1f} minutes")
    logger.info(f"  Output      : {out_path}")
    logger.info("=" * 60)

    # Save to CSV
    df.to_csv(out_path, index=False)
    logger.info(f"File saved successfully.")
    
    print(f"\nNext Steps:")
    print(f"1. python train_pipeline/train_ensemble_gpu.py --data \"{out_path}\" --use-gpu")

if __name__ == "__main__":
    main()
