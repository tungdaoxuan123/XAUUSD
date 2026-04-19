#!/usr/bin/env python3
"""
fetch_data_mt5.py - Fetch up to 2,000,000 bars from MT5 in 100,000-bar batches.

MetaTrader5's copy_rates_from_pos() is capped per request (usually 100,000).
This script fetches data iteratively by walking start_pos backwards from the
most recent bar, then deduplicates and sorts ascending before saving.

NOTE: The actual number of bars available depends entirely on the broker's
"Max bars in chart" setting in MetaTrader5 > Tools > Options > Charts.
For XAUUSD M1 on FTMO Demo this is typically ~100,000 bars (~70 days).
Even if you request 2,000,000, you will receive however many are stored.

Usage:
    python train_pipeline/fetch_data_mt5.py
    python train_pipeline/fetch_data_mt5.py --total-bars 2000000 --batch-size 100000
    python train_pipeline/fetch_data_mt5.py --symbol XAUUSD --timeframe M1 --out data/big.csv
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
from config import Settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MT5Fetcher")

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

OUTPUT_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


# ---------------------------------------------------------------------------
# MT5 connection
# ---------------------------------------------------------------------------

def connect_mt5(symbol: str) -> str:
    """Initialise MT5, login via config, and return the resolved symbol name."""
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
        mt5.shutdown(); sys.exit(1)

    account = mt5.account_info()
    logger.info(f"Connected - Account: {account.login} | Server: {account.server} | Balance: ${account.balance:,.2f}")

    # Resolve symbol (handles XAUUSD vs XAUUSD.sim)
    if mt5.symbol_info(symbol) is None:
        logger.warning(f"Symbol '{symbol}' not found. Searching...")
        symbols = mt5.symbols_get()
        match = next((s.name for s in symbols if symbol.replace(".", "") in s.name.upper()), None)
        if not match:
            logger.error(f"No symbol matching '{symbol}' found. Exiting.")
            mt5.shutdown(); sys.exit(1)
        symbol = match
        logger.info(f"Resolved to: {symbol}")

    mt5.symbol_select(symbol, True)
    return symbol


# ---------------------------------------------------------------------------
# Batched fetcher
# ---------------------------------------------------------------------------

def fetch_all_rates(
    symbol: str,
    timeframe_str: str,
    total_bars: int,
    batch_size: int,
) -> pd.DataFrame:
    """
    Two-phase fetch strategy:
      Phase 1: copy_rates_from_pos(pos=0, count=batch_size)
               — gets the most recent batch, gives us the oldest timestamp.
      Phase 2: copy_rates_range(start, end) walking backwards from oldest timestamp.
               — extends history as far back as the broker allows.

    Why two-phase:
      - copy_rates_from_pos with large start_pos fails on FTMO Demo.
      - copy_rates_range with future end_date also fails with "Invalid params".
      - But copy_rates_range with past timestamps (from phase 1) works correctly.
    """
    import datetime as dt

    tf = TIMEFRAME_MAP.get(timeframe_str.upper())
    if tf is None:
        logger.error(f"Unknown timeframe: {timeframe_str}. Options: {list(TIMEFRAME_MAP.keys())}")
        sys.exit(1)

    tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
    mins_per_bar = tf_minutes.get(timeframe_str.upper(), 1)

    all_frames = []
    cumulative = 0

    logger.info(f"Target: {total_bars:,} bars | Batch: {batch_size:,} | Strategy: pos=0 then date-range walk-back")

    # === Phase 1: fetch most recent batch via pos=0 ===
    logger.info("Phase 1: Fetching most recent bars via copy_rates_from_pos(pos=0)...")
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, min(batch_size, total_bars))

    if rates is None or len(rates) == 0:
        logger.error(f"Phase 1 failed: {mt5.last_error()}. Cannot proceed.")
        mt5.shutdown(); sys.exit(1)

    chunk = pd.DataFrame(rates)
    chunk["time"] = pd.to_datetime(chunk["time"], unit="s")
    all_frames.append(chunk)
    cumulative = len(chunk)
    oldest_ts = chunk["time"].min()

    logger.info(
        f"  Phase 1 done | Got: {len(chunk):,} | "
        f"Range: {chunk['time'].min()} -> {chunk['time'].max()}"
    )

    if cumulative >= total_bars:
        logger.info("Target reached in phase 1.")
    else:
        # === Phase 2: walk backwards using copy_rates_range ===
        logger.info("Phase 2: Walking backwards using copy_rates_range...")
        consecutive_empty = 0
        batch_num = 1

        while cumulative < total_bars:
            end_date = oldest_ts - dt.timedelta(minutes=1)  # 1 bar before oldest we have
            start_date = end_date - dt.timedelta(minutes=mins_per_bar * batch_size)

            if end_date.year < 2020:
                logger.info("Reached year 2020 — stopping walk-back.")
                break

            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)

            if rates is None or len(rates) == 0:
                err = mt5.last_error()
                logger.warning(
                    f"  Batch {batch_num}: No data for "
                    f"{start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')} | "
                    f"Error: {err}"
                )
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    logger.info("3 consecutive empty batches. Broker history exhausted.")
                    break
                oldest_ts = start_date
                batch_num += 1
                continue

            consecutive_empty = 0
            chunk = pd.DataFrame(rates)
            chunk["time"] = pd.to_datetime(chunk["time"], unit="s")
            all_frames.append(chunk)
            cumulative += len(chunk)
            oldest_ts = chunk["time"].min()

            logger.info(
                f"  Batch {batch_num:>3} | Got: {len(chunk):>7,} | "
                f"Cumulative: {cumulative:>8,} | "
                f"Range: {chunk['time'].min()} -> {chunk['time'].max()}"
            )

            batch_num += 1
            if len(rates) < batch_size * 0.3:
                logger.warning("Batch much smaller than expected. History likely exhausted.")
                break

    if not all_frames:
        logger.error("No data fetched.")
        mt5.shutdown(); sys.exit(1)

    logger.info("Combining and deduplicating batches...")
    df = pd.concat(all_frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    if before != len(df):
        logger.info(f"Removed {before - len(df):,} duplicate rows")

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[OUTPUT_COLUMNS]

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_output(df: pd.DataFrame, total_bars: int):
    """Warn if final row count is far below the requested total."""
    ratio = len(df) / total_bars
    logger.info(f"Final row count: {len(df):,} / {total_bars:,} requested ({ratio:.1%})")
    logger.info(f"Date range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    if ratio < 0.05:
        logger.warning(
            "WARN: Fetched less than 5% of target bars. "
            "Increase MT5 'Max bars in chart' in Tools > Options > Charts."
        )
    elif ratio < 0.5:
        logger.warning(f"WARN: Only {ratio:.0%} of target bars available. Broker history may be limited.")
    else:
        logger.info("Row count looks good.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch large MT5 OHLCV datasets in 100k-bar batches"
    )
    parser.add_argument("--symbol",      type=str, default="XAUUSD",   help="Symbol (default: XAUUSD)")
    parser.add_argument("--timeframe",   type=str, default="M1",       choices=list(TIMEFRAME_MAP.keys()))
    parser.add_argument("--total-bars",  type=int, default=2_000_000,  help="Total bars to try fetching (default: 2000000)")
    parser.add_argument("--batch-size",  type=int, default=50_000,    help="Per-request batch size (default: 50000 — FTMO Demo rejects >50k)")
    parser.add_argument("--out",         type=str, default=None,       help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.out is None:
        out_dir = Path(__file__).parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = out_dir / f"{args.symbol.lower()}_{args.timeframe.lower()}_{ts}_bulk.csv"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  MT5 Bulk Fetcher")
    logger.info("=" * 65)
    logger.info(f"  Symbol      : {args.symbol}")
    logger.info(f"  Timeframe   : {args.timeframe}")
    logger.info(f"  Target bars : {args.total_bars:,}")
    logger.info(f"  Batch size  : {args.batch_size:,}")
    logger.info(f"  Output      : {out_path}")
    logger.info("=" * 65)

    symbol = connect_mt5(args.symbol)
    df = fetch_all_rates(symbol, args.timeframe, args.total_bars, args.batch_size)
    validate_output(df, args.total_bars)

    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df):,} rows -> {out_path}")
    mt5.shutdown()
    logger.info("MT5 disconnected.")

    print(f"\nData saved: {out_path}")
    print(f"Next step : python train_pipeline/train_ensemble_gpu.py --data \"{out_path}\" --use-gpu --gpu-backend auto")


if __name__ == "__main__":
    main()
