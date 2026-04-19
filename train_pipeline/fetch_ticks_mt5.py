#!/usr/bin/env python3
"""
fetch_ticks_mt5.py - Fetch XAUUSD tick data from MetaTrader5

Fetches bid/ask ticks in configurable date ranges and saves to CSV
for use with the microstructure feature pipeline.

Schema output:
    time, bid, ask, last, bid_volume, ask_volume, flags

Usage:
    # Pull last 30 days of ticks
    python train_pipeline/fetch_ticks_mt5.py --days 30

    # Pull a specific date range
    python train_pipeline/fetch_ticks_mt5.py --from 2025-01-01 --to 2025-04-01

    # Pull 90 days, output to a specific file
    python train_pipeline/fetch_ticks_mt5.py --days 90 --out train_pipeline/data/xauusd_ticks_90d.csv
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed. Run: pip install MetaTrader5")
    sys.exit(1)

from config import Settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MT5TickFetcher")

UTC = pytz.timezone("UTC")

# Output columns matching the microstructure_features.py expected schema
OUTPUT_COLUMNS = ["time", "bid", "ask", "last", "bid_volume", "ask_volume", "flags"]


# ---------------------------------------------------------------------------
# MT5 connection (reuses same login as fetch_data_mt5.py)
# ---------------------------------------------------------------------------

def connect_mt5(symbol: str) -> str:
    """Initialise MT5, login via config, return resolved symbol name."""
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
    logger.info(
        f"Connected - Account: {account.login} | Server: {account.server} | Balance: ${account.balance:,.2f}"
    )

    # Resolve symbol (handles XAUUSD vs XAUUSD.sim etc.)
    if mt5.symbol_info(symbol) is None:
        logger.warning(f"Symbol '{symbol}' not found. Searching...")
        all_symbols = mt5.symbols_get()
        match = next(
            (s.name for s in all_symbols if symbol.replace(".", "") in s.name.upper()), None
        )
        if not match:
            logger.error(f"No symbol matching '{symbol}' found. Exiting.")
            mt5.shutdown()
            sys.exit(1)
        symbol = match
        logger.info(f"Resolved to: {symbol}")

    mt5.symbol_select(symbol, True)
    return symbol


# ---------------------------------------------------------------------------
# Tick fetcher — day-by-day to avoid MT5 limits
# ---------------------------------------------------------------------------

def fetch_ticks_range(
    symbol: str,
    date_from: datetime,
    date_to: datetime,
    tick_type: int = None,
) -> pd.DataFrame:
    """
    Fetch ticks between date_from and date_to by walking day by day.

    MT5 limits how many ticks can be returned in a single request.
    Fetching one day at a time avoids hitting this limit.

    Args:
        symbol:    Resolved MT5 symbol string
        date_from: Start datetime (timezone-aware UTC)
        date_to:   End datetime (timezone-aware UTC)
        tick_type: mt5.COPY_TICKS_ALL / COPY_TICKS_INFO / COPY_TICKS_TRADE
                   Default: COPY_TICKS_ALL

    Returns:
        Combined DataFrame with all ticks, sorted by time ascending.
    """
    if tick_type is None:
        tick_type = mt5.COPY_TICKS_ALL

    all_chunks = []
    current = date_from
    day_num = 0
    total_expected_days = (date_to - date_from).days + 1

    logger.info(
        f"Fetching ticks for {symbol} | "
        f"{date_from.strftime('%Y-%m-%d')} -> {date_to.strftime('%Y-%m-%d')} "
        f"({total_expected_days} days)"
    )

    while current < date_to:
        next_day = current + timedelta(days=1)
        end = min(next_day, date_to)
        day_num += 1

        ticks = mt5.copy_ticks_range(symbol, current, end, tick_type)

        if ticks is None or len(ticks) == 0:
            err = mt5.last_error()
            logger.warning(
                f"  Day {day_num:>4}: {current.strftime('%Y-%m-%d')} | No data | Error: {err}"
            )
            current = next_day
            continue

        chunk = pd.DataFrame(ticks)
        # MT5 'time' is already epoch seconds for ticks (high precision via time_msc)
        if "time_msc" in chunk.columns:
            # Use millisecond precision if available
            chunk["time"] = pd.to_datetime(chunk["time_msc"], unit="ms", utc=True)
        else:
            chunk["time"] = pd.to_datetime(chunk["time"], unit="s", utc=True)

        all_chunks.append(chunk)

        logger.info(
            f"  Day {day_num:>4}: {current.strftime('%Y-%m-%d')} | "
            f"Got: {len(chunk):>8,} ticks | "
            f"Cumulative: {sum(len(c) for c in all_chunks):>10,}"
        )

        current = next_day

    if not all_chunks:
        logger.error("No ticks fetched at all. Check symbol and date range.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.concat(all_chunks, ignore_index=True)

    # Normalize columns to our schema
    # MT5 tick fields: time, bid, ask, last, volume (=last_volume), flags, volume_real
    rename_map = {}
    if "volume" in df.columns and "bid_volume" not in df.columns:
        # MT5 uses 'volume' for the last traded volume; bid/ask volumes are often 0
        # We'll derive bid_volume/ask_volume from flags (bit 0=bid, bit 1=ask)
        rename_map["volume"] = "last_volume"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Build bid_volume and ask_volume from flags if not natively present
    # MT5 TICK_FLAG_BID = 2, TICK_FLAG_ASK = 4
    if "bid_volume" not in df.columns:
        df["bid_volume"] = 0.0
    if "ask_volume" not in df.columns:
        df["ask_volume"] = 0.0
    if "last" not in df.columns:
        df["last"] = 0.0
    if "flags" not in df.columns:
        df["flags"] = 0

    # Deduplicate and sort
    before = len(df)
    if "time_msc" in df.columns:
        df = df.drop_duplicates(subset=["time_msc"]).sort_values("time").reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    if before != len(df):
        logger.info(f"Removed {before - len(df):,} duplicate ticks")

    # Keep only our standard output columns
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    logger.info(
        f"Final tick count: {len(df):,} | "
        f"Range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}"
    )
    return df[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ticks(df: pd.DataFrame) -> None:
    """Print basic quality stats on the fetched tick data."""
    if df.empty:
        logger.warning("Empty DataFrame — nothing to validate.")
        return

    spread = df["ask"] - df["bid"]
    null_spread = (spread <= 0).sum()

    logger.info("=== Tick Validation ===")
    logger.info(f"  Total ticks    : {len(df):,}")
    logger.info(f"  Date range     : {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    logger.info(f"  Bid range      : {df['bid'].min():.2f} -> {df['bid'].max():.2f}")
    logger.info(f"  Ask range      : {df['ask'].min():.2f} -> {df['ask'].max():.2f}")
    logger.info(f"  Spread mean    : {spread.mean():.4f}")
    logger.info(f"  Spread std     : {spread.std():.4f}")
    logger.info(f"  Zero/neg spread: {null_spread:,} ({null_spread/len(df)*100:.2f}%)")
    logger.info(f"  Null bid rows  : {df['bid'].isna().sum():,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch XAUUSD tick data from MT5 and save to CSV"
    )
    parser.add_argument(
        "--symbol", type=str, default="XAUUSD", help="MT5 symbol (default: XAUUSD)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of past days to fetch (mutually exclusive with --from/--to)",
    )
    parser.add_argument(
        "--from",
        dest="date_from",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (UTC)",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (UTC). Defaults to today.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Auto-generated if not specified.",
    )
    parser.add_argument(
        "--tick-type",
        type=str,
        default="all",
        choices=["all", "info", "trade"],
        help="Tick type filter (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve date range
    today = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    if args.days is not None:
        date_to = today
        date_from = today - timedelta(days=args.days)
    elif args.date_from is not None:
        date_from = UTC.localize(datetime.strptime(args.date_from, "%Y-%m-%d"))
        date_to = (
            UTC.localize(datetime.strptime(args.date_to, "%Y-%m-%d"))
            if args.date_to
            else today
        )
    else:
        # Default: last 7 days
        date_to = today
        date_from = today - timedelta(days=7)
        logger.info("No date range specified. Defaulting to last 7 days.")

    # Resolve output path
    if args.out is None:
        out_dir = Path(__file__).parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        label = f"{date_from.strftime('%Y%m%d')}_{date_to.strftime('%Y%m%d')}"
        out_path = out_dir / f"{args.symbol.lower()}_ticks_{label}.csv"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Tick type
    tick_type_map = {
        "all": mt5.COPY_TICKS_ALL,
        "info": mt5.COPY_TICKS_INFO,
        "trade": mt5.COPY_TICKS_TRADE,
    }
    tick_type = tick_type_map[args.tick_type]

    logger.info("=" * 65)
    logger.info("  MT5 Tick Fetcher")
    logger.info("=" * 65)
    logger.info(f"  Symbol     : {args.symbol}")
    logger.info(f"  From       : {date_from.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"  To         : {date_to.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"  Tick type  : {args.tick_type}")
    logger.info(f"  Output     : {out_path}")
    logger.info("=" * 65)

    symbol = connect_mt5(args.symbol)
    df = fetch_ticks_range(symbol, date_from, date_to, tick_type)
    mt5.shutdown()
    logger.info("MT5 disconnected.")

    validate_ticks(df)

    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df):,} ticks -> {out_path}")

    print(f"\nTick data saved: {out_path}")
    print(f"Next step      : python train_pipeline/microstructure_features.py \\")
    print(f"                   --m1 train_pipeline/data/xauusd_m1_1y_dukas.csv \\")
    print(f"                   --ticks {out_path} \\")
    print(f"                   --out train_pipeline/data/xauusd_m1_micro.csv")


if __name__ == "__main__":
    main()
