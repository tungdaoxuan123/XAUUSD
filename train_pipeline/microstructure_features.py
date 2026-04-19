#!/usr/bin/env python3
"""
microstructure_features.py
--------------------------
Enriches M1 OHLCV bars with microstructure features derived from real tick data.

Two feature families:
  1. Volume Profile (rolling N-bar window)
       vprof_poc_dist       – (close - POC_price) / ATR
       vprof_in_value_area  – 1 if close is inside the 70% value area
       vprof_hvn_flag       – 1 if current price bin is a high-volume node
       vprof_lvn_flag       – 1 if current price bin is a low-volume node

  2. Order-Flow Features (per M1 bar, from its ticks)
       tick_imbalance        – (up_ticks - down_ticks) / total
       bid_ask_vol_imbalance – derived from flag-based direction (0 for FTMO)
       spread_mean           – mean bid-ask spread in the bar
       spread_std            – std dev of bid-ask spread in the bar
       ofi_window            – normalized price-action OFI (bid/ask change direction)
       of_pressure_flag      – -1 / 0 / +1 summary

Usage:
    python train_pipeline/microstructure_features.py \
        --m1   train_pipeline/data/xauusd_m1_1y_dukas.csv \
        --ticks train_pipeline/data/xauusd_ticks_20260102_20260412.csv \
        --out   train_pipeline/data/xauusd_m1_micro.csv \
        --vp-window 240 \
        --bin-size 0.10

Notes:
  - bid_volume / ask_volume are 0 in FTMO tick data (CFD broker).
    OFI is derived from bid and ask price movements instead (standard method).
  - Bars outside the tick date range get NaN for all micro features.
  - This script does NOT add RSI/MACD. Run train_ensemble_gpu.py afterward.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Microstructure")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICK_IMBALANCE_STRONG = 0.30   # |tick_imbalance| > this  => strong pressure
VALUE_AREA_FRACTION   = 0.70   # 70% of volume defines the value area
HVN_PERCENTILE        = 80     # top X% of volume nodes = HVN
LVN_PERCENTILE        = 20     # bottom X% of volume nodes = LVN


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_m1(path: str) -> pd.DataFrame:
    logger.info(f"Loading M1 bars from {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    # Floor to the minute to ensure exact alignment with tick grouping
    df["bar_time"] = df["time"].dt.floor("min")
    logger.info(f"  M1 bars: {len(df):,} | {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    return df


def load_ticks(path: str) -> pd.DataFrame:
    """Load tick CSV, keeping only the columns we need. Uses efficient dtypes."""
    logger.info(f"Loading ticks from {path} (large file — please wait)...")
    dtype = {
        "bid": "float32",
        "ask": "float32",
        "last": "float32",
        "bid_volume": "float32",
        "ask_volume": "float32",
        "flags": "int32",
    }
    df = pd.read_csv(path, dtype=dtype)
    # Tick CSV has mixed timestamp formats (some have .%f, some don't)
    # Use format='mixed' with utc=True to handle both gracefully
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    # Snap each tick to its M1 bar
    df["bar_time"] = df["time"].dt.floor("min")
    df["mid"] = ((df["bid"] + df["ask"]) / 2).astype("float32")
    logger.info(f"  Ticks:   {len(df):,} | {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    return df


# ---------------------------------------------------------------------------
# Order-Flow Features (per M1 bar from its own ticks)
# ---------------------------------------------------------------------------

def compute_orderflow_per_bar(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Group ticks by bar_time and compute order-flow metrics per bar.

    FTMO ticks don't have real bid_volume/ask_volume so we derive direction
    from the bid and ask price changes (standard price-based OFI).

    Returns DataFrame indexed by bar_time with columns:
        tick_imbalance, bid_ask_vol_imbalance, spread_mean, spread_std,
        ofi_window, tick_count
    """
    logger.info("Computing per-bar order-flow features...")

    # Work with the relevant columns only
    t = ticks[["bar_time", "bid", "ask", "mid", "bid_volume", "ask_volume"]].copy()

    # Direction from mid-price change within each group
    t["prev_mid"] = t.groupby("bar_time")["mid"].shift(1)
    t["up_tick"]   = (t["mid"] > t["prev_mid"]).astype("float32")
    t["down_tick"] = (t["mid"] < t["prev_mid"]).astype("float32")

    # Bid/ask change for OFI (price-based proxy)
    # OFI_t = sign(Δbid) * 1  if Δbid > 0 (aggressor bidding up)
    #        -sign(Δask) * 1  if Δask < 0 (aggressor lifting ask down)
    t["prev_bid"] = t.groupby("bar_time")["bid"].shift(1)
    t["prev_ask"] = t.groupby("bar_time")["ask"].shift(1)
    t["delta_bid"] = t["bid"] - t["prev_bid"]
    t["delta_ask"] = t["ask"] - t["prev_ask"]
    # OFI = delta_bid contribution + delta_ask contribution
    t["ofi_tick"] = np.where(
        t["delta_bid"] > 0, t["delta_bid"],
        np.where(t["delta_bid"] < 0, t["delta_bid"], 0.0)
    ) - np.where(
        t["delta_ask"] > 0, t["delta_ask"],
        np.where(t["delta_ask"] < 0, t["delta_ask"], 0.0)
    )

    t["spread"] = t["ask"] - t["bid"]

    # Aggregate per bar
    grp = t.groupby("bar_time")
    total_ticks = grp["up_tick"].count().rename("tick_count")
    up_sum   = grp["up_tick"].sum()
    down_sum = grp["down_tick"].sum()

    of = pd.DataFrame({
        "tick_count":           total_ticks,
        "tick_imbalance":       (up_sum - down_sum) / (total_ticks + 1e-9),
        "spread_mean":          grp["spread"].mean(),
        "spread_std":           grp["spread"].std().fillna(0),
        "ofi_window":           grp["ofi_tick"].sum(),
        # bid_volume/ask_volume are 0 from FTMO — compute direction-based proxy
        "bid_ask_vol_imbalance": (up_sum - down_sum) / (up_sum + down_sum + 1e-9),
    })

    # Normalize OFI by tick count to make it comparable across bars
    of["ofi_window"] = of["ofi_window"] / (of["tick_count"] + 1e-9)

    # Overall pressure flag: -1 sell | 0 neutral | +1 buy
    of["of_pressure_flag"] = np.select(
        [
            of["tick_imbalance"] >  TICK_IMBALANCE_STRONG,
            of["tick_imbalance"] < -TICK_IMBALANCE_STRONG,
        ],
        [1, -1],
        default=0,
    ).astype("int8")

    logger.info(f"  Order-flow computed for {len(of):,} bars")
    return of


# ---------------------------------------------------------------------------
# Volume Profile Features (rolling N-bar window from tick bin counts)
# ---------------------------------------------------------------------------

def compute_volume_profile(
    ticks: pd.DataFrame,
    m1_bars: pd.DataFrame,
    vp_window: int,
    bin_size: float,
) -> pd.DataFrame:
    """
    Build rolling volume-profile features for each M1 bar using a
    memory-efficient sliding-window approach.

    Instead of materializing a huge pivot matrix, we maintain a rolling
    deque of per-bar {price_bin: tick_count} dicts and build the profile
    by summing across the window. This uses kilobytes instead of gigabytes.

    Returns DataFrame indexed by bar_time.
    """
    logger.info(f"Computing volume profile | window={vp_window} bars | bin_size={bin_size}")

    # --- Step 1: aggregate ticks to (bar_time, price_bin) tick counts ---
    tck = ticks[["bar_time", "mid"]].copy()
    tck["price_bin"] = (tck["mid"] / bin_size).round() * bin_size

    # dict: bar_time -> {price_bin -> count}
    logger.info("  Building per-bar bin count map...")
    bar_bin_counts = (
        tck.groupby(["bar_time", "price_bin"])
        .size()
        .reset_index(name="cnt")
    )

    # Convert to nested dict for fast lookup: bar_time -> {bin: cnt}
    per_bar: dict = {}
    for bar_t, group in bar_bin_counts.groupby("bar_time"):
        per_bar[bar_t] = dict(zip(group["price_bin"], group["cnt"]))

    # --- Step 2: prepare sorted list of M1 bar times in the tick range ---
    bar_times_sorted = sorted(per_bar.keys())

    # Close price map and ATR map
    close_map = m1_bars.set_index("bar_time")["close"]
    if "ATR" in m1_bars.columns:
        atr_map = m1_bars.set_index("bar_time")["ATR"]
    else:
        hl_series = pd.Series(
            (m1_bars["high"] - m1_bars["low"]).values,
            index=m1_bars["bar_time"]
        )
        atr_map = hl_series.rolling(14, min_periods=1).mean()

    # --- Step 3: sliding window over bar_times ---
    from collections import defaultdict, deque

    logger.info(f"  Sliding window over {len(bar_times_sorted):,} active bars...")

    window_q: deque = deque()                 # (bar_time, {bin: cnt}) pairs in window
    rolling_vol: defaultdict = defaultdict(float)  # running sum across window

    results = {}

    for bar_t in bar_times_sorted:
        # Add current bar to window
        cur_bins = per_bar.get(bar_t, {})
        window_q.append((bar_t, cur_bins))
        for b, c in cur_bins.items():
            rolling_vol[b] += c

        # Evict oldest bar if window exceeds size
        while len(window_q) > vp_window:
            old_t, old_bins = window_q.popleft()
            for b, c in old_bins.items():
                rolling_vol[b] -= c
                if rolling_vol[b] <= 0:
                    del rolling_vol[b]

        if not rolling_vol:
            continue

        # Compute features from rolling_vol dict
        total = sum(rolling_vol.values())
        if total == 0:
            continue

        poc_bin = max(rolling_vol, key=rolling_vol.get)
        poc_vol = rolling_vol[poc_bin]

        # Value Area: expand from POC until 70% of volume is included
        sorted_bins = sorted(rolling_vol.keys())
        poc_i = sorted_bins.index(poc_bin)

        accumulated = poc_vol
        lo_i = poc_i
        hi_i = poc_i
        target = total * VALUE_AREA_FRACTION

        while accumulated < target:
            can_lo = lo_i > 0
            can_hi = hi_i < len(sorted_bins) - 1
            if not can_lo and not can_hi:
                break
            add_lo = rolling_vol.get(sorted_bins[lo_i - 1], 0) if can_lo else -1
            add_hi = rolling_vol.get(sorted_bins[hi_i + 1], 0) if can_hi else -1
            if add_hi >= add_lo:
                hi_i += 1
                accumulated += add_hi
            else:
                lo_i -= 1
                accumulated += add_lo

        va_high = sorted_bins[hi_i]
        va_low  = sorted_bins[lo_i]

        # Current close and its bin
        cur_close = close_map.get(bar_t, np.nan)
        if np.isnan(cur_close):
            continue

        cur_bin = round(cur_close / bin_size) * bin_size
        cur_bin_vol = rolling_vol.get(cur_bin, 0)

        # HVN/LVN
        non_zero_vols = [v for v in rolling_vol.values() if v > 0]
        if non_zero_vols:
            hvn_thresh = np.percentile(non_zero_vols, HVN_PERCENTILE)
            lvn_thresh = np.percentile(non_zero_vols, LVN_PERCENTILE)
            hvn = int(cur_bin_vol >= hvn_thresh)
            lvn = int(0 < cur_bin_vol <= lvn_thresh)
        else:
            hvn = lvn = 0

        # POC distance
        atr_val = atr_map.get(bar_t, np.nan)
        poc_dist = (cur_close - poc_bin) / atr_val if (not np.isnan(atr_val) and atr_val > 0) else 0.0

        results[bar_t] = {
            "vprof_poc_price":     poc_bin,
            "vprof_poc_dist":      poc_dist,
            "vprof_va_high":       va_high,
            "vprof_va_low":        va_low,
            "vprof_in_value_area": int(va_low <= cur_close <= va_high),
            "vprof_hvn_flag":      hvn,
            "vprof_lvn_flag":      lvn,
        }

    result_df = pd.DataFrame.from_dict(results, orient="index")
    result_df.index.name = "bar_time"
    logger.info(f"  Volume profile done for {len(result_df):,} bars")
    return result_df


# ---------------------------------------------------------------------------
# Main Assembly
# ---------------------------------------------------------------------------

def build_microstructure_features(
    m1_path: str,
    tick_path: str,
    out_path: str,
    vp_window: int = 240,
    bin_size: float = 0.10,
):
    """
    Load M1 and tick data, compute all microstructure features, merge and save.
    """
    m1 = load_m1(m1_path)
    ticks = load_ticks(tick_path)

    # Find overlap between tick data and M1 bars
    tick_start = ticks["bar_time"].min()
    tick_end   = ticks["bar_time"].max()
    logger.info(f"Tick coverage: {tick_start} -> {tick_end}")

    m1_in_range = m1[
        (m1["bar_time"] >= tick_start) &
        (m1["bar_time"] <= tick_end)
    ].copy()
    logger.info(f"M1 bars in tick range: {len(m1_in_range):,} / {len(m1):,} total")

    if len(m1_in_range) == 0:
        logger.error("No M1 bars overlap with tick data! Check date ranges.")
        sys.exit(1)

    # Only use ticks that fall within an M1 bar we have
    ticks_filtered = ticks[
        (ticks["bar_time"] >= m1_in_range["bar_time"].min()) &
        (ticks["bar_time"] <= m1_in_range["bar_time"].max())
    ]
    logger.info(f"Using {len(ticks_filtered):,} ticks (inside M1 range)")

    # 1. Compute order-flow per bar
    of_features = compute_orderflow_per_bar(ticks_filtered)

    # 2. Compute volume profile rolling features
    vp_features = compute_volume_profile(
        ticks_filtered, m1_in_range, vp_window, bin_size
    )

    # 3. Merge all features back onto full M1 dataset
    logger.info("Merging microstructure features onto M1 bars...")
    m1_result = m1.copy()
    m1_result = m1_result.set_index("bar_time")

    # Order-flow merge
    of_cols = [
        "tick_count", "tick_imbalance", "bid_ask_vol_imbalance",
        "spread_mean", "spread_std", "ofi_window", "of_pressure_flag"
    ]
    m1_result = m1_result.join(of_features[of_cols], how="left")

    # Volume profile merge
    vp_cols = [
        "vprof_poc_dist", "vprof_in_value_area",
        "vprof_hvn_flag", "vprof_lvn_flag",
        "vprof_poc_price", "vprof_va_high", "vprof_va_low"
    ]
    m1_result = m1_result.join(vp_features[vp_cols], how="left")

    m1_result = m1_result.reset_index(drop=False)
    m1_result = m1_result.rename(columns={"bar_time": "time"})

    # Report coverage
    n_total = len(m1_result)
    n_micro = m1_result["tick_imbalance"].notna().sum()
    logger.info(f"Coverage: {n_micro:,} / {n_total:,} bars have microstructure data ({n_micro/n_total*100:.1f}%)")
    logger.info(f"Null counts:\n{m1_result[of_cols + vp_cols].isnull().sum().to_string()}")

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    m1_result.to_csv(out_path, index=False)
    logger.info(f"Saved enriched M1 -> {out_path} | {len(m1_result):,} rows")

    return m1_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build microstructure features from tick data and attach to M1 bars"
    )
    parser.add_argument("--m1",        required=True,  help="M1 OHLCV CSV path")
    parser.add_argument("--ticks",     required=True,  help="Tick CSV path (from fetch_ticks_mt5.py)")
    parser.add_argument("--out",       required=True,  help="Output enriched CSV path")
    parser.add_argument("--vp-window", type=int,   default=240,   help="Volume profile rolling window in bars (default: 240 = 4h)")
    parser.add_argument("--bin-size",  type=float, default=0.10,  help="Price bin size in USD (default: 0.10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_microstructure_features(
        m1_path   = args.m1,
        tick_path = args.ticks,
        out_path  = args.out,
        vp_window = args.vp_window,
        bin_size  = args.bin_size,
    )
