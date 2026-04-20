#!/usr/bin/env python3
"""
live_features.py
----------------
Single source of truth for feature engineering at inference time.

The bug in the old live_sota_trading.prepare_sota_data() was that several
"synthetic microstructure" columns were replaced by placeholders
(e.g. vprof_in_value_area=1, vprof_lvn_flag=0, bid_ask_vol_imbalance
aliased to tick_imbalance). Distribution shift between train and live
is the #1 reason a trained model collapses to p≈1 on one class in
production.

This module delegates to the SAME functions used during training so
features are guaranteed identical.

Public API
----------
    build_live_features(rates_df, feature_list) -> DataFrame
        * rates_df: raw MT5 bars with [time, open, high, low, close, tick_volume]
        * feature_list: columns the model was trained on
        Returns a DataFrame with exactly those columns (NaN-safe).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# Reuse training-time feature builders
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from synthetic_microstructure import (  # noqa: E402
    corwin_schultz_spread,
    abdi_ranaldo_spread,
    signed_tick_volume,
    rolling_ofi_proxy,
    liquidity_proxies,
    rolling_volume_profile_from_bars,
    jump_and_regime,
)


def _classical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI/MACD/ATR/VWAP/BB — identical to train_ensemble_gpu.py."""
    # ATR
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # RSI-14
    d = df["close"].diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    rs = up / (dn + 1e-9)
    df["RSI"] = 100 - 100 / (1 + rs)

    # MACD(12,26,9)
    e12 = df["close"].ewm(span=12, adjust=False).mean()
    e26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]

    # VWAP (rolling 100) + dist
    vol = df["tick_volume"].astype("float64")
    df["VWAP"] = (df["close"]*vol).rolling(100).sum() / (vol.rolling(100).sum() + 1e-9)
    df["close_minus_vwap"] = df["close"] - df["VWAP"]

    # Bollinger width (20, 2σ)
    ma20 = df["close"].rolling(20).mean()
    sd20 = df["close"].rolling(20).std()
    df["BB_width"] = (4 * sd20) / (ma20 + 1e-9)
    return df


def _session_features(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    h = t.dt.hour
    df["is_asian"]   = ((h >= 0) & (h < 8)).astype("int8")
    df["is_london"]  = ((h >= 8) & (h < 16)).astype("int8")
    df["is_ny"]      = ((h >= 13) & (h < 21)).astype("int8")
    df["is_overlap"] = ((h >= 13) & (h < 16)).astype("int8")
    df["dow_sin"]  = np.sin(2*np.pi*t.dt.dayofweek/7).astype("float32")
    df["dow_cos"]  = np.cos(2*np.pi*t.dt.dayofweek/7).astype("float32")
    df["hour_sin"] = np.sin(2*np.pi*h/24).astype("float32")
    df["hour_cos"] = np.cos(2*np.pi*h/24).astype("float32")
    return df


def build_live_features(rates: pd.DataFrame,
                        feature_list: list[str],
                        vp_window: int = 240,
                        bin_size: float = 0.10,
                        ofi_window: int = 20) -> pd.DataFrame:
    """Build all columns needed by the model using TRAINING-TIME code."""
    df = rates.copy()
    df.columns = [c.lower() for c in df.columns]
    if "time" in df.columns and df["time"].dtype.kind in ("i", "u", "f"):
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # 1) Classical indicators
    df = _classical_indicators(df)

    # 2) Synthetic microstructure — same functions as training
    cs = corwin_schultz_spread(df)
    ar = abdi_ranaldo_spread(df)
    df["cs_spread"] = cs
    df["ar_spread"] = ar
    df["spread_mean"] = cs.rolling(20, min_periods=1).mean().astype("float32")
    df["spread_std"]  = cs.rolling(20, min_periods=1).std().fillna(0).astype("float32")

    sv = signed_tick_volume(df)
    df["buy_vol"] = sv["buy_vol"]
    df["sell_vol"] = sv["sell_vol"]
    df["signed_vol"] = sv["signed_vol"]
    df["tick_sign"] = sv["tick_sign"]
    num = sv["tick_sign"].rolling(ofi_window, min_periods=1).sum()
    df["tick_imbalance"] = (num / ofi_window).astype("float32")
    df["bid_ask_vol_imbalance"] = (
        (sv["buy_vol"] - sv["sell_vol"])
        / (sv["buy_vol"] + sv["sell_vol"] + 1e-9)
    ).astype("float32")
    df["ofi_window"] = rolling_ofi_proxy(sv["signed_vol"], ofi_window)
    df["of_pressure_flag"] = np.select(
        [df["tick_imbalance"] > 0.30, df["tick_imbalance"] < -0.30],
        [1, -1], default=0,
    ).astype("int8")

    liq = liquidity_proxies(df)
    df["kyle_lambda"] = liq["kyle_lambda"]
    df["amihud_illiq"] = liq["amihud_illiq"]

    # 3) Volume profile from bars (short window at live time to keep latency low)
    vp = rolling_volume_profile_from_bars(df, vp_window=min(vp_window, max(60, len(df)-1)),
                                          bin_size=bin_size)
    for c in vp.columns:
        df[c] = vp[c].values

    # 4) Regime
    jr = jump_and_regime(df)
    for c in jr.columns:
        df[c] = jr[c].values

    # 5) Session
    df = _session_features(df)

    # Final fill + guarantee schema
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    missing = [c for c in feature_list if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    return df
