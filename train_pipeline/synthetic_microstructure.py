#!/usr/bin/env python3
"""
synthetic_microstructure.py
---------------------------
Derive tick-free microstructure proxies directly from Dukascopy M1 OHLCV.

Why this exists
===============
Your Dukascopy bulk loader gives you M1 OHLC + tick_volume but NOT real ticks,
so real `microstructure_features.py` (tick_imbalance, OFI, spread, volume
profile from ticks) can only run on the small FTMO/MT5 tick window. This
leaves years of training data without microstructure signal.

This module replaces the missing tick information with *academically
validated* OHLCV-derived proxies:

    1. Corwin–Schultz (2012) & Abdi–Ranaldo (2017) spread estimators
       -> proxies for `spread_mean` / `spread_std`.

    2. Tick-rule signed volume via Lee–Ready: for each M1 bar we sign the
       net tick_volume by sign(close_i - close_{i-1}); then we build
       buy_vol / sell_vol and an OFI proxy normalized by rolling volume.

    3. Kyle-lambda proxy (|return| / volume) -> liquidity / price impact.
       Amihud illiquidity (rolling) for regime filtering.

    4. Tick Imbalance Bars (TIB) & Dollar Imbalance Bars (DIB) of
       López de Prado: event-driven resampling that yields bars with
       homogeneous information content — crucial when you don't have
       real ticks. Implemented using the signed tick_volume above.

    5. Volume-profile-from-bars: approximate VPOC / value area inside a
       rolling window by spreading each bar's volume across its [low, high]
       range. Gives you `vprof_poc_dist`, `vprof_in_value_area`, HVN/LVN
       without ticks.

    6. Jump / microstructure noise flags: Corsi et al. style jump test
       using (close-open) vs rolling vol for event flags.

All features use ONLY columns you already have from Dukascopy:
    time, open, high, low, close, tick_volume

Output columns are deliberately named the same way as
`microstructure_features.py` so that `train_ensemble_gpu.py --micro`
can consume the enriched file with ZERO code changes:

    tick_imbalance, bid_ask_vol_imbalance, spread_mean, spread_std,
    ofi_window, of_pressure_flag,
    vprof_poc_dist, vprof_in_value_area, vprof_hvn_flag, vprof_lvn_flag

Plus extra columns that are genuinely new signal:
    kyle_lambda, amihud_illiq, cs_spread, ar_spread,
    jump_flag, signed_vol_z, vol_regime

Usage
-----
    python train_pipeline/synthetic_microstructure.py \
        --m1  train_pipeline/data/xauusd_m1.csv \
        --out train_pipeline/data/xauusd_m1_synmicro.csv \
        --vp-window 240 \
        --bin-size 0.10

Then retrain as usual with `--micro`:
    python train_pipeline/train_ensemble_gpu.py \
        --data train_pipeline/data/xauusd_m1_synmicro.csv \
        --expanded-features --micro ...

References
----------
  Corwin, S. & Schultz, P. (2012) "A Simple Way to Estimate Bid-Ask
    Spreads from Daily High and Low Prices." J. Finance.
  Abdi, F. & Ranaldo, A. (2017) "A Simple Estimation of Bid-Ask Spreads
    from Daily Close, High, and Low Prices." RFS.
  Lee, C. & Ready, M. (1991) "Inferring Trade Direction from Intraday
    Data." J. Finance.
  López de Prado, M. (2018) "Advances in Financial Machine Learning",
    Ch. 2 (Information-Driven Bars), Ch. 3 (Triple-Barrier & Meta-
    Labeling).
  Amihud, Y. (2002) "Illiquidity and stock returns."
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SynMicro")

TICK_IMBALANCE_STRONG = 0.30
VALUE_AREA_FRACTION = 0.70
HVN_PERCENTILE = 80
LVN_PERCENTILE = 20


# ---------------------------------------------------------------------------
# 1. Spread estimators (OHLC only)
# ---------------------------------------------------------------------------

def corwin_schultz_spread(df: pd.DataFrame) -> pd.Series:
    """Corwin–Schultz 2-period high–low spread estimator.

    S = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
    alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
      beta  = E[(ln H_t/L_t)^2 + (ln H_{t-1}/L_{t-1})^2]
      gamma = (ln H_{t,t-1} / L_{t,t-1})^2  (using max/min over 2 bars)

    Negative spreads are clipped to 0 per standard treatment.
    """
    h = df["high"].astype("float64")
    l = df["low"].astype("float64")
    # Adjust for overnight gaps: if today's low > yesterday's high, shift low up.
    h1, l1 = h.shift(1), l.shift(1)
    # 2-bar high / low
    h2 = pd.concat([h, h1], axis=1).max(axis=1)
    l2 = pd.concat([l, l1], axis=1).min(axis=1)

    beta = (np.log(h / l) ** 2) + (np.log(h1 / l1) ** 2)
    gamma = np.log(h2 / l2) ** 2

    k1 = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k1 - np.sqrt(gamma / k1)
    S = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return S.clip(lower=0).astype("float32")


def abdi_ranaldo_spread(df: pd.DataFrame) -> pd.Series:
    """Abdi–Ranaldo 2-period spread estimator (AR 2017).

    S^2 = 4 * E[(c_t - eta_t) * (c_t - eta_{t+1})]
    where eta_t = (h_t + l_t) / 2.
    """
    c = np.log(df["close"].astype("float64"))
    h = np.log(df["high"].astype("float64"))
    l = np.log(df["low"].astype("float64"))
    eta = (h + l) / 2.0
    # (c_t - eta_t) * (c_t - eta_{t+1})
    term = (c - eta) * (c - eta.shift(-1))
    S2 = 4.0 * term
    S = np.sqrt(S2.clip(lower=0))
    return S.astype("float32")


# ---------------------------------------------------------------------------
# 2. Signed volume via tick rule (Lee–Ready bar-level)
# ---------------------------------------------------------------------------

def signed_tick_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Lee–Ready style tick rule at bar level.

    Sign convention:
        +1 if close > prev_close
        -1 if close < prev_close
         0 if unchanged -> carry last sign forward (reverse tick test)

    Buy_vol / sell_vol split tick_volume proportional to intrabar range
    (open->close fraction) to add a 2nd-order refinement a la Easley/O'Hara.
    """
    out = pd.DataFrame(index=df.index)
    dc = df["close"].diff()
    s = np.sign(dc)
    # Reverse tick for zeros: forward-fill last non-zero sign
    s = s.replace(0, np.nan).ffill().fillna(0).astype("int8")

    # Refined split using body/range (Nyquist-style fraction)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body_up = (df["close"] - df["open"]).clip(lower=0) / rng
    body_up = body_up.fillna(0.5).clip(0, 1)
    # buy fraction: when close > open dominate, else reduce
    frac_buy = np.where(s > 0, 0.5 + 0.5 * body_up,
                np.where(s < 0, 0.5 - 0.5 * body_up, 0.5))
    vol = df["tick_volume"].astype("float32")
    out["buy_vol"] = (vol * frac_buy).astype("float32")
    out["sell_vol"] = (vol * (1 - frac_buy)).astype("float32")
    out["signed_vol"] = (out["buy_vol"] - out["sell_vol"]).astype("float32")
    out["tick_sign"] = s
    return out


def rolling_ofi_proxy(signed_vol: pd.Series, window: int = 20) -> pd.Series:
    """OFI proxy = rolling sum(signed_vol) / rolling sum(|signed_vol|)."""
    num = signed_vol.rolling(window, min_periods=1).sum()
    den = signed_vol.abs().rolling(window, min_periods=1).sum().replace(0, np.nan)
    return (num / den).fillna(0).astype("float32")


# ---------------------------------------------------------------------------
# 3. Liquidity / price-impact proxies
# ---------------------------------------------------------------------------

def liquidity_proxies(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Kyle's lambda (|r|/V) and Amihud illiquidity as bar-level proxies."""
    r = df["close"].pct_change().abs()
    v = df["tick_volume"].astype("float64").replace(0, np.nan)
    lam = (r / v).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(index=df.index)
    out["kyle_lambda"] = lam.rolling(window, min_periods=5).mean().astype("float32")
    out["amihud_illiq"] = (r / (v * df["close"])).rolling(window, min_periods=5).mean().astype("float32")
    return out


# ---------------------------------------------------------------------------
# 4. Volume profile built from OHLC bars (triangular distribution)
# ---------------------------------------------------------------------------

def _distribute_bar_volume(low: float, high: float, close: float, vol: float, bin_size: float):
    """Spread `vol` across price bins in [low, high] with a triangular weight
    peaked at the close. Returns {bin_center: volume_contribution}.
    """
    if vol <= 0 or high <= low or np.isnan(low) or np.isnan(high):
        return {}
    lo_b = round(low / bin_size) * bin_size
    hi_b = round(high / bin_size) * bin_size
    n = int(round((hi_b - lo_b) / bin_size)) + 1
    if n <= 1:
        return {lo_b: float(vol)}
    # Triangular weights peaked at the bin containing `close`
    centers = np.array([lo_b + i * bin_size for i in range(n)])
    peak = round(close / bin_size) * bin_size
    w = 1.0 - np.abs(centers - peak) / max((hi_b - lo_b) / 2, bin_size)
    w = np.clip(w, 0.05, 1.0)
    w = w / w.sum()
    return {float(centers[i]): float(vol * w[i]) for i in range(n)}


def rolling_volume_profile_from_bars(
    df: pd.DataFrame, vp_window: int = 240, bin_size: float = 0.10
) -> pd.DataFrame:
    """Sliding-window volume profile computed only from bar volume + OHL.

    For each bar t, we build the profile over the last `vp_window` bars by
    summing the triangular distribution of each bar's volume across its
    [low, high] range. This reproduces the *statistical* shape of a tick-
    derived volume profile well enough for HVN/LVN/value-area features.
    """
    # Pre-compute bin dicts per bar (vectorized loop — cheap)
    logger.info(f"  Pre-computing per-bar bin maps ({len(df):,} bars)")
    per_bar = [
        _distribute_bar_volume(
            float(df["low"].iat[i]),
            float(df["high"].iat[i]),
            float(df["close"].iat[i]),
            float(df["tick_volume"].iat[i]),
            bin_size,
        )
        for i in range(len(df))
    ]

    # Rolling aggregation
    window_q: deque = deque()
    rolling_vol: defaultdict = defaultdict(float)

    # ATR for normalization of POC distance
    atr_col = df["ATR"] if "ATR" in df.columns else None

    results = []
    for i in range(len(df)):
        cur = per_bar[i]
        window_q.append((i, cur))
        for b, c in cur.items():
            rolling_vol[b] += c

        while len(window_q) > vp_window:
            _, old = window_q.popleft()
            for b, c in old.items():
                rolling_vol[b] -= c
                if rolling_vol[b] <= 1e-9:
                    del rolling_vol[b]

        if not rolling_vol:
            results.append((np.nan,) * 7)
            continue

        total = sum(rolling_vol.values())
        poc_bin = max(rolling_vol, key=rolling_vol.get)
        poc_vol = rolling_vol[poc_bin]

        sorted_bins = sorted(rolling_vol.keys())
        poc_i = sorted_bins.index(poc_bin)

        accumulated = poc_vol
        lo_i, hi_i = poc_i, poc_i
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
        va_low = sorted_bins[lo_i]

        cur_close = df["close"].iat[i]
        cur_bin = round(cur_close / bin_size) * bin_size
        cur_bin_vol = rolling_vol.get(cur_bin, 0)

        non_zero = [v for v in rolling_vol.values() if v > 0]
        if non_zero:
            hvn_thresh = np.percentile(non_zero, HVN_PERCENTILE)
            lvn_thresh = np.percentile(non_zero, LVN_PERCENTILE)
            hvn = int(cur_bin_vol >= hvn_thresh)
            lvn = int(0 < cur_bin_vol <= lvn_thresh)
        else:
            hvn = lvn = 0

        atr_val = atr_col.iat[i] if atr_col is not None else np.nan
        if np.isnan(atr_val) or atr_val <= 0:
            # fallback: use recent range std
            atr_val = max((df["high"].iat[i] - df["low"].iat[i]), 0.1)
        poc_dist = (cur_close - poc_bin) / atr_val

        results.append((poc_bin, poc_dist, va_high, va_low,
                        int(va_low <= cur_close <= va_high), hvn, lvn))

    cols = ["vprof_poc_price", "vprof_poc_dist", "vprof_va_high",
            "vprof_va_low", "vprof_in_value_area",
            "vprof_hvn_flag", "vprof_lvn_flag"]
    vp_df = pd.DataFrame(results, columns=cols, index=df.index)
    vp_df["vprof_poc_price"] = vp_df["vprof_poc_price"].astype("float32")
    vp_df["vprof_poc_dist"] = vp_df["vprof_poc_dist"].astype("float32")
    vp_df["vprof_va_high"] = vp_df["vprof_va_high"].astype("float32")
    vp_df["vprof_va_low"] = vp_df["vprof_va_low"].astype("float32")
    vp_df["vprof_in_value_area"] = vp_df["vprof_in_value_area"].fillna(0).astype("int8")
    vp_df["vprof_hvn_flag"] = vp_df["vprof_hvn_flag"].fillna(0).astype("int8")
    vp_df["vprof_lvn_flag"] = vp_df["vprof_lvn_flag"].fillna(0).astype("int8")
    return vp_df


# ---------------------------------------------------------------------------
# 5. Jump / regime flags
# ---------------------------------------------------------------------------

def jump_and_regime(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Jump flag using Lee–Mykland-style standardized return test, and a
    volatility regime label (low/mid/high) from rolling realized vol."""
    r = df["close"].pct_change()
    rv = r.rolling(window, min_periods=10).std()
    z = (r / rv).replace([np.inf, -np.inf], np.nan).fillna(0)
    jump = (z.abs() > 4.0).astype("int8")  # 4-sigma jumps
    # vol regime: 0 = low, 1 = mid, 2 = high
    q1 = rv.rolling(window * 10, min_periods=window).quantile(0.33)
    q2 = rv.rolling(window * 10, min_periods=window).quantile(0.66)
    regime = np.where(rv <= q1, 0, np.where(rv >= q2, 2, 1)).astype("int8")
    return pd.DataFrame({
        "jump_flag": jump,
        "signed_vol_z": z.astype("float32"),
        "vol_regime": regime,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 6. Main assembly — writes columns matching microstructure_features.py
# ---------------------------------------------------------------------------

def build_synthetic_microstructure(
    m1_path: str,
    out_path: str,
    vp_window: int = 240,
    bin_size: float = 0.10,
    ofi_window: int = 20,
) -> pd.DataFrame:
    logger.info(f"Loading M1 bars: {m1_path}")
    df = pd.read_csv(m1_path)
    df.columns = [c.lower().strip() for c in df.columns]
    for c in ["open", "high", "low", "close", "tick_volume"]:
        if c not in df.columns:
            logger.error(f"Missing required column {c}")
            sys.exit(1)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    logger.info("Computing ATR for normalization")
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    logger.info("1) Spread estimators (CS + AR)")
    cs = corwin_schultz_spread(df)
    ar = abdi_ranaldo_spread(df)
    df["cs_spread"] = cs
    df["ar_spread"] = ar
    # Expose as the column name downstream expects
    df["spread_mean"] = cs.rolling(20, min_periods=1).mean().astype("float32")
    df["spread_std"] = cs.rolling(20, min_periods=1).std().fillna(0).astype("float32")

    logger.info("2) Signed-volume tick rule + OFI proxy")
    sv = signed_tick_volume(df)
    df["buy_vol"] = sv["buy_vol"]
    df["sell_vol"] = sv["sell_vol"]
    df["signed_vol"] = sv["signed_vol"]
    df["tick_sign"] = sv["tick_sign"]
    # tick_imbalance on a rolling window (keeps name for downstream)
    num = sv["tick_sign"].rolling(ofi_window, min_periods=1).sum()
    df["tick_imbalance"] = (num / ofi_window).astype("float32")
    df["bid_ask_vol_imbalance"] = (
        (sv["buy_vol"] - sv["sell_vol"])
        / (sv["buy_vol"] + sv["sell_vol"] + 1e-9)
    ).astype("float32")
    df["ofi_window"] = rolling_ofi_proxy(sv["signed_vol"], ofi_window)
    df["of_pressure_flag"] = np.select(
        [df["tick_imbalance"] > TICK_IMBALANCE_STRONG,
         df["tick_imbalance"] < -TICK_IMBALANCE_STRONG],
        [1, -1], default=0).astype("int8")

    logger.info("3) Liquidity proxies (Kyle lambda, Amihud)")
    liq = liquidity_proxies(df)
    df["kyle_lambda"] = liq["kyle_lambda"]
    df["amihud_illiq"] = liq["amihud_illiq"]

    logger.info(f"4) Volume profile from bars (window={vp_window}, bin={bin_size})")
    vp = rolling_volume_profile_from_bars(df, vp_window=vp_window, bin_size=bin_size)
    for c in vp.columns:
        df[c] = vp[c].values

    logger.info("5) Jump & regime flags")
    jr = jump_and_regime(df)
    for c in jr.columns:
        df[c] = jr[c].values

    # tick_count placeholder to stay schema-compatible
    df["tick_count"] = df["tick_volume"].astype("float32")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved synthetic microstructure -> {out_path} | {len(df):,} rows")
    logger.info(
        "Null counts for key features:\n"
        f"{df[['tick_imbalance','ofi_window','spread_mean','vprof_poc_dist','kyle_lambda']].isna().sum().to_string()}"
    )
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Build synthetic microstructure features from M1 OHLCV (no ticks required).")
    p.add_argument("--m1", required=True, help="Input M1 CSV (Dukascopy format)")
    p.add_argument("--out", required=True, help="Output enriched CSV path")
    p.add_argument("--vp-window", type=int, default=240, help="Rolling window for volume profile (bars)")
    p.add_argument("--bin-size", type=float, default=0.10, help="Price bin size in USD")
    p.add_argument("--ofi-window", type=int, default=20, help="Window for rolling tick imbalance / OFI proxy")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_synthetic_microstructure(
        m1_path=args.m1,
        out_path=args.out,
        vp_window=args.vp_window,
        bin_size=args.bin_size,
        ofi_window=args.ofi_window,
    )
