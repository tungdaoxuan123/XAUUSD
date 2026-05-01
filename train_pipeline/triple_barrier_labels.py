#!/usr/bin/env python3
"""
triple_barrier_labels.py
------------------------
Replace the current fixed-horizon {BUY, HOLD, SELL} labels with
López de Prado's Triple-Barrier Method + Meta-Labeling.

Why it beats your current labeling
----------------------------------
Your `build_labels()` in train_ensemble_gpu.py uses:

    label = 1 if ret(h=5) > 0.0005 else (-1 if ret < -0.0005 else 0)

Problems:
  * Fixed horizon ignores the path — you label a +10 pip outcome even if
    you would have stopped out at -30 pips first.
  * Symmetric thresholds ignore realized volatility (ATR). In calm markets
    nothing is actionable; in news spikes, everything is a 1.
  * Same threshold for XAUUSD across all regimes biases the model.

Triple-Barrier fixes all three by simulating a *real* trade: an upper
barrier (take-profit), a lower barrier (stop-loss), and a vertical
barrier (timeout). The label is the *first* barrier hit.

Cost model (v2)
---------------
All barriers are computed from the *deteriorated* entry price (ask for
longs, bid for shorts) and all exit checks compare against the
*deteriorated* exit price (bid for long TP/SL, ask for short TP/SL).

This means:
  * TP requires MORE momentum to hit (correct — harder to win)
  * SL requires LESS of an adverse move to hit (correct — easier to lose)
  * Commission is expressed as synthetic spread widening so the math is
    internally consistent and not just a patch on the raw barriers.

Cost constants
--------------
  COMMISSION_PIPS = 0.00006   FTMO raw account: $6/lot round-trip on GBPUSD
                               ($3/lot/side * 2 sides = $6 = 0.6 pips)
  DEFAULT_SPREAD  = 0.00008   0.8 pip fallback when cs_spread column absent

Spread column resolution order
-------------------------------
  cs_spread  ->  spread_mean  ->  spread_est  ->  DEFAULT_SPREAD constant

Usage
-----
    # 1. Make triple-barrier labels
    python train_pipeline/triple_barrier_labels.py \\
        --data train_pipeline/data/xauusd_m1_synmicro.csv \\
        --out  train_pipeline/data/xauusd_m1_tb.csv \\
        --pt-atr 1.5 \\
        --sl-atr 1.0 \\
        --max-hold 30

    # 2. Train primary model on labels as before
    python train_pipeline/train_ensemble_gpu.py \\
        --data train_pipeline/data/xauusd_m1_tb.csv \\
        --label-col tb_label ...

    # 3. Train meta filter (after primary predictions are available)
    python train_pipeline/triple_barrier_labels.py --meta \\
        --data  train_pipeline/data/xauusd_m1_tb.csv \\
        --preds train_pipeline/reports/primary_preds.csv \\
        --out-model train_pipeline/models_gpu/meta_filter.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TripleBarrier")


# ---------------------------------------------------------------------------
# Cost model constants
# ---------------------------------------------------------------------------

# FTMO Raw account: $3/lot/side commission on GBPUSD
# $10/pip/lot => $6 round-trip = 0.6 pips = 0.00006 in price terms
COMMISSION_PIPS: float = 0.00006

# Fallback spread when the cs_spread / spread_mean / spread_est column is
# absent from the input CSV (e.g. raw MT5 data not yet through Step 1)
DEFAULT_SPREAD: float = 0.00008   # 0.8 pips


def _resolve_spread_series(df: pd.DataFrame) -> pd.Series:
    """Return the per-bar spread series from the DataFrame.

    Resolution order:
        cs_spread  ->  spread_mean  ->  spread_est  ->  DEFAULT_SPREAD constant

    The returned series is aligned to df.index.
    """
    for col in ("cs_spread", "spread_mean", "spread_est"):
        if col in df.columns:
            logger.info(f"[CostModel] Using spread column '{col}'")
            return df[col].astype("float64")
    logger.warning(
        f"[CostModel] No spread column found — falling back to "
        f"DEFAULT_SPREAD = {DEFAULT_SPREAD} ({DEFAULT_SPREAD * 1e4:.1f} pips). "
        "Run synthetic_microstructure.py first for per-bar spread estimates."
    )
    return pd.Series(DEFAULT_SPREAD, index=df.index, dtype="float64")


# ---------------------------------------------------------------------------
# Triple-barrier labeler
# ---------------------------------------------------------------------------

def triple_barrier_labels(
    df: pd.DataFrame,
    pt_atr: float = 1.5,
    sl_atr: float = 1.0,
    max_hold: int = 30,
    side_col: str | None = None,
) -> pd.DataFrame:
    """Compute triple-barrier outcomes for each bar.

    Barriers (cost-adjusted, v2)
    ----------------------------
    All barrier levels and hit-checks account for the bid/ask spread and
    round-trip commission.  Entry is at the worse price (ask for longs,
    bid for shorts); exit checks compare against the worse exit price
    (bid for longs, ask for shorts).

    effective_spread = cs_spread[i] + COMMISSION_PIPS
    half_spread      = effective_spread / 2

    Long:
        entry_ask = close[i] + half_spread
        upper     = entry_ask + pt_atr * ATR[i]   (TP — further away)
        lower     = entry_ask - sl_atr * ATR[i]   (SL — closer to bid)
        TP hit when  high[j] - half_spread >= upper
        SL hit when  low[j]  - half_spread <= lower

    Short:
        entry_bid = close[i] - half_spread
        upper     = entry_bid - pt_atr * ATR[i]   (TP at lower price)
        lower     = entry_bid + sl_atr * ATR[i]   (SL at higher price)
        TP hit when  low[j]  + half_spread <= upper
        SL hit when  high[j] + half_spread >= lower

    If `side_col` is provided (primary model's +/-1 signals), barriers are
    mirrored for shorts and the label becomes the meta-labeling target:
        +1  trade profitable (TP barrier hit first)
         0  vertical barrier expired
        -1  adverse barrier hit first

    If `side_col` is None, a long-side default is used and the three-class
    label (-1/0/+1) matches your existing convention but is now path-aware,
    volatility-scaled, and cost-adjusted.
    """
    required = {"close", "high", "low", "ATR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    close = df["close"].values.astype("float64")
    high  = df["high"].values.astype("float64")
    low   = df["low"].values.astype("float64")
    atr   = df["ATR"].values.astype("float64")
    n     = len(df)

    spread_arr = _resolve_spread_series(df).values.astype("float64")

    side = (
        df[side_col].values.astype("int8")
        if side_col
        else np.ones(n, dtype="int8")
    )

    tb_label = np.zeros(n, dtype="int8")
    tb_ret   = np.zeros(n, dtype="float32")
    tb_hit   = np.full(n, -1, dtype="int32")

    for i in range(n):
        s = side[i]
        if s == 0:
            continue
        atr_i = atr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        # --- Cost model ---------------------------------------------------
        effective_spread = spread_arr[i] + COMMISSION_PIPS
        half_spread      = effective_spread / 2.0

        if s > 0:
            # Long: enter at ask, exit (TP and SL) at bid
            entry    = close[i] + half_spread
            upper    = entry + pt_atr * atr_i
            lower    = entry - sl_atr * atr_i
        else:
            # Short: enter at bid, exit (TP and SL) at ask
            entry    = close[i] - half_spread
            upper    = entry - pt_atr * atr_i   # TP: price moves down
            lower    = entry + sl_atr * atr_i   # SL: price moves up

        # --- Barrier scan -------------------------------------------------
        end     = min(i + 1 + max_hold, n)
        hit     = -1
        outcome = 0

        for j in range(i + 1, end):
            hi_j, lo_j = high[j], low[j]

            if s > 0:
                # Long exit checks — compare against bid
                high_bid = hi_j - half_spread
                low_bid  = lo_j - half_spread
                if high_bid >= upper:
                    outcome = 1
                    hit = j
                    break
                if low_bid <= lower:
                    outcome = -1
                    hit = j
                    break
            else:
                # Short exit checks — compare against ask
                high_ask = hi_j + half_spread
                low_ask  = lo_j + half_spread
                if low_ask <= upper:    # TP: exit at ask, price moved down
                    outcome = 1
                    hit = j
                    break
                if high_ask >= lower:   # SL: exit at ask, price moved up
                    outcome = -1
                    hit = j
                    break

        if hit == -1:
            # Vertical barrier: use sign of terminal return in signal direction
            j       = end - 1
            ret     = (close[j] - entry) / entry * (1 if s > 0 else -1)
            outcome = 0
            hit     = j
        else:
            ret = (close[hit] - entry) / entry * (1 if s > 0 else -1)

        tb_label[i] = outcome
        tb_ret[i]   = ret
        tb_hit[i]   = hit

    out = df.copy()
    out["tb_label"]   = tb_label
    out["tb_return"]  = tb_ret
    out["tb_hit_idx"] = tb_hit
    return out


# ---------------------------------------------------------------------------
# Meta-labeling: given primary side predictions, produce secondary labels
# ---------------------------------------------------------------------------

def meta_labels_from_primary(
    df: pd.DataFrame,
    primary_col: str = "primary_pred",
    pt_atr: float = 1.5,
    sl_atr: float = 1.0,
    max_hold: int = 30,
) -> pd.DataFrame:
    """Compute triple-barrier outcomes along the primary model's side.

    meta_label = 1 if trade hit TP, else 0 (SL or timeout)
    This becomes the target for a binary "should we act" filter model.
    """
    if primary_col not in df.columns:
        raise ValueError(f"Primary column '{primary_col}' not found")
    out = triple_barrier_labels(df, pt_atr, sl_atr, max_hold, side_col=primary_col)
    out["meta_label"]  = (out["tb_label"] == 1).astype("int8")
    out["has_signal"]  = (out[primary_col] != 0).astype("int8")
    return out


# ---------------------------------------------------------------------------
# Sample-weighting utilities (AFML §4): uniqueness + time-decay
# ---------------------------------------------------------------------------

def compute_uniqueness_weights(tb_hit_idx: np.ndarray, n: int) -> np.ndarray:
    """Weight each sample by 1 / number of concurrent labels.

    Two labels are concurrent if their [entry, first-touch] windows overlap.
    This prevents the classifier from over-weighting periods where many
    samples share the same outcome.
    """
    concurrency = np.zeros(n, dtype="int32")
    for i in range(n):
        end = tb_hit_idx[i]
        if end < 0:
            continue
        concurrency[i:end + 1] += 1
    w = np.zeros(n, dtype="float32")
    for i in range(n):
        end = tb_hit_idx[i]
        if end < 0:
            w[i] = 0
            continue
        seg  = concurrency[i:end + 1]
        w[i] = float(np.mean(1.0 / np.maximum(seg, 1)))
    return w


def time_decay_weights(n: int, decay: float = 0.5) -> np.ndarray:
    """Linear time decay: oldest sample weight = `decay`, newest = 1.0."""
    return np.linspace(decay, 1.0, n, dtype="float32")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     required=True, help="Input CSV with ATR column")
    p.add_argument("--out",      default=None,  help="Output CSV path")
    p.add_argument("--pt-atr",   type=float,    default=1.5)
    p.add_argument("--sl-atr",   type=float,    default=1.0)
    p.add_argument("--max-hold", type=int,       default=30)
    p.add_argument("--meta",     action="store_true",
                   help="Run meta-labeling (requires --preds column 'primary_pred')")
    p.add_argument("--preds",    default=None,
                   help="CSV with primary_pred column aligned to data")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df.columns = [c.lower().strip() for c in df.columns]
    if "atr" in df.columns and "ATR" not in df.columns:
        df.rename(columns={"atr": "ATR"}, inplace=True)
    if "ATR" not in df.columns:
        logger.info("ATR not found — computing from OHLC")
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"]  - df["close"].shift()).abs()
        df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    if args.meta:
        if not args.preds:
            sys.exit("--meta requires --preds")
        preds              = pd.read_csv(args.preds)
        df["primary_pred"] = preds["primary_pred"].astype("int8")
        out = meta_labels_from_primary(
            df, pt_atr=args.pt_atr, sl_atr=args.sl_atr, max_hold=args.max_hold
        )
    else:
        out = triple_barrier_labels(
            df, pt_atr=args.pt_atr, sl_atr=args.sl_atr, max_hold=args.max_hold
        )

    # Uniqueness & time-decay weights for downstream trainer
    w_u              = compute_uniqueness_weights(out["tb_hit_idx"].values, len(out))
    w_t              = time_decay_weights(len(out))
    out["sample_weight"] = (w_u * w_t).astype("float32")

    out_path = args.out or args.data.replace(".csv", "_tb.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(f"Saved {len(out):,} rows with triple-barrier labels -> {out_path}")
    if "tb_label" in out.columns:
        vc = out["tb_label"].value_counts().to_dict()
        logger.info(f"Label distribution: {vc}")
        logger.info(
            f"[CostModel] COMMISSION_PIPS={COMMISSION_PIPS} "
            f"DEFAULT_SPREAD={DEFAULT_SPREAD}"
        )


if __name__ == "__main__":
    main()
