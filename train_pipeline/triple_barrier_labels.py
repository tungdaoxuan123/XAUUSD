#!/usr/bin/env python3
"""
triple_barrier_labels.py — LONG-ONLY binary edition
---------------------------------------------------
Path-aware, volatility-scaled labels for a BUY/LONG-only strategy with
fixed 2:1 risk-reward ratio.

Every bar is evaluated as a potential LONG entry only. The outcome is
binary: 0 = wait (no long trade), 1 = valid long entry.

Why long-only triple-barrier
----------------------------
The old 3-class {-1, 0, +1} labeling forced the model to spend capacity
distinguishing bearish continuation from neutral chop. Both are now
collapsed into class 0 ("do nothing"), producing cleaner gradients,
tighter confidence estimates, and an execution layer that never opens
shorts.

Fixed 2:1 RR with dynamic pip distance
--------------------------------------
Stop distance is ATR-scaled (--sl-atr) and take profit is exactly twice
that value (--pt-atr = 2 * --sl-atr). This preserves a hard 2:1 ratio
while allowing the actual pip distance to expand and contract with
market volatility. On quiet sessions ATR contracts → tighter stops and
smaller targets; during high-volatility moves ATR expands → the trade
gets more room automatically.

Cost model
----------
All barriers use the deteriorated entry price (ask for longs) and
deteriorated exit checks (bid for long TP/SL). Commission is expressed
as synthetic spread widening so the math is internally consistent.

Cost constants
--------------
<<<<<<< HEAD
  COMMISSION_PIPS = 0.00006   FTMO raw account: $6/lot round-trip
  DEFAULT_SPREAD  = 0.00008   0.8 pip fallback
=======
  COMMISSION_PIPS = args.commission   FTMO raw account: $6/lot round-trip on GBPUSD
                               ($3/lot/side * 2 sides = $6 = 0.6 pips)
  DEFAULT_SPREAD  = 0.00008   0.8 pip fallback when cs_spread column absent
>>>>>>> 722a9a899a0e51657b071f342974d45d54fbb6b2

Spread column resolution order
-------------------------------
  cs_spread  ->  spread_mean  ->  spread_est  ->  DEFAULT_SPREAD constant

Usage
-----
    # 1. Make long-only triple-barrier labels (default 2:1 RR)
    python train_pipeline/triple_barrier_labels.py \\
        --data train_pipeline/data/xauusd_m1_synmicro.csv \\
        --out  train_pipeline/data/xauusd_m1_tb.csv \\
        --pt-atr 2.0 \\
        --sl-atr 1.0 \\
        --max-hold 30

    # 2. Train binary primary model on labels
    python train_pipeline/train_ensemble_gpu.py \\
        --data train_pipeline/data/xauusd_m1_tb.csv \\
        --label-col tb_label ...

    # 3. Train binary meta filter (after primary predictions are available)
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
# NOTE: overwritten in main() via --commission CLI arg
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
    pt_atr: float = 2.0,
    sl_atr: float = 1.0,
    max_hold: int = 30,
    side_col: str | None = None,
    direction: int = 1,
) -> pd.DataFrame:
    """Compute triple-barrier outcomes for each bar (long or short).

    Parameters
    ----------
    direction : int
        +1 = evaluate long entries (TP above, SL below)
        -1 = evaluate short entries (TP below, SL above)

    Barriers (cost-adjusted)
    ------------------------
    Long (direction=+1):
        entry_ask = close[i] + half_spread
        upper     = entry_ask + pt_atr * ATR[i]   (TP)
        lower     = entry_ask - sl_atr * ATR[i]   (SL)

    Short (direction=-1):
        entry_bid = close[i] - half_spread
        upper     = entry_bid - pt_atr * ATR[i]   (TP: price moves down)
        lower     = entry_bid + sl_atr * ATR[i]   (SL: price moves up)

    Outcome mapping (binary):
        1  = valid trade (TP hit first)
        0  = wait / do nothing (SL hit or timeout)
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

    if side_col:
        raw_side = df[side_col].values.astype("int8")
        # Only evaluate entries matching our direction
        side = np.where(raw_side == direction, direction, 0).astype("int8")
    else:
        side = np.full(n, direction, dtype="int8")

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

        effective_spread = spread_arr[i] + COMMISSION_PIPS
        half_spread      = effective_spread / 2.0

        if s > 0:
            entry    = close[i] + half_spread
            upper    = entry + pt_atr * atr_i
            lower    = entry - sl_atr * atr_i
        else:
            entry    = close[i] - half_spread
            upper    = entry - pt_atr * atr_i
            lower    = entry + sl_atr * atr_i

        end     = min(i + 1 + max_hold, n)
        hit     = -1
        outcome = 0

        for j in range(i + 1, end):
            hi_j, lo_j = high[j], low[j]

            if s > 0:
                high_bid = hi_j - half_spread
                low_bid  = lo_j - half_spread
                if high_bid >= upper:
                    outcome = 1; hit = j; break
                if low_bid <= lower:
                    outcome = 0; hit = j; break
            else:
                high_ask = hi_j + half_spread
                low_ask  = lo_j + half_spread
                if low_ask <= upper:
                    outcome = 1; hit = j; break
                if high_ask >= lower:
                    outcome = 0; hit = j; break

        if hit == -1:
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
    pt_atr: float = 2.0,
    sl_atr: float = 1.0,
    max_hold: int = 30,
    direction: int = 1,
) -> pd.DataFrame:
    """Compute triple-barrier outcomes along the primary model's side."""
    if primary_col not in df.columns:
        raise ValueError(f"Primary column '{primary_col}' not found")
    out = triple_barrier_labels(df, pt_atr, sl_atr, max_hold,
                                side_col=primary_col, direction=direction)
    out["meta_label"]  = (out["tb_label"] == 1).astype("int8")
    out["has_signal"]  = (out[primary_col] > 0).astype("int8")
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
    p.add_argument("--pt-atr",   type=float,    default=2.0)
    p.add_argument("--sl-atr",   type=float,    default=1.0)
    p.add_argument("--max-hold", type=int,       default=30)
    p.add_argument("--meta",     action="store_true",
                   help="Run meta-labeling (requires --preds column 'primary_pred')")
    p.add_argument("--preds",    default=None,
                   help="CSV with primary_pred column aligned to data")
    p.add_argument("--commission", type=float, default=0.00006,
                   help="Commission in price terms. Use 0.00003 for JPY pairs.")
    p.add_argument("--side", type=str, default="long", choices=["long", "short"],
                   help="Direction: long (PT above, SL below) or short (PT below, SL above)")
    args = p.parse_args()

    direction = 1 if args.side == "long" else -1

    # Wire CLI commission arg to the module-level constant used in the labeler
    global COMMISSION_PIPS
    COMMISSION_PIPS = args.commission

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
            df, pt_atr=args.pt_atr, sl_atr=args.sl_atr, max_hold=args.max_hold,
            direction=direction,
        )
    else:
        out = triple_barrier_labels(
            df, pt_atr=args.pt_atr, sl_atr=args.sl_atr, max_hold=args.max_hold,
            direction=direction,
        )

    side_label = args.side.upper()
    # Uniqueness & time-decay weights for downstream trainer
    w_u              = compute_uniqueness_weights(out["tb_hit_idx"].values, len(out))
    w_t              = time_decay_weights(len(out))
    out["sample_weight"] = (w_u * w_t).astype("float32")

    out_path = args.out or args.data.replace(".csv", "_tb.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(f"Saved {len(out):,} rows with {side_label}-only triple-barrier labels -> {out_path}")
    if "tb_label" in out.columns:
        vc = out["tb_label"].value_counts().to_dict()
        n_one = vc.get(1, 0)
        n_total = vc.get(0, 0) + n_one
        logger.info(
            f"Label distribution: {vc}  "
            f"(positive rate: {n_one / max(n_total, 1):.1%})"
        )
        logger.info(
            f"[CostModel] COMMISSION_PIPS={COMMISSION_PIPS} "
            f"DEFAULT_SPREAD={DEFAULT_SPREAD}"
        )


if __name__ == "__main__":
    main()
