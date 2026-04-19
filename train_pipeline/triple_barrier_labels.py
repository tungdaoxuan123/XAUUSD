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

We then add Meta-Labeling (AFML Chapter 3): a PRIMARY model proposes a
direction (e.g. your current LightGBM trend model), then a SECONDARY
"meta" model decides whether to act on the primary signal. The meta
label is simply whether the triple-barrier outcome was profitable.

This gives you:
  - Better calibrated primary labels (volatility-scaled, path-aware)
  - A filter model that dramatically reduces false positives
  - Precision that climbs at the cost of recall — ideal for FTMO where
    sub-4.8% daily loss is a hard constraint.

Usage
-----
    # 1. Make triple-barrier labels
    python train_pipeline/triple_barrier_labels.py \
        --data train_pipeline/data/xauusd_m1_synmicro.csv \
        --out  train_pipeline/data/xauusd_m1_tb.csv \
        --pt-atr 1.5 \
        --sl-atr 1.0 \
        --max-hold 30

    # 2. Train primary model on labels as before
    python train_pipeline/train_ensemble_gpu.py \
        --data train_pipeline/data/xauusd_m1_tb.csv \
        --label-col tb_label ...

    # 3. Train meta filter (after primary predictions are available)
    python train_pipeline/triple_barrier_labels.py --meta \
        --data  train_pipeline/data/xauusd_m1_tb.csv \
        --preds train_pipeline/reports/primary_preds.csv \
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

    Barriers:
        upper = close_t + pt_atr * ATR_t   (take-profit)
        lower = close_t - sl_atr * ATR_t   (stop-loss)
        vertical = t + max_hold bars

    If `side_col` is provided (e.g. primary model's +/-1 signals), barriers
    are mirrored for shorts and the label becomes:
        +1 if trade profitable (barrier in signal's favor hit first)
         0 if vertical expired
        -1 if adverse barrier hit first
    This is the meta-labeling target.

    If `side_col` is None, we use a long-side default and return a
    three-class label that matches your current convention (-1/0/+1) but
    is path-aware and volatility-scaled.
    """
    required = {"close", "high", "low", "ATR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    close = df["close"].values.astype("float64")
    high = df["high"].values.astype("float64")
    low = df["low"].values.astype("float64")
    atr = df["ATR"].values.astype("float64")
    n = len(df)

    side = df[side_col].values.astype("int8") if side_col else np.ones(n, dtype="int8")

    tb_label = np.zeros(n, dtype="int8")
    tb_ret = np.zeros(n, dtype="float32")
    tb_hit = np.full(n, -1, dtype="int32")  # index where first barrier hit

    for i in range(n):
        s = side[i]
        if s == 0:
            continue
        atr_i = atr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue
        entry = close[i]
        if s > 0:
            upper = entry + pt_atr * atr_i
            lower = entry - sl_atr * atr_i
        else:
            upper = entry - pt_atr * atr_i  # TP for short is lower price
            lower = entry + sl_atr * atr_i  # SL for short is higher price

        end = min(i + 1 + max_hold, n)
        hit = -1
        outcome = 0
        for j in range(i + 1, end):
            hi_j, lo_j = high[j], low[j]
            if s > 0:
                # long: TP at upper, SL at lower
                if hi_j >= upper:
                    outcome = 1
                    hit = j
                    break
                if lo_j <= lower:
                    outcome = -1
                    hit = j
                    break
            else:
                # short: TP at upper(=lower price), SL at lower(=higher price)
                if lo_j <= upper:
                    outcome = 1
                    hit = j
                    break
                if hi_j >= lower:
                    outcome = -1
                    hit = j
                    break
        if hit == -1:
            # vertical barrier: use sign of terminal return in signal direction
            j = end - 1
            ret = (close[j] - entry) / entry * (1 if s > 0 else -1)
            outcome = 0
            hit = j
        else:
            ret = (close[hit] - entry) / entry * (1 if s > 0 else -1)

        tb_label[i] = outcome
        tb_ret[i] = ret
        tb_hit[i] = hit

    out = df.copy()
    out["tb_label"] = tb_label
    out["tb_return"] = tb_ret
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
    out["meta_label"] = (out["tb_label"] == 1).astype("int8")
    # Only keep rows where primary actually produced a signal (+/-1)
    out["has_signal"] = (out[primary_col] != 0).astype("int8")
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
        # Average 1/concurrency over the life of the label
        seg = concurrency[i:end + 1]
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
    p.add_argument("--data", required=True, help="Input CSV with ATR column")
    p.add_argument("--out", default=None, help="Output CSV path")
    p.add_argument("--pt-atr", type=float, default=1.5)
    p.add_argument("--sl-atr", type=float, default=1.0)
    p.add_argument("--max-hold", type=int, default=30)
    p.add_argument("--meta", action="store_true",
                   help="Run meta-labeling (requires --preds column 'primary_pred')")
    p.add_argument("--preds", default=None,
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
        lc = (df["low"] - df["close"].shift()).abs()
        df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    if args.meta:
        if not args.preds:
            sys.exit("--meta requires --preds")
        preds = pd.read_csv(args.preds)
        df["primary_pred"] = preds["primary_pred"].astype("int8")
        out = meta_labels_from_primary(df, pt_atr=args.pt_atr,
                                       sl_atr=args.sl_atr, max_hold=args.max_hold)
    else:
        out = triple_barrier_labels(df, pt_atr=args.pt_atr,
                                    sl_atr=args.sl_atr, max_hold=args.max_hold)

    # Uniqueness & time-decay weights for downstream trainer
    w_u = compute_uniqueness_weights(out["tb_hit_idx"].values, len(out))
    w_t = time_decay_weights(len(out))
    out["sample_weight"] = (w_u * w_t).astype("float32")

    out_path = args.out or args.data.replace(".csv", "_tb.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(f"Saved {len(out):,} rows with triple-barrier labels -> {out_path}")
    if "tb_label" in out.columns:
        vc = out["tb_label"].value_counts().to_dict()
        logger.info(f"Label distribution: {vc}")


if __name__ == "__main__":
    main()
