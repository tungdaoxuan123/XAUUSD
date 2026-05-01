#!/usr/bin/env python3
"""
prune_features.py
-----------------
SHAP-based feature importance diagnostic for the GBPUSD LightGBM model.

Purpose
-------
Identify the bottom N% of features by total absolute SHAP impact across
all three prediction classes (+1 / 0 / -1).  These are candidates for
dropping from the training feature matrix via DROP_FEATURES in
train_sota_v2.py.

Why SHAP over native Gain?
--------------------------
LightGBM's built-in Gain importance is computed on training data and
overstates the importance of high-cardinality or continuous features.
SHAP values measure actual output contribution on the validation set,
making them the correct tool for pruning decisions.

Multi-class handling
--------------------
For a 3-class model, shap.TreeExplainer returns a list of arrays
[shap_class_0, shap_class_1, shap_class_2], each of shape (n, p).
We sum the mean absolute values across all classes to get total
feature impact, then rank ascending (worst first).

Workflow
--------
1. Regenerate honest labels (MUST do before running this script):

       python train_pipeline/triple_barrier_labels.py \\
           --data train_pipeline/data/gbpusd_m1_synmicro.csv \\
           --out  train_pipeline/data/gbpusd_m1_tb.csv \\
           --pt-atr 2.0 --sl-atr 1.0 --max-hold 15

2. Retrain on honest labels:

       python train_pipeline/train_sota_v2.py \\
           --data    train_pipeline/data/gbpusd_m1_tb.csv \\
           --out-dir train_pipeline/reports/gbpusd \\
           --seq-len 60 --patch-len 8 --epochs 40 --gpu

3. Run this script:

       python train_pipeline/prune_features.py

4. Paste the terminal output (bottom 20% list) back to the review
   session so DROP_FEATURES can be hardcoded into train_sota_v2.py.

Output
------
  <out-dir>/feature_shap_ranking.csv   full ranking, all features
  Terminal: bottom --prune-pct% (prune candidates) + top 10 (sanity check)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("lightgbm not installed. Run: pip install lightgbm")

try:
    import shap
except ImportError:
    sys.exit("shap not installed. Run: pip install shap")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_shap_analysis(
    model_path: str,
    data_path: str,
    num_samples: int = 10_000,
    prune_pct: float = 0.20,
    out_dir: str | None = None,
) -> pd.DataFrame:
    """
    Run SHAP analysis and return a DataFrame ranked worst-to-best.

    Parameters
    ----------
    model_path  : path to lgb_model.txt
    data_path   : path to gbpusd_m1_tb.csv (triple-barrier labeled)
    num_samples : number of recent rows to use as validation set
    prune_pct   : fraction of features to flag as prune candidates (default 0.20)
    out_dir     : directory to save feature_shap_ranking.csv; None = same dir as model

    Returns
    -------
    pd.DataFrame with columns [feature, shap_importance] sorted ascending
    """
    # --- Load model ---------------------------------------------------------
    model_path = Path(model_path)
    if not model_path.exists():
        sys.exit(
            f"Model not found: {model_path}\n"
            "Run triple_barrier_labels.py + train_sota_v2.py first."
        )
    print(f"Loading model: {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    features = model.feature_name()
    print(f"Model has {len(features)} features")

    # --- Load validation data -----------------------------------------------
    data_path = Path(data_path)
    if not data_path.exists():
        sys.exit(
            f"Data not found: {data_path}\n"
            "Run triple_barrier_labels.py first."
        )
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    missing = [f for f in features if f not in df.columns]
    if missing:
        sys.exit(
            f"The following model features are missing from the CSV:\n"
            f"{missing}\n"
            "Ensure the CSV was generated AFTER the latest "
            "synthetic_microstructure.py run."
        )

    # Use the most recent `num_samples` rows as out-of-sample validation
    X_val = df[features].tail(num_samples).reset_index(drop=True)
    actual_n = len(X_val)
    print(f"Using {actual_n:,} rows for SHAP analysis (requested {num_samples:,})")

    # --- SHAP values --------------------------------------------------------
    print("Computing SHAP values (this may take 1–2 minutes)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # Multi-class: shap_values is a list of (n, p) arrays, one per class
    # Binary fallback: shap_values is a single (n, p) array
    if isinstance(shap_values, list):
        mean_abs_shap = np.zeros(len(features))
        for class_shap in shap_values:
            mean_abs_shap += np.abs(class_shap).mean(axis=0)
        print(f"Multi-class SHAP: summed across {len(shap_values)} classes")
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        print("Binary SHAP: single class")

    # --- Build ranking ------------------------------------------------------
    importance_df = pd.DataFrame({
        "feature":          features,
        "shap_importance":  mean_abs_shap,
    }).sort_values("shap_importance", ascending=True).reset_index(drop=True)

    # --- Save CSV -----------------------------------------------------------
    save_dir = Path(out_dir) if out_dir else model_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "feature_shap_ranking.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"\nFull ranking saved: {csv_path}")

    # --- Print prune candidates ---------------------------------------------
    prune_count = max(1, int(len(features) * prune_pct))
    prune_df    = importance_df.head(prune_count)
    top_df      = importance_df.tail(10).sort_values("shap_importance", ascending=False)

    sep = "=" * 60
    print(f"\n{sep}")
    print(
        f"BOTTOM {prune_count} FEATURES TO PRUNE "
        f"(bottom {prune_pct*100:.0f}% by SHAP — pure noise candidates)"
    )
    print(sep)
    print(prune_df.to_string(index=False))

    print(f"\n{sep}")
    print("TOP 10 FEATURES (sanity check — these should stay)")
    print(sep)
    print(top_df.to_string(index=False))

    print(f"\n{sep}")
    print("NEXT STEP: copy the feature names above into train_sota_v2.py:")
    print(sep)
    print("DROP_FEATURES = [")
    for feat in prune_df["feature"].tolist():
        print(f'    "{feat}",')
    print("]")
    print(sep)

    return importance_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="SHAP feature importance analysis for GBPUSD LightGBM model"
    )
    p.add_argument(
        "--model",
        default="train_pipeline/reports/gbpusd/lgb_model.txt",
        help="Path to trained LightGBM model file",
    )
    p.add_argument(
        "--data",
        default="train_pipeline/data/gbpusd_m1_tb.csv",
        help="Path to triple-barrier labeled CSV",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=10_000,
        help="Number of recent rows to use as validation set (default: 10000)",
    )
    p.add_argument(
        "--prune-pct",
        type=float,
        default=0.20,
        help="Fraction of features to flag for pruning (default: 0.20 = bottom 20%%)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save feature_shap_ranking.csv (default: same dir as model)",
    )
    args = p.parse_args()

    run_shap_analysis(
        model_path=args.model,
        data_path=args.data,
        num_samples=args.samples,
        prune_pct=args.prune_pct,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
