#!/usr/bin/env python3
"""
evaluate_on_ftmo.py - Cross-Broker Model Performance Tester

Loads an ensemble trained on Dukascopy data and evaluates its performance
on a separate CSV from your FTMO broker. This helps detect broker-mismatch overhead.

Usage:
    python train_pipeline/evaluate_on_ftmo.py \
        --models train_pipeline/models_gpu_1y \
        --data train_pipeline/data/xauusd_m1_2m.csv \
        --horizon 10 \
        --buy-threshold 0.0003 \
        --sell-threshold 0.0003
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Add project root to path for local imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import indicators and Ensemble logic
try:
    from train_pipeline.ensemble_gpu import EnsembleGPU, _add_indicators
except ImportError:
    from ensemble_gpu import EnsembleGPU, _add_indicators

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Evaluator")

# ---------------------------------------------------------------------------
# Labeling Logic (Mirroring train_ensemble_gpu.py)
# ---------------------------------------------------------------------------

def create_labels(df: pd.DataFrame, horizon: int, buy_threshold: float, sell_threshold: float):
    """Create forward-return labels for evaluation."""
    future_close = df["close"].shift(-horizon)
    fwd_return = (future_close - df["close"]) / df["close"]
    
    labels = np.zeros(len(df), dtype=int)
    labels[fwd_return > buy_threshold] = 1
    labels[fwd_return < -sell_threshold] = -1
    
    # Drop rows without future data (the last 'horizon' rows)
    return labels

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate Ensemble on FTMO data")
    parser.add_argument("--models", required=True, help="Path to models directory")
    parser.add_argument("--data",   required=True, help="Path to FTMO data CSV")
    parser.add_argument("--horizon", type=int, default=10, help="Forward horizon for labels")
    parser.add_argument("--buy-threshold",  type=float, default=0.0003, help="Buy threshold")
    parser.add_argument("--sell-threshold", type=float, default=0.0003, help="Sell threshold")
    args = parser.parse_args()

    # 1. Load Ensemble
    logger.info(f"Loading ensemble from {args.models}...")
    try:
        ens = EnsembleGPU.load(args.models)
        expanded = ens.expanded
    except Exception as e:
        logger.error(f"Failed to load ensemble: {e}")
        return

    # 2. Load and Prepare Data
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    df["time"] = pd.to_datetime(df["time"])
    
    logger.info("Computing technical indicators...")
    df_clean = _add_indicators(df)
    
    logger.info(f"Generating labels (horizon={args.horizon})...")
    y_true = create_labels(df_clean, args.horizon, args.buy_threshold, args.sell_threshold)
    
    # Drop the tail where we don't have future data
    df_eval = df_clean.iloc[:-args.horizon].copy()
    y_true = y_true[:-args.horizon]

    # 3. Build Observations (Batch mode)
    logger.info("Building observations batch...")
    
    # Get lookback from ensemble
    lookback = getattr(ens, "lookback", 10)
    logger.info(f"Using lookback: {lookback}")

    # Lagged closes
    price_cols = []
    for i in range(lookback - 1, -1, -1):
        col_name = f"close_lag_{i}"
        df_eval[col_name] = df_eval["close"].shift(i)
        price_cols.append(col_name)
    
    df_eval = df_eval.dropna().reset_index(drop=True)
    y_true = y_true[lookback-1:] # Sync with lag drop
    
    # Add dummy account features
    df_eval["current_position"] = 0.0
    df_eval["current_balance"] = 10000.0
    
    # Get dynamic feature names
    from train_pipeline.ensemble_gpu import get_feature_names
    input_cols = get_feature_names(lookback, expanded)
    X = df_eval[input_cols]

    # 4. Predict
    logger.info(f"Running predictions on {len(X):,} rows...")
    
    # We do a batch prediction using the internal probabilities
    # We can't use ens.predict() directly for batches because it expects single rows
    # We re-implement soft voting for the whole matrix here
    
    all_role_probs = []
    for role, model in ens.models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X) # Should return (N, 3) where columns are [SELL, HOLD, BUY]
            
            # Map probabilities if it's sklearn (where classes might not be consistent)
            if not hasattr(model, "booster"):
                 # Reconstruct (N, 3) order manually if classes are missing
                 # This is handled by _get_probs in the class, but for efficiency:
                 classes = model.classes_
                 full_probs = np.zeros((len(X), 3))
                 for idx, c in enumerate(classes):
                     if c == -1: full_probs[:, 0] = probs[:, idx]
                     elif c == 0: full_probs[:, 1] = probs[:, idx]
                     elif c == 1: full_probs[:, 2] = probs[:, idx]
                 probs = full_probs

            all_role_probs.append(probs)
    
    # Average probabilities
    avg_probs = np.mean(all_role_probs, axis=0) # (N, 3)
    
    # Final decisions: max probability
    # Indices: 0=SELL, 1=HOLD, 2=BUY
    y_pred_idx = np.argmax(avg_probs, axis=1)
    mapping = {0: -1, 1: 0, 2: 1}
    y_pred = np.array([mapping[idx] for idx in y_pred_idx])

    # 5. Output Report
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    report = classification_report(y_true, y_pred, target_names=["SELL", "HOLD", "BUY"], labels=[-1, 0, 1])

    logger.info("\n" + "="*65)
    logger.info("  FTMO CROSS-TEST EVALUATION RESULTS")
    logger.info("="*65)
    logger.info(f"  Model Set      : {args.models}")
    logger.info(f"  Test File      : {args.data}")
    logger.info(f"  Total Samples  : {len(X):,}")
    logger.info(f"  Accuracy       : {acc:.4f}")
    logger.info(f"  Macro F1       : {f1:.4f}")
    logger.info("-" * 65)
    logger.info(f"  Confusion Matrix:\n{cm}")
    logger.info("-" * 65)
    logger.info(f"  Classification Report:\n{report}")
    logger.info("="*65)

if __name__ == "__main__":
    run_evaluation()
