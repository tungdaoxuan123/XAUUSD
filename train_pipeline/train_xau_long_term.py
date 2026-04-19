#!/usr/bin/env python3
"""
train_xau_long_term.py - High-Performance Training for 13-Year XAUUSD Dataset

This script trains a 3-model LightGBM ensemble on the massive cleaned dataset
from HuggingFace (2011-2024). It uses exactly the same algorithm as the main
GPU trainer.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import lightgbm as lgb
    from train_pipeline.train_ensemble_gpu import (
        load_data, add_technical_indicators, build_features, 
        build_labels, compute_sample_weights, walk_forward_lgbm,
        save_lgbm_model, save_ensemble_metadata, get_lgbm_params,
        encode_labels, LABEL_MAP
    )
except ImportError:
    import lightgbm as lgb
    from train_ensemble_gpu import (
        load_data, add_technical_indicators, build_features, 
        build_labels, compute_sample_weights, walk_forward_lgbm,
        save_lgbm_model, save_ensemble_metadata, get_lgbm_params,
        encode_labels, LABEL_MAP
    )

# --- Configuration ---
DATA_PATH = "train_pipeline/data/xauusd_m1.csv"
OUT_DIR = "train_pipeline/models_xau_long_term"
HORIZON = 5
LOOKBACK = 10
BUY_THR = 0.0005
SELL_THR = 0.0005
USE_GPU = True
EXPANDED_FEATURES = True
N_SPLITS = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainXAU_LongTerm")

def main():
    logger.info("--- Starting Long-Term XAUUSD Ensemble Training ---")
    
    # 1. Load and Preprocess
    df_raw = load_data(DATA_PATH)
    df = add_technical_indicators(df_raw)
    
    # 2. Build Features
    logger.info("Building feature matrix...")
    X = build_features(df, lookback=LOOKBACK, expanded=EXPANDED_FEATURES, micro=False)
    feature_names = list(X.columns)
    
    # 3. Build Labels
    logger.info(f"Building labels (Horizon: {HORIZON})...")
    y = build_labels(df, lookback=LOOKBACK, horizon=HORIZON, buy_threshold=BUY_THR, sell_threshold=SELL_THR)
    
    # Align X and y
    valid_mask = y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True).astype(int)
    
    logger.info(f"Final training set size: {len(X):,} rows")
    
    device_type = "gpu" if USE_GPU else "cpu"
    model_paths = {}

    # 4. Train each role
    for role in ["trend", "structure", "regime"]:
        logger.info(f"\nTraining [{role}] model...")
        
        # Cross-validation
        model, summary = walk_forward_lgbm(X, y, role, device_type, N_SPLITS)
        
        # Refit on full dataset
        logger.info(f"Refitting [{role}] on full dataset...")
        y_enc = encode_labels(y)
        full_weight = compute_sample_weights(y)
        params = get_lgbm_params(role, device_type)
        n_est = params.pop("n_estimators", 500)
        
        full_data = lgb.Dataset(X.astype(np.float32).values, label=y_enc, weight=full_weight)
        model = lgb.train(params, full_data, num_boost_round=n_est)
        
        # Save
        path = save_lgbm_model(
            model, feature_names, summary, role, OUT_DIR, 
            EXPANDED_FEATURES, HORIZON, BUY_THR, SELL_THR, 
            device_type, LOOKBACK
        )
        model_paths[role] = path

    # 5. Metadata
    save_ensemble_metadata(
        OUT_DIR, model_paths, feature_names, EXPANDED_FEATURES, 
        USE_GPU, "lightgbm", HORIZON, BUY_THR, SELL_THR, LOOKBACK
    )

    logger.info(f"\nSUCCESS: Fully trained 3-model ensemble saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
