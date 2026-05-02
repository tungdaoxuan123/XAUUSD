#!/usr/bin/env python3
"""
prune_features.py
-----------------
SHAP-based feature importance diagnostic for the GBPUSD LightGBM and PyTorch models.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    pass

try:
    import torch
    try:
        import torch_directml
    except ImportError:
        pass
except ImportError:
    sys.exit("torch not installed. Run: pip install torch")

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
    model_path = Path(model_path)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")
    print(f"Loading model: {model_path}")

    # Detect model type from file extension
    if model_path.suffix == ".pt":
        print("Detected PyTorch model (.pt) — using DeepExplainer")
        device = torch.device("cpu")  # Force CPU for SHAP
        ckpt = torch.load(model_path, map_location=device)
        
        # Import the model class
        script_dir = Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        try:
            from train_sota_v2 import PatchTSTLite
        except ImportError:
            try:
                from sota_signal_generator import PatchTSTLite
            except ImportError:
                sys.exit("Could not import PatchTSTLite from train_sota_v2.py or sota_signal_generator.py")
        
        # Instantiate model using the saved checkpoint config
        model = PatchTSTLite(
            n_features=ckpt["n_features"],
            seq_len=ckpt.get("seq_len", 60),
            patch_len=ckpt.get("patch_len", 12),
            d_model=ckpt.get("d_model", 64),
            n_heads=ckpt.get("n_heads", 4),
            n_layers=ckpt.get("n_layers", 2)
        )
        model.load_state_dict(ckpt["state"])
        model.to(device)
        model.eval()
        
        features = ckpt.get("features", None)
        seq_len = ckpt.get("seq_len", 60)
        is_pytorch = True
    else:
        print("Detected LightGBM model (.txt) — using TreeExplainer")
        model = lgb.Booster(model_file=str(model_path))
        features = model.feature_name()
        is_pytorch = False

    if features is None:
        sys.exit("Could not determine features from model.")
    print(f"Model has {len(features)} features")

    # --- Load validation data -----------------------------------------------
    data_path = Path(data_path)
    if not data_path.exists():
        sys.exit(f"Data not found: {data_path}")
    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = [c.lower() if c != "ATR" else c for c in df.columns]
    if "atr" in df.columns and "ATR" not in df.columns:
        df.rename(columns={"atr": "ATR"}, inplace=True)

    if is_pytorch:
        try:
            from train_sota_v2 import add_session_features
            df = add_session_features(df)
        except Exception as e:
            print(f"Warning: could not add session features: {e}")

    missing = [f for f in features if f not in df.columns]
    if missing:
        sys.exit(f"Missing features in CSV:\n{missing}")

    # --- SHAP values --------------------------------------------------------
    print("Computing SHAP values (this may take 1–2 minutes)...")
    
    if is_pytorch:
        # We need num_samples + seq_len rows to build the sliding windows
        # so the final output has num_samples sequences
        rows_needed = num_samples + seq_len
        X_full = df[features].tail(rows_needed).values
        actual_n = len(X_full) - seq_len
        if actual_n <= 0:
            sys.exit(f"Not enough data to build {seq_len} sliding windows.")
            
        print(f"Using {actual_n:,} validation sequences for SHAP analysis")
        
        # Build sliding windows
        def build_windows(data, length):
            return np.array([data[i:i+length] for i in range(len(data) - length + 1)])
            
        windows = build_windows(X_full, seq_len)
        
        # We need background data (e.g. 20 samples) and validation data
        # DeepExplainer scales linearly with background samples; 100 was still too slow for a Transformer.
        bg_samples = min(20, len(windows))
        # Use num_samples for validation, defaulting to 1000 if user doesn't specify otherwise
        val_samples = min(num_samples, len(windows))
        
        bg_samples = min(2, len(windows))
        val_samples = min(10, len(windows))
        
        bg_tensor = torch.tensor(windows[:bg_samples], dtype=torch.float32).to(device)
        val_tensor = torch.tensor(windows[:val_samples], dtype=torch.float32).to(device)
        
        explainer = shap.DeepExplainer(model, bg_tensor)
        # Disable additivity check because DeepLIFT hooks don't perfectly support LayerNorm/Attention ops
        shap_values = explainer.shap_values(val_tensor, check_additivity=False)
        
        # DeepExplainer returns a list of tensors for classification
        if isinstance(shap_values, list):
            shap_values = [v.detach().cpu().numpy() if torch.is_tensor(v) else v for v in shap_values]
        elif torch.is_tensor(shap_values):
            shap_values = shap_values.detach().cpu().numpy()
            
        # IMPORTANT: DeepExplainer returns varying tensor shapes for sequence classification models
        # (e.g. (batch, seq_len, features, classes), (batch, classes, seq_len, features), or lists of tensors).
        # We must reduce over all non-feature dimensions to get per-feature importances!
        if isinstance(shap_values, list):
            # If it's a list, it's [class0, class1, class2], each of shape (batch, seq_len, features)
            shap_values = np.stack(shap_values, axis=0)  # (classes, batch, seq_len, features)
            
        mean_abs_shap = np.abs(shap_values)
        
        # Find the dimension that corresponds to the number of features
        feat_dim = None
        # We search from the end backwards, because batch/classes might accidentally match len(features)
        for i in reversed(range(len(mean_abs_shap.shape))):
            if mean_abs_shap.shape[i] == len(features):
                feat_dim = i
                break
                
        if feat_dim is None:
            sys.exit(f"Could not find feature dimension in SHAP output. Shape: {mean_abs_shap.shape}, expected features: {len(features)}")
            
        # Mean/Sum over all other axes
        axes_to_reduce = tuple(i for i in range(len(mean_abs_shap.shape)) if i != feat_dim)
        mean_abs_shap = mean_abs_shap.mean(axis=axes_to_reduce)
        print(f"SHAP squashed to 1D feature array: {mean_abs_shap.shape}")

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
        description="SHAP feature importance analysis for GBPUSD models"
    )
    p.add_argument(
        "--model",
        default="train_pipeline/reports/gbpusd/lgb_model.txt",
        help="Path to trained model file (.txt or .pt)",
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
        help="Number of recent rows/sequences to use as validation set (default: 10000)",
    )
    p.add_argument(
        "--prune-pct",
        type=float,
        default=0.20,
        help="Fraction of features to flag for pruning (default: 0.20 = bottom 20%)",
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
