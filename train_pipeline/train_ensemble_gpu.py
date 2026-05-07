#!/usr/bin/env python3
"""
train_ensemble_gpu.py - AMD GPU-accelerated 3-Model LONG-ONLY Binary Ensemble

Uses LightGBM with OpenCL backend for AMD GPUs on Windows (RX 6700 XT).
Falls back to CPU LightGBM cleanly if GPU is unavailable.

GPU Backend Priority:
    1. LightGBM with device_type='gpu' (OpenCL — works on AMD on Windows)
    2. LightGBM with device_type='cpu' (CPU fallback)
    3. sklearn RandomForest (last resort CPU fallback)

Label Encoding (binary):
    Labels are 0 = WAIT / 1 = LONG from triple-barrier or build_labels().
    LightGBM binary classification: objective='binary'.

3-Model Architecture:
    1. Trend Model     -> LightGBM (directional bias, MACD/VWAP features)
    2. Structure Model -> LightGBM (nonlinear patterns, different depth/leaves)
    3. Regime Model    -> LightGBM (volatility/regime detection, ATR/BB features)

Usage:
    python train_pipeline/train_ensemble_gpu.py         --data train_pipeline/data/xauusd_m1_tb.csv         --label-col tb_label         --expanded-features         --use-gpu         --gpu-backend auto         --out-dir train_pipeline/models_gpu

    # Simple binary labels with default thresholds:
    python train_pipeline/train_ensemble_gpu.py         --data train_pipeline/data/xauusd_m1.csv         --horizon 5         --long-threshold 0.0005         --expanded-features         --use-gpu         --out-dir train_pipeline/models_gpu
"""

import argparse
import json
import logging
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import guard for LightGBM
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Sklearn fallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
Path("train_pipeline/reports").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "reports" / "training_gpu.log", mode="w"),
    ],
)
logger = logging.getLogger("TrainGPU")

# ==========================================================================='
# LABEL ENCODING (binary: 0=WAIT, 1=LONG)
# ==========================================================================='

LABEL_NAMES = ["WAIT", "LONG"]


def encode_labels(y: pd.Series) -> np.ndarray:
    return y.values.astype(int)


# ==========================================================================='
# GPU DETECTION
# ==========================================================================='


def detect_gpu_backend(requested_backend: str, use_gpu: bool) -> tuple:
    """Detect available GPU backend. Returns (backend_name, device_type, gpu_active)."""
    if not use_gpu:
        logger.info("GPU disabled by CLI flag. Using CPU.")
        return "cpu", "cpu", False

    if not LIGHTGBM_AVAILABLE:
        logger.warning("LightGBM not installed. Falling back to CPU sklearn.")
        return "sklearn_cpu", "cpu", False

    # Try LightGBM GPU
    if requested_backend in ("lightgbm", "auto"):
        try:
            probe_data = lgb.Dataset(
                np.random.randn(100, 5).astype(np.float32),
                label=np.random.randint(0, 2, 100)
            )
            probe_params = {
                "objective": "binary",
                "device_type": "gpu",
                "verbose": -1,
                "num_leaves": 4,
            }
            lgb.train(probe_params, probe_data, num_boost_round=2)
            logger.info("LightGBM GPU (OpenCL) detected and working. Using GPU.")
            return "lightgbm", "gpu", True
        except Exception as e:
            logger.warning(f"LightGBM GPU probe failed: {e}")
            logger.warning("Falling back to LightGBM CPU.")
            return "lightgbm", "cpu", False

    # XGBoost AMD GPU path (optional)
    if requested_backend == "xgboost":
        try:
            import xgboost as xgb  # noqa: F401
            logger.warning("XGBoost AMD GPU (ROCm) requested. Verify driver manually.")
            return "xgboost", "gpu", True
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            logger.warning("Falling back to LightGBM CPU.")

    logger.info("No GPU backend active. Running LightGBM on CPU.")
    return "lightgbm", "cpu", False


# ==========================================================================='
# DATA LOADING (reused from train_ensemble.py)
# ==========================================================================='

REQUIRED_COLUMNS = {"time", "open", "high", "low", "close", "tick_volume"}


def load_data(csv_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from: {csv_path}")
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        sys.exit(1)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} rows | {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    return df


# ==========================================================================='
# TECHNICAL INDICATORS (exact mirror of live_ensemble_trading.py)
# ==========================================================================='


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of LiveEnsembleTrader.add_technical_indicators()."""
    df = df.copy()
    close = df["close"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = close.ewm(span=8, adjust=False).mean()
    exp2 = close.ewm(span=24, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]

    df["SMA20"] = close.rolling(20).mean()
    df["STD20"] = close.rolling(20).std()
    df["Upper_Band"] = df["SMA20"] + df["STD20"] * 2
    df["Lower_Band"] = df["SMA20"] - df["STD20"] * 2
    df["BB_width"] = df["Upper_Band"] - df["Lower_Band"]

    low_min = df["low"].rolling(14).min()
    high_max = df["high"].rolling(14).max()
    df["%K"] = 100 * ((close - low_min) / (high_max - low_min))
    df["%D"] = df["%K"].rolling(3).mean()

    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["tick_volume"].replace(0, np.nan).ffill().fillna(1)
    df["TPV"] = tp * vol
    df["Cum_TPV"] = df["TPV"].cumsum()
    df["Cum_Volume"] = vol.cumsum()
    df["VWAP"] = df["Cum_TPV"] / df["Cum_Volume"].replace(0, np.nan)
    df["close_minus_vwap"] = close - df["VWAP"]

    hl = df["high"] - df["low"]
    hc = (df["high"] - close.shift()).abs()
    lc = (df["low"] - close.shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df = df.replace([np.inf, -np.inf], np.nan)
    # Only drop rows where the indicators we just computed are NaN (first 20-30 bars)
    # This prevents dropping thousands of bars if microstructure data is partially missing.
    indicator_cols = ["RSI", "MACD", "Signal_Line", "ATR", "BB_width", "VWAP"]
    df = df.dropna(subset=[c for c in indicator_cols if c in df.columns]).reset_index(drop=True)

    # ---- Regime / session features (Problem 2) ----
    # ATR ratio: short-term vs long-term volatility
    atr_100 = pd.concat([df["high"] - df["low"],
                         (df["high"] - df["close"].shift()).abs(),
                         (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1).rolling(100).mean()
    df["atr_ratio"] = (df["ATR"] / (atr_100 + 1e-9)).clip(0, 10)

    # ADX
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    plus_dm = (df["high"] - df["high"].shift()).clip(lower=0)
    minus_dm = (df["low"].shift() - df["low"]).clip(lower=0)
    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    df["adx_14"] = dx.rolling(14).mean().fillna(0)

    # Session / time features
    if "time" in df.columns:
        dt_idx = pd.to_datetime(df["time"])
        hour = dt_idx.dt.hour
    else:
        hour = pd.Series(0, index=df.index)
    df["is_london"]  = ((hour >= 7)  & (hour < 16)).astype(int)
    df["is_ny"]      = ((hour >= 13) & (hour < 21)).astype(int)
    df["is_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)
    df["hour_sin"]   = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hour / 24)
    df["dow"]        = pd.to_datetime(df["time"]).dt.dayofweek if "time" in df.columns else 0

    return df


# ==========================================================================='
# FEATURES & PROCESSED DATA
# ==========================================================================='

def get_feature_names(lookback: int, expanded: bool = False, micro: bool = False) -> list:
    """Generate dynamic feature names based on lookback window."""
    base = [f"close_lag_{i}" for i in range(lookback - 1, -1, -1)]
    common = base + ["RSI", "MACD", "Signal_Line", "current_position", "current_balance"]
    if expanded:
        common += ["MACD_Hist", "VWAP", "close_minus_vwap", "ATR", "BB_width"]
        common += ["atr_ratio", "adx_14", "is_london", "is_ny", "is_overlap",
                   "hour_sin", "hour_cos", "dow"]
    if micro:
        common += [
            "tick_imbalance", "bid_ask_vol_imbalance", "spread_mean", "ofi_window", "of_pressure_flag",
            "vprof_poc_dist", "vprof_in_value_area", "vprof_hvn_flag", "vprof_lvn_flag"
        ]
    return common


def build_features(df: pd.DataFrame, lookback: int = 10, expanded: bool = False, micro: bool = False) -> pd.DataFrame:
    """Create feature matrix from indicators with variable price lags."""
    # List of micro features to extract if present
    micro_cols = [
        "tick_imbalance", "bid_ask_vol_imbalance", "spread_mean", "ofi_window", "of_pressure_flag",
        "vprof_poc_dist", "vprof_in_value_area", "vprof_hvn_flag", "vprof_lvn_flag"
    ]
    
    # Fill NAs for micro features with median + clip (Problem 1)
    MICRO_COLS = ["tick_imbalance", "ofi_window", "cs_spread", "kyle_lambda",
                  "vprof_poc_dist", "spread_mean", "bid_ask_vol_imbalance",
                  "of_pressure_flag", "vprof_in_value_area", "vprof_hvn_flag", "vprof_lvn_flag"]
    if micro:
        for col in micro_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().sum() > 0 else 0)
                if df[col].notna().sum() > 10:
                    lo = df[col].quantile(0.01)
                    hi = df[col].quantile(0.99)
                    if lo < hi:
                        df[col] = df[col].clip(lo, hi)
            else:
                logger.warning(f"Micro feature column {col} missing from data!")
                df[col] = 0.0

    rows = []
    n = len(df)
    
    # We start from lookback-1 to have enough data for the first lag set
    for i in range(lookback - 1, n):
        price_lags = df["close"].iloc[i - lookback + 1: i + 1].values
        latest = df.iloc[i]
        
        row = list(price_lags) + [
            float(latest["RSI"]),
            float(latest["MACD"]),
            float(latest["Signal_Line"]),
            0.0,      # current_position placeholder
            10000.0,  # current_balance placeholder
        ]
        
        if expanded:
            row += [
                float(latest["MACD_Hist"]),
                float(latest["VWAP"]),
                float(latest["close_minus_vwap"]),
                float(latest["ATR"]),
                float(latest["BB_width"]),
                float(latest.get("atr_ratio", 0)),
                float(latest.get("adx_14", 0)),
                int(latest.get("is_london", 0)),
                int(latest.get("is_ny", 0)),
                int(latest.get("is_overlap", 0)),
                float(latest.get("hour_sin", 0)),
                float(latest.get("hour_cos", 0)),
                int(latest.get("dow", 0)),
            ]
            
        if micro:
            for col in micro_cols:
                row.append(float(latest[col]))
                
        rows.append(row)

    cols = get_feature_names(lookback, expanded, micro)
    return pd.DataFrame(rows, columns=cols)


def build_labels(df, lookback=10, horizon=5, long_threshold=0.0005) -> pd.Series:
    """Sync labels with the lookback shift in build_features. Binary: 0=WAIT, 1=LONG."""
    close = df["close"].values
    n = len(close)
    labels = []
    
    # Must match the range(lookback-1, n) loop in build_features
    for i in range(lookback - 1, n):
        fi = i + horizon
        if fi >= n:
            labels.append(np.nan)
        else:
            ret = (close[fi] - close[i]) / close[i]
            labels.append(1 if ret > long_threshold else 0)
    return pd.Series(labels, name="label")


# ==========================================================================='
# CLASS WEIGHTING FOR IMBALANCED MULTICLASS LABELS
# ==========================================================================='


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    """Compute per-sample weights inversely proportional to class frequency."""
    counts = Counter(y.tolist())
    total = sum(counts.values())
    num_classes = len(counts)
    class_weight = {cls: total / (num_classes * cnt) for cls, cnt in counts.items()}
    logger.info(f"Computed class_weight (orig labels): {class_weight}")
    return np.array([class_weight[v] for v in y])


# ==========================================================================='
# LIGHTGBM MODEL DEFINITIONS (3 roles)
# ==========================================================================='


def get_lgbm_params(role: str, device_type: str) -> dict:
    """Each model has purpose-tuned hyperparameters. Binary classification."""
    base = {
        "objective": "binary",
        "device_type": device_type,
        "verbose": -1,
        "random_state": 42,
        "max_bin": 63,           # CRITICAL for AMD OpenCL
        "min_child_samples": 100,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    if role == "trend":
        return {**base,
            "n_estimators": 500,
            "num_leaves": 63,
            "learning_rate": 0.05,
        }
    elif role == "structure":
        return {**base,
            "n_estimators": 500,
            "num_leaves": 63,
            "learning_rate": 0.03,
        }
    elif role == "regime":
        return {**base,
            "n_estimators": 500,
            "num_leaves": 63,
            "learning_rate": 0.02,
        }
    return base


ENSEMBLE_ROLES = {
    "trend":     "Trend Model (LightGBM - directional)",
    "structure": "Structure Model (LightGBM - nonlinear patterns)",
    "regime":    "Regime Model (LightGBM - volatility/regime)",
}


# ==========================================================================='
# SKLEARN FALLBACK PIPELINES
# ==========================================================================='


def get_sklearn_fallback(role: str) -> Pipeline:
    if role == "trend":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
    elif role == "structure":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=42
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# ==========================================================================='
# WALK-FORWARD TRAINING
# ==========================================================================='


def walk_forward_lgbm(
    X: pd.DataFrame, y: pd.Series, role: str, device_type: str, n_splits: int = 5
) -> tuple:
    """Walk-forward training for LightGBM (binary). Returns (fitted_model, summary)."""
    label = ENSEMBLE_ROLES[role]
    logger.info(f"\n{'='*65}\n  Training [{role}]: {label} | device={device_type}\n{'='*65}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    last_model = None
    y_enc = y.values.astype(int)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train = X.iloc[train_idx].astype(np.float32).values
        X_val   = X.iloc[val_idx].astype(np.float32).values
        y_train_enc, y_val_enc = y_enc[train_idx], y_enc[val_idx]

        # weights based on class imbalance
        y_train_orig = y.iloc[train_idx].reset_index(drop=True)
        sample_weight = compute_sample_weights(y_train_orig)

        params = get_lgbm_params(role, device_type)
        n_est = params.pop("n_estimators", 500)

        # Per-fold scale_pos_weight (Problem 5)
        n_neg = (y_train_enc == 0).sum()
        n_pos = max((y_train_enc == 1).sum(), 1)
        params["scale_pos_weight"] = n_neg / n_pos

        train_data = lgb.Dataset(X_train, label=y_train_enc, weight=sample_weight)
        val_data = lgb.Dataset(X_val, label=y_val_enc, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_est,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )

        proba = model.predict(X_val)  # shape (n_val,) for binary
        y_pred = (proba >= 0.5).astype(int)
        y_val_orig = y_val_enc

        acc = accuracy_score(y_val_orig, y_pred)
        f1 = f1_score(y_val_orig, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_val_orig, y_pred, labels=[0, 1])
        report = classification_report(
            y_val_orig, y_pred, labels=[0, 1],
            target_names=LABEL_NAMES, zero_division=0
        )

        logger.info(
            f"\n  [{role}] Fold {fold + 1}/{n_splits} | "
            f"Train: {len(X_train):,}  Val: {len(X_val):,}\n"
            f"  Accuracy: {acc:.4f}  Binary F1: {f1:.4f}\n"
            f"  Confusion Matrix:\n{cm}\n{report}"
        )

        fold_metrics.append({"fold": fold + 1, "accuracy": acc, "binary_f1": f1})
        last_model = model

    acc_mean = np.mean([m["accuracy"] for m in fold_metrics])
    f1_mean = np.mean([m["binary_f1"] for m in fold_metrics])
    logger.info(f"  [{role}] AGGREGATE -> Mean Acc: {acc_mean:.4f} | Mean F1: {f1_mean:.4f}")

    summary = {
        "model": role,
        "label": label,
        "backend": "lightgbm",
        "device_type": device_type,
        "micro": any("tick" in str(f) for f in X.columns),
        "n_splits": n_splits,
        "mean_accuracy": acc_mean,
        "mean_binary_f1": f1_mean,
        "fold_detail": fold_metrics,
    }
    return last_model, summary


def walk_forward_sklearn(
    X: pd.DataFrame, y: pd.Series, role: str, n_splits: int = 5
) -> tuple:
    """Walk-forward training for sklearn fallback. Returns (fitted_pipeline, summary)."""
    logger.info(f"\n  [{role}] Using sklearn CPU fallback")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    last_pipeline = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline = get_sklearn_fallback(role)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="binary", zero_division=0)
        logger.info(f"  [{role}] Fold {fold+1} | Acc: {acc:.4f} | F1: {f1:.4f}")
        fold_metrics.append({"fold": fold + 1, "accuracy": acc, "binary_f1": f1})
        last_pipeline = pipeline

    summary = {
        "model": role,
        "backend": "sklearn_cpu",
        "n_splits": n_splits,
        "mean_accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
        "mean_binary_f1": np.mean([m["binary_f1"] for m in fold_metrics]),
        "fold_detail": fold_metrics,
    }
    return last_pipeline, summary


# ==========================================================================='
# SAVE ARTIFACTS
# ==========================================================================='


def save_lgbm_model(
    model,
    feature_names: list,
    summary: dict,
    role: str,
    out_dir: str,
    expanded: bool,
    horizon: int,
    long_thr: float,
    device_type: str,
    lookback: int,
):
    os.makedirs(out_dir, exist_ok=True)
    suffix = "expanded" if expanded else "default"
    tag = f"{role}_{suffix}"

    model_path = os.path.join(out_dir, f"lgbm_{tag}.txt")
    features_path = os.path.join(out_dir, f"features_{tag}.json")
    summary_path = os.path.join(out_dir, f"summary_{tag}.json")

    model.save_model(model_path)

    with open(features_path, "w") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "feature_count": len(feature_names),
                "lookback": lookback,
                "expanded_features": expanded,
                "microstructure_features": summary.get("micro", False),
                "label_horizon": horizon,
                "long_threshold": long_thr,
                "classification": "binary",
                "class_names": LABEL_NAMES,
                "backend": "lightgbm",
                "device_type": device_type,
            },
            f,
            indent=2,
        )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"  Saved model    : {model_path}")
    logger.info(f"  Saved features : {features_path}")
    logger.info(f"  Saved summary  : {summary_path}")
    return model_path


def save_sklearn_model(
    pipeline,
    feature_names: list,
    summary: dict,
    role: str,
    out_dir: str,
    expanded: bool,
    horizon: int,
    long_thr: float,
    lookback: int,
):
    import joblib

    os.makedirs(out_dir, exist_ok=True)
    suffix = "expanded" if expanded else "default"
    tag = f"{role}_{suffix}"

    model_path = os.path.join(out_dir, f"sklearn_{role}_{suffix}.joblib")
    joblib.dump(pipeline, model_path)

    features_path = os.path.join(out_dir, f"features_{tag}.json")
    with open(features_path, "w") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "feature_count": len(feature_names),
                "lookback": lookback,
                "expanded_features": expanded,
                "microstructure_features": summary.get("micro", False),
                "label_horizon": horizon,
                "long_threshold": long_thr,
                "classification": "binary",
                "class_names": LABEL_NAMES,
                "backend": "sklearn_cpu",
            },
            f,
            indent=2,
        )

    summary_path = os.path.join(out_dir, f"summary_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"  Saved sklearn fallback : {model_path}")
    return model_path


def save_ensemble_metadata(
    out_dir: str,
    model_paths: dict,
    feature_names: list,
    expanded: bool,
    gpu_active: bool,
    backend: str,
    horizon: int,
    long_thr: float,
    lookback: int,
):
    meta_path = os.path.join(out_dir, "ensemble_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "backend": backend,
                "gpu_used": gpu_active,
                "model_files": model_paths,
                "model_names": list(model_paths.keys()),
                "feature_names": feature_names,
                "feature_count": len(feature_names),
                "lookback": lookback,
                "expanded_features": expanded,
                "microstructure_features": "micro" in out_dir.lower() or any("vprof" in f for f in feature_names),
                "label_horizon": horizon,
                "long_threshold": long_thr,
                "classification": "binary",
                "class_names": LABEL_NAMES,
            },
            f,
            indent=2,
        )
    logger.info(f"Ensemble metadata saved: {meta_path}")


# ==========================================================================='
# CLI
# ==========================================================================='


def parse_args():
    parser = argparse.ArgumentParser(
        description="AMD GPU 3-Model LONG-ONLY Binary Ensemble Training — LightGBM OpenCL with CPU fallback"
    )
    parser.add_argument("--data",              required=True)
    parser.add_argument("--horizon",           type=int,   default=5)
    parser.add_argument("--lookback",          type=int,   default=10)
    parser.add_argument("--long-threshold",    type=float, default=0.0005)
    parser.add_argument("--label-col",         type=str,   default=None, help="Use existing label column from CSV")
    parser.add_argument("--expanded-features", action="store_true")
    parser.add_argument("--microstructure-features", action="store_true", help="Include microstructure columns if present in data")
    parser.add_argument("--use-gpu",           action="store_true")
    parser.add_argument(
        "--gpu-backend", type=str, default="auto", choices=["lightgbm", "xgboost", "auto"]
    )
    parser.add_argument("--n-splits",          type=int,   default=5)
    parser.add_argument("--out-dir",           type=str,   default="train_pipeline/models_gpu")
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = "expanded" if args.expanded_features else "default"

    logger.info("=" * 65)
    logger.info("  XAUUSD GPU LONG-ONLY Binary Ensemble Training (LightGBM + AMD OpenCL)")
    logger.info("=" * 65)
    logger.info(f"  Data          : {args.data}")
    if args.label_col:
        logger.info(f"  Label Column  : {args.label_col}")
    else:
        logger.info(f"  Horizon       : {args.horizon} bars")
        logger.info(f"  Long threshold: {args.long_threshold:.4%}")
    logger.info(f"  Lookback      : {args.lookback} bars")
    logger.info(f"  Features      : {suffix}")
    logger.info(f"  GPU requested : {args.use_gpu}")
    logger.info(f"  GPU backend   : {args.gpu_backend}")
    logger.info(f"  Folds         : {args.n_splits}")
    logger.info(f"  Output dir    : {args.out_dir}")
    logger.info("=" * 65)

    # Detect GPU
    backend, device_type, gpu_active = detect_gpu_backend(args.gpu_backend, args.use_gpu)
    logger.info(f"GPU active: {gpu_active} | Backend: {backend} | Device: {device_type}")

    # Load & process data
    df_raw = load_data(args.data)
    logger.info("Computing technical indicators...")
    df = add_technical_indicators(df_raw)
    logger.info(f"After indicators: {len(df):,} rows")

    logger.info("Building feature matrix...")
    X = build_features(df, lookback=args.lookback, expanded=args.expanded_features, micro=args.microstructure_features)
    feature_names = list(X.columns)
    logger.info(f"Features: {len(feature_names)} | {feature_names[:15]}...")

    if args.label_col:
        logger.info(f"Using existing labels from column: {args.label_col}")
        y = df[args.label_col].iloc[args.lookback - 1:].reset_index(drop=True)
    else:
        logger.info("Building labels...")
        y = build_labels(df, lookback=args.lookback, horizon=args.horizon, long_threshold=args.long_threshold)
    
    valid_mask = y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True).astype(int)
    logger.info(f"Dataset: {len(X):,} rows | Labels: {y.value_counts().sort_index().to_dict()}")

    # Train all 3 models
    model_paths = {}
    for role in ["trend", "structure", "regime"]:
        if backend == "lightgbm" and LIGHTGBM_AVAILABLE:
            model, summary = walk_forward_lgbm(X, y, role, device_type, args.n_splits)
            logger.info(f"\nRefitting [{role}] on full dataset...")
            y_enc = y.values.astype(int)
            full_weight = compute_sample_weights(y)
            params = get_lgbm_params(role, device_type)
            n_neg_full = (y_enc == 0).sum()
            n_pos_full = max((y_enc == 1).sum(), 1)
            params["scale_pos_weight"] = n_neg_full / n_pos_full
            n_est = params.pop("n_estimators", 500)
            full_data = lgb.Dataset(X.astype(np.float32).values, label=y_enc, weight=full_weight)
            model = lgb.train(params, full_data, num_boost_round=n_est)
            path = save_lgbm_model(
                model,
                feature_names,
                summary,
                role,
                args.out_dir,
                args.expanded_features,
                args.horizon,
                args.long_threshold,
                device_type,
                args.lookback,
            )
        else:
            pipeline, summary = walk_forward_sklearn(X, y, role, args.n_splits)
            pipeline.fit(X, y)
            path = save_sklearn_model(
                pipeline,
                feature_names,
                summary,
                role,
                args.out_dir,
                args.expanded_features,
                args.horizon,
                args.long_threshold,
                args.lookback,
            )

        model_paths[role] = path

    # Save ensemble metadata
    save_ensemble_metadata(
        args.out_dir,
        model_paths,
        feature_names,
        args.expanded_features,
        gpu_active,
        backend,
        args.horizon,
        args.long_threshold,
        args.lookback,
    )

    logger.info("\n" + "=" * 65)
    logger.info("  ALL 3 MODELS TRAINED AND SAVED")
    logger.info("=" * 65)
    for role, path in model_paths.items():
        logger.info(f"  {role:12s} -> {path}")
    logger.info("\nLoad in live bot:")
    logger.info("  from train_pipeline.ensemble_gpu import EnsembleGPU")
    logger.info(f"  ens = EnsembleGPU.load('{args.out_dir}')")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()