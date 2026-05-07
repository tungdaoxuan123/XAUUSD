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
# MODEL CONFIGS — each role uses different features & hyperparams (1d)
# ==========================================================================='

MODEL_CONFIGS = {
    "trend": {
        "features": ["return_lag_0", "return_lag_1", "return_lag_2", "return_lag_3",
                     "return_lag_4", "return_lag_5", "return_lag_6", "return_lag_7",
                     "return_lag_8", "return_lag_9",
                     "RSI", "MACD", "MACD_Hist", "Signal_Line",
                     "atr_norm", "ema_ratio"],
        "num_leaves": 63, "max_depth": 6, "min_child_samples": 300,
        "scale_pos_weight": 1.3, "lambda_l1": 0.1, "lambda_l2": 0.1,
    },
    "structure": {
        "features": ["ofi_window", "tick_imbalance", "cs_spread",
                     "kyle_lambda", "vprof_poc_dist", "amihud"],
        "num_leaves": 127, "max_depth": 8, "min_child_samples": 100,
        "scale_pos_weight": 1.3, "lambda_l1": 0.05, "lambda_l2": 0.05,
    },
    "regime": {
        "features": ["atr_pct", "amihud", "jump_flag",
                     "regime_flag", "vol_zscore"],
        "num_leaves": 31, "max_depth": 4, "min_child_samples": 500,
        "scale_pos_weight": 1.2, "lambda_l1": 0.2, "lambda_l2": 0.2,
    }
}

LABEL_NAMES = ["WAIT", "LONG"]


def encode_labels(y: pd.Series) -> np.ndarray:
    return y.values.astype(int)


# All synmicro columns from synthetic_microstructure.py (1c)
MICRO_COLS = [
    "tick_imbalance", "ofi_window", "cs_spread",
    "kyle_lambda", "vprof_poc_dist", "amihud",
    "jump_flag", "regime_flag",
]


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

    # Derived features for regime/structure models (1d)
    df["atr_norm"] = df["ATR"] / (close + 1e-9)                    # ATR as % of price
    df["atr_pct"]  = df["ATR"].pct_change(20).fillna(0)             # ATR trend
    df["ema_ratio"] = (close.ewm(span=5, adjust=False).mean() /
                       close.ewm(span=20, adjust=False).mean()).fillna(1)  # short/long EMA
    df["vol_zscore"] = ((df["ATR"] - df["ATR"].rolling(100).mean()) /
                        (df["ATR"].rolling(100).std() + 1e-9)).fillna(0)  # vol regime z-score

    df = df.replace([np.inf, -np.inf], np.nan)
    indicator_cols = ["RSI", "MACD", "Signal_Line", "ATR", "BB_width", "VWAP"]
    df = df.dropna(subset=[c for c in indicator_cols if c in df.columns]).reset_index(drop=True)
    return df


# ==========================================================================='
# FEATURES & PROCESSED DATA
# ==========================================================================='

def get_feature_names_for_role(role: str) -> list:
    """Return the feature list for a specific model role (1d)."""
    return list(MODEL_CONFIGS[role]["features"])


def get_all_feature_names() -> list:
    """Union of all feature names across all roles."""
    seen = set()
    out = []
    for role in ["trend", "structure", "regime"]:
        for f in MODEL_CONFIGS[role]["features"]:
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


def build_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Create feature matrix with ATR-normalized return lags (1b) + synmicro columns (1c).

    Removes current_position / current_balance (1a).
    """
    close = df["close"].values
    atr   = df["ATR"].values
    n     = len(df)

    # Fill NaN in synmicro columns from MICRO_COLS (1c)
    for col in MICRO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0.0

    rows = []
    for i in range(lookback, n):
        latest = df.iloc[i]

        # Return lags: normalized by ATR (1b)
        ret_lags = []
        for k in range(lookback):
            lag_i = i - k
            if lag_i < 1:
                ret_lags.append(0.0)
            else:
                atr_prev = atr[lag_i - 1] if np.isfinite(atr[lag_i - 1]) and atr[lag_i - 1] > 0 else 0.01
                ret_lags.append(np.clip((close[lag_i] - close[lag_i - 1]) / atr_prev, -10, 10))

        row = list(ret_lags) + [
            float(latest["RSI"]),
            float(latest["MACD"]),
            float(latest["MACD_Hist"]),
            float(latest["Signal_Line"]),
            float(latest.get("atr_norm", 0)),
            float(latest.get("ema_ratio", 1)),
            float(latest.get("atr_pct", 0)),
            float(latest.get("vol_zscore", 0)),
        ]

        # Synmicro columns — always appended (1c)
        for col in MICRO_COLS:
            row.append(float(latest[col]))

        rows.append(row)

    all_names = (["return_lag_{}".format(i) for i in range(lookback)]
                 + ["RSI", "MACD", "MACD_Hist", "Signal_Line",
                    "atr_norm", "ema_ratio", "atr_pct", "vol_zscore"]
                 + MICRO_COLS)
    return pd.DataFrame(rows, columns=all_names)


def build_labels(df, lookback=60, horizon=5, long_threshold=0.0005) -> pd.Series:
    """Sync labels with the lookback shift in build_features. Binary: 0=WAIT, 1=LONG."""
    close = df["close"].values
    n = len(close)
    labels = []

    # Must match range(lookback, n) in build_features
    for i in range(lookback, n):
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
    """Hyperparameters from MODEL_CONFIGS (1d)."""
    cfg = MODEL_CONFIGS[role]
    return {
        "objective": "binary",
        "device_type": device_type,
        "verbose": -1,
        "random_state": 42,
        "max_bin": 63,
        "n_estimators": 500,
        "num_leaves": cfg["num_leaves"],
        "max_depth": cfg["max_depth"],
        "min_child_samples": cfg["min_child_samples"],
        "scale_pos_weight": cfg["scale_pos_weight"],
        "lambda_l1": cfg["lambda_l1"],
        "lambda_l2": cfg["lambda_l2"],
    }


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
    X: pd.DataFrame, y: pd.Series, role: str, device_type: str, n_splits: int = 5,
    sample_weight: pd.Series = None,
) -> tuple:
    """Walk-forward training for LightGBM (binary). Uses per-role feature subset from MODEL_CONFIGS."""
    label = ENSEMBLE_ROLES[role]
    role_features = MODEL_CONFIGS[role]["features"]
    available = [f for f in role_features if f in X.columns]
    missing = [f for f in role_features if f not in X.columns]
    if missing:
        logger.warning(f"[{role}] Missing features: {missing}")
    X_role = X[available]
    n_zero_filled = 0
    for f in available:
        vals = X_role[f].values if hasattr(X_role[f], "values") else X_role[f]
        if isinstance(vals, np.ndarray) and np.all(np.abs(vals) < 1e-9):
            n_zero_filled += 1
    if n_zero_filled > len(available) * 0.5 and len(available) > 0:
        logger.error(
            f"[{role}] {n_zero_filled}/{len(available)} features are zero-filled! "
            f"Run synthetic_microstructure.py first."
        )
    logger.info(f"\n{'='*65}\n  Training [{role}]: {label} | device={device_type}")
    logger.info(f"  Features ({len(available)}): {available}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    last_model = None
    y_enc = y.values.astype(int)
    sw = sample_weight.values.astype("float32") if sample_weight is not None else None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_role)):
        X_train = X_role.iloc[train_idx].astype(np.float32).values
        X_val   = X_role.iloc[val_idx].astype(np.float32).values
        y_train_enc, y_val_enc = y_enc[train_idx], y_enc[val_idx]

        params = get_lgbm_params(role, device_type)
        n_est = params.pop("n_estimators", 500)

        if sw is not None:
            train_data = lgb.Dataset(X_train, label=y_train_enc, weight=sw[train_idx])
        else:
            train_data = lgb.Dataset(X_train, label=y_train_enc)
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
    role: str,
    out_dir: str,
    horizon: int,
    long_thr: float,
    device_type: str,
    lookback: int,
    summary: dict = None,
):
    os.makedirs(out_dir, exist_ok=True)
    tag = role
    role_features = [f for f in MODEL_CONFIGS[role]["features"] if len(f) > 0]

    model_path = os.path.join(out_dir, f"lgbm_{tag}.txt")
    features_path = os.path.join(out_dir, f"features_{tag}.json")
    summary_path = os.path.join(out_dir, f"summary_{tag}.json")

    model.save_model(model_path)

    with open(features_path, "w") as f:
        json.dump(
            {
                "feature_names": role_features,
                "feature_count": len(role_features),
                "lookback": lookback,
                "label_horizon": horizon,
                "long_threshold": long_thr,
                "classification": "binary",
                "class_names": LABEL_NAMES,
                "backend": "lightgbm",
                "device_type": device_type,
                "role": role,
            },
            f,
            indent=2,
        )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"  Saved model    : {model_path}")
    logger.info(f"  Saved features : {features_path}")
    if summary:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
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
    gpu_active: bool,
    backend: str,
    horizon: int,
    long_thr: float,
    lookback: int,
    side: str = "long",
):
    meta_path = os.path.join(out_dir, "ensemble_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "backend": backend,
                "gpu_used": gpu_active,
                "model_files": model_paths,
                "model_names": list(model_paths.keys()),
                "lookback": lookback,
                "label_horizon": horizon,
                "long_threshold": long_thr,
                "classification": "binary",
                "side": side,
                "class_names": LABEL_NAMES,
                "model_configs": {r: {k: v for k, v in MODEL_CONFIGS[r].items() if k != "features"}
                                  for r in model_paths},
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
    parser.add_argument("--lookback",          type=int,   default=60)
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
    parser.add_argument("--side", type=str, default="long", choices=["long", "short"],
                        help="Trade direction — saved in metadata for bot dispatcher")
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
    X = build_features(df, lookback=args.lookback)
    all_feature_names = get_all_feature_names()
    logger.info(f"All available features ({len(all_feature_names)}): {all_feature_names}")

    if args.label_col:
        logger.info(f"Using existing labels from column: {args.label_col}")
        y = df[args.label_col].iloc[args.lookback:].reset_index(drop=True)
    else:
        logger.warning("=" * 65)
        logger.warning("  NO --label-col PROVIDED — using fixed-horizon build_labels()")
        logger.warning("  This ignores path-dependent outcomes. Use triple_barrier_labels.py")
        logger.warning("  to generate tb_label and pass --label-col tb_label instead.")
        logger.warning("=" * 65)
        logger.info("Building labels...")
        y = build_labels(df, lookback=args.lookback, horizon=args.horizon, long_threshold=args.long_threshold)
    
    valid_mask = y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True).astype(int)
    logger.info(f"Dataset: {len(X):,} rows | Labels: {y.value_counts().sort_index().to_dict()}")
    if "sample_weight" in df.columns:
        sw = df["sample_weight"].iloc[args.lookback:].reset_index(drop=True)
        sw = sw[valid_mask].reset_index(drop=True)
        logger.info("Using sample_weight column from triple-barrier labels")
    else:
        sw = None
        logger.info("No sample_weight column found — using uniform weights")

    # Train all 3 models
    model_paths = {}
    for role in ["trend", "structure", "regime"]:
        if backend == "lightgbm" and LIGHTGBM_AVAILABLE:
            model, summary = walk_forward_lgbm(X, y, role, device_type, args.n_splits, sample_weight=sw)
            logger.info(f"\nRefitting [{role}] on full dataset...")
            y_enc = y.values.astype(int)
            params = get_lgbm_params(role, device_type)
            n_est = params.pop("n_estimators", 500)
            role_feats = [f for f in MODEL_CONFIGS[role]["features"] if f in X.columns]
            X_role = X[role_feats].astype(np.float32).values
            sw_full = sw.values.astype("float32") if sw is not None else None
            full_data = lgb.Dataset(X_role, label=y_enc, weight=sw_full) if sw_full is not None else lgb.Dataset(X_role, label=y_enc)
            model = lgb.train(params, full_data, num_boost_round=n_est)
            path = save_lgbm_model(
                model, role, args.out_dir, args.horizon,
                args.long_threshold, device_type, args.lookback,
                summary=summary,
            )
        else:
            pipeline, summary = walk_forward_sklearn(X, y, role, args.n_splits)
            pipeline.fit(X, y)
            path = save_sklearn_model(
                pipeline,
                all_feature_names,
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
        gpu_active,
        backend,
        args.horizon,
        args.long_threshold,
        args.lookback,
        side=args.side,
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