#!/usr/bin/env python3
"""
train_ensemble_gpu.py — Meta-Labeling v3 Single Model Trainer

Loads events CSV labeled with 3-outcome triple barriers (tb_label: 2=TP, 0=SL, 1=timeout).
Drops timeouts, remaps {2:1, 0:0}. Trains one binary LGBM or LogisticRegression model
with snapshot + velocity + synmicro features. Walk-forward + gain-based feature filter
+ calibration + optimal threshold search.

Usage:
    python train_pipeline/train_ensemble_gpu.py --data train_pipeline/data/events_long_labeled.csv --label-col tb_label --use-gpu --out-dir train_pipeline/models_gpu_long --side long
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

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

# ---- Feature definitions ----
SNAPSHOT_FEATURES = [
    "atr_norm", "bb_position", "candle_body",
    "upper_wick", "lower_wick", "range_vs_atr",
]

VELOCITY_FEATURES = [
    "pullback_speed", "vwap_slope_5", "volume_ratio",
]

MICRO_COLS = [
    "tick_imbalance", "ofi_window", "cs_spread",
    "kyle_lambda", "vprof_poc_dist",
]

MIN_CONTEXT_BARS = 500


# ---- GPU Detection ----
def detect_gpu():
    if not LIGHTGBM_AVAILABLE:
        return "cpu", False
    try:
        d = lgb.Dataset(np.random.randn(100, 5).astype(np.float32), label=np.random.randint(0, 2, 100))
        lgb.train({"objective": "binary", "device_type": "gpu", "verbose": -1, "num_leaves": 4}, d, num_boost_round=2)
        logger.info("LightGBM GPU (OpenCL) active.")
        return "gpu", True
    except Exception as e:
        logger.warning(f"GPU probe failed: {e}. Using CPU.")
        return "cpu", False


# ---- Feature computation ----
def compute_features(df: pd.DataFrame, zscore_window: int = 500) -> pd.DataFrame:
    """Compute snapshot + velocity features from OHLCV + indicator data."""
    close = df["close"].values.astype("float64")
    high = df["high"].values.astype("float64")
    low = df["low"].values.astype("float64")
    open_v = df["open"].values.astype("float64")
    vol = df["tick_volume"].values.astype("float64") if "tick_volume" in df.columns else np.ones(len(df))

    atr = df["ATR"].values.astype("float64") if "ATR" in df.columns else np.ones(len(df))
    rsi = df["RSI"].values.astype("float64") if "RSI" in df.columns else np.full(len(df), 50.0)
    vwap = df["VWAP"].values.astype("float64") if "VWAP" in df.columns else close
    macd_h = df["MACD_Hist"].values.astype("float64") if "MACD_Hist" in df.columns else np.zeros(len(df))
    ema5 = df["EMA5"].values.astype("float64") if "EMA5" in df.columns else close
    ema20 = df["EMA20"].values.astype("float64") if "EMA20" in df.columns else close

    # BB
    if all(c in df.columns for c in ["Upper_Band", "Lower_Band", "BB_width"]):
        bb_width = df["BB_width"].values.astype("float64")
        bb_pos = np.clip((close - df["Lower_Band"].values) / (bb_width + 1e-9), 0, 1)
    else:
        sma20 = pd.Series(close).rolling(20).mean().values
        std20 = pd.Series(close).rolling(20).std().values
        bb_width = std20 * 4
        lb = sma20 - std20 * 2
        bb_pos = np.clip((close - lb) / (bb_width + 1e-9), 0, 1)

    df_out = pd.DataFrame(index=df.index)

    # Snapshot (active only)
    df_out["atr_norm"] = atr / (close + 1e-9)
    df_out["bb_position"] = bb_pos
    df_out["candle_body"] = (close - open_v) / (atr + 1e-9)
    df_out["upper_wick"] = (high - np.maximum(open_v, close)) / (atr + 1e-9)
    df_out["lower_wick"] = (np.minimum(open_v, close) - low) / (atr + 1e-9)
    df_out["range_vs_atr"] = (high - low) / (atr + 1e-9)

    # Velocity (active only)
    n = len(df)
    for i in range(n):
        if i < MIN_CONTEXT_BARS:
            df_out.loc[i, "pullback_speed"] = 0.0
            df_out.loc[i, "vwap_slope_5"] = 0.0
            df_out.loc[i, "volume_ratio"] = 1.0
            continue

        df_out.loc[i, "pullback_speed"] = np.clip((close[i] - close[i - 5]) / (atr[i] * 5 + 1e-9), -10, 10)
        df_out.loc[i, "vwap_slope_5"] = (vwap[i] - vwap[i - 5]) / (atr[i] + 1e-9)
        df_out.loc[i, "volume_ratio"] = vol[i] / (np.mean(vol[max(0, i - 5):i]) + 1e-9)

    df_out = df_out.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Synmicro (active only)
    for col in MICRO_COLS:
        df_out[col] = df[col].values.astype("float64") if col in df.columns else 0.0

    # Rolling z-score normalization — regime-agnostic features
    ROLLING = zscore_window
    zscore_cols = ["atr_norm", "kyle_lambda", "vprof_poc_dist", "ofi_window", "tick_imbalance"]
    for col in zscore_cols:
        if col not in df_out.columns:
            continue
        raw = df_out[col].values
        rm = pd.Series(raw).rolling(ROLLING, min_periods=max(50, ROLLING // 5)).mean().values
        rs = pd.Series(raw).rolling(ROLLING, min_periods=max(50, ROLLING // 5)).std().values
        df_out[col] = np.clip((raw - rm) / (rs + 1e-9), -4, 4)
        df_out[col] = np.nan_to_num(df_out[col], nan=0.0)

    return df_out


def get_feature_list(df_out: pd.DataFrame) -> list:
    feats = SNAPSHOT_FEATURES + VELOCITY_FEATURES
    return [f for f in feats if f in df_out.columns] + [c for c in MICRO_COLS if c in df_out.columns]


# ---- Training ----
def train_model(
    X: pd.DataFrame, y: pd.Series, device_type: str, use_gpu: bool, side: str,
    skip_filter: bool = False,
):
    n_clean = len(X)
    logger.info(f"Training on {n_clean:,} clean events | side={side}")
    logger.info(f"Features ({len(X.columns)}): {list(X.columns)}")

    if n_clean >= 500 and LIGHTGBM_AVAILABLE:
        logger.info("N >= 500 -> LightGBM")
        model, feat_importance, fold_metrics = _train_lgbm(X, y, device_type, use_gpu, skip_filter)
        backend = "lightgbm"
    else:
        logger.warning(f"N={n_clean} < 500 -> LogisticRegression")
        model, feat_importance, fold_metrics = _train_logreg(X, y)
        backend = "sklearn"

    return model, feat_importance, fold_metrics, backend


def _train_lgbm(X: pd.DataFrame, y: pd.Series, device_type: str, use_gpu: bool,
                 skip_filter: bool = False):
    feats = list(X.columns)
    y_enc = y.values.astype(int)

    # Time-split: 70% train, 30% validation
    cut = int(len(X) * 0.7)
    X_tr = X.iloc[:cut].astype(np.float32).values
    X_va = X.iloc[cut:].astype(np.float32).values
    y_tr = y_enc[:cut]
    y_va = y_enc[cut:]

    logger.info(f"Time-split: train={len(X_tr):,} (first 70%) | val={len(X_va):,} (last 30%)")

    n_neg = (y_tr == 0).sum()
    n_pos = max((y_tr == 1).sum(), 1)
    scale_pos_weight = min(n_neg / n_pos, 3.0)

    params = {
        "objective": "binary",
        "device_type": device_type,
        "verbose": -1,
        "random_state": 42,
        "max_bin": 63,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 30,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "scale_pos_weight": scale_pos_weight,
        "learning_rate": 0.02,
    }

    train_ds = lgb.Dataset(X_tr, label=y_tr)
    val_ds = lgb.Dataset(X_va, label=y_va, reference=train_ds)

    model = lgb.train(
        params, train_ds, num_boost_round=500,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
    )

    p = model.predict(X_va)
    y_pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y_va, y_pred)
    f1 = f1_score(y_va, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_va, y_pred, labels=[0, 1])
    logger.info(f"  Val | Acc: {acc:.4f} | F1: {f1:.4f}")
    logger.info(f"  Confusion:\n{cm}")
    fold_metrics = [{"accuracy": acc, "binary_f1": f1}]

    imp = model.feature_importance(importance_type="gain")
    mean_imp = imp / max(imp.sum(), 1)
    imp_df = pd.DataFrame({"feature": feats, "importance": mean_imp}).sort_values("importance", ascending=False)
    logger.info(f"Feature importance:\n{imp_df.to_string()}")

    if skip_filter:
        survivors = feats[:]
        logger.info("Feature filter skipped — training on all features")
    else:
        threshold = 0.001 * max(mean_imp.max(), 1e-9)
        survivors = [feats[i] for i in range(len(feats)) if mean_imp[i] > threshold]
        logger.info(f"Surviving features ({len(survivors)}/{len(feats)}): {survivors}")

    # Retrain on full data
    if len(survivors) > 0:
        X_surv = X[survivors].astype(np.float32).values
    else:
        X_surv = X.astype(np.float32).values
        survivors = feats

    n_neg_f = (y_enc == 0).sum()
    n_pos_f = max((y_enc == 1).sum(), 1)
    params["scale_pos_weight"] = min(n_neg_f / n_pos_f, 3.0)

    full_ds = lgb.Dataset(X_surv, label=y_enc)
    full_model = lgb.train(params, full_ds, num_boost_round=500)

    return full_model, survivors, fold_metrics


def _train_logreg(X: pd.DataFrame, y: pd.Series):
    feats = list(X.columns)
    y_enc = y.values.astype(int)
    tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X) // 50)))
    fold_metrics = []

    for fold, (tr, va) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y_enc[tr], y_enc[va]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_va)
        acc = accuracy_score(y_va, y_pred)
        f1 = f1_score(y_va, y_pred, average="binary", zero_division=0)
        logger.info(f"  Fold {fold+1} | Acc: {acc:.4f} | F1: {f1:.4f}")
        fold_metrics.append({"fold": fold + 1, "accuracy": acc, "binary_f1": f1})

    # Retrain on full
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    pipe.fit(X, y_enc)
    return pipe, feats, fold_metrics


# ---- Calibration ----
class CalibratedWrapper(BaseEstimator, ClassifierMixin):
    """Wraps a raw binary model for sklearn CalibratedClassifierCV."""

    def __init__(self, base_model, backend: str):
        self.base_model = base_model
        self.backend = backend

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = _predict_proba(self.base_model, X, self.backend)
        return np.column_stack([1 - p, p])


def calibrate_and_threshold(model, X_calib: np.ndarray, y_calib: np.ndarray,
                            fold_metrics: list, side: str, backend: str):
    """Calibrate model on holdout and find optimal threshold."""
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve

    n = len(y_calib)
    p_raw = _predict_proba(model, X_calib, backend)

    if n >= 200:
        cal_method = "isotonic"
    else:
        cal_method = "sigmoid"
        logger.warning(f"Holdout N={n} < 200 -> using Platt Scaling (sigmoid)")

    wrapper = CalibratedWrapper(model, backend)
    wrapper.fit(X_calib, y_calib)
    cal = CalibratedClassifierCV(estimator=wrapper, method=cal_method, cv="prefit")
    cal.fit(X_calib, y_calib)
    p_cal = cal.predict_proba(X_calib)[:, 1]

    # Log calibration curve
    try:
        frac_pos, mean_pred = calibration_curve(y_calib, p_cal, n_bins=10)
        logger.info(f"Calibration curve ({cal_method}):")
        for mp, fp in zip(mean_pred, frac_pos):
            logger.info(f"  Model says {mp:.2f} -> Actual win rate {fp:.2f}")
    except Exception as e:
        logger.warning(f"Calibration curve failed: {e}")

    # Find optimal threshold
    best_t, best_exp, best_wr, best_n = 0.5, -999, 0, 0
    for t in np.arange(0.40, 0.81, 0.01):
        preds = (p_cal >= t).astype(int)
        n_trades = preds.sum()
        if n_trades < 10:
            continue
        wins = preds[y_calib == 1].sum() if preds.sum() > 0 else 0
        wr = wins / max(preds.sum(), 1)
        expectancy = wr * 2 - (1 - wr) * 1
        if expectancy > best_exp and n_trades >= 10:
            best_exp = expectancy
            best_t = t
            best_wr = wr
            best_n = n_trades

    logger.info(f"Optimal threshold: {best_t:.2f} | Expectancy: {best_exp:.3f}R | "
                f"Win rate: {best_wr:.1%} | N trades: {best_n}")

    # Log specific thresholds for manual inspection
    for t_check in [0.45, 0.50, 0.55]:
        preds = (p_cal >= t_check).astype(int)
        n_tr = preds.sum()
        if n_tr >= 10:
            wr = preds[y_calib == 1].sum() / max(preds.sum(), 1)
            exp = wr * 2 - (1 - wr) * 1
            logger.info(f"  Threshold {t_check:.2f}: Expectancy={exp:+.3f}R | "
                        f"Win rate={wr:.1%} | N={n_tr}")
        else:
            logger.info(f"  Threshold {t_check:.2f}: N={n_tr} (<10 — insufficient)")

    return cal, cal_method, best_t, best_exp


def _predict_proba(model, X, backend: str) -> np.ndarray:
    if backend == "lightgbm":
        p = model.predict(X.astype(np.float32))
        return p if p.ndim == 1 else p[:, 1]
    else:
        return model.predict_proba(X)[:, 1]


# ---- Save ----
def save_artifacts(model, calibrator, features, fold_metrics, args, backend, side,
                   best_threshold, best_expectancy):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Model
    if backend == "lightgbm":
        mpath = os.path.join(out_dir, "model.txt")
        model.save_model(mpath)
    else:
        import joblib
        mpath = os.path.join(out_dir, "model.joblib")
        joblib.dump(model, mpath)

    # Calibrator
    import joblib
    cpath = os.path.join(out_dir, f"calibrator_{side}.pkl")
    joblib.dump(calibrator, cpath)

    # Metadata
    meta = {
        "backend": backend,
        "side": side,
        "features": features,
        "n_features": len(features),
        "label_horizon": args.max_hold,
        "pt_atr": 2.0,
        "sl_atr": 1.0,
        "zscore_window": args.zscore_window,
        "threshold": best_threshold,
        "expectancy": best_expectancy,
        "fold_metrics": fold_metrics,
    }
    with open(os.path.join(out_dir, "ensemble_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved model -> {mpath}")
    logger.info(f"Saved calibrator -> {cpath}")
    logger.info(f"Saved metadata -> {out_dir}/ensemble_metadata.json")


# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Meta-Labeling v3 Single Model Trainer")
    ap.add_argument("--data", required=True)
    ap.add_argument("--label-col", type=str, default="tb_label")
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--out-dir", type=str, default="train_pipeline/models_gpu")
    ap.add_argument("--side", type=str, default="long", choices=["long", "short"])
    ap.add_argument("--max-hold", type=int, default=60)
    ap.add_argument("--expanded-features", action="store_true")
    ap.add_argument("--microstructure-features", action="store_true")
    ap.add_argument("--skip-feature-filter", action="store_true",
                    help="Skip gain-based feature filtering (train on all features)")
    ap.add_argument("--zscore-window", type=int, default=500,
                    help="Rolling window for z-score normalization (500 for long, 250 for small datasets)")
    args = ap.parse_args()

    logger.info("=" * 65)
    logger.info(f"  Meta-Labeling v3 Training | side={args.side}")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Output: {args.out_dir}")
    logger.info("=" * 65)

    device_type, gpu_active = detect_gpu() if args.use_gpu else ("cpu", False)

    df = pd.read_csv(args.data)
    df.columns = [c.lower().strip() for c in df.columns]
    logger.info(f"Loaded {len(df):,} rows")

    if args.label_col not in df.columns:
        logger.error(f"Label column '{args.label_col}' not found. Columns: {list(df.columns)[:20]}")
        sys.exit(1)

    # Drop timeouts (tb_label == 1), remap {2:1, 0:0}
    timeout_count = (df[args.label_col] == 1).sum()
    df = df[df[args.label_col] != 1].copy().reset_index(drop=True)
    df[args.label_col] = df[args.label_col].map({2: 1, 0: 0})
    logger.info(f"Training on {len(df):,} clean events (timeouts dropped: {timeout_count:,})")
    logger.info(f"Label dist: {df[args.label_col].value_counts().sort_index().to_dict()}")

    # Quality gate
    if len(df) < 100:
        logger.error(f"Only {len(df)} clean events — minimum 100 required. Abort.")
        sys.exit(1)

    # Compute features
    logger.info("Computing features...")
    X = compute_features(df, zscore_window=args.zscore_window)
    feats = get_feature_list(X)
    X = X[feats]
    y = df[args.label_col].reset_index(drop=True).astype(int)

    # Train
    model, survivors, fold_metrics, backend = train_model(X, y, device_type, gpu_active, args.side, skip_filter=args.skip_feature_filter)

    # Calibration: hold out final 10% time block (never seen in training or validation)
    n_calib = max(int(len(X) * 0.1), 50)
    X_calib = X[survivors].iloc[-n_calib:].values
    y_calib = y.iloc[-n_calib:].values
    logger.info(f"Calibration hold-out: {len(X_calib):,} rows (final 10% time block)")

    calibrator, cal_method, best_t, best_exp = calibrate_and_threshold(
        model, X_calib, y_calib, fold_metrics, args.side, backend,
    )

    # Save
    save_artifacts(
        model, calibrator, survivors, fold_metrics,
        args, backend, args.side, best_t, best_exp,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
