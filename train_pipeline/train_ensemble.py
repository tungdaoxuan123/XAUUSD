#!/usr/bin/env python3
"""
train_ensemble.py - 3-Model Ensemble Training Pipeline for XAUUSD Scalping Bot

Trains three independent sklearn pipelines in a single run:
  1. Logistic Regression  -> trend model     (linear directional bias)
  2. RandomForestClassifier -> structure model (nonlinear indicator interactions)
  3. GradientBoostingClassifier -> regime model (boosted volatility/regime detection)

Each model is saved independently. Live inference loads all three and combines
them via soft voting (averaged predict_proba) — see ensemble_sklearn.py.

Usage:
    python train_pipeline/train_ensemble.py ^
        --data train_pipeline/data/xauusd_m1.csv ^
        --horizon 5 ^
        --buy-threshold 0.0005 ^
        --sell-threshold 0.0005

    # Expanded features:
    python train_pipeline/train_ensemble.py ^
        --data train_pipeline/data/xauusd_m1.csv ^
        --expanded-features

Feature vector (default 15 features — matches live RL observation space):
    close_lag_9 ... close_lag_0  (10 bars)
    RSI, MACD, Signal_Line
    current_position = 0.0       (state placeholder, documented below)
    current_balance  = 10000.0   (state placeholder, documented below)

Feature vector (expanded 20 features):
    default + [MACD_Hist, VWAP, close_minus_vwap, ATR, BB_width]

Output per model (3 models x 3 files = 9 artifacts):
    train_pipeline/models/sklearn_logistic_<suffix>.joblib
    train_pipeline/models/sklearn_random_forest_<suffix>.joblib
    train_pipeline/models/sklearn_gradient_boosting_<suffix>.joblib
    train_pipeline/models/features_logistic_<suffix>.json
    ...
    train_pipeline/models/summary_logistic_<suffix>.json
    ...
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
Path("train_pipeline/reports").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "reports" / "training.log", mode="w"),
    ],
)
logger = logging.getLogger("TrainPipeline")


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

REQUIRED_COLUMNS = {"time", "open", "high", "low", "close", "tick_volume"}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load an MT5-exported CSV. Expected columns: time, open, high, low, close, tick_volume"""
    logger.info(f"Loading data from: {csv_path}")
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        sys.exit(1)

    df["time"] = pd.to_datetime(df["time"], infer_datetime_format=True)
    df = df.sort_values("time").reset_index(drop=True)

    logger.info(f"Loaded {len(df):,} rows | Range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    return df


# ===========================================================================
# 2. TECHNICAL INDICATORS  (exact mirror of live_ensemble_trading.py)
#    IMPORTANT: Any change to the live version MUST be reflected here.
# ===========================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of LiveEnsembleTrader.add_technical_indicators()."""
    df = df.copy()
    close = df["close"]

    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Faster MACD 8-24-9 (matches live scalper)
    exp1 = close.ewm(span=8, adjust=False).mean()
    exp2 = close.ewm(span=24, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]

    # Bollinger Bands
    df["SMA20"] = close.rolling(window=20).mean()
    df["STD20"] = close.rolling(window=20).std()
    df["Upper_Band"] = df["SMA20"] + (df["STD20"] * 2)
    df["Lower_Band"] = df["SMA20"] - (df["STD20"] * 2)
    df["BB_width"] = df["Upper_Band"] - df["Lower_Band"]

    # Stochastic
    low_min = df["low"].rolling(window=14).min()
    high_max = df["high"].rolling(window=14).max()
    df["%K"] = 100 * ((close - low_min) / (high_max - low_min))
    df["%D"] = df["%K"].rolling(window=3).mean()

    # VWAP (running cumulative — matches live implementation)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["tick_volume"].replace(0, np.nan).ffill().fillna(1)
    df["TPV"] = typical_price * volume
    df["Cum_TPV"] = df["TPV"].cumsum()
    df["Cum_Volume"] = volume.cumsum()
    df["VWAP"] = df["Cum_TPV"] / df["Cum_Volume"]
    df["close_minus_vwap"] = close - df["VWAP"]

    # ATR (14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    return df.dropna().reset_index(drop=True)


# ===========================================================================
# 3. FEATURE BUILDER  (mirrors get_observation_from_rates)
# ===========================================================================

# NOTE: Feature names are the single source of truth shared between
# train_ensemble.py and ensemble_sklearn.py. Change here = change there.
FEATURE_NAMES_DEFAULT = [
    f"close_lag_{i}" for i in range(9, -1, -1)
] + ["RSI", "MACD", "Signal_Line", "current_position", "current_balance"]

FEATURE_NAMES_EXPANDED = FEATURE_NAMES_DEFAULT + [
    "MACD_Hist", "VWAP", "close_minus_vwap", "ATR", "BB_width"
]


def build_features(df: pd.DataFrame, expanded: bool = False) -> pd.DataFrame:
    """
    Build the feature matrix row by row from indicator-enriched data.

    State placeholder note:
        'current_position' (0.0)  and 'current_balance' (10000.0) are
        live-state variables with no offline meaning. We fix them to safe
        constants so the model trains on a neutral account state. This is
        intentional and documented — signal classifiers should not depend on
        account state for entry direction.
    """
    rows = []
    n = len(df)
    lookback = 10

    for i in range(lookback - 1, n):
        price_lags = df["close"].iloc[i - lookback + 1: i + 1].values
        latest = df.iloc[i]

        row = list(price_lags) + [
            latest["RSI"],
            latest["MACD"],
            latest["Signal_Line"],
            0.0,       # current_position placeholder
            10000.0,   # current_balance placeholder
        ]

        if expanded:
            row += [
                latest["MACD_Hist"],
                latest["VWAP"],
                latest["close_minus_vwap"],
                latest["ATR"],
                latest["BB_width"],
            ]

        rows.append(row)

    names = FEATURE_NAMES_EXPANDED if expanded else FEATURE_NAMES_DEFAULT
    return pd.DataFrame(rows, columns=names)


# ===========================================================================
# 4. LABEL BUILDER
# ===========================================================================

def build_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    buy_threshold: float = 0.0005,
    sell_threshold: float = 0.0005,
) -> pd.Series:
    """
    Forward-return 3-class labels with dead zone.
      1  = BUY  (future return > buy_threshold)
     -1  = SELL (future return < -sell_threshold)
      0  = HOLD (noise zone)
    No leakage: label[i] uses close[i + horizon], features[i] use bars <= i.
    """
    close = df["close"].values
    n = len(close)
    lookback = 10
    labels = []

    for i in range(lookback - 1, n):
        future_idx = i + horizon
        if future_idx >= n:
            labels.append(np.nan)
        else:
            ret = (close[future_idx] - close[i]) / close[i]
            if ret > buy_threshold:
                labels.append(1)
            elif ret < -sell_threshold:
                labels.append(-1)
            else:
                labels.append(0)

    return pd.Series(labels, name="label")


# ===========================================================================
# 5. MODEL DEFINITIONS
# ===========================================================================

# Three base models — each role explained in plan.txt
ENSEMBLE_MODELS = {
    "logistic": {
        "label": "Trend Model (Logistic Regression)",
        "clf": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs", C=0.1
        ),
    },
    "random_forest": {
        "label": "Structure Model (Random Forest)",
        "clf": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=15,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
    },
    "gradient_boosting": {
        "label": "Regime Model (Gradient Boosting)",
        "clf": GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, random_state=42
        ),
    },
}


def build_pipeline(model_key: str) -> Pipeline:
    """Build a fresh sklearn Pipeline (StandardScaler + classifier)."""
    clf = ENSEMBLE_MODELS[model_key]["clf"]
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# ===========================================================================
# 6. WALK-FORWARD TRAINING
# ===========================================================================

def walk_forward_train(
    X: pd.DataFrame,
    y: pd.Series,
    model_key: str,
    n_splits: int = 5,
) -> tuple:
    """
    TimeSeriesSplit walk-forward validation. Returns (fitted_pipeline, summary_dict).
    Training always precedes validation chronologically — no leakage, no shuffling.
    """
    label = ENSEMBLE_MODELS[model_key]["label"]
    logger.info(f"\n{'='*60}\n  Training: {label}\n{'='*60}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    last_pipeline = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline = build_pipeline(model_key)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_val, y_pred, labels=[-1, 0, 1])
        report = classification_report(
            y_val, y_pred, labels=[-1, 0, 1],
            target_names=["SELL", "HOLD", "BUY"], zero_division=0
        )

        logger.info(
            f"\n  [{model_key}] Fold {fold + 1}/{n_splits}\n"
            f"  Train: {len(X_train):,}  Val: {len(X_val):,}\n"
            f"  Train dist: {y_train.value_counts().sort_index().to_dict()}\n"
            f"  Val   dist: {y_val.value_counts().sort_index().to_dict()}\n"
            f"  Accuracy : {acc:.4f}  |  Macro F1 : {f1:.4f}\n"
            f"  Confusion Matrix:\n{cm}\n{report}"
        )

        fold_metrics.append({"fold": fold + 1, "accuracy": acc, "macro_f1": f1})
        last_pipeline = pipeline

    acc_mean = np.mean([m["accuracy"] for m in fold_metrics])
    f1_mean = np.mean([m["macro_f1"] for m in fold_metrics])

    logger.info(
        f"\n  [{model_key}] AGGREGATE — Mean Accuracy: {acc_mean:.4f}  |  Mean Macro F1: {f1_mean:.4f}"
    )

    summary = {
        "model": model_key,
        "label": label,
        "n_splits": n_splits,
        "mean_accuracy": acc_mean,
        "mean_macro_f1": f1_mean,
        "fold_detail": fold_metrics,
    }

    return last_pipeline, summary


# ===========================================================================
# 7. SAVE ARTIFACT
# ===========================================================================

def save_model(
    pipeline: Pipeline,
    feature_names: list,
    summary: dict,
    model_key: str,
    out_dir: str,
    expanded: bool,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
) -> str:
    """Save model joblib + features JSON + summary JSON. Returns model file path."""
    os.makedirs(out_dir, exist_ok=True)
    suffix = "expanded" if expanded else "default"
    tag = f"{model_key}_{suffix}"

    model_path = os.path.join(out_dir, f"sklearn_{tag}.joblib")
    features_path = os.path.join(out_dir, f"features_{tag}.json")
    summary_path = os.path.join(out_dir, f"summary_{tag}.json")

    joblib.dump(pipeline, model_path)

    with open(features_path, "w") as f:
        json.dump({
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "expanded_features": expanded,
            "label_horizon": horizon,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
        }, f, indent=2)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"  Saved model    : {model_path}")
    logger.info(f"  Saved features : {features_path}")
    logger.info(f"  Saved summary  : {summary_path}")
    return model_path


# ===========================================================================
# 8. FEATURE COUNT VERIFICATION
# ===========================================================================

def verify_feature_count(feature_names: list, expanded: bool):
    expected = 20 if expanded else 15
    if len(feature_names) != expected:
        logger.error(f"Feature mismatch: got {len(feature_names)}, expected {expected}")
        sys.exit(1)
    logger.info(f"Feature count OK: {len(feature_names)} ({'expanded' if expanded else 'default'})")


# ===========================================================================
# 9. CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 3-model sklearn ensemble for XAUUSD (Logistic + RF + GBM)"
    )
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--horizon", type=int, default=5, help="Forward bars (default: 5)")
    parser.add_argument("--buy-threshold", type=float, default=0.0005, help="Buy label threshold")
    parser.add_argument("--sell-threshold", type=float, default=0.0005, help="Sell label threshold")
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-forward folds (default: 5)")
    parser.add_argument("--expanded-features", action="store_true", help="Use 20-feature expanded mode")
    parser.add_argument("--out-dir", type=str, default="train_pipeline/models", help="Artifact output dir")
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = "expanded" if args.expanded_features else "default"

    logger.info("=" * 60)
    logger.info("  XAUUSD - 3-Model Ensemble Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Data         : {args.data}")
    logger.info(f"  Horizon      : {args.horizon} bars")
    logger.info(f"  Buy thresh   : {args.buy_threshold:.4%}")
    logger.info(f"  Sell thresh  : {args.sell_threshold:.4%}")
    logger.info(f"  Folds        : {args.n_splits}")
    logger.info(f"  Features     : {suffix}")
    logger.info(f"  Models       : logistic, random_forest, gradient_boosting")
    logger.info("=" * 60)

    # Step 1: Load
    df_raw = load_data(args.data)

    # Step 2: Indicators
    logger.info("Computing technical indicators...")
    df = add_technical_indicators(df_raw)
    logger.info(f"After indicators: {len(df):,} rows remaining")

    # Step 3: Features
    logger.info("Building feature matrix...")
    X = build_features(df, expanded=args.expanded_features)
    feature_names = list(X.columns)
    verify_feature_count(feature_names, args.expanded_features)

    # Step 4: Labels
    logger.info("Building labels...")
    y = build_labels(
        df,
        horizon=args.horizon,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )

    valid_mask = y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True).astype(int)

    logger.info(f"Final dataset: {len(X):,} rows")
    logger.info(f"Label distribution: {y.value_counts().sort_index().rename({-1: 'SELL', 0: 'HOLD', 1: 'BUY'}).to_dict()}")

    if len(X) < 500:
        logger.warning("Small dataset (<500 rows). Results may not be reliable.")

    # Step 5: Train all three models
    trained_models = {}
    for model_key in ENSEMBLE_MODELS:
        pipeline, summary = walk_forward_train(X, y, model_key, n_splits=args.n_splits)

        logger.info(f"\nRefitting [{model_key}] on full dataset...")
        pipeline.fit(X, y)

        model_path = save_model(
            pipeline, feature_names, summary,
            model_key, args.out_dir, args.expanded_features,
            args.horizon, args.buy_threshold, args.sell_threshold,
        )
        trained_models[model_key] = model_path

    # Step 6: Final summary
    logger.info("\n" + "=" * 60)
    logger.info("  ALL MODELS TRAINED SUCCESSFULLY")
    logger.info("=" * 60)
    for key, path in trained_models.items():
        logger.info(f"  {key:25s} -> {path}")
    logger.info(f"\nLoad in live bot with:")
    logger.info(f"  SklearnEnsemble.load_all('{args.out_dir}', '{suffix}')")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
