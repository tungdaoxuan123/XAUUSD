#!/usr/bin/env python3
"""
ensemble_sklearn.py - 3-Model LONG-ONLY Binary Soft-Voting Inference Wrapper

Loads the three trained sklearn pipelines (trend / structure / regime) and
combines them via soft voting (averaged predict_proba) for binary long/wait.

Usage in live_ensemble_trading.py:
    from train_pipeline.ensemble_sklearn import SklearnEnsemble, build_obs_from_rates

    # In __init__:
    self.sklearn_signal = SklearnEnsemble.load_all("train_pipeline/models", "default")

    # In run_live_trading:
    obs_df = build_obs_from_rates(rates, expanded=False)
    sk_action, sk_confidence = self.sklearn_signal.predict(obs_df)
    # sk_action: 1.0 (long), 0.0 (wait)
    # sk_confidence: 0.0 - 1.0
"""

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("FTMO_Trader")

# ---------------------------------------------------------------------------
# Feature name constants — must match train_ensemble.py exactly
# ---------------------------------------------------------------------------
FEATURE_NAMES_DEFAULT = [
    f"close_lag_{i}" for i in range(9, -1, -1)
] + ["RSI", "MACD", "Signal_Line", "current_position", "current_balance"]

FEATURE_NAMES_EXPANDED = FEATURE_NAMES_DEFAULT + [
    "MACD_Hist", "VWAP", "close_minus_vwap", "ATR", "BB_width"
]

# Model keys must match the filenames produced by train_ensemble.py
MODEL_KEYS = ["logistic", "random_forest", "gradient_boosting"]


# ---------------------------------------------------------------------------
# Indicator computation (standalone — no class dependency)
# Mirror of LiveEnsembleTrader.add_technical_indicators()
# ---------------------------------------------------------------------------

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute exact same indicators as in live_ensemble_trading.py."""
    df = df.copy()
    close = df["close"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = close.ewm(span=8, adjust=False).mean()
    exp2 = close.ewm(span=24, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]

    df["SMA20"] = close.rolling(window=20).mean()
    df["STD20"] = close.rolling(window=20).std()
    df["Upper_Band"] = df["SMA20"] + (df["STD20"] * 2)
    df["Lower_Band"] = df["SMA20"] - (df["STD20"] * 2)
    df["BB_width"] = df["Upper_Band"] - df["Lower_Band"]

    low_min = df["low"].rolling(window=14).min()
    high_max = df["high"].rolling(window=14).max()
    df["%K"] = 100 * ((close - low_min) / (high_max - low_min))
    df["%D"] = df["%K"].rolling(window=3).mean()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["tick_volume"].replace(0, np.nan).ffill().fillna(1)
    df["TPV"] = typical_price * volume
    df["Cum_TPV"] = df["TPV"].cumsum()
    df["Cum_Volume"] = volume.cumsum()
    df["VWAP"] = df["Cum_TPV"] / df["Cum_Volume"]
    df["close_minus_vwap"] = close - df["VWAP"]

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    return df.dropna().reset_index(drop=True)


def build_obs_from_rates(rates, expanded: bool = False) -> "pd.DataFrame | None":
    """
    Convert MT5 rate array -> single-row feature DataFrame for inference.
    Mirrors get_observation_from_rates() in live_ensemble_trading.py.

    Args:
        rates:    numpy structured array from mt5_interface.get_rates()
        expanded: must match the mode used during training

    Returns:
        pd.DataFrame with shape (1, n_features), or None if insufficient data
    """
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = _add_indicators(df)

    if len(df) < 10:
        return None

    price_lags = df["close"].iloc[-10:].values
    latest = df.iloc[-1]

    row = list(price_lags) + [
        latest["RSI"],
        latest["MACD"],
        latest["Signal_Line"],
        0.0,     # current_position placeholder (matches training constant)
        10000.0, # current_balance placeholder  (matches training constant)
    ]

    if expanded:
        row += [
            latest["MACD_Hist"],
            latest["VWAP"],
            latest["close_minus_vwap"],
            latest["ATR"],
            latest["BB_width"],
        ]

    cols = FEATURE_NAMES_EXPANDED if expanded else FEATURE_NAMES_DEFAULT
    return pd.DataFrame([row], columns=cols)


# ---------------------------------------------------------------------------
# Soft-voting ensemble
# ---------------------------------------------------------------------------

class SklearnEnsemble:
    """
    Combines three sklearn pipelines via soft voting for binary long/wait.

    Output:
        action     : 1.0 (LONG) | 0.0 (WAIT)
        confidence : averaged long probability (0.0 - 1.0)
    """

    def __init__(self, pipelines: dict, expanded: bool = False):
        """
        Args:
            pipelines: dict {model_key: fitted_sklearn_Pipeline}
            expanded:  whether models were trained in expanded-feature mode
        """
        self.pipelines = pipelines   # {str: Pipeline}
        self.expanded = expanded

    @classmethod
    def load_all(cls, model_dir: str, suffix: str = "default") -> "SklearnEnsemble":
        """
        Load all three model artifacts from model_dir.

        Args:
            model_dir: directory containing sklearn_*.joblib files
            suffix:    'default' or 'expanded', must match training mode

        Example:
            ensemble = SklearnEnsemble.load_all("train_pipeline/models", "default")
        """
        pipelines = {}
        for key in MODEL_KEYS:
            path = os.path.join(model_dir, f"sklearn_{key}_{suffix}.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model artifact not found: {path}\n"
                    f"Run train_ensemble.py first to generate artifacts."
                )
            pipelines[key] = joblib.load(path)
            logger.info(f"SklearnEnsemble: loaded [{key}] from {path}")

        expanded = suffix == "expanded"
        logger.info(f"SklearnEnsemble: {len(pipelines)} models loaded (mode: {suffix})")
        return cls(pipelines, expanded)

    def _get_class_probs(self, pipeline, obs_df: pd.DataFrame) -> dict:
        proba = pipeline.predict_proba(obs_df)[0]
        classes = pipeline.classes_

        prob_map = {"long": 0.0, "wait": 0.0}
        for cls, p in zip(classes, proba):
            if cls == 1:
                prob_map["long"] = p
            elif cls == 0:
                prob_map["wait"] = p

        return prob_map

    def predict(self, obs_df: "pd.DataFrame | None") -> tuple:
        if obs_df is None or obs_df.empty:
            return 0.0, 0.0

        all_long = []

        for key, pipeline in self.pipelines.items():
            try:
                probs = self._get_class_probs(pipeline, obs_df)
                all_long.append(probs["long"])
            except Exception as e:
                logger.warning(f"SklearnEnsemble [{key}] predict_proba failed: {e}")
                all_long.append(0.0)

        avg_long = float(np.mean(all_long))
        confidence = avg_long

        if avg_long > 0.5:
            return 1.0, confidence
        else:
            return 0.0, 1.0 - confidence

    def predict_detail(self, obs_df: "pd.DataFrame | None") -> dict:
        if obs_df is None or obs_df.empty:
            return {}

        detail = {}
        all_long = []

        for key, pipeline in self.pipelines.items():
            probs = self._get_class_probs(pipeline, obs_df)
            detail[key] = probs
            all_long.append(probs["long"])

        detail["avg"] = {
            "long": float(np.mean(all_long)),
            "wait": 1.0 - float(np.mean(all_long)),
        }

        return detail
