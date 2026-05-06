#!/usr/bin/env python3
"""
ensemble_gpu.py - Live Inference Wrapper for GPU-trained LONG-ONLY Binary Ensemble

Loads three LightGBM models (or sklearn fallbacks) trained by train_ensemble_gpu.py
and combines them via soft voting (averaged predict_proba) to produce a
(long_action, long_confidence) output.

This wrapper auto-detects whether saved models are LightGBM (.txt) or
sklearn (.joblib) based on the ensemble_metadata.json.

Usage in live_ensemble_trading.py:
    from train_pipeline.ensemble_gpu import EnsembleGPU, build_obs_from_rates

    # In __init__:
    self.gpu_signal = EnsembleGPU.load("train_pipeline/models_gpu")

    # In run_live_trading:
    obs_df = build_obs_from_rates(rates, expanded=False)
    gp_action, gp_confidence = self.gpu_signal.predict(obs_df)
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("FTMO_Trader")

def get_feature_names(lookback: int, expanded: bool = False, micro: bool = False) -> list:
    """Generate dynamic feature names matching train_ensemble_gpu.py."""
    base = [f"close_lag_{i}" for i in range(lookback - 1, -1, -1)]
    common = base + ["RSI", "MACD", "Signal_Line", "current_position", "current_balance"]
    if expanded:
        common += ["MACD_Hist", "VWAP", "close_minus_vwap", "ATR", "BB_width"]
    if micro:
        common += [
            "tick_imbalance", "bid_ask_vol_imbalance", "spread_mean", "ofi_window", "of_pressure_flag",
            "vprof_poc_dist", "vprof_in_value_area", "vprof_hvn_flag", "vprof_lvn_flag"
        ]
    return common


# ---------------------------------------------------------------------------
# Indicator computation (standalone mirror of live_ensemble_trading.py)
# ---------------------------------------------------------------------------

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss.replace(0, np.nan))))

    exp1 = close.ewm(span=8, adjust=False).mean()
    exp2 = close.ewm(span=24, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal_Line"]

    df["SMA20"] = close.rolling(20).mean()
    df["STD20"] = close.rolling(20).std()
    df["BB_width"] = (df["SMA20"] + df["STD20"] * 2) - (df["SMA20"] - df["STD20"] * 2)

    # ATR (Standard 14-period)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - close.shift())
    low_close = np.abs(df['low'] - close.shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df["ATR"] = tr.rolling(14).mean()

    low_min = df["low"].rolling(14).min()
    high_max = df["high"].rolling(14).max()
    df["%K"] = 100 * ((close - low_min) / (high_max - low_min).replace(0, np.nan))
    df["%D"] = df["%K"].rolling(3).mean()

    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["tick_volume"].replace(0, np.nan).ffill().fillna(1)
    df["TPV"] = tp * vol
    df["VWAP"] = df["TPV"].cumsum() / vol.cumsum()
    df["close_minus_vwap"] = close - df["VWAP"]

    df = df.replace([np.inf, -np.inf], np.nan)
    # Only drop rows where the indicators we just computed are NaN
    indicator_cols = ["RSI", "MACD", "Signal_Line", "ATR", "BB_width", "VWAP"]
    df = df.dropna(subset=[c for c in indicator_cols if c in df.columns]).reset_index(drop=True)
    return df


def build_obs_from_rates(rates, lookback: int = 10, expanded: bool = False, micro: bool = False, current_pos=0.0, balance=10000.0):
    """Convert MT5 rate array -> single-row observation DataFrame with dynamic lookback."""
    df = pd.DataFrame(rates)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s")
    df = _add_indicators(df)

    if len(df) < lookback:
        logger.warning(f"BuildObs: Insufficient bars ({len(df)}) for lookback {lookback}")
        return None

    # Get the last 'lookback' prices
    price_lags = df["close"].iloc[-lookback:].values
    latest = df.iloc[-1]

    row = list(price_lags) + [
        float(latest["RSI"]), 
        float(latest["MACD"]), 
        float(latest["Signal_Line"]),
        float(current_pos), 
        float(balance),
    ]
    if expanded:
        row += [
            float(latest["MACD_Hist"]), 
            float(latest["VWAP"]),
            float(latest["close_minus_vwap"]), 
            float(latest["ATR"]), 
            float(latest["BB_width"]),
        ]
        
    if micro:
        # Placeholder for micro features in live mode (using 0.0)
        # Real-time tick integration will fill these in later
        row += [0.0] * 9

    cols = get_feature_names(lookback, expanded, micro)
    return pd.DataFrame([row], columns=cols)


# ---------------------------------------------------------------------------
# LightGBM model wrapper (predict_proba equivalent)
# ---------------------------------------------------------------------------

class _LGBMWrapper:
    """Wraps a raw LightGBM Booster to mimic sklearn predict_proba interface (binary)."""

    def __init__(self, booster):
        self.booster = booster
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        x_values = X.values if hasattr(X, "values") else X
        proba = self.booster.predict(x_values)  # shape (n_samples,) for binary
        if proba.ndim == 0:
            proba = np.array([proba])
        # Return as (n, 2) array: [p_wait, p_long]
        return np.stack([1.0 - proba, proba], axis=1)


# ---------------------------------------------------------------------------
# Ensemble loader + soft voting
# ---------------------------------------------------------------------------

class EnsembleGPU:
    """
    Soft-voting ensemble over three LightGBM or sklearn models (binary).

    Output:
        action     : 1.0 (LONG) | 0.0 (WAIT)
        confidence : averaged long probability (0.0-1.0)
    """

    def __init__(self, models: dict, metadata: dict):
        self.models = models
        self.metadata = metadata
        self.expanded = metadata.get("expanded_features", False)
        self.lookback = metadata.get("lookback", 10)
        self.backend = metadata.get("backend", "unknown")
        self.gpu_used = metadata.get("gpu_used", False)
        self.micro = metadata.get("microstructure_features", False) or metadata.get("micro", False)
        
        self.label_horizon = metadata.get("label_horizon")
        self.long_threshold = metadata.get("long_threshold", 0.0005)
        self.long_confidence = metadata.get("long_confidence", self.long_threshold)

    @classmethod
    def load(cls, model_dir: str) -> "EnsembleGPU":
        meta_path = os.path.join(model_dir, "ensemble_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"ensemble_metadata.json not found in {model_dir}.\n"
                "Run train_ensemble_gpu.py first."
            )

        with open(meta_path) as f:
            metadata = json.load(f)

        backend = metadata.get("backend", "lightgbm")
        model_files = metadata.get("model_files", {})
        models = {}

        for role, model_path in model_files.items():
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if model_path.endswith(".txt"):
                try:
                    import lightgbm as lgb
                    booster = lgb.Booster(model_file=model_path)
                    models[role] = _LGBMWrapper(booster)
                    logger.info(f"EnsembleGPU: loaded LightGBM [{role}] from {model_path}")
                except ImportError:
                    raise ImportError("LightGBM not installed. Run: pip install lightgbm")

            elif model_path.endswith(".joblib"):
                import joblib
                models[role] = joblib.load(model_path)
                logger.info(f"EnsembleGPU: loaded sklearn [{role}] from {model_path}")

            else:
                raise ValueError(f"Unknown model file type: {model_path}")

        logger.info(
            f"EnsembleGPU: {len(models)} models loaded | "
            f"backend={backend} | gpu={metadata.get('gpu_used', False)}"
        )
        return cls(models, metadata)

    def _get_probs(self, model, obs_df: pd.DataFrame) -> float:
        """Extract long-entry probability from a model (binary)."""
        try:
            proba = model.predict_proba(obs_df)[0]
            classes = model.classes_

            if hasattr(model, "booster") or list(classes) == [0, 1]:
                # Binary: proba = [p_wait, p_long]
                return float(proba[1])
            else:
                # sklearn with custom classes
                for cls, p in zip(classes, proba):
                    if cls == 1:
                        return float(p)
                return float(proba[-1])  # last column is assumed class 1

        except Exception as e:
            logger.warning(f"EnsembleGPU predict failed: {e}")
            return 0.0

    def predict(self, obs_df) -> tuple:
        """
        Soft-voting across all loaded models. Optimized for single observation.

        Returns:
            (action, confidence):
                action     = 1.0 (LONG) | 0.0 (WAIT)
                confidence = averaged long probability (0.0-1.0)
        """
        if obs_df is None or (hasattr(obs_df, "empty") and obs_df.empty) or (not hasattr(obs_df, "empty") and len(obs_df) == 0):
            return 0.0, 0.0

        all_long = []
        for role, model in self.models.items():
            all_long.append(self._get_probs(model, obs_df))

        avg_long = float(np.mean(all_long))
        confidence = avg_long

        if avg_long > 0.5:
            return 1.0, confidence
        else:
            return 0.0, 1.0 - confidence

    def predict_batch(self, X: pd.DataFrame) -> tuple:
        """
        Vectorized soft-voting across all loaded models.

        Returns:
            (actions, confidences): numpy arrays of shape (len(X),)
        """
        if X is None or (hasattr(X, "empty") and X.empty) or (not hasattr(X, "empty") and len(X) == 0):
            return np.array([]), np.array([])

        n_samples = len(X)
        sum_long = np.zeros(n_samples)

        for role, model in self.models.items():
            proba_arr = model.predict_proba(X)
            # proba_arr shape (n, 2): [p_wait, p_long]
            if hasattr(model, "booster"):
                sum_long += proba_arr[:, 1]
            else:
                for idx, cls in enumerate(model.classes_):
                    if cls == 1:
                        sum_long += proba_arr[:, idx]
                        break
                else:
                    sum_long += proba_arr[:, -1]

        n_models = len(self.models)
        avg_long = sum_long / n_models
        confidence = avg_long

        actions = np.where(avg_long > 0.5, 1.0, 0.0)

        return actions, confidence

    def predict_detail(self, obs_df) -> dict:
        """Returns per-model long probability breakdown. Useful for dry-run debugging."""
        if obs_df is None or obs_df.empty:
            return {}

        all_long = []
        detail = {}

        for role, model in self.models.items():
            p_long = self._get_probs(model, obs_df)
            detail[role] = {"long": p_long, "wait": 1.0 - p_long}
            all_long.append(p_long)

        detail["avg"] = {
            "long": float(np.mean(all_long)),
            "wait": 1.0 - float(np.mean(all_long)),
        }
        return detail
