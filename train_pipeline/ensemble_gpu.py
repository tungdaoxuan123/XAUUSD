#!/usr/bin/env python3
"""
ensemble_gpu.py - Live Inference Wrapper for GPU-trained LightGBM Ensemble

Loads three LightGBM models (or sklearn fallbacks) trained by train_ensemble_gpu.py
and combines them via soft voting (averaged predict_proba) to produce an
(action, confidence) output identical to EnsembleTrader.predict_ensemble().

This wrapper auto-detects whether saved models are LightGBM (.txt) or
sklearn (.joblib) based on the ensemble_metadata.json.

Usage in live_ensemble_trading.py:
    from train_pipeline.ensemble_gpu import EnsembleGPU, build_obs_from_rates

    # In __init__:
    self.gpu_signal = EnsembleGPU.load("train_pipeline/models_gpu")

    # In run_live_trading (as confluence filter):
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


# Label mapping (-1=SELL, 0=HOLD, 1=BUY -> lgbm classes 0,1,2)
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}


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
    """Wraps a raw LightGBM Booster to mimic sklearn predict_proba interface."""

    def __init__(self, booster, n_classes=3):
        self.booster = booster
        self.n_classes = n_classes
        # classes_ is always [0, 1, 2] in LightGBM multiclass
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X) -> np.ndarray:
        # X can be pd.DataFrame or np.ndarray from live bot
        x_values = X.values if hasattr(X, "values") else X
        proba = self.booster.predict(x_values)  # shape (n_samples, 3)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        return proba


# ---------------------------------------------------------------------------
# Ensemble loader + soft voting
# ---------------------------------------------------------------------------

class EnsembleGPU:
    """
    Soft-voting ensemble over three LightGBM or sklearn models.

    Output interface matches EnsembleTrader.predict_ensemble():
        action     : +0.5 (BUY) | -0.5 (SELL) | 0.0 (HOLD)
        confidence : highest averaged class probability
    """

    def __init__(self, models: dict, metadata: dict):
        """
        Args:
            models:   dict {role: model_object} where model has predict_proba(X)
            metadata: contents of ensemble_metadata.json
        """
        self.models = models
        self.metadata = metadata
        self.expanded = metadata.get("expanded_features", False)
        self.lookback = metadata.get("lookback", 10)
        self.backend = metadata.get("backend", "unknown")
        self.gpu_used = metadata.get("gpu_used", False)
        self.micro = metadata.get("microstructure_features", False) or metadata.get("micro", False)
        
        self.label_horizon = metadata.get("label_horizon")
        self.buy_threshold = metadata.get("buy_threshold")
        self.sell_threshold = metadata.get("sell_threshold")

    @classmethod
    def load(cls, model_dir: str) -> "EnsembleGPU":
        """
        Load all models from model_dir using ensemble_metadata.json.

        Args:
            model_dir: directory containing ensemble_metadata.json + model files

        Example:
            ens = EnsembleGPU.load("train_pipeline/models_gpu")
        """
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
                # LightGBM text model
                try:
                    import lightgbm as lgb
                    booster = lgb.Booster(model_file=model_path)
                    models[role] = _LGBMWrapper(booster)
                    logger.info(f"EnsembleGPU: loaded LightGBM [{role}] from {model_path}")
                except ImportError:
                    raise ImportError("LightGBM not installed. Run: pip install lightgbm")

            elif model_path.endswith(".joblib"):
                # sklearn Pipeline fallback
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

    def _get_probs(self, model, obs_df: pd.DataFrame) -> dict:
        """
        Extract buy/hold/sell probabilities from a model.
        Handles both LightGBM (classes_ = [0,1,2] = SELL,HOLD,BUY)
        and sklearn (classes_ can be a subset of [-1,0,1]).
        """
        try:
            proba = model.predict_proba(obs_df)[0]
            classes = model.classes_

            prob_map = {"buy": 0.0, "hold": 0.0, "sell": 0.0}

            # LightGBM path: classes are LightGBM-encoded (0=SELL,1=HOLD,2=BUY)
            if hasattr(model, "booster"):
                prob_map["sell"] = float(proba[0])
                prob_map["hold"] = float(proba[1])
                prob_map["buy"]  = float(proba[2])
            else:
                # sklearn path: classes_ are original labels (-1, 0, 1)
                for cls, p in zip(classes, proba):
                    if cls == 1:
                        prob_map["buy"] = float(p)
                    elif cls == 0:
                        prob_map["hold"] = float(p)
                    elif cls == -1:
                        prob_map["sell"] = float(p)

            return prob_map

        except Exception as e:
            logger.warning(f"EnsembleGPU predict failed: {e}")
            return {"buy": 0.0, "hold": 1.0, "sell": 0.0}

    def predict(self, obs_df) -> tuple:
        """
        Soft-voting across all loaded models. Optimized for single observation.

        Returns:
            (action, confidence):
                action     = +0.5 (BUY) | -0.5 (SELL) | 0.0 (HOLD)
                confidence = highest averaged class probability (0.0-1.0)
        """
        if obs_df is None or (hasattr(obs_df, "empty") and obs_df.empty) or (not hasattr(obs_df, "empty") and len(obs_df) == 0):
            return 0.0, 0.0

        all_buy, all_hold, all_sell = [], [], []

        for role, model in self.models.items():
            probs = self._get_probs(model, obs_df)
            all_buy.append(probs["buy"])
            all_hold.append(probs["hold"])
            all_sell.append(probs["sell"])

        avg_buy  = float(np.mean(all_buy))
        avg_hold = float(np.mean(all_hold))
        avg_sell = float(np.mean(all_sell))
        confidence = max(avg_buy, avg_hold, avg_sell)

        if avg_buy > avg_sell and avg_buy > avg_hold:
            return 0.5, confidence
        elif avg_sell > avg_buy and avg_sell > avg_hold:
            return -0.5, confidence
        else:
            return 0.0, confidence

    def predict_batch(self, X: pd.DataFrame) -> tuple:
        """
        Vectorized soft-voting across all loaded models. Useful for backtesting.
        
        Returns:
            (actions, confidences): numpy arrays of shape (len(X),)
        """
        if X is None or (hasattr(X, "empty") and X.empty) or (not hasattr(X, "empty") and len(X) == 0):
            return np.array([]), np.array([])

        n_samples = len(X)
        sum_buy  = np.zeros(n_samples)
        sum_hold = np.zeros(n_samples)
        sum_sell = np.zeros(n_samples)

        for role, model in self.models.items():
            # Get vectorized probabilities (n_samples, 3)
            proba_arr = model.predict_proba(X)
            
            # Identify which index is which class
            if hasattr(model, "booster"):
                # LightGBM: 0=SELL, 1=HOLD, 2=BUY
                sum_sell += proba_arr[:, 0]
                sum_hold += proba_arr[:, 1]
                sum_buy  += proba_arr[:, 2]
            else:
                # sklearn: classes_ are labels (-1, 0, 1)
                for idx, cls in enumerate(model.classes_):
                    if cls == 1:
                        sum_buy += proba_arr[:, idx]
                    elif cls == 0:
                        sum_hold += proba_arr[:, idx]
                    elif cls == -1:
                        sum_sell += proba_arr[:, idx]

        n_models = len(self.models)
        avg_buy  = sum_buy / n_models
        avg_hold = sum_hold / n_models
        avg_sell = sum_sell / n_models
        
        # Stack to find max across classes
        stacked = np.stack([avg_sell, avg_hold, avg_buy], axis=1) # (N, 3)
        confidence = np.max(stacked, axis=1)
        winning_class = np.argmax(stacked, axis=1) # 0=SELL, 1=HOLD, 2=BUY
        
        actions = np.zeros(n_samples)
        actions[winning_class == 0] = -0.5
        actions[winning_class == 2] = 0.5
        
        # If HOLD is winner, action is already 0.0
        return actions, confidence

    def predict_detail(self, obs_df) -> dict:
        """Returns per-model probability breakdown. Useful for dry-run debugging."""
        if obs_df is None or obs_df.empty:
            return {}

        all_buy, all_hold, all_sell = [], [], []
        detail = {}

        for role, model in self.models.items():
            probs = self._get_probs(model, obs_df)
            detail[role] = probs
            all_buy.append(probs["buy"])
            all_hold.append(probs["hold"])
            all_sell.append(probs["sell"])

        detail["avg"] = {
            "buy": float(np.mean(all_buy)),
            "hold": float(np.mean(all_hold)),
            "sell": float(np.mean(all_sell)),
        }
        return detail
