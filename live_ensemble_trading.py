"""
live_ensemble_trading.py — Meta-Labeling v3 Live Bot
=====================================================

Event-based execution with calibrated dual-model dispatch.
Supports long, short, or both simultaneously.

On each bar:
  1. Check Setup A (long) and Setup B (short) primary signal conditions
  2. If condition fires -> compute features -> model predict -> calibrate
  3. If calibrated p >= threshold -> place order (2:1 TP/SL via ATR)
  4. Block signals while position open
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import Settings, setup_logging
from mt5_interface import MT5Interface
from risk_manager import FTMORiskManager

from sklearn.base import BaseEstimator, ClassifierMixin

class CalibratedWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model=None, backend=""):
        self.base_model = base_model
        self.backend = backend
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self
    def predict_proba(self, X):
        if self.base_model is None:
            p = np.ones(len(X)) * 0.5
            return np.column_stack([1 - p, p])
        if hasattr(self.base_model, 'predict'):
            p = self.base_model.predict(X.astype(np.float32))
            p = np.atleast_1d(np.asarray(p, dtype=float))
            return np.column_stack([1 - p, p])
        p = np.ones(len(X)) * 0.5
        return np.column_stack([1 - p, p])

logger = setup_logging()


class DirectionModel:
    """Loads a single trained model + calibrator + metadata."""

    def __init__(self, model_dir: str, side: str):
        self.side = side
        self.dir_val = 1 if side == "long" else -1

        meta_path = os.path.join(model_dir, "ensemble_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.features = self.meta.get("features", [])
        self.threshold = self.meta.get("threshold", 0.5)
        self.zscore_window = self.meta.get("zscore_window", 500)
        logger.info(f"[{side.upper()}] Loaded | features={len(self.features)} | "
                    f"threshold={self.threshold:.2f} | zscore_win={self.zscore_window}")

        self.model = self._load_model(model_dir)
        self.calibrator = self._load_calibrator(model_dir)

    def _load_model(self, d: str):
        import lightgbm as lgb
        try:
            mp = os.path.join(d, "model.txt")
            return lgb.Booster(model_file=mp)
        except Exception:
            import joblib
            return joblib.load(os.path.join(d, "model.joblib"))

    def _load_calibrator(self, d: str):
        import joblib
        cp = os.path.join(d, f"calibrator_{self.side}.pkl")
        if os.path.exists(cp):
            return joblib.load(cp)
        return None

    def predict_calibrated(self, X: np.ndarray) -> tuple:
        """Return (p_raw, p_cal)."""
        try:
            p_raw = float(self.model.predict(X.astype(np.float32).reshape(1, -1)))
        except Exception:
            try:
                p_raw = float(self.model.predict_proba(X.reshape(1, -1))[0, 1])
            except Exception:
                return 0.0, 0.0

        if self.calibrator is not None:
            try:
                p_cal = float(self.calibrator.predict_proba(X.reshape(1, -1))[0, 1])
                return p_raw, p_cal
            except Exception:
                pass
        return p_raw, p_raw


class LiveEnsembleTrader:
    """Dual-model live bot with primary signal filter + calibration."""

    def __init__(self, long_dir: str = None, short_dir: str = None):
        self.settings = Settings
        self.interface = MT5Interface()
        self.risk_mgr = FTMORiskManager(self.interface)

        self.long_model = DirectionModel(long_dir, "long") if long_dir else None
        self.short_model = DirectionModel(short_dir, "short") if short_dir else None

        if not self.interface.authorized:
            self.interface.initialize()
        self.symbol = self.interface.symbol

        # Rolling z-score buffers for regime-agnostic features
        self._zscore_bufs: Dict[str, list] = {}
        self._zscore_win = 500  # overridden by model metadata at predict time
        for k in ["atr_norm", "kyle_lambda", "vprof_poc_dist", "ofi_window", "tick_imbalance"]:
            self._zscore_bufs[k] = []

    def _zscore(self, key: str, raw_val: float, win: int = None) -> float:
        w = win or self._zscore_win
        buf = self._zscore_bufs[key]
        buf.append(raw_val)
        if len(buf) > w:
            buf.pop(0)
        if len(buf) < max(50, w // 5):
            return raw_val
        m = np.mean(buf)
        s = np.std(buf)
        if s < 1e-9:
            return 0.0
        return float(np.clip((raw_val - m) / s, -4, 4))

    def _compute_indicators(self, df):
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["tick_volume"].replace(0, np.nan).ffill().fillna(1)

        df["EMA5"] = close.ewm(span=5, adjust=False).mean()
        df["EMA20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA50"] = close.ewm(span=50, adjust=False).mean()
        df["EMA200"] = close.ewm(span=200, adjust=False).mean()
        tp = (high + low + close) / 3
        df["VWAP"] = (tp * vol).cumsum() / vol.cumsum()

        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        df["ATR"] = np.maximum(hl, np.maximum(hc, lc))
        df["ATR"] = df["ATR"].rolling(14).mean()

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["Upper_Band"] = sma20 + std20 * 2
        df["Lower_Band"] = sma20 - std20 * 2
        df["BB_width"] = df["Upper_Band"] - df["Lower_Band"]

        return df.dropna().reset_index(drop=True)

    def _check_setup(self, df, side: int) -> bool:
        if len(df) < 60:
            return False
        close = df["close"].values
        ema20 = df["EMA20"].values
        ema50 = df["EMA50"].values
        ema200 = df["EMA200"].values
        atr14 = df["ATR"].values

        i = len(df) - 1
        atr50 = np.nanmean(atr14[max(0, i - 50):i + 1])
        vol_ok = atr14[i] > atr50 * 0.5
        if not vol_ok:
            return False

        if side > 0:
            ema20_slope = ema20[i] - ema20[max(0, i - 3)]
            return (close[i] > ema20[i] and ema20[i] > ema50[i] and
                    ema50[i] > ema200[i] and ema20_slope > 0)
        else:
            return False  # SHORT disabled for now

    def _compute_live_features(self, df):
        close = df["close"].values.astype("float64")
        high = df["high"].values.astype("float64")
        low = df["low"].values.astype("float64")
        open_v = df["open"].values.astype("float64")
        atr = df["ATR"].values.astype("float64")
        vwap = df["VWAP"].values.astype("float64")
        ema20 = df["EMA20"].values.astype("float64")
        ema200 = df["EMA200"].values.astype("float64")
        vol = df.get("tick_volume", pd.Series(np.ones(len(df)))).values.astype("float64")

        i = len(df) - 1
        if i < 60:
            return None

        feat = {}
        feat["atr_norm"] = atr[i] / (close[i] + 1e-9)
        feat["bb_position"] = np.clip((close[i] - df["Lower_Band"].values[i]) / (df["BB_width"].values[i] + 1e-9), 0, 1)
        feat["candle_body"] = (close[i] - open_v[i]) / (atr[i] + 1e-9)
        feat["upper_wick"] = (high[i] - max(open_v[i], close[i])) / (atr[i] + 1e-9)
        feat["lower_wick"] = (min(open_v[i], close[i]) - low[i]) / (atr[i] + 1e-9)
        feat["range_vs_atr"] = (high[i] - low[i]) / (atr[i] + 1e-9)
        feat["pullback_speed"] = (close[i] - close[i - 5]) / (atr[i] * 5 + 1e-9)
        feat["vwap_slope_5"] = (vwap[i] - vwap[i - 5]) / (atr[i] + 1e-9)
        feat["volume_ratio"] = vol[i] / (np.mean(vol[max(0, i - 5):i]) + 1e-9)
        feat["above_ema200"] = 1.0 if close[i] > ema200[i] else 0.0

        # Return lags (ATR-normalized, clipped)
        for k in range(15):
            lag_i = i - (k + 1)
            if lag_i >= 0 and atr[lag_i] > 0:
                feat[f"return_lag_{k}"] = np.clip((close[i] - close[lag_i]) / (atr[lag_i] + 1e-9), -10, 10)
            else:
                feat[f"return_lag_{k}"] = 0.0

        # Microstructure from rates (fallback 0)
        for mc in ["tick_imbalance", "ofi_window", "cs_spread", "kyle_lambda", "vprof_poc_dist"]:
            feat[mc] = float(df.iloc[i][mc]) if mc in df.columns else 0.0

        return feat

    def run(self, interval_s: int = 10, dry_run: bool = False):
        logger.info(f"Event-based live loop ({interval_s}s, dry_run={dry_run})")
        if not self.interface.initialize():
            logger.error("MT5 init failed"); return
        self.risk_mgr.initialize_balance()

        last_update = datetime.now() - timedelta(seconds=interval_s)
        try:
            while True:
                now = datetime.now()
                if (now - last_update).total_seconds() < interval_s:
                    time.sleep(1); continue
                last_update = now

                # Skip if position open
                positions = self.interface.get_positions()
                if positions and len(positions) > 0:
                    continue

                if not self.risk_mgr.can_trade():
                    time.sleep(10); continue

                # Fetch data
                rates = self.interface.get_rates(count=700)
                if rates is None or len(rates) < 60:
                    continue
                df = pd.DataFrame(rates)
                df = self._compute_indicators(df)
                if len(df) < 60:
                    continue

                atr = df["ATR"].iloc[-1]
                current_price = df["close"].iloc[-1]

                # Check both setups
                for model, side, condition in [
                    (self.long_model, 1, self._check_setup(df, 1)),
                    (self.short_model, -1, self._check_setup(df, -1)),
                ]:
                    if model is None or not condition:
                        continue

                    feat = self._compute_live_features(df)
                    if feat is None:
                        continue

                    # Apply rolling z-score with model-specific window
                    for zk in ["atr_norm", "kyle_lambda", "vprof_poc_dist", "ofi_window", "tick_imbalance"]:
                        feat[zk] = self._zscore(zk, feat.get(zk, 0.0), win=model.zscore_window)

                    # Build feature vector in model's expected order
                    f_list = [feat.get(f, 0.0) for f in model.features]
                    X = np.array(f_list, dtype=np.float32)

                    p_raw, p_cal = model.predict_calibrated(X)
                    side_label = "LONG" if side > 0 else "SHORT"
                    logger.info(f"Setup {side_label}: raw={p_raw:.3f} cal={p_cal:.3f} (thresh={model.threshold:.2f})")

                    if p_cal < 0.45:
                        logger.info(f"Skip {side_label}: cal={p_cal:.3f} < 0.45")
                        continue

                    # Place order: SL = close -/+ 1*ATR, TP = close +/- 2*ATR
                    sl_dist = atr * 1.0
                    tp_dist = 2.0 * sl_dist
                    if side > 0:
                        sl = current_price - sl_dist
                        tp = current_price + tp_dist
                    else:
                        sl = current_price + sl_dist
                        tp = current_price - tp_dist

                    lots = self.risk_mgr.calculate_position_size(float(p_cal), sl_dist)
                    if lots <= 0:
                        continue

                    if dry_run:
                        logger.info(f"DRY RUN: {side_label} {lots:.2f} lots @ {current_price:.2f} "
                                    f"SL={sl:.2f} TP={tp:.2f}")
                    else:
                        self.interface.send_order(float(side), lots, sl, tp)
                        time.sleep(2)
                    break

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            self.interface.shutdown()


def main():
    ap = argparse.ArgumentParser(description="Meta-Labeling v3 Live Bot")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--long-model", type=str, default=None, help="Path to long model dir")
    ap.add_argument("--short-model", type=str, default=None, help="Path to short model dir")
    ap.add_argument("--interval", type=int, default=10)
    args = ap.parse_args()

    long_dir = args.long_model or "train_pipeline/models_gpu_long_lb15_momentum"
    short_dir = args.short_model or "train_pipeline/models_gpu_short_lb60"

    if not os.path.exists(long_dir):
        logger.warning(f"Long model dir not found: {long_dir}")
        long_dir = None
    if not os.path.exists(short_dir):
        logger.warning(f"Short model dir not found: {short_dir}")
        short_dir = None

    trader = LiveEnsembleTrader(long_dir=long_dir, short_dir=short_dir)
    trader.run(interval_s=args.interval, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
