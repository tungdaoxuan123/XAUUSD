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
        logger.info(f"[{side.upper()}] Loaded | features={len(self.features)} | threshold={self.threshold:.2f}")

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

    def predict_calibrated(self, X: np.ndarray) -> float:
        try:
            p_raw = self.model.predict(X.astype(np.float32))
            if hasattr(p_raw, "__len__"):
                p_raw = float(p_raw[0])
            else:
                p_raw = float(p_raw)
        except Exception:
            try:
                p_raw = float(self.model.predict_proba(X)[0, 1])
            except Exception:
                return 0.0

        if self.calibrator is not None:
            try:
                p_cal = self.calibrator.predict_proba(X.reshape(1, -1))[0, 1]
                return float(p_cal)
            except Exception:
                pass
        return float(p_raw)


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
        logger.info(f"Dual-model bot ready | symbol={self.symbol}")

    def _compute_indicators(self, df):
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["tick_volume"].replace(0, np.nan).ffill().fillna(1)

        df["EMA5"] = close.ewm(span=5, adjust=False).mean()
        df["EMA20"] = close.ewm(span=20, adjust=False).mean()
        tp = (high + low + close) / 3
        df["VWAP"] = (tp * vol).cumsum() / vol.cumsum()

        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        df["ATR"] = np.maximum(hl, np.maximum(hc, lc))
        df["ATR"] = df["ATR"].rolling(14).mean()

        # RSIt
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta).clip(lower=0).rolling(14).mean()
        df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        ema8 = close.ewm(span=8, adjust=False).mean()
        ema24 = close.ewm(span=24, adjust=False).mean()
        df["MACD_Hist"] = (ema8 - ema24) - (ema8 - ema24).ewm(span=9, adjust=False).mean()

        # BBands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["Upper_Band"] = sma20 + std20 * 2
        df["Lower_Band"] = sma20 - std20 * 2
        df["BB_width"] = df["Upper_Band"] - df["Lower_Band"]

        return df.dropna().reset_index(drop=True)

    def _check_setup(self, df, side: int) -> bool:
        """Check if the EMA pullback setup fires on the latest bar."""
        if len(df) < 3:
            return False
        close = df["close"].values
        ema5 = df["EMA5"].values
        ema20 = df["EMA20"].values
        vwap = df["VWAP"].values
        atr14 = df["ATR"].values

        i = len(df) - 1
        atr50 = np.nanmean(atr14[max(0, i - 50):i + 1])
        vol_ok = atr14[i] > atr50 * 0.8
        if not vol_ok:
            return False

        if side > 0:
            return (close[i] > vwap[i] and ema5[i] > ema20[i] and
                    close[i - 1] < ema20[i - 1] and close[i] > ema20[i])
        else:
            return (close[i] < vwap[i] and ema5[i] < ema20[i] and
                    close[i - 1] > ema20[i - 1] and close[i] < ema20[i])

    def _compute_live_features(self, df):
        """Mirror of train_ensemble_gpu.compute_features on a small window."""
        close = df["close"].values.astype("float64")
        high = df["high"].values.astype("float64")
        low = df["low"].values.astype("float64")
        open_v = df["open"].values.astype("float64")
        atr = df["ATR"].values.astype("float64")
        rsi = df["RSI"].values.astype("float64")
        vwap = df["VWAP"].values.astype("float64")
        macd_h = df["MACD_Hist"].values.astype("float64")
        ema5 = df["EMA5"].values.astype("float64")
        ema20 = df["EMA20"].values.astype("float64")
        vol = df.get("tick_volume", pd.Series(np.ones(len(df)))).values.astype("float64")

        i = len(df) - 1
        if i < 15:
            return None

        feat = {}
        feat["rsi_14"] = rsi[i]
        feat["macd_hist"] = macd_h[i]
        feat["atr_norm"] = atr[i] / (close[i] + 1e-9)
        feat["ema_ratio"] = ema5[i] / (ema20[i] + 1e-9)
        feat["close_minus_vwap_norm"] = (close[i] - vwap[i]) / (atr[i] + 1e-9)
        feat["bb_position"] = np.clip((close[i] - (close[i - 20:i].mean() - close[i - 20:i].std() * 2))
                                      / (close[i - 20:i].std() * 4 + 1e-9), 0, 1)
        feat["candle_body"] = (close[i] - open_v[i]) / (atr[i] + 1e-9)
        feat["upper_wick"] = (high[i] - max(open_v[i], close[i])) / (atr[i] + 1e-9)
        feat["lower_wick"] = (min(open_v[i], close[i]) - low[i]) / (atr[i] + 1e-9)
        feat["vol_zscore"] = (atr[i] - np.nanmean(atr[max(0, i - 100):i]))
        feat["vol_zscore"] /= (np.nanstd(atr[max(0, i - 100):i]) + 1e-9)
        feat["range_vs_atr"] = (high[i] - low[i]) / (atr[i] + 1e-9)

        feat["pullback_speed"] = (close[i] - close[i - 5]) / (atr[i] * 5 + 1e-9)
        feat["atr_ratio"] = atr[i] / (atr[i - 10] + 1e-9)
        feat["vwap_slope_5"] = (vwap[i] - vwap[i - 5]) / (atr[i] + 1e-9)
        feat["rsi_delta_5"] = rsi[i] - rsi[i - 5]
        feat["volume_ratio"] = vol[i] / (np.mean(vol[max(0, i - 5):i]) + 1e-9)
        feat["ema_gap"] = (ema5[i] - ema20[i]) / (atr[i] + 1e-9)
        feat["ema_gap_delta"] = feat["ema_gap"] - (ema5[i - 5] - ema20[i - 5]) / (atr[i - 5] + 1e-9)

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
                rates = self.interface.get_rates(count=200)
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

                    # Build feature vector in model's expected order
                    f_list = [feat.get(f, 0.0) for f in model.features]
                    X = np.array(f_list, dtype=np.float32)

                    p_cal = model.predict_calibrated(X)
                    side_label = "LONG" if side > 0 else "SHORT"
                    logger.info(f"Setup {side_label}: p_cal={p_cal:.3f} (thresh={model.threshold:.2f})")

                    if p_cal < model.threshold:
                        logger.info(f"Skip {side_label}: p_cal={p_cal:.3f} < {model.threshold:.2f}")
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

    long_dir = args.long_model or "train_pipeline/models_gpu_long"
    short_dir = args.short_model or "train_pipeline/models_gpu_short"

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
