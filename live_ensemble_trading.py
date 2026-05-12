"""
live_ensemble_trading.py — Meta-Labeling v3 Live Bot
=====================================================

Event-based execution with calibrated dual-model dispatch.
Supports long, short, or both simultaneously (hedging).

On each bar:
  1. Manage open positions (time stop, BE+partial, chandelier, confidence decay)
  2. Check hedge-aware exits when both sides active
  3. Check macro exits (volatility, session, drawdown, news)
  4. Check Setup A (long) and Setup B (short) primary signal conditions
  5. If condition fires -> compute features -> model predict -> calibrate
  6. If calibrated p >= threshold -> place order (2:1 TP/SL via ATR)
  7. Respect cooldowns after closes
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import Settings, setup_logging
from mt5_interface import MT5Interface
from news_manager import NewsManager
from risk_manager import FTMORiskManager

from sklearn.base import BaseEstimator, ClassifierMixin

logger = setup_logging()


# ---------------------------------------------------------------------------
# CalibratedWrapper — required for unpickling calibrators
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PositionTracker — metadata per open position
# ---------------------------------------------------------------------------

@dataclass
class PositionTracker:
    ticket: int
    side: int                # +1 long, -1 short
    entry_price: float
    sl: float
    tp: float
    entry_bar: int
    entry_time: datetime
    best_price: float        # best price seen since entry
    highest_r: float = 0.0   # highest R-multiple reached
    partial_closed: bool = False


# ---------------------------------------------------------------------------
# DirectionModel
# ---------------------------------------------------------------------------

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
    """Dual-model live bot with hedging + multi-layer exit strategy."""

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
        self._zscore_win = 500
        for k in ["atr_norm", "kyle_lambda", "vprof_poc_dist", "ofi_window", "tick_imbalance"]:
            self._zscore_bufs[k] = []
        self.bar_counter = 0

        # Position tracking
        self._trackers: Dict[int, PositionTracker] = {}

        # Cooldowns
        self._flip_cooldown_bar = -100  # 3-bar cooldown after directional flip
        self._reentry_cooldowns: List[tuple] = []  # [(bar, side), ...] 10-bar cooldown

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
        for k in range(60):
            lag_i = i - (k + 1)
            if lag_i >= 0 and atr[lag_i] > 0:
                feat[f"return_lag_{k}"] = np.clip((close[i] - close[lag_i]) / (atr[lag_i] + 1e-9), -10, 10)
            else:
                feat[f"return_lag_{k}"] = 0.0

        # Microstructure from rates (fallback 0)
        for mc in ["tick_imbalance", "ofi_window", "cs_spread", "kyle_lambda", "vprof_poc_dist"]:
            feat[mc] = float(df.iloc[i][mc]) if mc in df.columns else 0.0

        return feat

    def _add_reentry_cooldown(self, side: int):
        self._reentry_cooldowns.append((self.bar_counter, side))

    def _get_volume(self, ticket: int) -> float:
        try:
            pos = next(p for p in (self.interface.get_positions() or []) if p.ticket == ticket)
            return float(pos.volume)
        except (StopIteration, AttributeError):
            return 0.0

    def _manage_positions(self, positions, df, dry_run: bool):
        """Multi-layer exit strategy. Called every bar with open positions."""
        atr_val = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 0.0
        close_val = float(df["close"].iloc[-1])

        # Sync trackers
        active_tickets = [p.ticket for p in positions]
        for tid in list(self._trackers.keys()):
            if tid not in active_tickets:
                del self._trackers[tid]

        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour; minute = now_utc.minute; weekday = now_utc.weekday()

        for p in positions:
            tid = p.ticket
            if tid not in self._trackers:
                self._trackers[tid] = PositionTracker(
                    ticket=tid, side=+1 if p.type == 0 else -1,
                    entry_price=float(p.price_open), sl=float(p.sl), tp=float(p.tp),
                    entry_bar=self.bar_counter, entry_time=datetime.now(),
                    best_price=float(p.price_open))
            tr = self._trackers[tid]

            # --- Macro: Volatility Spike ---
            atr50 = np.nanmean(df["ATR"].iloc[-50:].values) if len(df) >= 50 else 0
            if atr50 > 0 and atr_val > 3 * atr50:
                logger.warning(f"Vol spike: ATR={atr_val:.2f} > 3x ATR50={atr50:.2f} — closing ALL")
                if not dry_run: self.interface.close_all_positions()
                self._trackers.clear(); return

            # --- Macro: Session close (NY close 21 UTC / Fri 21 UTC) ---
            is_eod = (hour >= 20 and minute >= 55) or (weekday == 4 and hour >= 21)
            if is_eod:
                logger.info(f"EOD close {tid}: {tr.highest_r:.2f}R")
                if not dry_run: self.interface.close_position(tid)
                self._add_reentry_cooldown(tr.side)
                self._trackers.pop(tid, None); continue

            # --- Calculate R ---
            sl_dist = max(abs(tr.entry_price - tr.sl), 1e-9)
            if tr.side > 0:
                r = (close_val - tr.entry_price) / sl_dist
                tr.best_price = max(tr.best_price, close_val)
            else:
                r = (tr.entry_price - close_val) / sl_dist
                tr.best_price = min(tr.best_price, close_val)
            tr.highest_r = max(tr.highest_r, r)

            # --- Time Stop (60 bars) ---
            if self.bar_counter - tr.entry_bar >= 60:
                logger.info(f"Time stop {tid}: {r:.2f}R")
                if not dry_run: self.interface.close_position(tid)
                self._add_reentry_cooldown(tr.side)
                self._trackers.pop(tid, None); continue

            # --- BE + Partial Close at +1R ---
            if r >= 1.0 and not tr.partial_closed:
                half_vol = round(p.volume / 2, 2) if hasattr(p, 'volume') else 0.01
                if half_vol >= 0.01:
                    logger.info(f"BE+partial {tid}: {r:.2f}R — close {half_vol}, SL->BE")
                    if not dry_run:
                        self.interface.close_partial_position(tid, half_vol)
                        self.interface.modify_position(tid, tr.entry_price, tr.tp)
                    tr.partial_closed = True; continue

            # --- Chandelier Trail at +1.2R ---
            if r >= 1.2 and atr_val > 0:
                if tr.side > 0:
                    ns = close_val - atr_val * 1.0
                    if ns > tr.sl:
                        if not dry_run: self.interface.modify_position(tid, ns, tr.tp)
                        tr.sl = ns
                else:
                    ns = close_val + atr_val * 1.0
                    if ns < tr.sl:
                        if not dry_run: self.interface.modify_position(tid, ns, tr.tp)
                        tr.sl = ns

            # --- Confidence Decay ---
            try:
                feat = self._compute_live_features(df)
                model = self.long_model if tr.side > 0 else self.short_model
                if feat and model:
                    for zk in ["atr_norm", "kyle_lambda", "vprof_poc_dist", "ofi_window", "tick_imbalance"]:
                        feat[zk] = self._zscore(zk, feat.get(zk, 0.0), win=model.zscore_window)
                    f_list = [feat.get(f, 0.0) for f in model.features]
                    X = np.array(f_list, dtype=np.float32)
                    _, p_cal = model.predict_calibrated(X)
                    if p_cal < 0.40:
                        logger.info(f"Conf decay {tid}: p_cal={p_cal:.3f} — early close at {r:.2f}R")
                        if not dry_run: self.interface.close_position(tid)
                        self._add_reentry_cooldown(tr.side)
                        self._trackers.pop(tid, None); continue
            except Exception:
                pass

        # ---- Hedge-aware exits (both sides active) ----
        longs = [t for t in self._trackers.values() if t.side > 0]
        shorts = [t for t in self._trackers.values() if t.side < 0]
        if longs and shorts:
            lr = longs[0].highest_r; sr = shorts[0].highest_r

            if lr > 0.6 and sr > 0.6:
                logger.info(f"Hedge: both >0.6R (L={lr:.2f} S={sr:.2f}) — closing both")
                if not dry_run:
                    for t in longs + shorts: self.interface.close_position(t.ticket)
                self._trackers.clear(); return

            if lr + sr >= 2.0:
                logger.info(f"Hedge: net {lr+sr:.2f} >= 2.0 — closing both")
                if not dry_run:
                    for t in longs + shorts: self.interface.close_position(t.ticket)
                self._trackers.clear(); return

            stronger = max(longs + shorts, key=lambda t: t.highest_r)
            weaker = min(longs + shorts, key=lambda t: t.highest_r)
            if stronger.highest_r > 1.2 and weaker.highest_r > 0.4:
                hv = round(self._get_volume(stronger.ticket) / 2, 2)
                logger.info(f"Hedge asym: close weak {weaker.ticket}({weaker.highest_r:.2f}R)"
                            f" + partial strong {stronger.ticket}({stronger.highest_r:.2f}R)")
                if not dry_run:
                    self.interface.close_position(weaker.ticket)
                    if hv >= 0.01: self.interface.close_partial_position(stronger.ticket, hv)
                self._trackers.pop(weaker.ticket, None); return

            if lr < -0.8 and sr < -0.8:
                worse = min(longs + shorts, key=lambda t: t.highest_r)
                logger.info(f"Hedge lock: close worse {worse.ticket}({worse.highest_r:.2f}R)")
                if not dry_run: self.interface.close_position(worse.ticket)
                self._trackers.pop(worse.ticket, None); return

    def run(self, interval_s: int = 10, dry_run: bool = False):
        logger.info(f"Event-based live loop ({interval_s}s, dry_run={dry_run})")
        if not self.interface.initialize():
            logger.error("MT5 init failed"); return
        self.risk_mgr.initialize_balance()

        # Pre-warm z-score buffers on startup
        logger.info("Fetching historical data to warm up Z-score buffers...")
        warmup_rates = self.interface.get_rates(count=600)
        if warmup_rates is not None and len(warmup_rates) > 0:
            warmup_df = pd.DataFrame(warmup_rates)
            warmup_df = self._compute_indicators(warmup_df)
            for i in range(len(warmup_df)):
                feat = self._compute_live_features(warmup_df.iloc[:i+1])
                if feat:
                    for zk in ["atr_norm", "kyle_lambda", "vprof_poc_dist", "ofi_window", "tick_imbalance"]:
                        self._zscore(zk, feat.get(zk, 0.0))
            logger.info("Z-score buffers warmed up successfully.")
        else:
            logger.warning("Could not fetch warmup rates. Z-score buffers will start empty.")

        last_update = datetime.now() - timedelta(seconds=interval_s)
        try:
            while True:
                now = datetime.now()
                if (now - last_update).total_seconds() < interval_s:
                    time.sleep(1); continue
                last_update = now
                self.bar_counter += 1

                # Fetch data for exit management + signal scanning
                rates = self.interface.get_rates(count=700)
                if rates is None or len(rates) < 60:
                    continue
                df = pd.DataFrame(rates)
                df = self._compute_indicators(df)
                if len(df) < 60:
                    continue

                # ---- Phase 1: Manage open positions ----
                positions = self.interface.get_positions() or []
                if positions:
                    self._manage_positions(positions, df, dry_run)
                    # Re-fetch positions after management actions
                    positions = self.interface.get_positions() or []

                # ---- Phase 2: Macro exits (news) ----
                if NewsManager.is_in_blackout():
                    if positions:
                        logger.info("News blackout — closing positions")
                        if not dry_run: self.interface.close_all_positions()
                        self._trackers.clear()
                        positions = []
                    else:
                        logger.info("News blackout — skipping entries")
                    continue  # skip entry scan during blackout

                # ---- Phase 3: Cooldown cleanup ----
                self._reentry_cooldowns = [(b, s) for b, s in self._reentry_cooldowns
                                           if self.bar_counter - b < 10]

                # ---- Phase 4: Check for new entries ----
                if not self.risk_mgr.can_trade():
                    time.sleep(10); continue

                has_long = any(p.type == 0 for p in positions)
                has_short = any(p.type == 1 for p in positions)
                if has_long and has_short:
                    continue  # both active, no new entries

                atr = df["ATR"].iloc[-1]
                current_price = df["close"].iloc[-1]

                for model, side in [(self.long_model, 1), (self.short_model, -1)]:
                    if model is None:
                        continue

                    # Cooldowns
                    if self.bar_counter - self._flip_cooldown_bar < 3:
                        continue
                    has_side = has_long if side > 0 else has_short
                    if has_side:
                        continue
                    if any(self.bar_counter - b < 10 for b, s in self._reentry_cooldowns if s == side):
                        continue

                    feat = self._compute_live_features(df)
                    if feat is None:
                        continue
                    if feat is None:
                        continue

                    setup_fired = self._check_setup(df, side)

                    # Apply rolling z-score with model-specific window
                    for zk in ["atr_norm", "kyle_lambda", "vprof_poc_dist", "ofi_window", "tick_imbalance"]:
                        feat[zk] = self._zscore(zk, feat.get(zk, 0.0), win=model.zscore_window)

                    f_list = [feat.get(f, 0.0) for f in model.features]
                    X = np.array(f_list, dtype=np.float32)
                    p_raw, p_cal = model.predict_calibrated(X)

                    side_label = "LONG" if side > 0 else "SHORT"
                    tag = "Setup" if setup_fired else "Scan"
                    logger.info(f"{tag} {side_label}: raw={p_raw:.3f} cal={p_cal:.3f} (thresh={model.threshold:.2f})")

                    if p_cal < model.threshold:
                        logger.info(f"Skip {side_label}: cal={p_cal:.3f} < {model.threshold:.2f}")
                        continue

                    # Directional flip: high-confidence signal opposite existing position
                    opp_has = has_short if side > 0 else has_long
                    if opp_has:
                        flip_tickets = [p.ticket for p in positions if (p.type == 1) == (side > 0)]
                        logger.info(f"DIRECTIONAL FLIP: {side_label} p_cal={p_cal:.3f} — "
                                    f"closing {len(flip_tickets)} opposing positions")
                        if not dry_run:
                            for ft in flip_tickets:
                                self.interface.close_position(ft)
                                self._trackers.pop(ft, None)
                        self._flip_cooldown_bar = self.bar_counter
                        time.sleep(1)
                    elif (side > 0 and has_long) or (side < 0 and has_short):
                        logger.info(f"Block {side_label}: position already open")
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

    long_dir = args.long_model or "train_pipeline/models_gpu_long_lb10_momentum"
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
