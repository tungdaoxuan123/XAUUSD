"""
live_ensemble_trading.py — LONG-ONLY binary ensemble trading bot

Uses GPU-trained LightGBM binary ensemble (trend/structure/regime)
with fixed 2:1 ATR-derived TP/SL. Never opens shorts.

Key changes from the old bidirectional version:
  - Binary long/wait signal (not buy/sell/hold)
  - Fixed 2:1 TP/SL at entry (SL = ATR × multiplier, TP = 2 × SL)
  - No trailing, no partial exits, no breakeven adjustments
  - TP/SL levels are locked at order placement time
"""

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta

from config import Settings, setup_logging
from mt5_interface import MT5Interface
from risk_manager import FTMORiskManager
from train_pipeline.ensemble_gpu import EnsembleGPU, _add_indicators

logger = setup_logging()


class LiveEnsembleTrader:
    """Long-only live trading bot with binary ensemble + fixed 2:1 TP/SL."""

    def __init__(self, ensemble_path=None):
        self.settings = Settings
        self.interface = MT5Interface()
        self.risk_mgr = FTMORiskManager(self.interface)

        model_path = ensemble_path or self.settings.ENSEMBLE_MODEL_PATH
        self.ensemble = EnsembleGPU.load(model_path)

        self.long_threshold = self.settings.BUY_THRESHOLD
        self.long_confidence = self.settings.BUY_CONFIDENCE

        self.current_symbol = self.interface.symbol
        if not self.interface.authorized:
            self.interface.initialize()
            self.current_symbol = self.interface.symbol

        logger.info(
            f"LONG-ONLY Ensemble Bot ready — Symbol: {self.current_symbol} | "
            f"Model: {model_path} | Micro: {self.ensemble.micro}"
        )

    def calculate_atr(self, rates, period=14):
        df = pd.DataFrame(rates)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(period).mean().iloc[-1]

    def build_micro_features(self, rates, ticks):
        if ticks is None or len(ticks) == 0:
            keys = ["tick_imbalance", "bid_ask_vol_imbalance", "spread_mean",
                    "ofi_window", "of_pressure_flag",
                    "vprof_poc_dist", "vprof_in_value_area",
                    "vprof_hvn_flag", "vprof_lvn_flag"]
            return {f: 0.0 for f in keys}

        t = pd.DataFrame(ticks)
        t["time"] = pd.to_datetime(t["time"], unit="s")
        t["mid"] = (t["bid"] + t["ask"]) / 2.0

        t_recent = t.iloc[-100:].copy()
        t_recent["prev_mid"] = t_recent["mid"].shift(1)
        up = (t_recent["mid"] > t_recent["prev_mid"]).sum()
        down = (t_recent["mid"] < t_recent["prev_mid"]).sum()
        total = len(t_recent)

        tick_imba = (up - down) / (total + 1e-9)
        spread_mean = (t_recent["ask"] - t_recent["bid"]).mean()

        t_recent["prev_bid"] = t_recent["bid"].shift(1)
        t_recent["prev_ask"] = t_recent["ask"].shift(1)
        ofi = ((t_recent["bid"] > t_recent["prev_bid"]).sum() -
               (t_recent["bid"] < t_recent["prev_bid"]).sum()) \
            - ((t_recent["ask"] > t_recent["prev_ask"]).sum() -
               (t_recent["ask"] < t_recent["prev_ask"]).sum())

        bin_dist = 0.10 if "XAU" in self.current_symbol else 10.0
        bins = (t["mid"] / bin_dist).round() * bin_dist
        counts = bins.value_counts()
        poc_bin = counts.idxmax()

        current_price = t["mid"].iloc[-1]
        atr = self.calculate_atr(rates)
        poc_dist = (current_price - poc_bin) / (atr + 1e-9)

        total_vol = counts.sum()
        sorted_counts = counts.sort_index()
        cumsum = sorted_counts.cumsum()
        va_low = float(sorted_counts.index[cumsum >= total_vol * 0.15][0])
        va_high = float(sorted_counts.index[cumsum >= total_vol * 0.85][0])

        return {
            "tick_imbalance": float(tick_imba),
            "bid_ask_vol_imbalance": float(tick_imba),
            "spread_mean": float(spread_mean),
            "ofi_window": float(ofi / 100.0),
            "of_pressure_flag": 1 if tick_imba > 0.2 else (-1 if tick_imba < -0.2 else 0),
            "vprof_poc_dist": float(poc_dist),
            "vprof_in_value_area": 1 if va_low <= current_price <= va_high else 0,
            "vprof_hvn_flag": 1 if counts[poc_bin] > counts.mean() * 2 else 0,
            "vprof_lvn_flag": 0,
        }

    def get_observation_from_rates(self, rates, ticks=None):
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = _add_indicators(df)

        lookback = self.ensemble.lookback
        if len(df) < lookback:
            return None

        latest_idx = len(df) - 1
        price_lags = df["close"].iloc[latest_idx - lookback + 1:latest_idx + 1].values
        feat_latest = df.iloc[latest_idx]

        obs = list(price_lags) + [
            float(feat_latest["RSI"]),
            float(feat_latest["MACD"]),
            float(feat_latest["Signal_Line"]),
            0.0,
            10000.0,
            float(feat_latest["MACD_Hist"]),
            float(feat_latest["VWAP"]),
            float(feat_latest["close_minus_vwap"]),
            float(feat_latest["ATR"]),
            float(feat_latest["BB_width"]),
        ]

        if self.ensemble.micro:
            micro = self.build_micro_features(rates, ticks)
            obs += [
                micro["tick_imbalance"], micro["bid_ask_vol_imbalance"],
                micro["spread_mean"], micro["ofi_window"],
                micro["of_pressure_flag"],
                micro["vprof_poc_dist"], micro["vprof_in_value_area"],
                micro["vprof_hvn_flag"], micro["vprof_lvn_flag"],
            ]

        return np.array(obs).reshape(1, -1)

    def run_live_trading(self, interval_seconds=10, dry_run=False):
        logger.info(f"LONG-ONLY FTMO Session ({interval_seconds}s intervals)...")
        if not self.interface.initialize():
            logger.error("MT5 init failed.")
            return
        if not self.risk_mgr.initialize_balance():
            logger.error("Balance fetch failed.")
            return

        self.current_symbol = self.interface.symbol
        last_update = datetime.now() - timedelta(seconds=interval_seconds)

        try:
            while True:
                now = datetime.now()

                if not self.risk_mgr.can_trade():
                    account = self.interface.get_account_info()
                    if account:
                        loss_pct = (self.risk_mgr.day_start_balance - account.equity) / max(self.risk_mgr.day_start_balance, 1) * 100
                        if loss_pct >= self.settings.MAX_DAILY_LOSS_PCT:
                            logger.critical("HARD STOP: Daily loss limit. Closing all.")
                            self.interface.close_all_positions()
                            break
                    time.sleep(10)
                    continue

                if (now - last_update).total_seconds() >= interval_seconds:
                    # Check for existing positions — long-only, no trailing
                    positions = self.interface.get_positions()
                    if positions and len(positions) > 0:
                        last_update = now
                        time.sleep(interval_seconds)
                        continue

                    rates = self.interface.get_rates(count=self.ensemble.lookback + 100)
                    ticks = self.interface.get_ticks(count=2000)

                    if rates is not None and len(rates) >= self.ensemble.lookback:
                        obs = self.get_observation_from_rates(rates, ticks)
                        if obs is None:
                            continue

                        long_action, long_confidence = self.ensemble.predict(obs)
                        current_price = rates[-1]['close']
                        atr = self.calculate_atr(rates)

                        # MACD + VWAP confluence for long-only
                        df_temp = pd.DataFrame(rates)
                        df_temp['time'] = pd.to_datetime(df_temp['time'], unit='s')
                        df_temp = _add_indicators(df_temp)
                        latest = df_temp.iloc[-1]
                        bull_confirm = (
                            latest['MACD'] > latest['Signal_Line'] and
                            latest['close'] > latest['VWAP']
                        )

                        is_long_signal = (
                            long_action == 1.0 and
                            bull_confirm and
                            long_confidence >= self.long_confidence
                        )

                        if is_long_signal:
                            # Fixed 2:1 ATR-derived TP/SL
                            sl_dist = atr * 2.0
                            tp_dist = 2.0 * sl_dist
                            sl = current_price - sl_dist
                            tp = current_price + tp_dist

                            lots = self.risk_mgr.calculate_position_size(
                                float(long_confidence), sl_dist
                            )

                            if lots > 0:
                                if dry_run:
                                    logger.info(
                                        f"DRY RUN: LONG {lots} lots @ {current_price:.2f} | "
                                        f"SL: {sl:.2f} | TP: {tp:.2f} | "
                                        f"Conf: {long_confidence:.3f}"
                                    )
                                else:
                                    self.interface.send_order(1.0, lots, sl, tp)
                                    time.sleep(2)

                        last_update = now
                        logger.info(
                            f"Status — Price: {current_price:.2f} | "
                            f"LongAction: {long_action:.0f} | "
                            f"LongConf: {long_confidence:.3f}"
                        )

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        except Exception as e:
            logger.error(f"Execution error: {e}")
        finally:
            self.interface.shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LONG-ONLY FTMO Live Trading")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--long-thresh", type=float)
    parser.add_argument("--long-conf", type=float)
    args = parser.parse_args()

    if args.symbol:
        Settings.SYMBOL = args.symbol

    model_path = args.model or Settings.ENSEMBLE_MODEL_PATH
    trader = LiveEnsembleTrader(ensemble_path=model_path)
    if args.long_thresh is not None:
        trader.long_threshold = args.long_thresh
    if args.long_conf is not None:
        trader.long_confidence = args.long_conf

    trader.run_live_trading(interval_seconds=args.interval, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
