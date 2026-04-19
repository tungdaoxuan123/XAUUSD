import pandas as pd
import numpy as np
from ensemble_trader import EnsembleTrader
from market_regime_detector import MarketRegimeDetector
from trading_env import TradingEnv
import time
import logging
from datetime import datetime, timedelta

from config import Settings, setup_logging
from mt5_interface import MT5Interface
from risk_manager import FTMORiskManager
from train_pipeline.ensemble_gpu import EnsembleGPU, _add_indicators

# Setup MT5-aware logging
logger = setup_logging()

class LiveEnsembleTrader:
    """
    Live trading system adapted for FTMO MT5 Challenge using ensemble predictions
    """

    def __init__(self, ensemble_path=None):
        self.settings = Settings
        self.interface = MT5Interface()
        self.risk_mgr = FTMORiskManager(self.interface)
        
        # Load the new GPU/Microstructure ensemble
        model_path = ensemble_path or self.settings.ENSEMBLE_MODEL_PATH
        self.ensemble = EnsembleGPU.load(model_path)

        # Initialize risk settings for scalping (FTMO 10k)
        # Load thresholds
        self.buy_threshold = self.settings.BUY_THRESHOLD
        self.sell_threshold = self.settings.SELL_THRESHOLD
        self.buy_confidence = self.settings.BUY_CONFIDENCE
        self.sell_confidence = self.settings.SELL_CONFIDENCE

        self.partial_closed_tickets = set()
        
        # Symbol discovery
        self.current_symbol = self.interface.symbol
        if not self.interface.authorized:
            self.interface.initialize()
            self.current_symbol = self.interface.symbol

        logger.info(f"Initialized BTC-Micro-Ready Live Bot - Symbol: {self.current_symbol}")
        logger.info(f"Model: {model_path} | Micro: {self.ensemble.micro}")

    def calculate_atr(self, rates, period=14):
        """Calculates Average True Range from MT5 rates"""
        df = pd.DataFrame(rates)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(period).mean().iloc[-1]

    def build_micro_features(self, rates, ticks):
        """Computes real-time microstructure features from raw MT5 ticks"""
        if ticks is None or len(ticks) == 0:
            return {f: 0.0 for f in ["tick_imbalance", "bid_ask_vol_imbalance", "spread_mean", "ofi_window", "of_pressure_flag", "vprof_poc_dist", "vprof_in_value_area", "vprof_hvn_flag", "vprof_lvn_flag"]}

        t = pd.DataFrame(ticks)
        t["time"] = pd.to_datetime(t["time"], unit="s")
        t["mid"] = (t["bid"] + t["ask"]) / 2.0
        
        # 1. Order Flow (latest 100 ticks)
        t_recent = t.iloc[-100:].copy()
        t_recent["prev_mid"] = t_recent["mid"].shift(1)
        up = (t_recent["mid"] > t_recent["prev_mid"]).sum()
        down = (t_recent["mid"] < t_recent["prev_mid"]).sum()
        total = len(t_recent)
        
        tick_imba = (up - down) / (total + 1e-9)
        spread_mean = (t_recent["ask"] - t_recent["bid"]).mean()
        
        # Price-based OFI
        t_recent["prev_bid"] = t_recent["bid"].shift(1)
        t_recent["prev_ask"] = t_recent["ask"].shift(1)
        ofi = ((t_recent["bid"] > t_recent["prev_bid"]).sum() - (t_recent["bid"] < t_recent["prev_bid"]).sum()) \
            - ((t_recent["ask"] > t_recent["prev_ask"]).sum() - (t_recent["ask"] < t_recent["prev_ask"]).sum())
        
        # 2. Volume Profile (last 2000 ticks)
        # Use symbol-aware bin size: 0.10 for XAUUSD, 10.0 for BTCUSD
        bin_dist = 0.10 if "XAU" in self.current_symbol else 10.0
        bins = (t["mid"] / bin_dist).round() * bin_dist
        counts = bins.value_counts()
        poc_bin = counts.idxmax()
        
        current_price = t["mid"].iloc[-1]
        atr = self.calculate_atr(rates)
        poc_dist = (current_price - poc_bin) / (atr + 1e-9)
        
        # Value Area (approximate)
        total_vol = counts.sum()
        sorted_counts = counts.sort_index()
        cumsum = sorted_counts.cumsum()
        va_low = sorted_counts.index[cumsum >= total_vol * 0.15][0]
        va_high = sorted_counts.index[cumsum >= total_vol * 0.85][0]
        
        return {
            "tick_imbalance": float(tick_imba),
            "bid_ask_vol_imbalance": float(tick_imba), # Proxy
            "spread_mean": float(spread_mean),
            "ofi_window": float(ofi / 100.0),
            "of_pressure_flag": 1 if tick_imba > 0.2 else (-1 if tick_imba < -0.2 else 0),
            "vprof_poc_dist": float(poc_dist),
            "vprof_in_value_area": 1 if va_low <= current_price <= va_high else 0,
            "vprof_hvn_flag": 1 if counts[poc_bin] > counts.mean() * 2 else 0,
            "vprof_lvn_flag": 0 # Not stable in real-time
        }

    def get_observation_from_rates(self, rates, ticks=None):
        """Prepare feature vector matching the 79-feature BTC micro model"""
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = _add_indicators(df)
        
        lookback = self.ensemble.lookback
        if len(df) < lookback:
            return None
            
        latest_idx = len(df) - 1
        price_lags = df["close"].iloc[latest_idx - lookback + 1 : latest_idx + 1].values
        feat_latest = df.iloc[latest_idx]
        
        # Basic + Expanded Features
        obs = list(price_lags) + [
            float(feat_latest["RSI"]),
            float(feat_latest["MACD"]),
            float(feat_latest["Signal_Line"]),
            0.0, # pos
            10000.0, # balance
            float(feat_latest["MACD_Hist"]),
            float(feat_latest["VWAP"]),
            float(feat_latest["close_minus_vwap"]),
            float(feat_latest["ATR"]),
            float(feat_latest["BB_width"]),
        ]
        
        # Micro Features
        if self.ensemble.micro:
            micro = self.build_micro_features(rates, ticks)
            obs += [
                micro["tick_imbalance"], micro["bid_ask_vol_imbalance"], micro["spread_mean"],
                micro["ofi_window"], micro["of_pressure_flag"],
                micro["vprof_poc_dist"], micro["vprof_in_value_area"], micro["vprof_hvn_flag"], micro["vprof_lvn_flag"]
            ]
            
        return np.array(obs).reshape(1, -1)

    def monitor_trailing(self):
        """Dynamic ATR + R-multiple trailing for XAUUSD volatility"""
        positions = self.interface.get_positions()
        
        if not positions:
            if np.random.random() < 0.05: # Heartbeat every ~2 mins
                 logger.info("Monitor: Scanning for opportunities...")
            # Clean up old tickets from tracking set
            if len(self.partial_closed_tickets) > 0:
                self.partial_closed_tickets.clear()
            return
        
        rates = self.interface.get_rates(count=50)  # Fresh ATR data
        if rates is None or len(rates) < 20:
            return
            
        atr = self.calculate_atr(rates, period=14)  # Current volatility
        
        for pos in positions:
            entry = pos.price_open
            current = pos.price_current
            sl = pos.sl or 0
            ticket = pos.ticket
            
            # XAUUSD 0.1 move = 1 pip (standard)
            # profit_pips calculation: 1.00 price move = 10 pips (for 100 points brokers)
            profit_pips = (current - entry) * 10 if pos.type == 0 else (entry - current) * 10
            
            # Calculate dynamic R based on actual ATR distance if SL is 0, else initial SL distance
            initial_sl_dist = abs(entry - sl) if sl != 0 else atr * 2
            current_r = profit_pips / (initial_sl_dist * 10) if initial_sl_dist > 0 else 0
            
            # 1. 2R+ -> Full Exit
            if current_r >= 2.0:
                logger.info(f"2R+ TARGET REACHED: Closing {ticket} | R: {current_r:.1f} | Pips: {profit_pips:.1f}")
                self.interface.close_position(ticket)
                continue
                
            # 2. 1.5R -> Partial close 50%
            if current_r >= 1.5 and ticket not in self.partial_closed_tickets:
                half_vol = round(pos.volume / 2, 2)
                if half_vol >= 0.01:
                    logger.info(f"1.5R PARTIAL EXIT: {ticket} | R: {current_r:.1f} | Vol: {half_vol}")
                    if self.interface.close_partial_position(ticket, half_vol):
                        self.partial_closed_tickets.add(ticket)
            
            # 3. 1R+ -> Dynamic ATR Trail (0.8 ATR distance)
            if current_r >= 1.0:
                trail_dist = atr * 0.8  # Adaptive to volatility
                
                if pos.type == 0:  # BUY
                    new_sl = current - trail_dist
                    if new_sl > sl + (atr * 0.1):  # Only move if meaningful
                        logger.info(f"ATR TRAIL BUY {ticket}: {sl:.2f} -> {new_sl:.2f} | ATR: {atr:.2f} | R: {current_r:.1f}")
                        self.interface.modify_position(ticket, new_sl, pos.tp)
                else:  # SELL
                    new_sl = current + trail_dist
                    if new_sl < sl - (atr * 0.1) or sl == 0:
                        logger.info(f"ATR TRAIL SELL {ticket}: {sl:.2f} -> {new_sl:.2f} | ATR: {atr:.2f} | R: {current_r:.1f}")
                        self.interface.modify_position(ticket, new_sl, pos.tp)
            
            # 4. 0.5R -> Breakeven + buffer
            elif current_r >= 0.5:
                buffer = atr * 0.3  # Dynamic buffer, not fixed pip
                
                if pos.type == 0:  # BUY
                    if sl < entry:
                        new_sl = entry + buffer
                        logger.info(f"DYNAMIC BE+: BUY {ticket} -> {new_sl:.2f} (Buffer: {buffer:.2f})")
                        self.interface.modify_position(ticket, new_sl, pos.tp)
                else: # SELL
                    if sl > entry or sl == 0:
                        new_sl = entry - buffer
                        logger.info(f"DYNAMIC BE+: SELL {ticket} -> {new_sl:.2f} (Buffer: {buffer:.2f})")
                        self.interface.modify_position(ticket, new_sl, pos.tp)

    def run_live_trading(self, interval_seconds=10, dry_run=False):
        """Main FTMO trading loop with specified intervals"""
        logger.info(f"Starting FTMO MT5 Trading Session ({interval_seconds}-second intervals)...")
        
        if not self.interface.initialize():
            logger.error("Failed to initialize MT5 interface. Exiting.")
            return

        if not self.risk_mgr.initialize_balance():
            logger.error("Failed to fetch initial balance. Exiting.")
            return

        self.current_symbol = self.interface.symbol
        last_update = datetime.now() - timedelta(seconds=interval_seconds)

        try:
            while True:
                now = datetime.now()
                # Check for trailing stops on active positions
                self.monitor_trailing()

                # Safety check: Daily/Total Drawdown
                if not self.risk_mgr.can_trade():
                    account = self.interface.get_account_info()
                    if account and (self.risk_mgr.day_start_balance - account.equity) / self.risk_mgr.day_start_balance * 100 >= self.settings.MAX_DAILY_LOSS_PCT:
                        logger.critical("HARD STOP: Daily loss limit breached. Closing all positions.")
                        self.interface.close_all_positions()
                        break
                    time.sleep(10) # Wait and check again
                    continue

                # Execution loop
                if (now - last_update).total_seconds() >= interval_seconds:
                    # Fetch data - using buffer for indicator warm-up
                    rates = self.interface.get_rates(count=self.ensemble.lookback + 100)
                    ticks = self.interface.get_ticks(count=2000)
                    
                    if rates is not None and len(rates) >= self.ensemble.lookback:
                        obs = self.get_observation_from_rates(rates, ticks)
                        if obs is None:
                            continue
                            
                        action, confidence = self.ensemble.predict(obs)
                        
                        current_price = rates[-1]['close']
                        atr = self.calculate_atr(rates)
                        
                        # Confluence Check (MACD 8-24-9 + VWAP)
                        df_temp = pd.DataFrame(rates)
                        df_temp['time'] = pd.to_datetime(df_temp['time'], unit='s')
                        df_temp = _add_indicators(df_temp)
                        latest = df_temp.iloc[-1]
                        
                        bull_confirm = latest['MACD'] > latest['Signal_Line'] and latest['close'] > latest['VWAP']
                        bear_confirm = latest['MACD'] < latest['Signal_Line'] and latest['close'] < latest['VWAP']
                        
                        # High-confidence signals with confluence
                        is_buy = action > 0 and bull_confirm
                        is_sell = action < 0 and bear_confirm
                        
                        # Buy/Sell Threshold Check
                        is_buy_signal = action >= self.buy_threshold and confidence >= self.buy_confidence
                        is_sell_signal = action <= -self.sell_threshold and confidence >= self.sell_confidence
                        
                        if is_buy_signal or is_sell_signal:
                            # Calculate SL and TP based on ATR (2*ATR SL, 3*ATR TP)
                            if action > 0: # BUY
                                sl = current_price - (2 * atr)
                                tp = current_price + (3 * atr)
                            else: # SELL
                                sl = current_price + (2 * atr)
                                tp = current_price - (3 * atr)
                                
                            # Calculate lots with ATR-based stop distance points
                            lots = self.risk_mgr.calculate_position_size(float(confidence), atr * 2)
                            
                            if lots > 0:
                                if dry_run:
                                    logger.info(f"DRY RUN: Would {'BUY' if action > 0 else 'SELL'} {lots} lots at {current_price} | SL: {sl:.2f} | TP: {tp:.2f}")
                                else:
                                    self.interface.send_order(action, lots, sl, tp)
                                    time.sleep(2)
                        
                        last_update = now
                        logger.info(f"Status - Price: {current_price:.2f} | Action: {float(action):.3f} | Confidence: {float(confidence):.3f}")

                time.sleep(10) # 10-second interval as requested

        except KeyboardInterrupt:
            logger.info("Stopped by user. Closing connection.")
        except Exception as e:
            logger.error(f"Execution error: {e}")
        finally:
            self.interface.shutdown()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live FTMO Trading with AI Ensemble")
    parser.add_argument("--dry-run", action="store_true", help="Monitor without placing trades")
    parser.add_argument("--symbol", type=str, help="Override symbol in config")
    parser.add_argument("--model", type=str, help="Override ensemble model path")
    parser.add_argument("--interval", type=int, default=10, help="Timeframe interval in seconds")
    parser.add_argument("--buy-thresh", type=float, help="Override BUY_THRESHOLD")
    parser.add_argument("--sell-thresh", type=float, help="Override SELL_THRESHOLD")
    parser.add_argument("--buy-conf", type=float, help="Override BUY_CONFIDENCE")
    parser.add_argument("--sell-conf", type=float, help="Override SELL_CONFIDENCE")
    args = parser.parse_args()

    if args.symbol:
        Settings.SYMBOL = args.symbol
    
    model_path = args.model or Settings.ENSEMBLE_MODEL_PATH

    trader = LiveEnsembleTrader(ensemble_path=model_path)
    
    # Apply overrides from CLI
    if args.buy_thresh is not None: trader.buy_threshold = args.buy_thresh
    if args.sell_thresh is not None: trader.sell_threshold = args.sell_thresh
    if args.buy_conf is not None: trader.buy_confidence = args.buy_conf
    if args.sell_conf is not None: trader.sell_confidence = args.sell_conf

    trader.run_live_trading(interval_seconds=args.interval, dry_run=args.dry_run)

if __name__ == "__main__":
    main()