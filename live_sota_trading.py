import pandas as pd
import numpy as np
import time
import logging
import json
import os
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
try:
    import torch_directml
except ImportError:
    pass
import MetaTrader5 as mt5

from config import Settings, setup_logging
from mt5_interface import MT5Interface
from risk_manager import FTMORiskManager
from train_pipeline.sota_signal_generator import PatchTSTLite, LABEL_UNMAP

# Setup dedicated SOTA logging
logger = setup_logging()

class SOTASignalGenerator:
    """In-process inference engine for PatchTST SOTA model"""
    def __init__(self, model_path, config_path, device="cpu"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SOTA config not found: {config_path}")
        
        with open(config_path) as f:
            cfg = json.load(f)
            
        self.features = cfg["features"]
        self.seq_len = cfg["seq_len"]
        self.patch_len = cfg["patch_len"]
        
        # Force CPU — DirectML silently returns zeros for Transformer ops
        if device is not None and "privateuseone" in str(device):
            logger.warning("DirectML detected — forcing SOTA to CPU (TransformerEncoderLayer unsupported on DirectML)")
            device = "cpu"

        self.device = device
        logger.info(f"Loading SOTA model from {model_path} on {self.device}...")
        
        # 1. Load checkpoint on CPU first to prevent striding errors
        ckpt = torch.load(model_path, map_location="cpu")
        
        # 2. Initialize model on self.device (enforced CPU if DirectML)
        self.model = PatchTSTLite(
            n_features=ckpt["n_features"],
            seq_len=ckpt["seq_len"],
            patch_len=ckpt["patch_len"]
        ).to(self.device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        
        # 3. Load normalization stats from checkpoint
        self.mu = torch.from_numpy(ckpt["mu"]).to(self.device) if "mu" in ckpt else None
        self.sd = torch.from_numpy(ckpt["sd"]).to(self.device) if "sd" in ckpt else None

    def predict(self, df):
        """Prepare window and run inference"""
        # Ensure we have all necessary features
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            logger.error(f"SOTA Predict: Missing features {missing}")
            return 0, 0.5, 0.5

        # Get the latest window
        X = df[self.features].astype("float32").values[-self.seq_len:]
        X = torch.tensor(X).to(self.device)
        
        # Normalize (Use saved stats if available, else batch-norm as fallback)
        if self.mu is not None and self.sd is not None:
            X = (X - self.mu) / self.sd
        else:
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
            
        X = X.unsqueeze(0) # (1, seq_len, F)
        
        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            
        signal = LABEL_UNMAP[int(probs.argmax())]
        return int(signal), float(probs[2]), float(probs[0]) # signal, p_buy, p_sell

class LiveSOTATrader:
    """Specialized Live Trading System for PatchTST SOTA signals"""
    
    def __init__(self, model_path=None, config_path=None, device=None):
        self.settings = Settings
        self.interface = MT5Interface()
        self.risk_mgr = FTMORiskManager(self.interface)
        
        # Load SOTA components
        m_path = model_path or self.settings.SOTA_MODEL_PATH
        c_path = config_path or self.settings.SOTA_CONFIG_PATH
        d_device = device or self.settings.SOTA_DEVICE
        
        self.sota = SOTASignalGenerator(m_path, c_path, d_device)
        
        self.current_symbol = self.interface.symbol
        if not self.interface.authorized:
            self.interface.initialize()
            self.current_symbol = self.interface.symbol
            
        self.buy_confidence = self.settings.BUY_CONFIDENCE
        self.sell_confidence = self.settings.SELL_CONFIDENCE
        
        logger.info(f"Initialized SOTA Live Trader - Symbol: {self.current_symbol}")

    def prepare_sota_data(self, rates):
        """Prepare raw rates for SOTA inference with all 21 trained features"""
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        
        # Price Sanity Check
        last_close = df["close"].iloc[-1]
        if last_close > 10000:
            logger.error(f"prepare_sota_data: Price {last_close} looks like BTC, not XAU! Check symbol.")
            return df 
        
        # 1. Technical Indicators
        # ATR (14-period)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = tr.rolling(window=14).mean()
        
        # VWAP Proxy
        df['VWAP'] = (df['close'] * df['tick_volume']).rolling(100).sum() / (df['tick_volume'].rolling(100).sum() + 1e-9)
        df['close_minus_vwap'] = df['close'] - df['VWAP']
        
        # 2. Synthetic Microstructure (Tick-free proxies)
        df["tick_imbalance"] = (df["close"] - df["open"]).rolling(5).mean()
        df["bid_ask_vol_imbalance"] = df["tick_imbalance"] 
        
        # Order Flow - Fixed float64 cast to prevent integer overflow
        df["tick_volume"] = df["tick_volume"].astype("float64") 
        df["ofi_window"] = df["tick_volume"].diff().fillna(0)
        df["of_pressure_flag"] = np.where(df["tick_imbalance"] > 0, 1, np.where(df["tick_imbalance"] < 0, -1, 0))
        
        # Spread & Liquidity
        df["spread_mean"] = (df["high"] - df["low"]).rolling(10).mean()
        df["spread_std"] = (df["high"] - df["low"]).rolling(10).std()
        df["kyle_lambda"] = abs(df["close"].diff()) / (df["tick_volume"] + 1e-6)
        df["amihud_illiq"] = abs(df["close"].diff() / df["close"]) / (df["tick_volume"] + 1e-9)
        
        # 3. Volume Profile Proxies
        vprof_window = 60
        df["poc_approx"] = df["close"].rolling(vprof_window).mean()
        df["vprof_poc_dist"] = (df["close"] - df["poc_approx"]) / (df["ATR"] + 1e-9)
        df["vprof_in_value_area"] = 1 # Approximation
        df["vprof_hvn_flag"] = np.where(df["tick_volume"] > df["tick_volume"].rolling(20).mean() * 1.5, 1, 0)
        df["vprof_lvn_flag"] = 0
        
        # 4. Regime
        df["jump_flag"] = np.where(abs(df["close"].diff()) > df["ATR"].shift(1) * 2.0, 1, 0)
        df["signed_vol_z"] = (df["tick_volume"] - df["tick_volume"].rolling(20).mean()) / (df["tick_volume"].rolling(20).std() + 1e-6)
        df["vol_regime"] = np.where(df["ATR"] > df["ATR"].rolling(50).mean(), 1, 0)
        
        # Final cleanup
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    def run_live_trading(self, interval_seconds=10, dry_run=False):
        """Main SOTA trading loop"""
        logger.info(f"Starting SOTA MT5 Session ({interval_seconds}s intervals)...")
        
        if not self.interface.initialize():
            logger.error("MT5 Initialization failed.")
            return

        self.risk_mgr.initialize_balance()
        last_update = datetime.now() - timedelta(seconds=interval_seconds)

        try:
            while True:
                now = datetime.now()
                
                # Check for opportunities
                if (now - last_update).total_seconds() >= interval_seconds:
                    # To enforce "one trade at a time", check for existing positions first
                    positions = self.interface.get_positions()
                    if positions is not None and len(positions) > 0:
                        # Log status occasionally but skip the scan
                        if now.second % 30 < 2:
                            logger.info("Scan skipped: Position currently open.")
                        last_update = now
                        continue

                    # Fetch 150 bars for warm-up
                    target_symbol = "XAUUSD" if "XAU" in self.current_symbol else self.current_symbol
                    rates = self.interface.get_rates(count=150, symbol=target_symbol)
                    
                    if rates is not None and len(rates) >= 60:
                        df_enriched = self.prepare_sota_data(rates)
                        
                        # Sanity check: ensure data wasn't rejected by price filter
                        if len(df_enriched.columns) < 20:
                            last_update = now
                            continue
                            
                        signal, p_buy, p_sell = self.sota.predict(df_enriched)
                        confidence = max(p_buy, p_sell)
                        current_price = rates[-1]['close']
                        atr = df_enriched['ATR'].iloc[-1]
                        
                        logger.info(f"Scan - Signal: {signal} | Conf: {confidence:.3f} | Price: {current_price:.2f}")

                        if signal != 0 and (p_buy >= self.buy_confidence or p_sell >= self.sell_confidence):
                            # Risk management
                            sl_dist = atr * 2.0
                            lots = self.risk_mgr.calculate_position_size(float(confidence), sl_dist)
                            
                            if lots > 0:
                                sl = current_price - sl_dist if signal > 0 else current_price + sl_dist
                                tp = current_price + (atr * 3.0) if signal > 0 else current_price - (atr * 3.0)
                                
                                if dry_run:
                                    logger.info(f"DRY RUN: New Trade {'BUY' if signal > 0 else 'SELL'} {lots} lots at {current_price}")
                                else:
                                    self.interface.send_order(signal, lots, sl, tp)
                                    time.sleep(2)
                    
                    last_update = now
                
                time.sleep(2)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            self.interface.shutdown()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live SOTA Trading with PatchTST")
    parser.add_argument("--dry-run", action="store_true", help="Monitor without placing trades")
    parser.add_argument("--device", type=str, help="Override SOTA_DEVICE")
    parser.add_argument("--symbol", type=str, help="Override symbol")
    args = parser.parse_args()

    if args.symbol:
        Settings.SYMBOL = args.symbol

    trader = LiveSOTATrader(device=args.device)
    trader.run_live_trading(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
