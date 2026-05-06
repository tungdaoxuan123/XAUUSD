#!/usr/bin/env python3
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import List, Dict

# Ensure local imports work
sys.path.append(os.getcwd())

from config import Settings, setup_logging
from mt5_interface import MT5Interface
from train_pipeline.sota_signal_generator import PatchTSTLite
from train_pipeline.live_features import build_live_features
from train_pipeline.dynamic_exits import ExitPlanner, ExitConfig, OpenPosition

try:
    import torch_directml
except ImportError:
    pass

logger = setup_logging()

class SimManager:
    """Tracks balance, open positions, and simulated order execution."""
    def __init__(self, initial_balance=10000.0):
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: Dict[int, OpenPosition] = {}
        self.history: List[dict] = []
        self.ticket_counter = 1000

    def open_trade(self, side, lots, entry, sl, tp, bar_idx):
        ticket = self.ticket_counter
        self.ticket_counter += 1
        pos = OpenPosition(
            side=side, entry=entry, sl=sl, tp=tp,
            entry_bar=bar_idx, best_price=entry
        )
        self.positions[ticket] = pos
        return ticket

    def close_trade(self, ticket, exit_price, reason, symbol="XAUUSD"):
        pos = self.positions.pop(ticket)
        # Contract size: 100,000 for Forex, 100 for Gold
        contract_size = 100000.0 if "GBP" in symbol or "EUR" in symbol else 100.0
        
        pnl = (exit_price - pos.entry) * pos.side * contract_size * getattr(pos, 'lots', 0.01)
        
        self.balance += pnl
        self.history.append({
            "ticket": ticket, "side": pos.side, "entry": pos.entry,
            "exit": exit_price, "pnl": pnl, "reason": reason, "lots": getattr(pos, 'lots', 0.01)
        })
        return pnl

def lots_multiplier_safeguard(pos, exit_price):
    return 1.0 # placeholder

class SOTABacktester:
    def __init__(self, model_path, config_path, device="cpu"):
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.features = self.cfg["features"]
        self.seq_len = self.cfg["seq_len"]
        
        ckpt = torch.load(model_path, map_location="cpu")
        self.model = PatchTSTLite(
            n_features=ckpt.get("n_features", len(self.features)),
            seq_len=ckpt.get("seq_len", self.cfg["seq_len"]),
            patch_len=ckpt.get("patch_len", self.cfg["patch_len"]),
            d_model=ckpt.get("d_model", 96),
            n_heads=ckpt.get("n_heads", 4),
            n_layers=ckpt.get("n_layers", 3)
        ).to(device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self.device = device
        
        self.mu = torch.from_numpy(np.asarray(ckpt["mu"], dtype=np.float32)).to(device)
        self.sd = torch.from_numpy(np.asarray(ckpt["sd"], dtype=np.float32)).to(device)
        self.temperature = ckpt.get("temperature", 1.0)

        self.planner = ExitPlanner(ExitConfig(
            min_prob_long=0.55,
            min_prob_short=0.55,
            max_hold_bars=30
        ))

    def predict(self, window_df: pd.DataFrame):
        X = window_df[self.features].values[-self.seq_len:]
        xt = torch.from_numpy(X.astype("float32")).to(self.device)
        xt = (xt - self.mu) / self.sd
        
        with torch.no_grad():
            logits = self.model(xt.unsqueeze(0))
            probs = torch.softmax(logits / self.temperature, dim=-1).cpu().numpy()[0]
        
        p_sell, p_hold, p_buy = probs[0], probs[1], probs[2]
        signal = -1 if p_sell > p_buy and p_sell > p_hold else 1 if p_buy > p_sell and p_buy > p_hold else 0
        conf = p_buy if signal > 0 else p_sell if signal < 0 else p_hold
        return signal, conf, p_buy, p_sell, p_hold

    def run(self, df: pd.DataFrame, initial_balance=10000.0, max_positions=3, symbol="XAUUSD"):
        sim = SimManager(initial_balance)
        n = len(df)
        
        # --- Speed Optimization: Pre-calculate all features once ---
        print(f"Pre-calculating features for {n} bars...")
        full_feat_df = build_live_features(df, self.features)
        lookback = self.seq_len
        # -----------------------------------------------------------

        print(f"Starting simulation over {n} bars...")
        
        for i in range(lookback, n):
            current_bar = df.iloc[i]
            # Use pre-calculated window slice
            window_slice = full_feat_df.iloc[i - lookback + 1 : i + 1]
            
            # 1. Manage open positions
            tickets = list(sim.positions.keys())
            for tid in tickets:
                pos = sim.positions[tid]
                # Note: ExitPlanner still needs raw prices for some checks, 
                # but we use the pre-calculated window slice for speed
                if pos.side > 0:
                    if current_bar['low'] <= pos.sl:
                        sim.close_trade(tid, pos.sl, "SL", symbol=symbol)
                        continue
                    if current_bar['high'] >= pos.tp:
                        sim.close_trade(tid, pos.tp, "TP", symbol=symbol)
                        continue
                else:
                    if current_bar['high'] >= pos.sl:
                        sim.close_trade(tid, pos.sl, "SL", symbol=symbol)
                        continue
                    if current_bar['low'] <= pos.tp:
                        sim.close_trade(tid, pos.tp, "TP", symbol=symbol)
                        continue
                
                # Pass pre-calculated features to manage_open
                upd = self.planner.manage_open(pos, window_slice, i)
                if upd.close_reason == "time":
                    sim.close_trade(tid, current_bar['close'], "TIME", symbol=symbol)
                elif upd.new_sl is not None:
                    pos.sl = upd.new_sl

            # 2. Entries
            if len(sim.positions) < max_positions:
                try:
                    # No more build_live_features inside the loop!
                    sig, conf, pb, ps, ph = self.predict(window_slice)
                    
                    if sig != 0:
                        plan = self.planner.build_plan(sig, conf, window_slice, sim.balance, 0.05)
                        if not plan.skip:
                            tid = sim.open_trade(plan.side, plan.lots, plan.entry, plan.sl, plan.tp, i)
                            sim.positions[tid].lots = plan.lots # attach lots to tracker
                            # print(f"Bar {i}: [ENTRY] {'BUY' if plan.side>0 else 'SELL'} @ {plan.entry:.2f} lots={plan.lots:.2f} conf={conf:.2f}")
                except Exception as e:
                    # print(f"Error at bar {i}: {e}")
                    pass

        print("\n" + "="*40)
        print("SOTA BACKTEST REPORT")
        print("="*40)
        print(f"Total Trades: {len(sim.history)}")
        if sim.history:
            wins = [t for t in sim.history if t['pnl'] > 0]
            total_pnl = sum([t['pnl'] for t in sim.history])
            print(f"Net Profit: ${total_pnl:.2f}")
            print(f"Win Rate: {len(wins)/len(sim.history)*100:.1f}%")
            print(f"Final Balance: ${sim.balance:.2f}")
        else:
            print("No trades.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--min-prob", type=float, default=0.55)
    args = parser.parse_args()

    iface = MT5Interface()
    if not iface.initialize(): return
    rates = iface.get_rates(count=1440 * args.days, symbol=args.symbol)
    iface.shutdown()
    if rates is None: return

    df = pd.DataFrame(rates)
    df.columns = [c.lower() for c in df.columns]
    
    # Calculate ATR since MT5 raw rates do not include it natively
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df["atr"] = df["ATR"] # For compatibility with lowercased components

    tester = SOTABacktester(Settings.SOTA_MODEL_PATH, Settings.SOTA_CONFIG_PATH)
    tester.planner.cfg.min_prob_long = args.min_prob
    tester.planner.cfg.min_prob_short = args.min_prob
    tester.run(df, max_positions=Settings.MAX_POSITIONS, symbol=args.symbol)

if __name__ == "__main__":
    main()
