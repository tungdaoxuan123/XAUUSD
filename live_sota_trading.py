"""
live_sota_trading.py — LONG-ONLY SOTA live bot
===============================================

Uses PatchTST-lite transformer with temperature-calibrated probabilities
for LONG-ONLY entry decisions. Fixed 2:1 ATR-derived TP/SL via ExitPlanner.

Key changes for long-only redesign:
  - binary exit planning (no short side)
  - Entry filter: p_long > min_prob_long only
  - 2:1 RR: tp_dist = 2 × sl_dist
  - No trailing / in-trade management beyond exit
"""

from __future__ import annotations

import time
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import torch_directml  # noqa: F401
except ImportError:
    pass

import MetaTrader5 as mt5  # noqa: F401  (used indirectly via interface)

from config import Settings, setup_logging
from mt5_interface import MT5Interface
from risk_manager import FTMORiskManager

from train_pipeline.sota_signal_generator import PatchTSTLite
from train_pipeline.live_features import build_live_features
from train_pipeline.dynamic_exits import ExitPlanner, ExitConfig, OpenPosition

logger = setup_logging()


# ---------------------------------------------------------------------------
# Signal generator (now with temperature-scaled probabilities)
# ---------------------------------------------------------------------------

class SOTASignalGenerator:
    """PatchTST inference with calibrated probabilities."""

    def __init__(self, model_path: str, config_path: str, device: str = "cpu"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SOTA config not found: {config_path}")
        with open(config_path) as f:
            cfg = json.load(f)

        self.features = cfg["features"]
        self.seq_len = cfg["seq_len"]
        self.patch_len = cfg["patch_len"]
        # Temperature from v2 trainer; default 1.0 if v1 checkpoint
        self.temperature = float(cfg.get("temperature", 1.0))

        if device is not None and "privateuseone" in str(device):
            logger.warning("DirectML detected — forcing CPU for TransformerEncoderLayer")
            device = "cpu"
        self.device = device

        logger.info(f"Loading SOTA model: {model_path}  T={self.temperature:.3f}")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        # Initialize model with robust metadata retrieval
        n_features = ckpt.get("n_features", len(ckpt.get("features", [])))
        if n_features == 0:
            n_features = len(self.features)

        # Retrieve architectural settings (defaulting to v2 trainer defaults)
        # If metadata is missing (v1 models), infer from state dict shapes
        state = ckpt.get("state", {})
        
        # d_model detection
        d_model = ckpt.get("d_model")
        if d_model is None:
            if "pos_embed" in state:
                d_model = state["pos_embed"].shape[-1]
            elif "patch_embed.weight" in state:
                d_model = state["patch_embed.weight"].shape[0]
            else:
                d_model = 96 # Final fallback
        
        # n_layers detection
        n_layers = ckpt.get("n_layers")
        if n_layers is None:
            # Count the highest layer index in the state dict
            layer_keys = [k for k in state.keys() if "encoder.layers." in k]
            if layer_keys:
                n_layers = max([int(k.split(".")[2]) for k in layer_keys]) + 1
            else:
                n_layers = 3 # Final fallback

        n_heads = ckpt.get("n_heads", 4)

        self.model = PatchTSTLite(
            n_features=n_features,
            seq_len=ckpt.get("seq_len", self.seq_len),
            patch_len=ckpt.get("patch_len", self.patch_len),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        ).to(self.device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

        mu = ckpt.get("mu"); sd = ckpt.get("sd")
        self.mu = torch.from_numpy(np.asarray(mu, dtype=np.float32)).to(self.device) if mu is not None else None
        self.sd = torch.from_numpy(np.asarray(sd, dtype=np.float32)).to(self.device) if sd is not None else None
        if self.mu is None:
            logger.warning("Checkpoint lacks mu/sd — using per-window normalization (less safe)")

    def predict(self, df: pd.DataFrame):
        """Return (signal, p_long, p_wait, entropy)."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            logger.error(f"Predict: missing features {missing[:5]}… ({len(missing)})")
            return 0, 0.5, 0.5, np.log(2)

        X = df[self.features].astype("float32").values[-self.seq_len:]
        if len(X) < self.seq_len:
            return 0, 0.5, 0.5, np.log(2)

        X = torch.tensor(X, device=self.device)
        if self.mu is not None and self.sd is not None:
            X = (X - self.mu) / self.sd
        else:
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        X = X.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits / max(self.temperature, 1e-3), dim=-1)
            p = probs.cpu().numpy()[0]

        ent = float(-(p * np.log(p + 1e-12)).sum())
        # Binary: class 1 = LONG, class 0 = WAIT
        if p.shape[0] >= 2:
            p_long = float(p[1])
            p_wait = float(p[0])
        else:
            p_long = float(p[0])
            p_wait = 1.0 - p_long
        signal = 1 if p_long > 0.5 else 0
        return signal, p_long, p_wait, ent


# ---------------------------------------------------------------------------
# Live trader
# ---------------------------------------------------------------------------

class LiveSOTATrader:
    def __init__(self, model_path=None, config_path=None, device=None,
                 exit_config: Optional[ExitConfig] = None, direction: int = 1):
        self.settings = Settings
        self.interface = MT5Interface()
        self.risk_mgr = FTMORiskManager(self.interface)

        m = model_path or self.settings.SOTA_MODEL_PATH
        c = config_path or self.settings.SOTA_CONFIG_PATH
        d = device or self.settings.SOTA_DEVICE
        self.sota = SOTASignalGenerator(m, c, d)

        if not self.interface.authorized:
            self.interface.initialize()
        self.current_symbol = self.interface.symbol
        self.direction = direction

        if "GBP" in self.current_symbol:
            base_cfg = ExitConfig(
                atr_mult_sl=1.0,
                min_sl_pips=0.00050,
                contract_size=100000.0,
                max_hold_bars=45,
            )
        else:
            base_cfg = ExitConfig(
                min_sl_pips=1.5,
                contract_size=100.0,
            )

        if exit_config:
            base_cfg.min_prob = exit_config.min_prob

        self.planner = ExitPlanner(base_cfg, direction=direction)

        self._trackers: Dict[int, OpenPosition] = {}
        self._bar_counter = 0
        self._last_order_bar = -1
        side_label = "LONG" if direction > 0 else "SHORT"
        logger.info(
            f"{side_label}-ONLY SOTA Trader ready — symbol={self.current_symbol} "
            f"T={self.sota.temperature:.3f} "
            f"min_p={self.planner.cfg.min_prob:.2f}"
        )

    # ---- helpers ----------------------------------------------------------

    def _get_equity(self) -> float:
        try:
            return float(self.interface.account_info().equity)
        except Exception:
            return float(getattr(self.settings, "DEFAULT_EQUITY", 10000.0))

    def _current_spread(self) -> float:
        try:
            tick = self.interface.get_tick()
            if tick is None:
                return 0.0
            return float(tick.ask - tick.bid)
        except Exception:
            return 0.0

    # ---- main loop --------------------------------------------------------

    def run_live_trading(self, interval_seconds: int = 10, dry_run: bool = False):
        side_label = "LONG" if self.direction > 0 else "SHORT"
        logger.info(f"{side_label}-ONLY SOTA live loop ({interval_seconds}s, dry_run={dry_run})")
        if not self.interface.initialize():
            logger.error("MT5 init failed"); return
        self.risk_mgr.initialize_balance()

        last_update = datetime.now() - timedelta(seconds=interval_seconds)
        try:
            while True:
                now = datetime.now()
                if (now - last_update).total_seconds() < interval_seconds:
                    time.sleep(1); continue
                last_update = now
                self._bar_counter += 1

                positions = self.interface.get_positions() or []
                self._manage_open_positions(positions, dry_run=dry_run)

                if len(positions) >= self.settings.MAX_POSITIONS:
                    continue

                target = "XAUUSD" if "XAU" in self.current_symbol else self.current_symbol
                rates = self.interface.get_rates(
                    count=max(300, self.sota.seq_len + 50), symbol=target
                )
                if rates is None or len(rates) < self.sota.seq_len + 30:
                    logger.warning(f"Not enough rates ({0 if rates is None else len(rates)})")
                    continue

                rates_df = pd.DataFrame(rates)
                df = build_live_features(rates_df, self.sota.features)

                signal, p_long, p_wait, entropy = self.sota.predict(df)

                side_label = "LONG" if self.direction > 0 else "SHORT"
                logger.info(
                    f"scan signal={side_label if signal>0 else 'WAIT'} "
                    f"p_long={p_long:.3f} p_wait={p_wait:.3f} H={entropy:.3f}"
                )

                if signal == 0:
                    continue

                equity = self._get_equity()
                spread = self._current_spread()
                plan = self.planner.build_plan(signal, p_long, df, equity, spread)
                if plan.skip:
                    logger.info(f"skip: {plan.reason}")
                    continue

                logger.info(
                    f"PLAN LONG lots={plan.lots:.2f} "
                    f"entry={plan.entry:.2f} sl={plan.sl:.2f} tp={plan.tp:.2f} "
                    f"RR={plan.rr:.2f} risk={plan.equity_risk*100:.2f}%"
                )

                if dry_run:
                    continue

                ticket = self.interface.send_order(
                    plan.side, plan.lots, plan.sl, plan.tp,
                    max_slippage_pts=self.interface.dynamic_deviation(spread_pts=spread)
                )
                if ticket:
                    self._trackers[ticket] = OpenPosition(
                        side=plan.side, entry=plan.entry, sl=plan.sl, tp=plan.tp,
                        entry_bar=self._bar_counter, best_price=plan.entry,
                    )
                    self._last_order_bar = self._bar_counter
                time.sleep(2)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            self.interface.shutdown()

    # ---- in-trade management ---------------------------------------------

    def _manage_open_positions(self, positions, dry_run: bool = False):
        if not positions:
            self._trackers.clear()
            return

        active_tickets = [p.ticket for p in positions]
        dead = [tid for tid in self._trackers if tid not in active_tickets]
        for tid in dead:
            del self._trackers[tid]

        target = "XAUUSD" if "XAU" in self.current_symbol else self.current_symbol
        rates = self.interface.get_rates(
            count=max(200, self.sota.seq_len + 30), symbol=target
        )
        if rates is None:
            return
        df = build_live_features(pd.DataFrame(rates), self.sota.features)

        for p in positions:
            tid = p.ticket
            tracker = self._trackers.get(tid)
            if tracker is None:
                tracker = OpenPosition(
                    side=self.direction,
                    entry=float(p.price_open),
                    sl=float(p.sl),
                    tp=float(p.tp),
                    entry_bar=self._bar_counter,
                    best_price=float(p.price_open),
                )
                self._trackers[tid] = tracker

            upd = self.planner.manage_open(tracker, df, self._bar_counter)
            if upd.close_reason == "time":
                logger.info(f"Ticket {tid}: time stop hit -> closing")
                if not dry_run:
                    self.interface.close_position(tid)
                del self._trackers[tid]
                continue

            if upd.new_sl is not None:
                fmt = ".5f" if "GBP" in self.current_symbol else ".2f"
                logger.info(f"Ticket {tid}: SL update -> {upd.new_sl:{fmt}}")
                if not dry_run:
                    self.interface.modify_sl(tid, upd.new_sl)
                tracker.sl = upd.new_sl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Live SOTA trading (long or short)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", type=str)
    ap.add_argument("--symbol", type=str)
    ap.add_argument("--min-prob", type=float, default=None,
                    help="Override min probability threshold")
    ap.add_argument("--side", type=str, default="long", choices=["long", "short"],
                    help="Trade direction")
    args = ap.parse_args()

    if args.symbol:
        Settings.SYMBOL = args.symbol

    exit_cfg = ExitConfig()
    if args.min_prob is not None:
        exit_cfg.min_prob = args.min_prob

    direction = 1 if args.side == "long" else -1
    trader = LiveSOTATrader(device=args.device, exit_config=exit_cfg, direction=direction)
    trader.run_live_trading(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
