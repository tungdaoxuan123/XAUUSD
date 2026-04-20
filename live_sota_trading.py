"""
live_sota_trading.py (rewritten)
================================

Fixes three issues reported in production:

  1. Model F1 ~0.39 and "confidence stuck at 1".
       -> Trainer v2 (`train_sota_v2.py`) uses label smoothing + class-
          balanced CE + temperature scaling. This file applies the saved
          temperature at inference time, so `confidence` is a calibrated
          probability, not a saturated argmax.

  2. "Don't know when to enter".
       -> Decisions now use the CALIBRATED probability against a
          configurable threshold, plus spread/session/regime gates from
          `train_pipeline.dynamic_exits.ExitPlanner`.

  3. "Entries use fixed 2-ATR / 3-ATR exits".
       -> All SL/TP + sizing now come from ExitPlanner (volatility-
          targeted risk, confidence-scaled RR, half-Kelly sizing,
          in-trade breakeven + chandelier trail + time stop).

  Additionally, live features are now built by `train_pipeline.
  live_features.build_live_features`, which delegates to the SAME
  functions used during training — fixing the silent feature
  distribution shift in the old prepare_sota_data().
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

from train_pipeline.sota_signal_generator import PatchTSTLite, LABEL_UNMAP
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
        ckpt = torch.load(model_path, map_location="cpu")
        self.model = PatchTSTLite(
            n_features=ckpt["n_features"],
            seq_len=ckpt["seq_len"],
            patch_len=ckpt["patch_len"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

        mu = ckpt.get("mu"); sd = ckpt.get("sd")
        self.mu = torch.from_numpy(np.asarray(mu, dtype=np.float32)).to(self.device) if mu is not None else None
        self.sd = torch.from_numpy(np.asarray(sd, dtype=np.float32)).to(self.device) if sd is not None else None
        if self.mu is None:
            logger.warning("Checkpoint lacks mu/sd — using per-window normalization (less safe)")

    def predict(self, df: pd.DataFrame):
        """Return (signal, p_buy, p_sell, p_hold, entropy)."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            logger.error(f"Predict: missing features {missing[:5]}… ({len(missing)})")
            return 0, 1/3, 1/3, 1/3, np.log(3)

        X = df[self.features].astype("float32").values[-self.seq_len:]
        if len(X) < self.seq_len:
            return 0, 1/3, 1/3, 1/3, np.log(3)

        X = torch.tensor(X, device=self.device)
        if self.mu is not None and self.sd is not None:
            X = (X - self.mu) / self.sd
        else:
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        X = X.unsqueeze(0)

        with torch.no_grad():
            logits = self.model(X)
            # Apply temperature calibration
            probs = F.softmax(logits / max(self.temperature, 1e-3), dim=-1)
            p = probs.cpu().numpy()[0]

        # Entropy as a secondary confidence measure (nats)
        ent = float(-(p * np.log(p + 1e-12)).sum())
        signal = int(LABEL_UNMAP[int(p.argmax())])
        return signal, float(p[2]), float(p[0]), float(p[1]), ent


# ---------------------------------------------------------------------------
# Live trader
# ---------------------------------------------------------------------------

class LiveSOTATrader:
    def __init__(self, model_path=None, config_path=None, device=None,
                 exit_config: Optional[ExitConfig] = None):
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

        self.planner = ExitPlanner(exit_config or ExitConfig(
            risk_per_trade=getattr(self.settings, "RISK_PER_TRADE", 0.005),
            max_risk_frac=getattr(self.settings, "MAX_RISK_FRAC", 0.01),
            min_prob_long=getattr(self.settings, "BUY_CONFIDENCE", 0.55),
            min_prob_short=getattr(self.settings, "SELL_CONFIDENCE", 0.55),
            max_hold_bars=getattr(self.settings, "MAX_HOLD_BARS", 30),
        ))
        self._open_tracker: Optional[OpenPosition] = None
        self._bar_counter = 0

        logger.info(f"LiveSOTATrader ready — symbol={self.current_symbol} "
                    f"T={self.sota.temperature:.3f} "
                    f"min_p={self.planner.cfg.min_prob_long}")

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
        logger.info(f"SOTA live loop ({interval_seconds}s intervals, dry_run={dry_run})")
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

                # Skip new-entry scan while we have an open position, BUT
                # still run in-trade management.
                positions = self.interface.get_positions() or []
                if len(positions) > 0:
                    self._manage_open_positions(positions, dry_run=dry_run)
                    continue

                # ----- Fresh scan -------------------------------------------------
                target = "XAUUSD" if "XAU" in self.current_symbol else self.current_symbol
                rates = self.interface.get_rates(count=max(300, self.sota.seq_len + 50),
                                                 symbol=target)
                if rates is None or len(rates) < self.sota.seq_len + 30:
                    logger.warning(f"Not enough rates ({0 if rates is None else len(rates)})")
                    continue

                rates_df = pd.DataFrame(rates)
                df = build_live_features(rates_df, self.sota.features)

                signal, p_buy, p_sell, p_hold, entropy = self.sota.predict(df)
                prob = max(p_buy, p_sell)
                direction_prob = p_buy if signal > 0 else p_sell if signal < 0 else p_hold

                logger.info(
                    f"scan signal={signal:+d} p_buy={p_buy:.3f} p_sell={p_sell:.3f} "
                    f"p_hold={p_hold:.3f} H={entropy:.3f}"
                )

                if signal == 0:
                    continue

                equity = self._get_equity()
                spread = self._current_spread()
                plan = self.planner.build_plan(signal, direction_prob, df, equity, spread)
                if plan.skip:
                    logger.info(f"skip: {plan.reason}")
                    continue

                logger.info(
                    f"PLAN {'BUY' if plan.side>0 else 'SELL'} lots={plan.lots:.2f} "
                    f"entry={plan.entry:.2f} sl={plan.sl:.2f} tp={plan.tp:.2f} "
                    f"RR={plan.rr:.2f} risk={plan.equity_risk*100:.2f}%"
                )

                if dry_run:
                    continue

                ok = self.interface.send_order(plan.side, plan.lots, plan.sl, plan.tp)
                if ok:
                    self._open_tracker = OpenPosition(
                        side=plan.side, entry=plan.entry, sl=plan.sl, tp=plan.tp,
                        entry_bar=self._bar_counter, best_price=plan.entry,
                    )
                time.sleep(2)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            self.interface.shutdown()

    # ---- in-trade management ---------------------------------------------

    def _manage_open_positions(self, positions, dry_run: bool = False):
        if self._open_tracker is None:
            # We were restarted mid-trade; reconstruct minimal tracker from broker
            p = positions[0]
            self._open_tracker = OpenPosition(
                side=+1 if p.type == 0 else -1,
                entry=float(p.price_open),
                sl=float(p.sl),
                tp=float(p.tp),
                entry_bar=self._bar_counter,
                best_price=float(p.price_open),
            )
        target = "XAUUSD" if "XAU" in self.current_symbol else self.current_symbol
        rates = self.interface.get_rates(count=max(150, self.sota.seq_len + 20),
                                         symbol=target)
        if rates is None:
            return
        df = build_live_features(pd.DataFrame(rates), self.sota.features)

        upd = self.planner.manage_open(self._open_tracker, df, self._bar_counter)
        if upd.close_reason == "time":
            logger.info("time stop hit -> closing position")
            if not dry_run:
                self.interface.close_all()
            self._open_tracker = None
            return
        if upd.new_sl is not None:
            logger.info(f"SL update -> {upd.new_sl:.2f}")
            if not dry_run:
                pos_id = positions[0].ticket
                self.interface.modify_sl(pos_id, upd.new_sl)
            self._open_tracker.sl = upd.new_sl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Live SOTA trading (calibrated)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", type=str)
    ap.add_argument("--symbol", type=str)
    ap.add_argument("--min-prob", type=float, default=None,
                    help="Override calibrated probability threshold (both sides)")
    args = ap.parse_args()

    if args.symbol:
        Settings.SYMBOL = args.symbol

    exit_cfg = ExitConfig()
    if args.min_prob is not None:
        exit_cfg.min_prob_long = exit_cfg.min_prob_short = args.min_prob

    trader = LiveSOTATrader(device=args.device, exit_config=exit_cfg)
    trader.run_live_trading(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
