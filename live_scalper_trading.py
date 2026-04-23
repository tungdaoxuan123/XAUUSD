"""
live_scalper_trading.py
=======================
High-frequency variant of live_sota_trading.py designed for scalp-style
trading. Defaulted to 24h trading, but can be restricted to specific sessions.

Goal: many small wins, not one big swing.

Differences vs live_sota_trading.py
-----------------------------------
  * Uses ScalperPlanner (tight SL ~1.1*ATR, TP1 +0.5R partial, TP2 +1.2R)
  * Default 24h window (can be restricted via --session-start-utc)
  * Daily goal/stop: stop trading at +3R or -2R realized
  * Cooldown after each close (3 bars)
  * Max 20 trades per day
  * Partial close at TP1 (50% of size) + move SL to BE
  * Flat-by-close support (if session end is set)
  * Slightly looser min_prob (0.52) because scalper wants volume;
    edge comes from risk mgmt, not prob threshold.

Usage
-----
    # Dry run (no orders sent) — recommended first
    python live_scalper_trading.py --dry-run --min-prob 0.52

    # Live
    python live_scalper_trading.py --min-prob 0.52

    # Tighter/looser profile
    python live_scalper_trading.py --rr-final 1.0 --cooldown 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import Settings, setup_logging
from mt5_interface import MT5Interface
from risk_manager import FTMORiskManager

from train_pipeline.sota_signal_generator import PatchTSTLite, LABEL_UNMAP
from train_pipeline.live_features import build_live_features
from train_pipeline.scalper_exits import (
    ScalperPlanner, ScalperConfig, ScalpOpen,
)

logger = setup_logging()


# ---------------------------------------------------------------------------
# Model inference (calibrated probabilities via temperature)
# ---------------------------------------------------------------------------

class CalibratedSignal:
    def __init__(self, model_path: str, config_path: str, device: str = "cpu"):
        with open(config_path) as f:
            cfg = json.load(f)
        self.features = cfg["features"]
        self.seq_len = cfg["seq_len"]
        self.patch_len = cfg["patch_len"]
        self.temperature = float(cfg.get("temperature", 1.0))
        self.device = device if "privateuseone" not in str(device) else "cpu"

        logger.info(f"Loading SOTA model: {model_path}  T={self.temperature:.3f}")
        # Nuclear Option: Monkey-patch the internal rebuilder to ignore DirectML devices
        import torch._utils
        orig_rebuild = torch._utils._rebuild_device_tensor_from_numpy
        def patched_rebuild(data, dtype, device, *args):
            # args[0] might be requires_grad in some torch versions
            return orig_rebuild(data, dtype, torch.device('cpu'), *args)
        
        try:
            torch._utils._rebuild_device_tensor_from_numpy = patched_rebuild
            ckpt = torch.load(model_path, map_location='cpu')
        finally:
            torch._utils._rebuild_device_tensor_from_numpy = orig_rebuild

        self.model = PatchTSTLite(
            n_features=ckpt["n_features"],
            seq_len=ckpt["seq_len"],
            patch_len=ckpt["patch_len"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

        mu, sd = ckpt.get("mu"), ckpt.get("sd")
        self.mu = torch.from_numpy(np.asarray(mu, dtype=np.float32)).to(self.device) if mu is not None else None
        self.sd = torch.from_numpy(np.asarray(sd, dtype=np.float32)).to(self.device) if sd is not None else None

    def predict(self, df: pd.DataFrame):
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            logger.error(f"predict: missing feats {missing[:5]}… ({len(missing)})")
            return 0, 1/3, 1/3, 1/3
        X = df[self.features].astype("float32").values[-self.seq_len:]
        if len(X) < self.seq_len:
            return 0, 1/3, 1/3, 1/3
        X = torch.tensor(X, device=self.device)
        if self.mu is not None:
            X = (X - self.mu) / self.sd
        else:
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        X = X.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(X)
            probs = F.softmax(logits / max(self.temperature, 1e-3), dim=-1)
            p = probs.cpu().numpy()[0]
        signal = int(LABEL_UNMAP[int(p.argmax())])
        return signal, float(p[2]), float(p[0]), float(p[1])


# ---------------------------------------------------------------------------
# Scalper live loop
# ---------------------------------------------------------------------------

class LiveScalper:
    def __init__(self, model_path=None, config_path=None, device=None,
                 scalp_cfg: Optional[ScalperConfig] = None):
        self.settings = Settings
        self.interface = MT5Interface()
        self.risk_mgr = FTMORiskManager(self.interface)

        self.sota = CalibratedSignal(
            model_path or self.settings.SOTA_MODEL_PATH,
            config_path or self.settings.SOTA_CONFIG_PATH,
            device or self.settings.SOTA_DEVICE,
        )
        if not self.interface.authorized:
            self.interface.initialize()
        self.symbol = self.interface.symbol
        self.planner = ScalperPlanner(scalp_cfg or ScalperConfig())
        self.pos: Optional[ScalpOpen] = None
        self.bar_counter = 0

        cfg = self.planner.cfg
        logger.info(
            f"LiveScalper ready symbol={self.symbol} "
            f"session={cfg.session_start_utc:02d}-{cfg.session_end_utc:02d}UTC "
            f"(={cfg.session_start_utc+7:02d}-{cfg.session_end_utc+7:02d}VN) "
            f"RR={cfg.rr_partial}/{cfg.rr_final} cooldown={cfg.cooldown_bars} "
            f"goal=+{cfg.daily_goal_R}R stop=-{cfg.daily_stop_R}R "
            f"maxHold={cfg.max_hold_bars} maxTrades={cfg.max_trades_per_day}"
        )

    # ---- helpers ----------------------------------------------------------

    def _equity(self) -> float:
        try:
            info = self.interface.get_account_info()
            return float(info.equity) if info else float(self.settings.INITIAL_BALANCE)
        except Exception:
            return float(getattr(self.settings, "INITIAL_BALANCE", 10000.0))

    def _spread(self) -> float:
        try:
            ticks = self.interface.get_ticks(count=1)
            if ticks is None or len(ticks) == 0:
                return 0.0
            t = ticks[-1]
            return float(t["ask"] - t["bid"]) if hasattr(t, "__getitem__") else float(t.ask - t.bid)
        except Exception:
            return 0.0

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _compute_pnl_R(self, pos: ScalpOpen, exit_price: float) -> float:
        sl_dist = abs(pos.entry - pos.sl)
        if sl_dist <= 0:
            return 0.0
        if pos.side > 0:
            return (exit_price - pos.entry) / sl_dist
        return (pos.entry - exit_price) / sl_dist

    # ---- main loop --------------------------------------------------------

    def run(self, interval_s: int = 10, dry_run: bool = False):
        logger.info(f"scalper loop interval={interval_s}s dry_run={dry_run}")
        if not self.interface.initialize():
            logger.error("MT5 init failed"); return
        self.risk_mgr.initialize_balance()

        last_tick = datetime.min
        try:
            while True:
                now = datetime.now()
                if (now - last_tick).total_seconds() < interval_s:
                    time.sleep(1); continue
                last_tick = now
                self.bar_counter += 1
                now_utc = self._now_utc()

                # ---- 1) Manage any open position first ----
                positions = self.interface.get_positions() or []
                if len(positions) > 0:
                    self._manage_open(positions, now_utc, dry_run=dry_run)
                    continue

                # No broker-side position: clear local tracker if needed
                self.pos = None

                # ---- 2) Session & daily checks ----
                if not self.planner.session_open(now_utc):
                    if self.bar_counter % 30 == 0:
                        logger.info(f"waiting for session (now {now_utc:%H:%M}UTC)")
                    continue
                done = self.planner.daily_done(now_utc)
                if done:
                    if self.bar_counter % 30 == 0:
                        logger.info(f"daily done: {done}")
                    continue

                # ---- 3) Feature pipeline ----
                target = "XAUUSD" if "XAU" in self.symbol else self.symbol
                rates = self.interface.get_rates(
                    count=max(300, self.sota.seq_len + 50), symbol=target,
                )
                if rates is None or len(rates) < self.sota.seq_len + 30:
                    continue
                df = build_live_features(pd.DataFrame(rates), self.sota.features)

                # ---- 4) Signal ----
                signal, p_buy, p_sell, p_hold = self.sota.predict(df)
                prob_dir = p_buy if signal > 0 else p_sell if signal < 0 else p_hold
                logger.info(
                    f"scan {signal:+d} p_b={p_buy:.3f} p_s={p_sell:.3f} p_h={p_hold:.3f} "
                    f"R_today={self.planner.day.realized_R:+.2f} trades={self.planner.day.trades_taken}"
                )
                if signal == 0:
                    continue

                # ---- 5) Build scalp plan ----
                plan = self.planner.build_plan(
                    signal, prob_dir, df, self._equity(),
                    self._spread(), now_utc, self.bar_counter,
                )
                if plan.skip:
                    logger.info(f"skip: {plan.reason}")
                    continue

                logger.info(
                    f"PLAN {'BUY' if plan.side>0 else 'SELL'} "
                    f"lots={plan.lots:.2f} entry={plan.entry:.2f} "
                    f"sl={plan.sl:.2f} tp1={plan.tp1:.2f} tp2={plan.tp2:.2f} "
                    f"risk={plan.equity_risk*100:.2f}%"
                )
                if dry_run:
                    self.planner.record_open(now_utc)
                    continue

                # Try to send with TP2; TP1 is handled by manage_open (partial)
                ok = self.interface.send_order(plan.side, plan.lots, plan.sl, plan.tp2)
                if not ok:
                    continue

                self.planner.record_open(now_utc)
                self.pos = ScalpOpen(
                    side=plan.side, entry=plan.entry, sl=plan.sl,
                    tp1=plan.tp1, tp2=plan.tp2,
                    entry_bar=self.bar_counter, entry_time=now_utc,
                    best_price=plan.entry, initial_lots=plan.lots,
                )
                time.sleep(2)

        except KeyboardInterrupt:
            logger.info("stopped by user")
            if self.pos is not None and not dry_run:
                logger.info("flattening open position on exit")
                self.interface.close_all_positions()
        finally:
            self.interface.shutdown()

    # ---- in-trade management ---------------------------------------------

    def _manage_open(self, positions, now_utc: datetime, dry_run: bool = False):
        if self.pos is None:
            p = positions[0]
            self.pos = ScalpOpen(
                side=+1 if p.type == 0 else -1,
                entry=float(p.price_open),
                sl=float(p.sl), tp1=float(p.tp), tp2=float(p.tp),
                entry_bar=self.bar_counter, entry_time=now_utc,
                best_price=float(p.price_open),
                initial_lots=float(p.volume),
            )
        target = "XAUUSD" if "XAU" in self.symbol else self.symbol
        rates = self.interface.get_rates(
            count=max(150, self.sota.seq_len + 20), symbol=target,
        )
        if rates is None:
            return
        df = build_live_features(pd.DataFrame(rates), self.sota.features)
        upd = self.planner.manage_open(self.pos, df, self.bar_counter, now_utc)

        # Partial close at TP1 + move SL to BE
        if upd.partial_close_frac > 0:
            part = round(self.pos.initial_lots * upd.partial_close_frac, 2)
            logger.info(f"TP1 hit -> partial close {part} lots, SL->BE")
            if not dry_run:
                try:
                    self.interface.close_partial_position(positions[0].ticket, part)
                except Exception as e:
                    logger.warning(f"partial close failed: {e}")
            if upd.new_sl is not None:
                if not dry_run:
                    try:
                        self.interface.modify_position(
                            positions[0].ticket, upd.new_sl, self.pos.tp2)
                    except Exception as e:
                        logger.warning(f"modify_position failed: {e}")
                self.pos.sl = upd.new_sl
            # record TP1 profit as 0.5R * partial fraction (rough bookkeeping)
            self.planner.day.realized_R += (
                self.planner.cfg.rr_partial * upd.partial_close_frac
            )
            return

        # Trailing SL
        if upd.new_sl is not None and upd.close_reason is None:
            if not dry_run:
                try:
                    self.interface.modify_position(
                        positions[0].ticket, upd.new_sl, self.pos.tp2)
                except Exception as e:
                    logger.warning(f"trail modify failed: {e}")
            self.pos.sl = upd.new_sl
            return

        # Hard close
        if upd.close_reason is not None:
            last_px = float(df["close"].iloc[-1])
            pnl_R = self._compute_pnl_R(self.pos, last_px)
            # The partial already booked its portion; attribute remainder here
            remaining = 1.0 - (self.planner.cfg.partial_size if self.pos.tp1_hit else 0.0)
            logger.info(
                f"CLOSE {upd.close_reason} @ {last_px:.2f} pnl_R~{pnl_R:.2f} "
                f"(remaining={remaining:.2f})"
            )
            if not dry_run:
                self.interface.close_all_positions()
            self.planner.record_close(pnl_R * remaining, self.bar_counter, now_utc)
            self.pos = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="XAUUSD scalper — VN evening session")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--min-prob", type=float, default=None)
    ap.add_argument("--rr-final", type=float, default=None)
    ap.add_argument("--rr-partial", type=float, default=None)
    ap.add_argument("--max-hold", type=int, default=None)
    ap.add_argument("--cooldown", type=int, default=None)
    ap.add_argument("--goal-r", type=float, default=None)
    ap.add_argument("--stop-r", type=float, default=None)
    ap.add_argument("--session-start-utc", type=int, default=None)
    ap.add_argument("--session-end-utc", type=int, default=None)
    args = ap.parse_args()

    cfg = ScalperConfig()
    if args.min_prob is not None:
        cfg.min_prob_long = cfg.min_prob_short = args.min_prob
    if args.rr_final is not None:    cfg.rr_final = args.rr_final
    if args.rr_partial is not None:  cfg.rr_partial = args.rr_partial
    if args.max_hold is not None:    cfg.max_hold_bars = args.max_hold
    if args.cooldown is not None:    cfg.cooldown_bars = args.cooldown
    if args.goal_r is not None:      cfg.daily_goal_R = args.goal_r
    if args.stop_r is not None:      cfg.daily_stop_R = args.stop_r
    if args.session_start_utc is not None: cfg.session_start_utc = args.session_start_utc
    if args.session_end_utc is not None:   cfg.session_end_utc = args.session_end_utc

    LiveScalper(device=args.device, scalp_cfg=cfg).run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
