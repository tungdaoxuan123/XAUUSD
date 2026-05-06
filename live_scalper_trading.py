"""
live_scalper_trading.py — LONG-ONLY high-frequency scalper
==========================================================

Designed for London-NY overlap (13:00-17:00 UTC = 20:00-00:00 VN).
Uses PatchTST-lite + calibrated probabilities for LONG-ONLY entries.

Differences from live_sota_trading.py:
  * ScalperPlanner (tight SL, TP1 +0.5R partial, TP2 +1.2R)
  * Default session: 13:00-17:00 UTC
  * Daily goal/stop: +3R / -2R
  * Max 20 trades/day, cooldown after each close (3 bars)
  * Partial close at TP1 (50% of size) + move SL to BE
  * Hot-window v2: tick-level entry timing

Usage
-----
    python live_scalper_trading.py --dry-run --min-prob 0.52
    python live_scalper_trading.py --min-prob 0.52
    python live_scalper_trading.py --session-start-utc 0 --session-end-utc 24
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

from train_pipeline.sota_signal_generator import PatchTSTLite
from train_pipeline.live_features import build_live_features
from train_pipeline.scalper_exits import (
    ScalperPlanner, ScalperConfig, ScalpOpen,
)
from train_pipeline.hot_window_executor import (
    HotWindowExecutor, HotWindowConfig,
)

logger = setup_logging()


class CalibratedSignal:
    """PatchTST inference with calibrated (temperature-scaled) binary probabilities."""

    def __init__(self, model_path: str, config_path: str, device: str = "cpu"):
        with open(config_path) as f:
            cfg = json.load(f)
        self.features = cfg["features"]
        self.seq_len = cfg["seq_len"]
        self.patch_len = cfg["patch_len"]
        self.temperature = float(cfg.get("temperature", 1.0))
        self.device = device if "privateuseone" not in str(device) else "cpu"

        logger.info(f"Loading SOTA model: {model_path}  T={self.temperature:.3f}")
        import torch._utils
        orig_rebuild = torch._utils._rebuild_device_tensor_from_numpy
        def patched_rebuild(data, dtype, device, *args):
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
            return 0, 0.5, 0.5, 0.5
        X = df[self.features].astype("float32").values[-self.seq_len:]
        if len(X) < self.seq_len:
            return 0, 0.5, 0.5, 0.5
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
        if p.shape[0] >= 2:
            p_long = float(p[1]); p_wait = float(p[0])
        else:
            p_long = float(p[0]); p_wait = 1.0 - p_long
        signal = 1 if p_long > 0.5 else 0
        return signal, p_long, p_wait, p_wait


class LiveScalper:
    def __init__(self, model_path=None, config_path=None, device=None,
                 scalp_cfg: Optional[ScalperConfig] = None,
                 hot_cfg: Optional[HotWindowConfig] = None,
                 use_hot_window: bool = True):
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

        self.use_hot_window = use_hot_window
        self.hot_executor = HotWindowExecutor(
            cfg=hot_cfg or HotWindowConfig(),
            tick_source=self.interface.get_ticks,
            position_check_fn=self.interface.get_positions,
        )

        cfg = self.planner.cfg
        logger.info(
            f"LONG-ONLY Scalper ready symbol={self.symbol} "
            f"session={cfg.session_start_utc:02d}-{cfg.session_end_utc:02d}UTC "
            f"RR={cfg.rr_partial}/{cfg.rr_final} cooldown={cfg.cooldown_bars} "
            f"goal=+{cfg.daily_goal_R}R stop=-{cfg.daily_stop_R}R "
            f"maxHold={cfg.max_hold_bars} maxTrades={cfg.max_trades_per_day}"
        )

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

    def _live_mid(self) -> float:
        try:
            ticks = self.interface.get_ticks(count=1)
            if ticks is None or len(ticks) == 0:
                return 0.0
            t = ticks[-1]
            bid = float(t["bid"]) if hasattr(t, "__getitem__") else float(t.bid)
            ask = float(t["ask"]) if hasattr(t, "__getitem__") else float(t.ask)
            if bid > 0 and ask > 0:
                return 0.5 * (bid + ask)
            return 0.0
        except Exception:
            return 0.0

    def _now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _compute_pnl_R(self, pos: ScalpOpen, exit_price: float) -> float:
        sl_dist = abs(pos.entry - pos.sl)
        if sl_dist <= 0:
            return 0.0
        return (exit_price - pos.entry) / sl_dist

    def run(self, interval_s: int = 10, dry_run: bool = False):
        logger.info(f"LONG-ONLY scalper loop interval={interval_s}s dry_run={dry_run}")
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

                positions = self.interface.get_positions() or []
                if len(positions) > 0:
                    self._manage_open(positions, now_utc, dry_run=dry_run)
                    continue

                self.pos = None

                if not self.planner.session_open(now_utc):
                    if self.bar_counter % 30 == 0:
                        logger.info(f"waiting for session (now {now_utc:%H:%M}UTC)")
                    continue
                done = self.planner.daily_done(now_utc)
                if done:
                    if self.bar_counter % 30 == 0:
                        logger.info(f"daily done: {done}")
                    continue

                target = "XAUUSD" if "XAU" in self.symbol else self.symbol
                rates = self.interface.get_rates(
                    count=max(300, self.sota.seq_len + 50), symbol=target,
                )
                if rates is None or len(rates) < self.sota.seq_len + 30:
                    continue
                df = build_live_features(pd.DataFrame(rates), self.sota.features)

                signal, p_long, p_wait, _ = self.sota.predict(df)
                logger.info(
                    f"scan {'LONG' if signal>0 else 'WAIT'} "
                    f"p_long={p_long:.3f} p_wait={p_wait:.3f} "
                    f"R_today={self.planner.day.realized_R:+.2f} "
                    f"trades={self.planner.day.trades_taken}"
                )
                if signal == 0:
                    continue

                plan = self.planner.build_plan(
                    signal, p_long, df, self._equity(),
                    self._spread(), now_utc, self.bar_counter,
                )
                if plan.skip:
                    logger.info(f"skip: {plan.reason}")
                    continue

                logger.info(
                    f"PLAN LONG lots={plan.lots:.2f} entry={plan.entry:.2f} "
                    f"sl={plan.sl:.2f} tp1={plan.tp1:.2f} tp2={plan.tp2:.2f} "
                    f"risk={plan.equity_risk*100:.2f}%"
                )

                fill_price = plan.entry
                if self.use_hot_window:
                    atr_val = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else plan.sl_dist / 1.1
                    live_spread = self._spread()
                    spread_baseline = max(live_spread, 0.05)
                    live_ref = self._live_mid()
                    ref_price = live_ref if live_ref > 0 else plan.entry

                    decision = self.hot_executor.run(
                        signal=plan.side,
                        ref_price=ref_price,
                        atr=atr_val,
                        spread_baseline=spread_baseline,
                    )
                    if not decision.fill:
                        logger.info(f"hot window skipped: {decision.reason}")
                        self.planner.day.last_close_bar = self.bar_counter
                        continue
                    fill_price = decision.fill_price or ref_price
                    sl_dist = plan.sl_dist
                    plan.sl  = fill_price - sl_dist
                    plan.tp1 = fill_price + self.planner.cfg.rr_partial * sl_dist
                    plan.tp2 = fill_price + self.planner.cfg.rr_final   * sl_dist
                    plan.entry = fill_price
                    logger.info(
                        f"hot fill @ {fill_price:.2f} ({decision.reason}, {decision.elapsed_seconds:.1f}s)"
                    )

                if dry_run:
                    self.planner.record_open(now_utc)
                    continue

                ok = self.interface.send_order(
                    plan.side, plan.lots, sl=0.0, tp=0.0,
                    max_slippage_pts=self.interface.dynamic_deviation(spread_pts=self._spread()),
                )
                if not ok:
                    continue

                time.sleep(0.2)
                my_pos = None
                positions = self.interface.get_positions()
                if positions:
                    my_pos = next((p for p in positions if p.ticket == ok.order), None)

                real_entry = my_pos.price_open if my_pos else plan.entry
                plan.entry = real_entry
                sl_dist = plan.sl_dist
                plan.sl  = real_entry - sl_dist
                plan.tp1 = real_entry + self.planner.cfg.rr_partial * sl_dist
                plan.tp2 = real_entry + self.planner.cfg.rr_final   * sl_dist

                if my_pos:
                    self.interface.modify_position(ok.order, plan.sl, plan.tp2)

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

    def _manage_open(self, positions, now_utc: datetime, dry_run: bool = False):
        if self.pos is None:
            p = positions[0]
            self.pos = ScalpOpen(
                side=+1,
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
                        self.interface.modify_position(positions[0].ticket, upd.new_sl, self.pos.tp2)
                    except Exception as e:
                        logger.warning(f"modify failed: {e}")
                self.pos.sl = upd.new_sl
            self.planner.day.realized_R += self.planner.cfg.rr_partial * upd.partial_close_frac
            return

        if upd.new_sl is not None and upd.close_reason is None:
            if not dry_run:
                try:
                    self.interface.modify_position(positions[0].ticket, upd.new_sl, self.pos.tp2)
                except Exception as e:
                    logger.warning(f"trail modify failed: {e}")
            self.pos.sl = upd.new_sl
            return

        if upd.close_reason is not None:
            last_px = float(df["close"].iloc[-1])
            pnl_R = self._compute_pnl_R(self.pos, last_px)
            remaining = 1.0 - (self.planner.cfg.partial_size if self.pos.tp1_hit else 0.0)
            logger.info(f"CLOSE {upd.close_reason} @ {last_px:.2f} pnl_R~{pnl_R:.2f}")
            if not dry_run:
                self.interface.close_all_positions()
            self.planner.record_close(pnl_R * remaining, self.bar_counter, now_utc)
            self.pos = None


def main():
    ap = argparse.ArgumentParser(description="XAUUSD LONG-ONLY scalper")
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
    ap.add_argument("--no-hot-window", action="store_true")
    ap.add_argument("--hot-seconds", type=float, default=None)
    ap.add_argument("--hot-trigger-now", type=float, default=None)
    ap.add_argument("--hot-trigger-fb", "--hot-fallback", dest="hot_trigger_fb", type=float, default=None)
    ap.add_argument("--hot-abort", type=float, default=None)
    ap.add_argument("--hot-abort-atr", type=float, default=None)
    args = ap.parse_args()

    cfg = ScalperConfig(symbol=Settings.SYMBOL)
    if args.min_prob is not None:  cfg.min_prob_long = args.min_prob
    if args.rr_final is not None:  cfg.rr_final = args.rr_final
    if args.rr_partial is not None: cfg.rr_partial = args.rr_partial
    if args.max_hold is not None:  cfg.max_hold_bars = args.max_hold
    if args.cooldown is not None:  cfg.cooldown_bars = args.cooldown
    if args.goal_r is not None:    cfg.daily_goal_R = args.goal_r
    if args.stop_r is not None:    cfg.daily_stop_R = args.stop_r
    if args.session_start_utc is not None: cfg.session_start_utc = args.session_start_utc
    if args.session_end_utc is not None:   cfg.session_end_utc = args.session_end_utc

    hot_cfg = HotWindowConfig()
    if args.hot_seconds is not None:     hot_cfg.window_seconds = args.hot_seconds
    if args.hot_trigger_now is not None: hot_cfg.trigger_now = args.hot_trigger_now
    if args.hot_trigger_fb is not None:  hot_cfg.trigger_fallback = args.hot_trigger_fb
    if args.hot_abort is not None:       hot_cfg.abort_hard = args.hot_abort
    if args.hot_abort_atr is not None:   hot_cfg.adverse_momentum_atr = args.hot_abort_atr

    LiveScalper(
        device=args.device, scalp_cfg=cfg, hot_cfg=hot_cfg,
        use_hot_window=not args.no_hot_window,
    ).run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
