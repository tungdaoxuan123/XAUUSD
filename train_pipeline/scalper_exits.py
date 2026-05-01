#!/usr/bin/env python3
"""
scalper_exits.py
----------------
High-frequency, small-wins exit controller for the VN evening session
(20:00-00:00 VN = 13:00-17:00 UTC, the London-NY overlap on XAUUSD).

Design philosophy
=================
This is the *opposite* of the v2 `ExitPlanner`:

  v2 swing mode            |  scalper mode
  ------------------------ |  ------------------------------
  RR 1.8 - 3.5             |  RR 0.8 - 1.4
  Hold up to 30 bars       |  Hold 5-12 bars max
  Single big TP            |  Partial TP1 (0.5R) + runner
  Fixed daily risk         |  Daily GOAL + daily STOP
  Entry once per signal    |  Entry + cooldown, many signals/hour
  No session restriction   |  Hard 13:00-17:00 UTC window

Session defaults (v2 guardrails restored)
-----------------------------------------
  session_start_utc = 13   (20:00 VN)
  session_end_utc   = 17   (00:00 VN)
  daily_goal_R      = 3.0  (lock in profit)
  daily_stop_R      = 2.0  (stop-loss for the day)
  max_trades_per_day= 20

These match what the module docstring promised but the old code had
accidentally set to infinity. Override via CLI --goal-r / --stop-r etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScalperConfig:
    # --- Session window (UTC) ---
    # Default: London-NY overlap 13:00-17:00 UTC = 20:00-00:00 VN
    # Pass --session-start-utc 0 --session-end-utc 24 for 24h mode.
    session_start_utc: int = 13
    session_end_utc:   int = 17
    flat_by_minutes_before_close: int = 5   # flatten 5 min before session end

    # --- Risk (per trade & per day) ---
    risk_per_trade:   float = 0.0015   # 0.15% per trade
    max_risk_frac:    float = 0.005    # hard cap 0.5%
    daily_goal_R:     float = 3.0      # stop trading at +3R (restored)
    daily_stop_R:     float = 2.0      # stop trading at -2R (unchanged)
    max_trades_per_day: int = 20       # restored from 1000

    # --- Entry gating ---
    min_prob_long:    float = 0.52
    min_prob_short:   float = 0.52
    cooldown_bars:    int   = 3
    max_spread_ratio: float = 1.4
    news_signed_z_abs: float = 3.5

    # --- SL / TP geometry ---
    atr_mult_sl:      float = 5.0
    rr_partial:       float = 0.5
    rr_final:         float = 1.2
    partial_size:     float = 0.5
    trail_mult_atr:   float = 1.0

    # --- Time / hold ---
    max_hold_bars:    int   = 60    # minutes (uses entry_time)
    min_hold_bars:    int   = 1

    # --- Symbol mechanics ---
    symbol:           str   = "XAUUSD"
    contract_size:    float = 100.0
    point_value:      float = 1.0
    min_sl_price_dist:float = 0.15

    def __post_init__(self):
        if "GBP" in self.symbol.upper():
            self.contract_size = 100000.0
            self.min_sl_price_dist = 0.00015
        else:
            self.contract_size = 100.0
            self.min_sl_price_dist = 0.15


# ---------------------------------------------------------------------------
# Plan / open-position dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScalpPlan:
    skip: bool = False
    reason: str = ""
    side: int = 0
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    lots: float = 0.0
    sl_dist: float = 0.0
    prob: float = 0.0
    equity_risk: float = 0.0


@dataclass
class ScalpOpen:
    side: int
    entry: float
    sl: float
    tp1: float
    tp2: float
    entry_bar: int
    entry_time: datetime
    best_price: float
    initial_lots: float
    initial_sl_dist: float = 0.0
    tp1_hit: bool = False
    be_moved: bool = False


@dataclass
class ScalpUpdate:
    partial_close_frac: float = 0.0
    new_sl: Optional[float] = None
    close_reason: Optional[str] = None  # "tp2" | "time" | "eod" | "trail"


@dataclass
class DayStats:
    date_utc: str = ""
    realized_R: float = 0.0
    trades_taken: int = 0
    last_close_bar: int = -10_000


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class ScalperPlanner:
    def __init__(self, cfg: Optional[ScalperConfig] = None):
        self.cfg = cfg or ScalperConfig()
        self.day = DayStats(date_utc="")

    def _today_key(self, now_utc: datetime) -> str:
        return now_utc.strftime("%Y-%m-%d")

    def _rollover(self, now_utc: datetime):
        k = self._today_key(now_utc)
        if self.day.date_utc != k:
            self.day = DayStats(date_utc=k)

    def session_open(self, now_utc: datetime) -> bool:
        cfg = self.cfg
        return cfg.session_start_utc <= now_utc.hour < cfg.session_end_utc

    def near_session_close(self, now_utc: datetime) -> bool:
        cfg = self.cfg
        if cfg.session_end_utc >= 24 and cfg.flat_by_minutes_before_close <= 0:
            return False
        end_h = cfg.session_end_utc
        close_minute = 60 - cfg.flat_by_minutes_before_close
        return (now_utc.hour == end_h - 1 and now_utc.minute >= close_minute) \
            or now_utc.hour >= end_h

    def daily_done(self, now_utc: datetime) -> Optional[str]:
        self._rollover(now_utc)
        d = self.day
        if d.realized_R >= self.cfg.daily_goal_R:
            return f"goal hit (+{d.realized_R:.2f}R)"
        if d.realized_R <= -self.cfg.daily_stop_R:
            return f"daily stop ({d.realized_R:.2f}R)"
        if d.trades_taken >= self.cfg.max_trades_per_day:
            return f"max trades ({d.trades_taken})"
        return None

    def _sl_distance(self, bar_df, atr: float) -> float:
        return max(self.cfg.atr_mult_sl * atr, self.cfg.min_sl_price_dist)

    def _size(self, equity: float, sl_dist: float) -> float:
        cfg = self.cfg
        dollar_risk = min(equity * cfg.risk_per_trade, equity * cfg.max_risk_frac)
        lot_value = sl_dist * cfg.point_value * cfg.contract_size
        if lot_value <= 0:
            return 0.0
        lots = dollar_risk / lot_value
        return float(max(0.01, round(lots, 2)))

    def build_plan(self, signal: int, prob: float, bar_df,
                   equity: float, spread_now: float,
                   now_utc: datetime, current_bar: int) -> ScalpPlan:
        cfg = self.cfg
        plan = ScalpPlan(prob=prob)
        self._rollover(now_utc)

        if signal == 0:
            plan.skip = True; plan.reason = "no signal"; return plan

        if not self.session_open(now_utc):
            plan.skip = True
            plan.reason = f"out of session ({now_utc.hour:02d}:{now_utc.minute:02d} UTC)"
            return plan
        if self.near_session_close(now_utc):
            plan.skip = True; plan.reason = "near session close"; return plan

        done = self.daily_done(now_utc)
        if done:
            plan.skip = True; plan.reason = done; return plan

        if current_bar - self.day.last_close_bar < cfg.cooldown_bars:
            plan.skip = True
            plan.reason = f"cooldown ({current_bar - self.day.last_close_bar}<{cfg.cooldown_bars})"
            return plan

        thr = cfg.min_prob_long if signal > 0 else cfg.min_prob_short
        if prob < thr:
            plan.skip = True; plan.reason = f"p {prob:.2f} < {thr}"; return plan

        if "spread_mean" in bar_df.columns:
            sm = float(bar_df["spread_mean"].iloc[-5:].mean())
            if sm > 0.05 and spread_now > cfg.max_spread_ratio * sm:
                plan.skip = True
                plan.reason = f"spread {spread_now:.3f}>{cfg.max_spread_ratio}*{sm:.3f}"
                return plan

        if "signed_vol_z" in bar_df.columns:
            z = float(bar_df["signed_vol_z"].iloc[-1])
            if abs(z) > cfg.news_signed_z_abs and np.sign(z) != signal:
                plan.skip = True; plan.reason = f"contrarian to z={z:.1f}"; return plan

        atr = float(bar_df["ATR"].iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            plan.skip = True; plan.reason = "bad ATR"; return plan

        entry = float(bar_df["close"].iloc[-1])
        sl_dist = self._sl_distance(bar_df, atr)
        if signal > 0:
            sl  = entry - sl_dist
            tp1 = entry + cfg.rr_partial * sl_dist
            tp2 = entry + cfg.rr_final   * sl_dist
        else:
            sl  = entry + sl_dist
            tp1 = entry - cfg.rr_partial * sl_dist
            tp2 = entry - cfg.rr_final   * sl_dist

        lots = self._size(equity, sl_dist)
        plan.side = signal
        plan.entry = entry; plan.sl = sl
        plan.tp1 = tp1; plan.tp2 = tp2
        plan.sl_dist = sl_dist
        plan.lots = lots
        plan.equity_risk = (lots * sl_dist * cfg.point_value * cfg.contract_size
                            / max(equity, 1e-9))
        return plan

    def manage_open(self, pos: ScalpOpen, bar_df, bar_idx: int,
                    now_utc: datetime) -> ScalpUpdate:
        cfg = self.cfg
        upd = ScalpUpdate()
        last = float(bar_df["close"].iloc[-1])
        atr = float(bar_df["ATR"].iloc[-1])

        if self.near_session_close(now_utc):
            upd.close_reason = "eod"; return upd

        if pos.entry_time is not None:
            elapsed_minutes = (now_utc - pos.entry_time).total_seconds() / 60.0
            if elapsed_minutes >= cfg.max_hold_bars:
                upd.close_reason = "time"; return upd
        elif bar_idx - pos.entry_bar >= cfg.max_hold_bars * 6:
            upd.close_reason = "time"; return upd

        if pos.side > 0:
            pos.best_price = max(pos.best_price, last)
            tp1_hit = last >= pos.tp1
            tp2_hit = last >= pos.tp2
        else:
            pos.best_price = min(pos.best_price, last)
            tp1_hit = last <= pos.tp1
            tp2_hit = last <= pos.tp2

        if tp2_hit:
            upd.close_reason = "tp2"; return upd

        if tp1_hit and not pos.tp1_hit:
            pos.tp1_hit = True
            pos.be_moved = True
            upd.partial_close_frac = cfg.partial_size
            upd.new_sl = pos.entry + (1e-4 if pos.side > 0 else -1e-4)
            return upd

        if pos.be_moved and atr > 0:
            trail_dist = cfg.trail_mult_atr * atr
            if pos.side > 0:
                new_sl = pos.best_price - trail_dist
                if new_sl > pos.sl:
                    upd.new_sl = new_sl
            else:
                new_sl = pos.best_price + trail_dist
                if new_sl < pos.sl:
                    upd.new_sl = new_sl
        return upd

    def record_close(self, pnl_R: float, close_bar: int, now_utc: datetime):
        self._rollover(now_utc)
        self.day.realized_R += float(pnl_R)
        self.day.last_close_bar = int(close_bar)

    def record_open(self, now_utc: datetime):
        self._rollover(now_utc)
        self.day.trades_taken += 1
