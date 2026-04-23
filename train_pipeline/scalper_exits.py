#!/usr/bin/env python3
"""
scalper_exits.py
----------------
High-frequency, small-wins exit controller for the VN evening session
(20:00–24:00 VN = 13:00–17:00 UTC, the London-NY overlap on XAUUSD).

Design philosophy
=================
This is the *opposite* of the v2 `ExitPlanner`:

  v2 swing mode            |  scalper mode
  ------------------------ |  ------------------------------
  RR 1.8 – 3.5             |  RR 0.8 – 1.4
  Hold up to 30 bars       |  Hold 5–12 bars max
  Single big TP            |  Partial TP1 (0.5R) + runner
  Fixed daily risk         |  Daily GOAL + daily STOP (whichever first)
  Entry once per signal    |  Entry + cooldown, many signals/hour
  No session restriction   |  Hard 13:00–17:00 UTC window

Why these choices work in 20:00–24:00 VN
----------------------------------------
* 13:00 UTC = London-NY overlap = ~40% of XAUUSD's daily range.
* Median M1 bar range at this hour is 30–70 cents.
* 0.5R scalps are reachable within 3–6 bars ~55% of the time
  historically — that's the core of the "many small wins" edge.
* Forcing flat-by-EOD (24:00 VN) eliminates overnight gap risk
  and avoids Asian-session chop where RR quickly degrades.

Three key mechanics added on top of v2:

  1. Partial take-profit at +0.5R:
        - Close 50% of position at 0.5R
        - Move remaining SL to break-even
        - Final TP is the original ~1.2R
     Expected value: 0.5*0.5R + 0.5 * (trailing from BE) ≈ +0.35R
     per winning trade, with LOSERS capped at -1R because SL hasn't
     moved yet. At 55% hit-rate on TP1, this produces a very stable
     equity curve with low variance — the "many small wins" feel.

  2. Cooldown clock:
        - After any close, block new entries for N bars.
        - Prevents revenge trading and stop-cascade overfitting.

  3. Daily goal / daily stop:
        - Stop trading when realized P&L today >= +DAILY_GOAL_R
        - Stop trading when realized P&L today <= -DAILY_STOP_R
        - Reset at UTC midnight.
     This is how prop traders actually run: lock in the win, avoid
     giving back. FTMO-friendly.

Public API
----------
    cfg = ScalperConfig()
    planner = ScalperPlanner(cfg)

    # Before scan:
    if not planner.session_open(now_utc):           # hard window
        sleep
    if planner.daily_done(today_pnl_r):             # goal/stop
        sleep

    plan = planner.build_plan(signal, prob, bar_df, equity, spread_now, now_utc)
    # Same fields as TradePlan from dynamic_exits; adds tp1 (partial), tp2 (final)

    upd = planner.manage_open(pos, bar_df, bar_idx, now_utc)
    # upd.partial_close=0.5, upd.new_sl=entry at TP1 hit
    # upd.close_reason in {"time", "eod", "trail", "be", None}
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
    # Defaulting to 24h (00:00–24:00 UTC)
    session_start_utc: int = 0
    session_end_utc:   int = 24      # exclusive
    flat_by_minutes_before_close: int = 0  # 0 = disabled for 24h mode
 
    # --- Risk (per trade & per day) ---
    risk_per_trade:   float = 0.003   # 0.3% per trade
    max_risk_frac:    float = 0.005   # hard cap
    daily_goal_R:     float = 999999.0 # Effectively removed
    daily_stop_R:     float = 2.0     # stop for the day at -2R realized
    max_trades_per_day: int = 1000    # Increased

    # --- Entry gating (looser than swing mode) ---
    min_prob_long:    float = 0.52    # lowered from 0.55 — scalper wants frequency
    min_prob_short:   float = 0.52
    cooldown_bars:    int   = 3       # wait 3 bars after any close
    max_spread_ratio: float = 1.4
    news_signed_z_abs: float = 3.5    # slightly looser — overlap has real flow

    # --- SL / TP geometry (tight!) ---
    atr_mult_sl:      float = 1.1     # tighter than v2's 2.0
    min_sl_pips:      float = 1.5     # 15 cents XAU floor
    rr_partial:       float = 0.5     # TP1 = +0.5R (partial)
    rr_final:         float = 1.2     # TP2 = +1.2R (runner)
    partial_size:     float = 0.5     # close 50% at TP1
    trail_mult_atr:   float = 1.0     # tighter trail after BE

    # --- Time / hold ---
    max_hold_bars:    int   = 12      # hard time stop (12 min on M1)
    min_hold_bars:    int   = 1       # allow almost immediate exit

    # --- Symbol mechanics (XAU: 1 lot = 100 oz, $1/point/lot) ---
    contract_size:    float = 100.0
    point_value:      float = 1.0


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
    partial_close_frac: float = 0.0     # e.g. 0.5 at TP1
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

    # -------------------- session / day control --------------------

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
        # e.g. 16:50 UTC or later
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

    # -------------------- SL / TP geometry --------------------

    def _sl_distance(self, bar_df, atr: float) -> float:
        return max(self.cfg.atr_mult_sl * atr, self.cfg.min_sl_pips * 0.01)  # pips -> price

    def _size(self, equity: float, sl_dist: float) -> float:
        cfg = self.cfg
        dollar_risk = min(equity * cfg.risk_per_trade,
                          equity * cfg.max_risk_frac)
        lot_value = sl_dist * cfg.point_value * cfg.contract_size
        if lot_value <= 0:
            return 0.0
        lots = dollar_risk / lot_value
        return float(max(0.01, round(lots, 2)))

    # -------------------- main entry --------------------

    def build_plan(self, signal: int, prob: float, bar_df,
                   equity: float, spread_now: float,
                   now_utc: datetime, current_bar: int) -> ScalpPlan:
        cfg = self.cfg
        plan = ScalpPlan(prob=prob)
        self._rollover(now_utc)

        if signal == 0:
            plan.skip = True; plan.reason = "no signal"; return plan

        # ---- session gate ----
        if not self.session_open(now_utc):
            plan.skip = True
            plan.reason = f"out of session ({now_utc.hour:02d}:{now_utc.minute:02d} UTC)"
            return plan
        if self.near_session_close(now_utc):
            plan.skip = True; plan.reason = "near session close (no new entries)"; return plan

        # ---- daily gate ----
        done = self.daily_done(now_utc)
        if done:
            plan.skip = True; plan.reason = done; return plan

        # ---- cooldown ----
        if current_bar - self.day.last_close_bar < cfg.cooldown_bars:
            plan.skip = True
            plan.reason = f"cooldown ({current_bar - self.day.last_close_bar}<{cfg.cooldown_bars})"
            return plan

        # ---- probability gate ----
        thr = cfg.min_prob_long if signal > 0 else cfg.min_prob_short
        if prob < thr:
            plan.skip = True; plan.reason = f"p {prob:.2f} < {thr}"; return plan

        # ---- spread gate ----
        if "spread_mean" in bar_df.columns:
            sm = float(bar_df["spread_mean"].iloc[-5:].mean())
            if sm > 0.05 and spread_now > cfg.max_spread_ratio * sm:
                plan.skip = True
                plan.reason = f"spread {spread_now:.3f}>{cfg.max_spread_ratio}*{sm:.3f}"
                return plan

        # ---- contrarian-to-aggressor flow gate ----
        if "signed_vol_z" in bar_df.columns:
            z = float(bar_df["signed_vol_z"].iloc[-1])
            if abs(z) > cfg.news_signed_z_abs and np.sign(z) != signal:
                plan.skip = True; plan.reason = f"contrarian to z={z:.1f}"; return plan

        # ---- geometry ----
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

    # -------------------- in-trade management --------------------

    def manage_open(self, pos: ScalpOpen, bar_df, bar_idx: int,
                    now_utc: datetime) -> ScalpUpdate:
        cfg = self.cfg
        upd = ScalpUpdate()
        last = float(bar_df["close"].iloc[-1])
        atr = float(bar_df["ATR"].iloc[-1])

        # End-of-session forced flat
        if self.near_session_close(now_utc):
            upd.close_reason = "eod"; return upd

        # Time stop
        if bar_idx - pos.entry_bar >= cfg.max_hold_bars:
            upd.close_reason = "time"; return upd

        # Update best / unrealized R
        if pos.side > 0:
            pos.best_price = max(pos.best_price, last)
            R = (last - pos.entry) / max(pos.entry - pos.sl, 1e-9)
            tp1_hit = last >= pos.tp1
            tp2_hit = last >= pos.tp2
        else:
            pos.best_price = min(pos.best_price, last)
            R = (pos.entry - last) / max(pos.sl - pos.entry, 1e-9)
            tp1_hit = last <= pos.tp1
            tp2_hit = last <= pos.tp2

        # TP2 full close
        if tp2_hit:
            upd.close_reason = "tp2"; return upd

        # TP1 partial + BE
        if tp1_hit and not pos.tp1_hit:
            pos.tp1_hit = True
            pos.be_moved = True
            upd.partial_close_frac = cfg.partial_size
            upd.new_sl = pos.entry + (1e-4 if pos.side > 0 else -1e-4)
            return upd

        # Chandelier trail after BE
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

    # -------------------- bookkeeping called by live loop --------------------

    def record_close(self, pnl_R: float, close_bar: int, now_utc: datetime):
        self._rollover(now_utc)
        self.day.realized_R += float(pnl_R)
        self.day.last_close_bar = int(close_bar)

    def record_open(self, now_utc: datetime):
        self._rollover(now_utc)
        self.day.trades_taken += 1
