#!/usr/bin/env python3
"""
dynamic_exits.py — LONG-ONLY exit planner with fixed 2:1 RR
----------------------------------------------------------

Simplified quant-style SL/TP + position sizing for long-only XAUUSD M1.

  1. Volatility-targeted risk: SL distance = max(chandelier(ATR*k), recent-swing)
  2. Dynamic TP = entry + 2 * SL_dist (fixed 2:1 ratio)
  3. Half-Kelly fractional sizing with calibrated probability
  4. In-trade: time stop at max_hold_bars (matches triple-barrier horizon)

API
---
    planner = ExitPlanner(config=ExitConfig(...))
    plan = planner.build_plan(signal=1, prob=p_long, bar_df, equity, spread)
    upd = planner.manage_open(position, bar_df, bar_idx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExitConfig:
    # Risk
    risk_per_trade: float = 0.005        # 0.5% of equity per trade
    max_risk_frac:  float = 0.01         # hard cap per trade
    kelly_fraction: float = 0.5          # half-Kelly
    kelly_cap:      float = 0.25         # never bet more than 25% Kelly fraction

    # SL
    atr_mult_sl:    float = 1.0          # base chandelier multiple (matches triple-barrier sl-atr)
    swing_lookback: int   = 20           # bars for recent swing low
    min_sl_pips:    float = 2.0          # safety floor (~$0.20 for XAU)

    # TP / Risk-Reward (fixed 2:1)
    rr_base:        float = 2.0          # fixed TP = 2 * SL

    # Probability gate
    min_prob:       float = 0.55

    # Microstructure / spread gating
    max_spread_ratio: float = 1.5
    news_signed_z_abs: float = 3.0

    # In-trade mgmt
    max_hold_bars: int = 30              # time stop (match triple-barrier)

    # Symbol mechanics
    contract_size: float = 100.0         # XAU: 1 lot = 100 oz
    point_value: float = 1.0             # $1 / point / lot


# ---------------------------------------------------------------------------
# Plan outputs
# ---------------------------------------------------------------------------

@dataclass
class TradePlan:
    skip: bool = False
    reason: str = ""
    side: int = 0                # +1 long / 0 none
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lots: float = 0.0
    sl_dist: float = 0.0
    rr: float = 0.0
    prob: float = 0.0
    equity_risk: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class ManageUpdate:
    new_sl: Optional[float] = None
    close_reason: Optional[str] = None   # "time" | None


@dataclass
class OpenPosition:
    side: int                     # +1 long
    entry: float
    sl: float
    tp: float
    entry_bar: int
    best_price: float


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class ExitPlanner:
    def __init__(self, config: Optional[ExitConfig] = None, direction: int = 1):
        self.cfg = config or ExitConfig()
        self.direction = direction  # +1=long, -1=short

    @staticmethod
    def _kelly(p: float, rr: float) -> float:
        q = 1.0 - p
        if rr <= 0:
            return 0.0
        f = (p * rr - q) / rr
        return max(0.0, f)

    def _dynamic_sl_distance(self, bar_df, atr: float) -> float:
        cfg = self.cfg
        last_close = float(bar_df["close"].iloc[-1])
        lb = min(cfg.swing_lookback, len(bar_df) - 1)
        if lb <= 2:
            return max(cfg.atr_mult_sl * atr, cfg.min_sl_pips)
        recent = bar_df.iloc[-lb:]
        if self.direction > 0:
            swing_low = float(recent["low"].min())
            d_swing = max(0.0, last_close - swing_low)
        else:
            swing_high = float(recent["high"].max())
            d_swing = max(0.0, swing_high - last_close)
        chand = cfg.atr_mult_sl * atr
        return max(chand, d_swing * 0.5, cfg.min_sl_pips)

    def _size(self, equity: float, sl_dist: float, prob: float, rr: float) -> float:
        cfg = self.cfg
        f_kelly = self._kelly(prob, rr)
        f_use = min(f_kelly * cfg.kelly_fraction, cfg.kelly_cap)
        risk_frac = min(cfg.risk_per_trade * (1 + f_use), cfg.max_risk_frac)
        dollar_risk = equity * risk_frac
        lot_value_per_dollar = sl_dist * cfg.point_value * cfg.contract_size
        if lot_value_per_dollar <= 0:
            return 0.0
        lots = dollar_risk / lot_value_per_dollar
        return float(max(0.01, round(lots, 2)))

    def build_plan(self, signal: int, prob: float,
                   bar_df, equity: float, spread_now: float) -> TradePlan:
        cfg = self.cfg
        plan = TradePlan(prob=prob)
        d = self.direction

        if signal <= 0:
            plan.skip = True
            plan.reason = f"no {'long' if d>0 else 'short'} signal"
            return plan

        if prob < cfg.min_prob:
            plan.skip = True
            plan.reason = f"p {prob:.2f} < {cfg.min_prob}"
            return plan

        if "spread_mean" in bar_df.columns:
            sm = float(bar_df["spread_mean"].iloc[-5:].mean())
            MIN_BASELINE = 0.05
            if sm > MIN_BASELINE and spread_now > cfg.max_spread_ratio * sm:
                plan.skip = True
                plan.reason = f"spread {spread_now:.3f}>{cfg.max_spread_ratio}*{sm:.3f}"
                return plan

        if "signed_vol_z" in bar_df.columns:
            z = float(bar_df["signed_vol_z"].iloc[-1])
            if abs(z) > cfg.news_signed_z_abs and np.sign(z) != d:
                plan.skip = True
                plan.reason = f"contrarian vol z={z:.1f}"
                return plan

        atr = float(bar_df["ATR"].iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            plan.skip = True; plan.reason = "invalid ATR"; return plan

        entry = float(bar_df["close"].iloc[-1])
        sl_dist = self._dynamic_sl_distance(bar_df, atr)
        rr = cfg.rr_base
        tp_dist = rr * sl_dist

        if d > 0:
            sl = entry - sl_dist; tp = entry + tp_dist
        else:
            sl = entry + sl_dist; tp = entry - tp_dist

        lots = self._size(equity, sl_dist, prob, rr)

        plan.side = d
        plan.entry = entry
        plan.sl = sl; plan.tp = tp
        plan.sl_dist = sl_dist; plan.rr = rr
        plan.lots = lots
        plan.equity_risk = (
            lots * sl_dist * cfg.point_value * cfg.contract_size
            / max(equity, 1e-9)
        )
        plan.meta = {
            "atr": atr, "conf": prob, "rr": rr,
            "sl_dist": sl_dist, "lots": lots,
        }
        return plan

    def manage_open(self, pos: OpenPosition, bar_df, bar_idx: int) -> ManageUpdate:
        cfg = self.cfg
        upd = ManageUpdate()
        last = float(bar_df["close"].iloc[-1])

        pos.best_price = max(pos.best_price, last) if pos.side > 0 else min(pos.best_price, last)

        if bar_idx - pos.entry_bar >= cfg.max_hold_bars:
            upd.close_reason = "time"
        return upd
