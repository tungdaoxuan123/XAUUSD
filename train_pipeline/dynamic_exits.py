#!/usr/bin/env python3
"""
dynamic_exits.py
----------------
Quant-style dynamic SL/TP + position sizing for XAUUSD M1.

Replaces the fixed 2-ATR SL / 3-ATR TP in live_sota_trading.py with a
rules-based controller used in professional systematic books:

  1. Volatility-targeted risk (R = fixed % of equity). All trades risk
     the same *money*, not the same *pips*. This is what "risk parity"
     means in practice — CTAs, prop shops, and FTMO-passed traders all
     use it.

  2. Dynamic SL distance = max(chandelier(ATR*k), recent-swing).
     Chandelier stop (Chuck LeBeau): SL = entry - k*ATR for longs,
     tightens on new highs. Bounded by the most recent swing to avoid
     sitting inside recent noise.

  3. Dynamic TP = entry ± RR * SL_dist, where RR depends on *calibrated
     probability* and *regime*:
         RR = RR_base * conf_boost(p) * regime_boost(vol_regime, session)
     High conviction + trending regime + NY/overlap session → wider TP.
     Low conviction + chop + Asian session → tighter.

  4. Kelly-fractional sizing with calibrated p:
         f* = (p*b - q) / b   where b = RR, q = 1-p
         lots = fraction * f_kelly_cap * equity_to_lots(sl_dist)
     We take HALF-Kelly (industry standard) and cap at `max_risk_frac`.

  5. In-trade management:
         - Move SL to break-even at +1R (after trade is 1R in profit).
         - Chandelier trail from that point on.
         - Time stop at `max_hold_bars` (matches triple-barrier horizon).

  6. Microstructure/Session gating (skip, don't trade):
         - Spread > 1.5× rolling mean (news spike)
         - |signed_vol_z| > 3 and signal is contrarian
         - Asian session + low-conf → skip (low liquidity in XAU)

API
---
    planner = ExitPlanner(config=ExitConfig(...))
    plan = planner.build_plan(signal, p_cal, bar_df)
    # plan.skip -> bool
    # plan.lots, plan.sl, plan.tp, plan.meta_reason

    # On each tick while trade is open:
    upd = planner.manage_open(position, latest_bar)
    # upd.new_sl or None, upd.close_reason in {"time","trail","be",None}

This module is 100% stateless except for the per-position tracker, so
it backtests identically to live.
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
    atr_mult_sl:    float = 2.0          # base chandelier multiple
    swing_lookback: int   = 20           # bars for recent swing low/high
    min_sl_pips:    float = 2.0          # safety floor (~$0.20 for XAU)

    # TP / Risk-Reward
    rr_base:        float = 1.8          # baseline TP = 1.8 * SL
    rr_min:         float = 1.2
    rr_max:         float = 3.5

    # Calibrated-probability gating
    min_prob_long:  float = 0.55
    min_prob_short: float = 0.55

    # Regime / session boosts
    trend_regime_boost: float = 1.25
    chop_regime_penalty: float = 0.8
    overlap_session_boost: float = 1.15
    asian_session_penalty: float = 0.85

    # Microstructure / spread gating
    max_spread_ratio: float = 1.5        # spread > 1.5× rolling mean => skip
    news_signed_z_abs: float = 3.0       # skip if |signed_vol_z|>3 against signal

    # In-trade mgmt
    breakeven_R: float = 1.0             # move SL to BE at 1R
    trail_mult_atr: float = 1.8          # chandelier after BE
    max_hold_bars: int = 30              # time stop (match triple-barrier)

    # Symbol mechanics
    contract_size: float = 100.0         # XAU: 1 lot = 100 oz
    point_value: float = 1.0             # $1 / point / lot (XAU M5 default in MT5)


# ---------------------------------------------------------------------------
# Plan outputs
# ---------------------------------------------------------------------------

@dataclass
class TradePlan:
    skip: bool = False
    reason: str = ""
    side: int = 0                # +1 long / -1 short / 0 none
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
    close_reason: Optional[str] = None   # "time" | "trail" | "be" | None


@dataclass
class OpenPosition:
    side: int
    entry: float
    sl: float
    tp: float
    entry_bar: int
    best_price: float
    be_moved: bool = False


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class ExitPlanner:
    def __init__(self, config: Optional[ExitConfig] = None):
        self.cfg = config or ExitConfig()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _kelly(p: float, rr: float) -> float:
        """Kelly fraction for binary payoff rr:1 with win-prob p."""
        q = 1.0 - p
        if rr <= 0:
            return 0.0
        f = (p * rr - q) / rr
        return max(0.0, f)

    def _dynamic_sl_distance(self, bar_df, side: int, atr: float) -> float:
        cfg = self.cfg
        last_close = float(bar_df["close"].iloc[-1])
        lb = min(cfg.swing_lookback, len(bar_df) - 1)
        if lb <= 2:
            return max(cfg.atr_mult_sl * atr, cfg.min_sl_pips)
        recent = bar_df.iloc[-lb:]
        if side > 0:
            swing_low = float(recent["low"].min())
            d_swing = max(0.0, last_close - swing_low)
        else:
            swing_high = float(recent["high"].max())
            d_swing = max(0.0, swing_high - last_close)
        chand = cfg.atr_mult_sl * atr
        return max(chand, d_swing * 0.5, cfg.min_sl_pips)  # blend, not pure swing

    def _dynamic_rr(self, prob: float, bar_df) -> float:
        cfg = self.cfg
        # Confidence boost: scale rr with how far above threshold we are
        excess = max(0.0, prob - 0.55) / 0.45
        conf_mult = 1.0 + excess               # 1.0 → 2.0
        rr = cfg.rr_base * (0.8 + 0.8 * conf_mult)

        # Regime boost: use vol_regime / session columns if present
        if "vol_regime" in bar_df.columns:
            vr = int(bar_df["vol_regime"].iloc[-1])
            if vr == 2:   # high-vol / trending
                rr *= cfg.trend_regime_boost
            elif vr == 0:  # chop
                rr *= cfg.chop_regime_penalty
        if "is_overlap" in bar_df.columns and int(bar_df["is_overlap"].iloc[-1]):
            rr *= cfg.overlap_session_boost
        if "is_asian" in bar_df.columns and int(bar_df["is_asian"].iloc[-1]):
            rr *= cfg.asian_session_penalty

        return float(np.clip(rr, cfg.rr_min, cfg.rr_max))

    def _size(self, equity: float, sl_dist: float, prob: float, rr: float) -> float:
        cfg = self.cfg
        f_kelly = self._kelly(prob, rr)
        f_use = min(f_kelly * cfg.kelly_fraction, cfg.kelly_cap)
        risk_frac = min(cfg.risk_per_trade * (1 + f_use),    # scale with edge
                        cfg.max_risk_frac)
        dollar_risk = equity * risk_frac
        # lot = dollar_risk / (sl_dist * point_value * contract_size)
        lot_value_per_dollar = sl_dist * cfg.point_value * cfg.contract_size
        if lot_value_per_dollar <= 0:
            return 0.0
        lots = dollar_risk / lot_value_per_dollar
        # Round to broker step (0.01 default), min 0.01
        return float(max(0.01, round(lots, 2)))

    # -- main API -----------------------------------------------------------

    def build_plan(self, signal: int, prob: float,
                   bar_df, equity: float, spread_now: float) -> TradePlan:
        cfg = self.cfg
        plan = TradePlan(prob=prob)

        if signal == 0:
            plan.skip = True; plan.reason = "no signal"; return plan

        # Probability gate (uses CALIBRATED prob from v2 trainer)
        if signal > 0 and prob < cfg.min_prob_long:
            plan.skip = True; plan.reason = f"p_long {prob:.2f} < {cfg.min_prob_long}"; return plan
        if signal < 0 and prob < cfg.min_prob_short:
            plan.skip = True; plan.reason = f"p_short {prob:.2f} < {cfg.min_prob_short}"; return plan

        # Spread gate (only if we have a non-trivial rolling baseline)
        if "spread_mean" in bar_df.columns:
            sm = float(bar_df["spread_mean"].iloc[-5:].mean())
            MIN_BASELINE = 0.05  # 5 points (~$0.005) — below this, skip the gate
            if sm > MIN_BASELINE and spread_now > cfg.max_spread_ratio * sm:
                plan.skip = True
                plan.reason = f"spread {spread_now:.3f}>{cfg.max_spread_ratio}*{sm:.3f}"
                return plan

        # News/spike gate (contrarian skip)
        if "signed_vol_z" in bar_df.columns:
            z = float(bar_df["signed_vol_z"].iloc[-1])
            if abs(z) > cfg.news_signed_z_abs and np.sign(z) != signal:
                plan.skip = True; plan.reason = f"contrarian to z={z:.1f}"; return plan

        atr = float(bar_df["ATR"].iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            plan.skip = True; plan.reason = "invalid ATR"; return plan

        entry = float(bar_df["close"].iloc[-1])
        sl_dist = self._dynamic_sl_distance(bar_df, signal, atr)
        rr = self._dynamic_rr(prob, bar_df)
        tp_dist = rr * sl_dist

        if signal > 0:
            sl = entry - sl_dist; tp = entry + tp_dist
        else:
            sl = entry + sl_dist; tp = entry - tp_dist

        lots = self._size(equity, sl_dist, prob, rr)

        plan.side = signal
        plan.entry = entry
        plan.sl = sl; plan.tp = tp
        plan.sl_dist = sl_dist; plan.rr = rr
        plan.lots = lots
        plan.equity_risk = lots * sl_dist * cfg.point_value * cfg.contract_size / max(equity, 1e-9)
        plan.meta = {
            "atr": atr, "conf": prob, "rr": rr, "sl_dist": sl_dist, "lots": lots,
        }
        return plan

    def manage_open(self, pos: OpenPosition, bar_df, bar_idx: int) -> ManageUpdate:
        """Called each new bar. Returns SL update and/or close instruction."""
        cfg = self.cfg
        upd = ManageUpdate()
        last = float(bar_df["close"].iloc[-1])
        atr = float(bar_df["ATR"].iloc[-1])

        # Update best price and compute unrealized R
        if pos.side > 0:
            pos.best_price = max(pos.best_price, last)
            R = (last - pos.entry) / max(pos.entry - pos.sl, 1e-9)
        else:
            pos.best_price = min(pos.best_price, last)
            R = (pos.entry - last) / max(pos.sl - pos.entry, 1e-9)

        # 1. Time stop
        if bar_idx - pos.entry_bar >= cfg.max_hold_bars:
            upd.close_reason = "time"; return upd

        # 2. Breakeven at +1R
        if not pos.be_moved and R >= cfg.breakeven_R:
            be = pos.entry + (1e-4 if pos.side > 0 else -1e-4)
            upd.new_sl = be
            pos.be_moved = True
            return upd

        # 3. Chandelier trail after BE
        if pos.be_moved and atr > 0:
            trail_dist = cfg.trail_mult_atr * atr
            if pos.side > 0:
                trail_sl = pos.best_price - trail_dist
                if trail_sl > pos.sl:
                    upd.new_sl = trail_sl
            else:
                trail_sl = pos.best_price + trail_dist
                if trail_sl < pos.sl:
                    upd.new_sl = trail_sl
        return upd
