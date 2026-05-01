#!/usr/bin/env python3
"""
hot_window_executor.py  (v2 — improved)
----------------------------------------
Tick-level execution layer that runs AFTER the model fires a signal.

Changes vs v1
=============
1.  **Async position guard** – the executor now accepts an optional
    `position_check_fn` callable.  On every poll it checks whether a
    broker-side position has already opened (can happen if a previous
    order is still processing).  If one exists the window aborts
    immediately with reason "abort: position already open".

2.  **Tick-rate filter** – if the market goes quiet (tick frequency
    drops below `min_ticks_per_second`) the window pauses scoring and
    logs a warning.  Very low tick rates on FTMO Demo often precede
    gapping / low-liquidity fills, so we avoid entries during them.

3.  **Lee-Ready OFI improvement** – the Lee-Ready fallback now uses a
    quote-midpoint test rather than a pure tick-direction test, making
    it more robust for brokers that don't set TICK_FLAG_BUY/SELL
    (e.g. FTMO Demo).  The tick-rule (previous tick direction) is added
    as a secondary tiebreaker when mid doesn't change.

4.  **Volume-weighted momentum** – micro-momentum is now volume-weighted
    so large-lot ticks count more than noise ticks at the same price.

5.  **Confirmed abort on consecutive bad scores** – a single bad score
    no longer aborts.  We require `abort_confirm_count` consecutive
    polls below `abort_hard` before aborting, reducing false aborts on
    transient spreads.

6.  **Tick-rate feature exposed in score** – a new `tick_rate` feature
    replaces nothing; it adds a small bonus (+0.05 max) when market is
    active, and a small penalty (-0.10) when tick rate falls below
    `min_ticks_per_second`.  Weight is taken from nothing (0.05 from
    momentum, 0.05 from orderflow — both slightly reduced).

7.  **Configurable position check interval** – position is checked every
    `position_check_interval_polls` polls, not every poll, to avoid
    excessive MT5 calls.

Public API (unchanged from v1):
    cfg      = HotWindowConfig()
    executor = HotWindowExecutor(cfg, tick_source=interface.get_ticks,
                                 position_check_fn=interface.get_positions)
    decision = executor.run(
        signal=+1, ref_price=plan.entry, atr=plan.sl_dist / cfg.atr_mult,
        spread_baseline=baseline,
    )
    if decision.fill:
        send_order(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HotWindowConfig:
    # Window timing
    window_seconds: float = 30.0
    poll_hz: float = 5.0

    # Tick lookback per poll
    ticks_per_poll: int = 200

    # Minimum tick quality gate
    min_ticks_for_features: int = 30

    # NEW: Tick-rate filter
    min_ticks_per_second: float = 1.5   # below this = low liquidity, pause scoring
    tick_rate_window_ticks: int = 50    # count ticks over last N ticks to estimate rate

    # Decision thresholds
    trigger_now: float = 0.55
    trigger_fallback: float = 0.20
    abort_hard: float = -0.40

    # NEW: require N consecutive bad polls before aborting (prevents false aborts)
    abort_confirm_count: int = 3

    # Pullback opportunity
    pullback_atr: float = 0.25
    pullback_flip_ofi: float = 0.30
    pullback_min_seconds: float = 5.0
    pullback_ofi_window_seconds: float = 3.0

    # Slippage cap
    max_slippage_atr: float = 0.50

    # Score weights (slightly adjusted — tick_rate takes 0.05 from mm+ofi)
    w_momentum: float = 0.45
    w_orderflow: float = 0.25
    w_spread: float = 0.20
    w_tick_rate: float = 0.10   # NEW

    # Spread gate
    max_spread_ratio: float = 1.5

    # Adverse-momentum hard abort
    adverse_momentum_atr: float = 0.50

    # NEW: Position guard — check every N polls
    position_check_interval_polls: int = 3

    # Diagnostic logging
    log_every: float = 2.0


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class HotDecision:
    fill: bool = False
    reason: str = ""
    fill_price: Optional[float] = None
    score_at_decision: float = 0.0
    elapsed_seconds: float = 0.0
    samples: int = 0
    history: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tick utilities
# ---------------------------------------------------------------------------

def _mid(tick) -> float:
    try:
        bid = float(tick["bid"]); ask = float(tick["ask"])
    except Exception:
        bid = float(getattr(tick, "bid", 0.0))
        ask = float(getattr(tick, "ask", 0.0))
    if bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    last = 0.0
    try:
        if "last" in tick.dtype.names:
            last = float(tick["last"])
    except Exception:
        pass
    return last if last > 0 else (bid or ask)


def _ts(tick) -> float:
    try:
        if "time_msc" in tick.dtype.names:
            return float(tick["time_msc"]) / 1000.0
        return float(tick["time"])
    except Exception:
        return float(getattr(tick, "time", 0.0))


def _tick_volume(tick) -> float:
    try:
        return float(tick["volume_real"]) if "volume_real" in tick.dtype.names \
            else float(tick["volume"])
    except Exception:
        return float(getattr(tick, "volume", 1.0))


def _is_buy_aggressor(tick, prev_mid: float, prev_side: int = 0) -> int:
    """
    Improved Lee-Ready with quote-midpoint test + tick-rule tiebreaker.

    Priority:
      1. MT5 TICK_FLAG_BUY / TICK_FLAG_SELL (most accurate, broker-dependent)
      2. Midpoint test: if mid crossed up vs prev_mid -> buy; crossed down -> sell
      3. Tick rule tiebreaker: use prev_side if mid unchanged (avoids 0)
    """
    try:
        flags = int(tick["flags"])
        TICK_FLAG_BUY = 0x4
        TICK_FLAG_SELL = 0x8
        if flags & TICK_FLAG_BUY:
            return +1
        if flags & TICK_FLAG_SELL:
            return -1
    except Exception:
        pass
    m = _mid(tick)
    if prev_mid > 0:
        if m > prev_mid:
            return +1
        if m < prev_mid:
            return -1
    # tiebreaker: carry previous direction (tick rule)
    return prev_side


def _tick_rate(ticks, window: int = 50) -> float:
    """Estimate ticks/second from the last `window` ticks."""
    if ticks is None or len(ticks) < 2:
        return 0.0
    batch = ticks[-min(window, len(ticks)):]
    t0 = _ts(batch[0])
    t1 = _ts(batch[-1])
    dt = t1 - t0
    if dt <= 0:
        return 0.0
    return len(batch) / dt


def _compute_ofi_vw(ticks, signal: int, start_idx: int = 0) -> float:
    """
    Volume-weighted OFI (improved from unweighted).
    Uses improved Lee-Ready with tick-rule tiebreaker.
    """
    batch = ticks[start_idx:]
    if len(batch) < 2:
        return 0.0
    buy_vol = sell_vol = 0.0
    prev_mid = _mid(batch[0])
    prev_side = 0
    for t in batch[1:]:
        v = _tick_volume(t)
        side = _is_buy_aggressor(t, prev_mid, prev_side)
        if side > 0:
            buy_vol += v
        elif side < 0:
            sell_vol += v
        prev_mid = _mid(t)
        if side != 0:
            prev_side = side
    total = buy_vol + sell_vol
    ofi_raw = (buy_vol - sell_vol) / total if total > 0 else 0.0
    return ofi_raw * (1 if signal > 0 else -1)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_micro_features(ticks, signal: int, atr: float,
                            ref_price: float, spread_baseline: float,
                            min_ticks: int = 30,
                            tick_rate_window: int = 50,
                            min_ticks_per_second: float = 1.5) -> dict:
    if ticks is None or len(ticks) < min_ticks:
        n = len(ticks) if ticks is not None else 0
        return {"score": 0.0, "mm": 0.0, "ofi": 0.0, "sq": 0.0,
                "tick_rate_feat": 0.0, "tick_rate": 0.0,
                "spread": 0.0, "mid": ref_price, "n": n,
                "quality": False, "low_liquidity": True}

    mids = np.array([_mid(t) for t in ticks])
    times = np.array([_ts(t) for t in ticks])
    spreads = []
    for t in ticks:
        try:
            s = float(t["ask"] - t["bid"])
        except Exception:
            s = 0.0
        if s > 0:
            spreads.append(s)
    spread_now = float(np.median(spreads[-min(20, len(spreads)):]) if spreads else 0.0)

    # Tick rate
    tr = _tick_rate(ticks, window=tick_rate_window)
    if tr >= min_ticks_per_second:
        tick_rate_feat = min(0.05, (tr - min_ticks_per_second) / 10.0)  # small bonus
    else:
        tick_rate_feat = -0.10  # penalty for low liquidity
    low_liquidity = tr < min_ticks_per_second

    # --- Volume-weighted micro-momentum (IMPROVED) ---
    now_t = float(times[-1])
    mid_now = float(mids[-1])
    a = atr if atr and atr > 0 else max(mid_now * 0.0005, 0.05)

    def vw_mid_at(delta: float) -> float:
        """Volume-weighted average mid in the last `delta` seconds."""
        target = now_t - delta
        idx = int(np.searchsorted(times, target))
        idx = max(0, min(idx, len(mids) - 1))
        seg_mids = mids[idx:]
        if len(seg_mids) == 0:
            return float(mids[idx])
        vols = np.array([_tick_volume(t) for t in ticks[idx:]])
        total_v = vols.sum()
        if total_v > 0:
            return float((seg_mids * vols).sum() / total_v)
        return float(seg_mids.mean())

    mm_1  = (mid_now - vw_mid_at(1.0))  / a
    mm_3  = (mid_now - vw_mid_at(3.0))  / a
    mm_10 = (mid_now - vw_mid_at(10.0)) / a
    mm = float(np.clip(0.5 * mm_1 + 0.3 * mm_3 + 0.2 * mm_10, -2, 2))
    mm_signed = mm * (1 if signal > 0 else -1)

    # --- OFI (volume-weighted + decorrelated) ---
    ofi_full = _compute_ofi_vw(ticks, signal, start_idx=0)
    mm_proxy = float(np.clip(mm_signed, -1.0, 1.0))
    DECORR = 0.4
    ofi_signed = float(np.clip(ofi_full - DECORR * mm_proxy, -1.0, 1.0))

    # --- Spread quality [-1, 1] ---
    if spread_baseline > 1e-6:
        sq = float(np.clip(1.0 - (spread_now / spread_baseline), -1.0, 1.0))
    else:
        sq = 0.0

    return {
        "score": None,
        "mm": float(mm_signed),
        "ofi": float(ofi_signed),
        "sq": float(sq),
        "tick_rate_feat": float(tick_rate_feat),
        "tick_rate": float(tr),
        "spread": float(spread_now),
        "mid": float(mid_now),
        "n": int(len(ticks)),
        "quality": True,
        "low_liquidity": bool(low_liquidity),
    }


def _pullback_ofi(ticks, signal: int, times: np.ndarray,
                  window_seconds: float) -> float:
    if ticks is None or len(ticks) < 2:
        return 0.0
    cutoff = float(times[-1]) - window_seconds
    start_idx = int(np.searchsorted(times, cutoff))
    start_idx = max(0, start_idx)
    return _compute_ofi_vw(ticks, signal, start_idx=start_idx)


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

class HotWindowExecutor:
    """
    30-second tick-level decision loop with:
    - Position guard (aborts if a position opens mid-window)
    - Tick-rate filter (avoids low-liquidity fills)
    - Volume-weighted momentum + OFI
    - Consecutive-bad-score abort (reduces false aborts)
    """

    def __init__(self,
                 cfg: Optional[HotWindowConfig] = None,
                 tick_source: Optional[Callable[[int], list]] = None,
                 position_check_fn: Optional[Callable[[], list]] = None,
                 sleep_fn: Callable[[float], None] = time.sleep,
                 clock_fn: Callable[[], float] = time.monotonic):
        self.cfg = cfg or HotWindowConfig()
        self.tick_source = tick_source
        self.position_check_fn = position_check_fn   # NEW
        self.sleep_fn = sleep_fn
        self.clock_fn = clock_fn

    def _score(self, feats: dict) -> float:
        c = self.cfg
        return float(
            c.w_momentum   * feats["mm"]
            + c.w_orderflow  * feats["ofi"]
            + c.w_spread     * feats["sq"]
            + c.w_tick_rate  * feats["tick_rate_feat"]   # NEW
        )

    def run(self, signal: int, ref_price: float, atr: float,
            spread_baseline: float) -> HotDecision:
        if signal == 0 or self.tick_source is None:
            return HotDecision(fill=False, reason="no signal or no tick source")

        cfg = self.cfg
        period = 1.0 / max(cfg.poll_hz, 0.1)
        deadline = self.clock_fn() + cfg.window_seconds
        start = self.clock_fn()
        last_log = 0.0
        decision = HotDecision()
        recent_scores: List[float] = []
        consecutive_aborts = 0   # NEW: for confirmed abort
        poll_count = 0

        logger.info(
            f"[hot] open window {cfg.window_seconds:.0f}s sig={signal:+d} "
            f"ref={ref_price:.2f} atr={atr:.3f} "
            f"now={cfg.trigger_now} fb={cfg.trigger_fallback} "
            f"abort={cfg.abort_hard} (confirm={cfg.abort_confirm_count})"
        )

        while True:
            now = self.clock_fn()
            elapsed = now - start
            if now >= deadline:
                break

            poll_count += 1

            # --- Position guard (NEW) ---
            if (self.position_check_fn is not None
                    and poll_count % cfg.position_check_interval_polls == 0):
                try:
                    existing = self.position_check_fn()
                    if existing and len(existing) > 0:
                        decision.reason = "abort: position already open mid-window"
                        decision.score_at_decision = 0.0
                        break
                except Exception as e:
                    logger.warning(f"[hot] position check failed: {e}")

            # --- Tick fetch ---
            ticks = None
            try:
                ticks = self.tick_source(cfg.ticks_per_poll)
            except Exception as e:
                logger.warning(f"[hot] tick fetch failed: {e}")

            feats = compute_micro_features(
                ticks, signal=signal, atr=atr,
                ref_price=ref_price, spread_baseline=spread_baseline,
                min_ticks=cfg.min_ticks_for_features,
                tick_rate_window=cfg.tick_rate_window_ticks,
                min_ticks_per_second=cfg.min_ticks_per_second,
            )
            score = self._score(feats)
            feats["score"] = score
            feats["t"] = elapsed
            decision.history.append(feats)
            decision.samples += 1
            recent_scores.append(score)
            if len(recent_scores) > 6:
                recent_scores.pop(0)

            # Throttled log
            if elapsed - last_log >= cfg.log_every:
                last_log = elapsed
                liq_flag = " [LOW-LIQ]" if feats.get("low_liquidity") else ""
                qual_flag = "" if feats.get("quality", True) else " [LOW-TICK]"
                logger.info(
                    f"[hot t={elapsed:4.1f}s]{qual_flag}{liq_flag} "
                    f"mid={feats['mid']:.2f} "
                    f"mm={feats['mm']:+.2f} ofi={feats['ofi']:+.2f} "
                    f"sq={feats['sq']:+.2f} tr={feats['tick_rate']:.1f}/s "
                    f"score={score:+.2f} spread={feats['spread']:.3f} n={feats['n']}"
                )

            # Skip scoring decisions during low-liquidity periods
            if feats.get("low_liquidity") and elapsed < cfg.window_seconds * 0.8:
                logger.debug(f"[hot] low liquidity pause (tr={feats['tick_rate']:.1f}/s)")
                self.sleep_fn(period)
                continue

            cur_price = feats["mid"]

            adverse_pos_signed = (
                ((ref_price - cur_price) / max(atr, 1e-9)) if signal > 0
                else ((cur_price - ref_price) / max(atr, 1e-9))
            )
            exec_slip = (
                ((cur_price - ref_price) / max(atr, 1e-9)) if signal > 0
                else ((ref_price - cur_price) / max(atr, 1e-9))
            )

            # --- Hard aborts (consecutive confirmation) ---
            if score <= cfg.abort_hard and elapsed >= 2.0:
                consecutive_aborts += 1
                if consecutive_aborts >= cfg.abort_confirm_count:
                    decision.reason = (
                        f"abort: score {score:.2f} <= {cfg.abort_hard} "
                        f"({consecutive_aborts} consecutive polls)"
                    )
                    decision.score_at_decision = score
                    break
            else:
                consecutive_aborts = 0  # reset on any non-abort poll

            if adverse_pos_signed >= cfg.adverse_momentum_atr:
                decision.reason = f"abort: adverse momentum {adverse_pos_signed:.2f}*ATR"
                decision.score_at_decision = score
                break

            if (spread_baseline > 1e-6
                    and feats["spread"] > cfg.max_spread_ratio * spread_baseline):
                decision.reason = (
                    f"abort: spread {feats['spread']:.3f}>"
                    f"{cfg.max_spread_ratio}*{spread_baseline:.3f}"
                )
                decision.score_at_decision = score
                break

            slip_too_wide = exec_slip > cfg.max_slippage_atr

            # --- Confirm-and-go ---
            if score >= cfg.trigger_now and not slip_too_wide:
                strong = score >= cfg.trigger_now + 0.15
                run2 = (len(recent_scores) >= 2
                        and all(s >= cfg.trigger_now for s in recent_scores[-2:]))
                if strong or run2:
                    decision.fill = True
                    decision.reason = f"confirm-go score={score:.2f}"
                    decision.fill_price = cur_price
                    decision.score_at_decision = score
                    break

            # --- Pullback-and-flip ---
            if (elapsed >= cfg.pullback_min_seconds
                    and adverse_pos_signed >= cfg.pullback_atr
                    and not slip_too_wide):
                times_arr = (np.array([_ts(t) for t in ticks])
                             if ticks is not None else np.array([]))
                pb_ofi = _pullback_ofi(
                    ticks, signal, times_arr,
                    window_seconds=cfg.pullback_ofi_window_seconds,
                )
                if pb_ofi >= cfg.pullback_flip_ofi:
                    decision.fill = True
                    decision.reason = (
                        f"pullback-fill retrace={adverse_pos_signed:.2f}*ATR "
                        f"pb_ofi={pb_ofi:+.2f}"
                    )
                    decision.fill_price = cur_price
                    decision.score_at_decision = score
                    break

            self.sleep_fn(period)

        # --- Time-up resolution ---
        if not decision.fill and not decision.reason:
            last_score = recent_scores[-1] if recent_scores else 0.0
            last_feats = decision.history[-1] if decision.history else None
            slip_now = 0.0
            if last_feats:
                slip_now = (
                    ((last_feats["mid"] - ref_price) / max(atr, 1e-9)) if signal > 0
                    else ((ref_price - last_feats["mid"]) / max(atr, 1e-9))
                )
            if last_score >= cfg.trigger_fallback and slip_now <= cfg.max_slippage_atr:
                decision.fill = True
                decision.reason = f"timeout-fill score={last_score:.2f} slip={slip_now:+.2f}*ATR"
                decision.fill_price = last_feats["mid"] if last_feats else ref_price
                decision.score_at_decision = last_score
            else:
                decision.reason = (
                    f"timeout-skip score={last_score:.2f} slip={slip_now:+.2f}*ATR"
                )
                decision.score_at_decision = last_score

        decision.elapsed_seconds = self.clock_fn() - start
        logger.info(
            f"[hot] decision fill={decision.fill} reason={decision.reason} "
            f"elapsed={decision.elapsed_seconds:.1f}s samples={decision.samples}"
        )
        return decision
