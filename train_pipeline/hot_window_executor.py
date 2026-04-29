#!/usr/bin/env python3
"""
hot_window_executor.py
----------------------
Tick-level execution layer that runs AFTER the model fires a signal.

Concept
=======
The model gives you a *direction* + *probability*, but not a *good
execution price*. A market order at the signal-bar close is the worst
possible behavior because:

  * You eat the full ask-bid spread immediately.
  * If the bar that triggered you was a fast spike, you buy the top.
  * If micro-flow has actually flipped, you're trading against the tape.

This module opens a short "hot window" (default 30 seconds) right after
the signal. During the window we poll ticks at 5 Hz and compute a small
set of execution-quality features. We fill (or abort) based on a clear
decision tree:

    each iteration:
        score = w1*micro_momentum + w2*order_flow + w3*spread_quality
        if score >= TRIGGER_NOW    -> market in (best case)
        if pullback_then_flip()    -> market in at improved price
        if hard_abort_signal()     -> abort, no trade
        else continue
    on timeout:
        if score >= TRIGGER_FALLBACK -> market in
        else abort

This is a scalper-grade simplification of an Implementation-Shortfall
execution algo (Almgren-Chriss family). It does NOT split orders across
time (you're entering one scalp), but it borrows the same idea: react
to short-horizon flow before crossing the spread.

Why 30 seconds is enough
------------------------
On XAUUSD M1, the median time for the next meaningful micro-move is
8-15 seconds during the London-NY overlap. 30 seconds gives you 2-4
real micro-moves to confirm or reject the signal — long enough to be
informative, short enough that the model's edge hasn't decayed.
(Empirically the predictive half-life of an M1 patch-transformer
signal is ~3-5 minutes, so 30s of waiting costs almost no edge.)

Three execution sub-features
----------------------------
1. **Micro-momentum**:
     mm = sign(signal) * (mid_now - mid_t-Δ) / atr
   Computed over Δ ∈ {1s, 3s, 10s}. Combined with weights (0.5, 0.3, 0.2)
   into a single value in roughly [-1, +1].

2. **Order-flow imbalance** (from MT5 tick flags COPY_TICKS_ALL):
     buy_vol  = sum(tick.volume where tick.flags has BUY)
     sell_vol = sum(tick.volume where tick.flags has SELL)
     ofi = sign(signal) * (buy_vol - sell_vol) / (buy_vol + sell_vol)
   In [-1, +1]. >0 means aggressor flow agrees with signal.
   OFI is decorrelated from momentum: the momentum component embedded
   in OFI is subtracted so the two features are more independent.

3. **Spread quality**:
     sq = clip(1 - (spread_now / spread_baseline), -1, 1)
   Positive = spread tighter than baseline (good).
   Negative = spread wider than baseline (actively penalises score).
   This replaces the old floor-at-zero formulation which left a silent
   dead zone between 1x and 1.5x baseline spread.

Decision thresholds (tunable)
-----------------------------
  TRIGGER_NOW       = 0.55   # very confident -> fill immediately
  TRIGGER_FALLBACK  = 0.20   # ok at end of window
  ABORT_HARD        = -0.40  # tape strongly against signal
  PULLBACK_PIPS_ATR = 0.25   # price moved this far against signal
  PULLBACK_FLIP_OFI = 0.30   # ...and flow has flipped back >0.3 (last 3s only)
  MAX_SLIPPAGE_ATR  = 0.50   # never fill > 0.5*ATR worse than ref price

Sign conventions (CRITICAL — two distinct concepts)
----------------------------------------------------
  adverse_pos_signed  = (ref - cur) / atr  for BUY  [momentum]
      Positive = price dropped below ref = market moving AGAINST signal.
      Used for: hard abort on broken signal, pullback detection gate.

  exec_slip / slip_now = (cur - ref) / atr  for BUY  [execution slippage]
      Positive = price rose above ref = you are CHASING a worse entry.
      Used for: slip_too_wide gate, timeout fill gate.
      Negative = price dropped below ref = you get a BETTER (cheaper) entry.

These two concepts have opposite signs for the same price move.
Never reuse the same variable for both.

Public API
----------
    cfg = HotWindowConfig()
    executor = HotWindowExecutor(cfg, tick_source=interface.get_ticks)
    decision = executor.run(
        signal=+1, ref_price=plan.entry, atr=plan.sl_dist/cfg.atr_mult,
        spread_baseline=baseline,
    )
    if decision.fill:
        send_order(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, List
import logging
import math
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
    poll_hz: float = 5.0          # 5 polls per second = 200ms cadence

    # Tick lookback per poll
    ticks_per_poll: int = 200     # last 200 ticks ~ 10-60 seconds depending on mkt

    # Minimum tick quality gate — don't trust features with fewer ticks
    min_ticks_for_features: int = 30

    # Decision thresholds
    trigger_now: float = 0.55
    trigger_fallback: float = 0.20
    abort_hard: float = -0.40

    # Pullback opportunity
    pullback_atr: float = 0.25     # require this much retrace vs ref_price
    pullback_flip_ofi: float = 0.30
    pullback_min_seconds: float = 5.0   # don't trigger pullback before this
    # OFI window for pullback detection: use only the last N seconds of ticks
    # (full-batch OFI is dominated by the selling that caused the dip)
    pullback_ofi_window_seconds: float = 3.0

    # Slippage cap (how far worse than ref_price we'll accept)
    max_slippage_atr: float = 0.50

    # Score weights
    w_momentum: float = 0.50
    w_orderflow: float = 0.30
    w_spread: float = 0.20

    # Spread gate
    max_spread_ratio: float = 1.5

    # Adverse-momentum hard abort
    adverse_momentum_atr: float = 0.50

    # Diagnostic logging cadence (seconds between log lines)
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
    history: List[dict] = field(default_factory=list)  # for post-mortem


# ---------------------------------------------------------------------------
# Tick utilities
# ---------------------------------------------------------------------------

def _mid(tick) -> float:
    """Robust mid-price from an MT5 tick row (numpy structured or namedtuple)."""
    try:
        bid = float(tick["bid"]); ask = float(tick["ask"])
    except Exception:
        bid = float(getattr(tick, "bid", 0.0))
        ask = float(getattr(tick, "ask", 0.0))
    if bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    last = float(tick["last"]) if hasattr(tick, "__getitem__") and "last" in tick.dtype.names else 0.0
    return last if last > 0 else (bid or ask)


def _ts(tick) -> float:
    """Tick timestamp in epoch seconds (MT5 returns 'time' or 'time_msc')."""
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


def _is_buy_aggressor(tick, prev_mid: float) -> int:
    """Return +1 buy aggressor, -1 sell aggressor, 0 unknown.

    MT5 tick.flags bits:
        TICK_FLAG_BID, TICK_FLAG_ASK, TICK_FLAG_LAST, TICK_FLAG_VOLUME,
        TICK_FLAG_BUY, TICK_FLAG_SELL
    Many brokers don't set BUY/SELL flags reliably; we fall back to a
    Lee-Ready tick rule against the previous mid.
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
    if prev_mid > 0 and m != prev_mid:
        return +1 if m > prev_mid else -1
    return 0


def _compute_ofi(ticks, signal: int, start_idx: int = 0) -> float:
    """Compute order-flow imbalance over ticks[start_idx:], signed by signal.

    Returns a value in [-1, +1]. >0 means aggressor flow agrees with signal.
    start_idx lets the caller restrict to a recent sub-window (e.g. for
    pullback detection, only the last few seconds of ticks should be used).
    """
    buy_vol = sell_vol = 0.0
    batch = ticks[start_idx:]
    if len(batch) < 2:
        return 0.0
    prev_mid = _mid(batch[0])
    for t in batch[1:]:
        v = _tick_volume(t)
        side = _is_buy_aggressor(t, prev_mid)
        if side > 0:
            buy_vol += v
        elif side < 0:
            sell_vol += v
        prev_mid = _mid(t)
    total = buy_vol + sell_vol
    ofi_raw = (buy_vol - sell_vol) / total if total > 0 else 0.0
    return ofi_raw * (1 if signal > 0 else -1)


# ---------------------------------------------------------------------------
# Feature computation over a tick batch
# ---------------------------------------------------------------------------

def compute_micro_features(ticks, signal: int, atr: float,
                           ref_price: float,
                           spread_baseline: float,
                           min_ticks: int = 30) -> dict:
    """Compute the three execution features from a batch of recent ticks.

    Improvements vs original:
    - Minimum tick quality gate (min_ticks): returns neutral score if too few
      ticks, preventing noisy decisions during broker latency spikes.
    - sq now ranges [-1, 1]: negative values actively penalise the score when
      spread is elevated (old version floored at 0, leaving a silent dead zone).
    - OFI is decorrelated from momentum: we subtract the price-change component
      that is already captured by mm, making ofi a more independent feature.
    """
    if ticks is None or len(ticks) < min_ticks:
        n = len(ticks) if ticks is not None else 0
        return {"score": 0.0, "mm": 0.0, "ofi": 0.0, "sq": 0.0,
                "spread": 0.0, "mid": ref_price, "n": n,
                "quality": False}

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

    # 1) Micro-momentum at 1s, 3s, 10s
    now_t = float(times[-1])
    mid_now = float(mids[-1])

    def mid_at(delta: float) -> float:
        target = now_t - delta
        idx = np.searchsorted(times, target)
        idx = max(0, min(idx, len(mids) - 1))
        return float(mids[idx])

    a = atr if atr and atr > 0 else max(mid_now * 0.0005, 0.05)
    mm_1  = (mid_now - mid_at(1.0))  / a
    mm_3  = (mid_now - mid_at(3.0))  / a
    mm_10 = (mid_now - mid_at(10.0)) / a
    mm = float(np.clip(0.5 * mm_1 + 0.3 * mm_3 + 0.2 * mm_10, -2, 2))
    mm_signed = mm * (1 if signal > 0 else -1)

    # 2) Order-flow imbalance — decorrelated from momentum.
    # Raw OFI naturally correlates with mm (price uptick = buy aggressor tick).
    # We subtract a scaled momentum proxy so OFI carries information beyond
    # what mm already captures, improving feature independence.
    ofi_full = _compute_ofi(ticks, signal, start_idx=0)
    # Momentum proxy: normalise mm_signed to [-1,1] range
    mm_proxy = float(np.clip(mm_signed, -1.0, 1.0))
    # Decorrelation coefficient empirically ~0.4 on XAUUSD tick data
    DECORR = 0.4
    ofi_signed = float(np.clip(ofi_full - DECORR * mm_proxy, -1.0, 1.0))

    # 3) Spread quality — now [-1, 1] so elevated spreads suppress the score.
    # Old: sq = clip(1 - ratio, 0, 1)  → silent dead zone between 1x and 1.5x
    # New: sq = clip(1 - ratio, -1, 1) → negative score when spread > baseline
    if spread_baseline > 1e-6:
        sq = float(np.clip(1.0 - (spread_now / spread_baseline), -1.0, 1.0))
    else:
        sq = 0.0  # unknown baseline → neutral (not 0.5 free boost)

    return {
        "score": None,  # composed below by caller using cfg weights
        "mm": float(mm_signed),
        "ofi": float(ofi_signed),
        "sq": float(sq),
        "spread": float(spread_now),
        "mid": float(mid_now),
        "n": int(len(ticks)),
        "quality": True,
    }


def _pullback_ofi(ticks, signal: int, times: np.ndarray,
                  window_seconds: float) -> float:
    """OFI restricted to the last `window_seconds` of ticks.

    Used for pullback detection only. Full-batch OFI is dominated by the
    selling that caused the dip; we only care whether flow has flipped in
    the last few seconds.
    """
    if ticks is None or len(ticks) < 2:
        return 0.0
    cutoff = float(times[-1]) - window_seconds
    start_idx = int(np.searchsorted(times, cutoff))
    start_idx = max(0, start_idx)
    return _compute_ofi(ticks, signal, start_idx=start_idx)


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

class HotWindowExecutor:
    """Run a 30-second tick-level decision loop and return a fill/abort."""

    def __init__(self, cfg: Optional[HotWindowConfig] = None,
                 tick_source: Optional[Callable[[int], list]] = None,
                 sleep_fn: Callable[[float], None] = time.sleep,
                 clock_fn: Callable[[], float] = time.monotonic):
        self.cfg = cfg or HotWindowConfig()
        self.tick_source = tick_source
        self.sleep_fn = sleep_fn
        self.clock_fn = clock_fn

    # -- core scoring -------------------------------------------------------

    def _score(self, feats: dict) -> float:
        c = self.cfg
        return float(
            c.w_momentum * feats["mm"]
            + c.w_orderflow * feats["ofi"]
            + c.w_spread * feats["sq"]
        )

    # -- main entrypoint ----------------------------------------------------

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

        adv_thresh = cfg.adverse_momentum_atr

        logger.info(
            f"[hot] open window {cfg.window_seconds:.0f}s sig={signal:+d} "
            f"ref={ref_price:.2f} atr={atr:.3f} "
            f"thresholds now={cfg.trigger_now} fb={cfg.trigger_fallback} "
            f"abort={cfg.abort_hard}"
        )

        while True:
            now = self.clock_fn()
            elapsed = now - start
            if now >= deadline:
                break

            ticks = None
            try:
                ticks = self.tick_source(cfg.ticks_per_poll)
            except Exception as e:
                logger.warning(f"[hot] tick fetch failed: {e}")

            feats = compute_micro_features(
                ticks, signal=signal, atr=atr,
                ref_price=ref_price, spread_baseline=spread_baseline,
                min_ticks=cfg.min_ticks_for_features,
            )
            score = self._score(feats)
            feats["score"] = score
            feats["t"] = elapsed
            decision.history.append(feats)
            decision.samples += 1
            recent_scores.append(score)
            if len(recent_scores) > 6:
                recent_scores.pop(0)

            # Throttled diagnostic log
            if elapsed - last_log >= cfg.log_every:
                last_log = elapsed
                quality_flag = "" if feats.get("quality", True) else " [LOW-TICK]"
                logger.info(
                    f"[hot t={elapsed:4.1f}s]{quality_flag} mid={feats['mid']:.2f} "
                    f"mm={feats['mm']:+.2f} ofi={feats['ofi']:+.2f} "
                    f"sq={feats['sq']:+.2f} score={score:+.2f} "
                    f"spread={feats['spread']:.3f} n={feats['n']}"
                )

            cur_price = feats["mid"]

            # Adverse momentum: positive = price moved AGAINST signal direction.
            #   BUY:  ref - cur  (positive when cur < ref, i.e. price dropped)
            #   SELL: cur - ref  (positive when cur > ref, i.e. price rose)
            # Used for: hard abort on broken signal, pullback detection gate.
            adverse_pos_signed = (
                ((ref_price - cur_price) / max(atr, 1e-9)) if signal > 0
                else ((cur_price - ref_price) / max(atr, 1e-9))
            )

            # Execution slippage: positive = fill price is WORSE than ref.
            #   BUY:  cur - ref  (positive when cur > ref, i.e. price rose = chasing)
            #   SELL: ref - cur  (positive when cur < ref, i.e. price fell = underselling)
            # Used for: slip_too_wide gate (blocks chasing, allows pullback entries).
            exec_slip = (
                ((cur_price - ref_price) / max(atr, 1e-9)) if signal > 0
                else ((ref_price - cur_price) / max(atr, 1e-9))
            )

            # 1) Hard aborts -------------------------------------------------
            if score <= cfg.abort_hard and elapsed >= 2.0:
                decision.reason = f"abort: score {score:.2f} <= {cfg.abort_hard}"
                decision.score_at_decision = score
                break
            if adverse_pos_signed >= adv_thresh:
                decision.reason = (
                    f"abort: adverse momentum {adverse_pos_signed:.2f}*ATR")
                decision.score_at_decision = score
                break
            if (spread_baseline > 1e-6
                    and feats["spread"] > cfg.max_spread_ratio * spread_baseline):
                decision.reason = (
                    f"abort: spread {feats['spread']:.3f}>"
                    f"{cfg.max_spread_ratio}*{spread_baseline:.3f}")
                decision.score_at_decision = score
                break

            # 2) Slippage cap: block if we are chasing too far above ref (BUY)
            #    or selling too far below ref (SELL). Pullback entries (negative
            #    exec_slip) are explicitly allowed — they improve fill price.
            slip_too_wide = exec_slip > cfg.max_slippage_atr

            # 3) Confirm-and-go ---------------------------------------------
            if score >= cfg.trigger_now and not slip_too_wide:
                # Require small confirmation: 2 consecutive scores >= now or
                # very strong instant score.
                strong = score >= cfg.trigger_now + 0.15
                run2 = len(recent_scores) >= 2 and all(
                    s >= cfg.trigger_now for s in recent_scores[-2:])
                if strong or run2:
                    decision.fill = True
                    decision.reason = f"confirm-go score={score:.2f}"
                    decision.fill_price = cur_price
                    decision.score_at_decision = score
                    break

            # 4) Pullback-and-flip opportunity -------------------------------
            # adverse_pos_signed >= pullback_atr: price has retraced enough
            # against signal to be a meaningful dip (not chasing — momentum
            # abort would have fired if it were a full signal failure).
            # slip_too_wide is False here because exec_slip = -adverse for BUY,
            # so a price drop always gives negative exec_slip (better entry).
            if (elapsed >= cfg.pullback_min_seconds
                    and adverse_pos_signed >= cfg.pullback_atr
                    and not slip_too_wide):
                times_arr = np.array([_ts(t) for t in ticks]) if ticks is not None else np.array([])
                pb_ofi = _pullback_ofi(
                    ticks, signal, times_arr,
                    window_seconds=cfg.pullback_ofi_window_seconds,
                )
                if pb_ofi >= cfg.pullback_flip_ofi:
                    decision.fill = True
                    decision.reason = (
                        f"pullback-fill retrace={adverse_pos_signed:.2f}*ATR "
                        f"pb_ofi={pb_ofi:+.2f} (last {cfg.pullback_ofi_window_seconds:.0f}s)"
                    )
                    decision.fill_price = cur_price
                    decision.score_at_decision = score
                    break

            self.sleep_fn(period)

        # 5) Time-up resolution ---------------------------------------------
        if not decision.fill and not decision.reason:
            last_score = recent_scores[-1] if recent_scores else 0.0
            last_feats = decision.history[-1] if decision.history else None
            slip_now = 0.0
            if last_feats:
                # slip_now positive = fill price is WORSE than ref (chasing).
                #   BUY:  mid - ref  (positive = price rose = expensive entry)
                #   SELL: ref - mid  (positive = price fell = underselling)
                # Negative slip_now = better-than-ref entry (pullback) — always allowed.
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
                    f"timeout-skip score={last_score:.2f} "
                    f"slip={slip_now:+.2f}*ATR"
                )
                decision.score_at_decision = last_score

        decision.elapsed_seconds = self.clock_fn() - start
        logger.info(
            f"[hot] decision fill={decision.fill} reason={decision.reason} "
            f"elapsed={decision.elapsed_seconds:.1f}s samples={decision.samples}"
        )
        return decision
