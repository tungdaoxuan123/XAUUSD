# Scalper mode — many small wins, VN evening session

This adds a **high-frequency scalping profile** on top of the calibrated
SOTA model. Designed for running 20:00–24:00 Vietnam time, which is
13:00–17:00 UTC — the London-NY overlap, the best XAUUSD window of the
day.

## What's different from the swing profile

| Mode | `dynamic_exits.ExitPlanner` (v2 swing) | `scalper_exits.ScalperPlanner` (new) |
|---|---|---|
| RR | 1.8 – 3.5 | 0.5 (partial) + 1.2 (final) |
| Hold | up to 30 bars | 5–12 bars max |
| Exits | single TP | **TP1 partial 50% + SL→BE, TP2 runner** |
| Session | any | hard 13:00–17:00 UTC, flat by 16:50 |
| Daily | fixed risk | **+3R goal / -2R stop / max 20 trades** |
| Cooldown | none | 3 bars after any close |
| Min prob | 0.55 | 0.52 (scalper needs volume) |
| Sizing | half-Kelly | flat 0.3%/trade (consistency > edge-sizing for HF) |

## Why these choices for small-wins frequency

1. **TP1 partial at +0.5R** is the core trick. You close 50% at 0.5R,
   move SL to breakeven on the remaining 50%, and let the runner go to
   1.2R or trail out. Math:
   - Win on TP1 only → +0.25R (banked) + 0 (BE stop) = +0.25R
   - Win TP1 then TP2 → +0.25R + 0.6R = +0.85R
   - Loss before TP1 → −1R
   - At a realistic 55% TP1-hit rate, expected value is positive and
     variance is *much* lower than a single 1.2R target. This is the
     "many small wins" feel quant scalpers chase.

2. **Hard session window** (13–17 UTC). Your local 20:00–24:00 VN is
   exactly when XAU has its best liquidity, tightest spreads, and
   highest directional follow-through. No reason to let the bot run
   outside it.

3. **Daily goal +3R / stop -2R**. Prop-trader psychology encoded: once
   the day is won, *stop*. Once it's gone, *stop*. This is what keeps
   FTMO accounts alive through bad streaks and prevents give-back on
   good days.

4. **Cooldown after close**. Prevents stop-hunt back-to-back entries
   where the same regime chops you twice.

## Usage

```bash
# Dry run (no orders sent) — always run this first
python live_scalper_trading.py --dry-run --min-prob 0.52

# Live (actually sends orders). Uses defaults from ScalperConfig.
python live_scalper_trading.py

# Tuning
python live_scalper_trading.py \
    --min-prob 0.54 \
    --rr-partial 0.5 --rr-final 1.3 \
    --max-hold 10 --cooldown 4 \
    --goal-r 2.5 --stop-r 1.5
```

## Session time math (VN = UTC+7)

| VN time | UTC | What's happening |
|---|---|---|
| 20:00 | 13:00 | **London-NY overlap starts** — XAU 30-70¢ M1 ranges |
| 21:30 | 14:30 | Peak overlap volume |
| 23:00 | 16:00 | London closes, NY still active |
| 23:50 | 16:50 | **Flat by this time** (no new entries, close open) |
| 24:00 | 17:00 | Bot goes idle until next day |

## Expected trade cadence

With `min_prob=0.52`, `cooldown=3`, `max_hold=12` on a typical overlap
session you'll see 8–15 trades per evening. Caps at 20/day.

## If you want the runner to mean-revert to TP1 only

Set `--rr-final` equal to `--rr-partial`:
```bash
python live_scalper_trading.py --rr-partial 0.6 --rr-final 0.6
```
This fully closes at a single tight TP, maximum frequency, pure scalping.

## What to check on the first live session

- Log shows `session 13-17 UTC (=20-24 VN)` at startup.
- `R_today=+0.00 trades=0` counter increments as trades fire.
- Each plan logs `tp1=... tp2=...` — confirm both are set.
- At TP1 the log shows `TP1 hit -> partial close X lots, SL->BE`.
- At 16:50+ UTC the log shows `near session close` and stops scanning.
