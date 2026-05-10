Corrected Agent Brief — SHORT Exhaustion Strategy
The bear regime filter approach is aborted before running. Reason: XAUUSD 2022–2026 is one of the strongest gold bull markets in history. A true H1 EMA200 bear filter would leave under 2,000 events — not enough to train LightGBM. We pivot directly to Exhaustion SHORT (blow-off top reversals).

Step 1 — New SHORT Signal in primary_signal_generator.py
Replace the EMA20 crossdown signal entirely:

python
# Exhaustion SHORT — blow-off top reversal
vwap = compute_vwap(close, volume)
atr50_mean = atr.rolling(50).mean()

# 5 consecutive green bars (momentum climax)
consecutive_green = (
    (close > close.shift(1)) &
    (close.shift(1) > close.shift(2)) &
    (close.shift(2) > close.shift(3)) &
    (close.shift(3) > close.shift(4)) &
    (close.shift(4) > close.shift(5))
)

short_signal = (
    (close > vwap + 2.5 * atr) &      # extended above VWAP
    consecutive_green &                 # 5 green bars = climax
    (volume_ratio > 2.0) &             # volume spike
    (atr > 0.8 * atr50_mean)          # volatility active
)
Step 2 — Regenerate SHORT Pipeline
text
events_raw_short → synmicro_short → triple_barrier_short
Same parameters: TP=2.0×ATR, SL=1.0×ATR, max hold=60 bars.

CRITICAL checkpoint: Log event count. Target: 5,000–20,000 events. If below 3,000, report back immediately — signal fires too rarely and we need to relax one condition. If above 30,000, the signal is too loose and we tighten volume_ratio to 2.5.

Step 3 — Train and Walk-Forward
text
python train_pipeline/train_ensemble_gpu.py \
  --data train_pipeline/data/events_short_events_short_labeled.csv \
  --label-col tb_label --use-gpu \
  --out-dir train_pipeline/models_gpu_short_exhaustion \
  --side short --zscore-window 250 --lookback 15 --recency-weight

python train_pipeline/walk_forward_backtest.py \
  --data train_pipeline/data/events_short_events_short_labeled.csv \
  --model-dir train_pipeline/models_gpu_short_exhaustion \
  --side short --lookback 15 --threshold 0.55 \
  --out walk_forward_short_exhaustion.csv
Step 4 — Report Back Only
Event count after new signal

Label dist {0: X, 1: Y} — target win rate > 48%

Walk-forward win rate across all months

Walk-forward R/trade

Success condition: walk-forward win rate ≥ 52% and R/trade ≥ +0.40R. Anything below that and we try tightening volume_ratio to 2.5 before exploring other approaches.

