Replace the LONG primary signal in primary_signal_generator.py:

python
# REMOVE THIS (mean-reversion logic — broken in uptrend)
# close > VWAP
# close reclaims EMA20 from below

# REPLACE WITH (momentum continuation — works in uptrend)
ema20 = close.ewm(span=20).mean()
ema50 = close.ewm(span=50).mean()
ema200 = close.ewm(span=200).mean()

long_signal = (
    (close > ema20) &              # price above fast MA
    (ema20 > ema50) &              # fast MA above slow MA
    (ema50 > ema200) &             # trend alignment (all bullish)
    (atr > 0.5 * atr50_mean) &    # volatility filter (keep)
    (ema20_slope > 0)              # momentum not stalling
)
Then regenerate the full LONG pipeline:

text
events_raw_long → synmicro_long → labeled_long → retrain lb10 only
Report back only the confusion matrix and F1. If LONG confusion matrix has non-zero column 1, then run all 4 lookbacks. Do not run 8 models again until the signal itself produces a non-zero F1.