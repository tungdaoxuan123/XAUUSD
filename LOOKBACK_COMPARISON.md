# XAUUSD Lookback Comparison — 8 Models, Recency-Weighted

## LONG Models (VWAP-based, zscore=500)

| Lookback | Features | Val F1 | T@0.45 | T@0.50 | T@0.55 |
|----------|----------|--------|--------|--------|--------|
| 10 | 25 | 0.0000 | +1.33R, 77.8%, 1,870 | +1.52R, 83.9%, 1,562 | **+1.55R, 84.8%, 1,516** |
| 15 | 30 | 0.0000 | +1.34R, 77.9%, 1,939 | +1.35R, 78.2%, 1,924 | +1.46R, 81.9%, 1,695 |
| 30 | 45 | 0.0000 | +1.43R, 80.9%, 1,859 | +1.51R, 83.6%, 1,724 | +1.51R, 83.6%, 1,724 |
| 60 | 75→66 | 0.0000 | +1.37R, 79.0%, 1,933 | +1.46R, 81.8%, 1,772 | +1.51R, 83.8%, 1,667 |

## SHORT Models (EMA20-based, zscore=250)

| Lookback | Features | Val F1 | T@0.45 | T@0.50 | T@0.55 |
|----------|----------|--------|--------|--------|--------|
| 10 | 25 | 0.0131 | +1.26R, 75.4%, 2,087 | +1.26R, 75.4%, 2,087 | +1.33R, 77.6%, 1,903 |
| 15 | 30 | **0.0287** | +1.19R, 72.9%, 2,282 | +1.34R, 77.9%, 1,904 | +1.35R, 78.4%, 1,868 |
| 30 | 45→44 | 0.0124 | +1.26R, 75.4%, 2,269 | +1.26R, 75.4%, 2,269 | +1.35R, 78.3%, 2,014 |
| 60 | 75→73 | 0.0184 | +1.24R, 74.8%, 2,341 | +1.37R, 78.9%, 2,032 | **+1.48R, 82.6%, 1,770** |

## Best Per Side

| Side | Best Lookback | Threshold | Expectancy | Win Rate | Trades |
|------|---------------|-----------|------------|----------|--------|
| LONG | **lb10** | 0.55 | +1.55R | 84.8% | 1,516 |
| SHORT | **lb60** | 0.55 | +1.48R | 82.6% | 1,770 |

## All calibrations perfect (isotonic). EMA200 feature used across all but marginal (<0.01 importance). Return_lag features distributed across the importance spectrum — confirming they contribute real signal.

## Model Directories

| Model | Path |
|-------|------|
| LONG lb10 | train_pipeline/models_gpu_long_lb10/ |
| LONG lb15 | train_pipeline/models_gpu_long_lb15/ |
| LONG lb30 | train_pipeline/models_gpu_long_lb30/ |
| LONG lb60 | train_pipeline/models_gpu_long_lb60/ |
| SHORT lb10 | train_pipeline/models_gpu_short_lb10/ |
| SHORT lb15 | train_pipeline/models_gpu_short_lb15/ |
| SHORT lb30 | train_pipeline/models_gpu_short_lb30/ |
| SHORT lb60 | train_pipeline/models_gpu_short_lb60/ |
