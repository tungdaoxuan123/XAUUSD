# XAUUSD Dual-Model Training Summary

## Data Pipeline

| Stage | Input | Output | Rows |
|-------|-------|--------|------|
| Dukascopy fetch | — | `xauusd_m1_4y_dukas.csv` | 2,286,720 |
| Indicator dropna | raw | filtered | 1,553,351 |
| Events (LONG, VWAP) | `--side long` | `events_raw_long.csv` | 38,526 |
| Events (SHORT, EMA20) | `--side short` | `events_raw_short.csv` | 45,195 |
| Synmicro (LONG) | events | `events_raw_long_synmicro.csv` | 38,526 |
| Synmicro (SHORT) | events | `events_raw_short_synmicro.csv` | 45,195 |
| Triple-barrier (LONG) | synmicro | `events_long_events_long_labeled.csv` | 38,526 |
| Triple-barrier (SHORT) | synmicro | `events_short_events_short_labeled.csv` | 45,195 |

## Labeling

| Parameter | Value |
|-----------|-------|
| TP multiplier | 2.0 × ATR |
| SL multiplier | 1.0 × ATR |
| Max hold | 60 bars |
| Label scheme | 2=TP, 0=SL, 1=timeout |
| Timeouts dropped | LONG: 18, SHORT: 22 |

## Primary Signal Conditions

| Setup A (LONG) | Setup B (SHORT) |
|----------------|-----------------|
| close > VWAP | close < EMA20 |
| EMA5 > EMA20 | EMA5 < EMA20 |
| close reclaims EMA20 from below | close loses EMA20 from above |
| ATR > 0.5 × ATR50 mean | ATR > 0.5 × ATR50 mean |

## Features (14, all z-scored except velocity/BB)

| Category | Features |
|----------|----------|
| Snapshot (6) | atr_norm, bb_position, candle_body, upper_wick, lower_wick, range_vs_atr |
| Velocity (3) | pullback_speed, vwap_slope_5, volume_ratio |
| Microstructure (5) | tick_imbalance, ofi_window, cs_spread, kyle_lambda, vprof_poc_dist |
| Z-scored subset | atr_norm, kyle_lambda, vprof_poc_dist, ofi_window, tick_imbalance |
| Z-score window | LONG: 500, SHORT: 250 |
| Clip range | [-4, 4] |

## Model Config

| Parameter | Value |
|-----------|-------|
| Algorithm | LightGBM (binary) |
| GPU | OpenCL (AMD RX 6700 XT) |
| num_leaves | 31 |
| max_depth | 6 |
| min_child_samples | 30 |
| n_estimators | 500 |
| learning_rate | 0.02 |
| lambda_l1 / lambda_l2 | 0.5 |
| subsample / colsample_bytree | 0.8 |
| max_bin | 63 |
| Early stopping patience | 100 |
| Train/Val/Cal split | 70% / 20% / 10% chronological |

## Results

| Metric | LONG | SHORT |
|--------|------|-------|
| Clean events | 38,508 | 45,173 |
| Label dist | 0: 21,773 / 1: 16,735 | 0: 26,543 / 1: 18,630 |
| Win rate (clean) | 43.5% | 41.2% |
| Features surviving | 14/14 | 14/14 |
| Calibration | Isotonic | Isotonic |

### Thresholds

| Threshold | LONG | SHORT |
|-----------|------|-------|
| 0.45 | +1.15R, 71.7% WR, 1,691 trades | +0.95R, 65.1% WR, 2,302 trades |
| 0.50 | +1.15R, 71.7% WR, 1,691 trades | +0.95R, 65.1% WR, 2,302 trades |
| 0.55 | +1.31R, 76.8% WR, 1,338 trades | +1.13R, 70.9% WR, 1,593 trades |
| Optimal | 0.80, +1.73R, 90.9% WR, 474 tr | 0.77, +1.65R, 88.4% WR, 414 tr |

## Commands

### Train LONG
```
python train_pipeline/train_ensemble_gpu.py --data train_pipeline/data/events_long_events_long_labeled.csv --label-col tb_label --use-gpu --out-dir train_pipeline/models_gpu_long --side long
```

### Train SHORT
```
python train_pipeline/train_ensemble_gpu.py --data train_pipeline/data/events_short_events_short_labeled.csv --label-col tb_label --use-gpu --out-dir train_pipeline/models_gpu_short --side short --zscore-window 250
```

### Live Trading
```
python live_ensemble_trading.py --dry-run --long-model train_pipeline/models_gpu_long --short-model train_pipeline/models_gpu_short
```

## Model Artifacts

| File | Contents |
|------|----------|
| `models_gpu_long/model.txt` | LightGBM binary model |
| `models_gpu_long/calibrator_long.pkl` | Isotonic calibrator |
| `models_gpu_long/ensemble_metadata.json` | Features, thresholds, config |
| `models_gpu_short/model.txt` | LightGBM binary model |
| `models_gpu_short/calibrator_short.pkl` | Isotonic calibrator |
| `models_gpu_short/ensemble_metadata.json` | Features, thresholds, config (zscore=250) |
