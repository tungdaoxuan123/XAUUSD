# LONG-ONLY XAUUSD/GBPUSD Trading System — Run Guide

## Architecture

4-layer stack, all long-only, fixed 2:1 TP/SL ratio with ATR-derived dynamic pip distances.

```
RAW DATA  ->  SYNTHETIC MICRO  ->  TRIPLE-BARRIER LABELS  ->  MODEL TRAINING  ->  LIVE EXECUTION
(MT5/Dukas)   (OFI/spread/VPOC)    (binary 0=wait 1=long)     (LGBM or PatchTST)   (fixed 2:1 TP/SL)
```

---

## Prerequisites

```
pip install -r requirements.txt
```

Create `.env`:
```
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=FTMO-Demo
SYMBOL=XAUUSD.sim
ENSEMBLE_MODEL_PATH=train_pipeline/models_gpu
```

---

## PATH A — LightGBM Ensemble (fast, GPU via OpenCL, AMD RX 6700 XT)

### Step 1: Fetch data
```powershell
python train_pipeline/fetch_data_mt5.py --symbol XAUUSD --timeframe M1 --bars 500000 --out train_pipeline/data/xauusd_m1.csv
```

### Step 2: Synthetic microstructure features
```powershell
python train_pipeline/synthetic_microstructure.py --m1 train_pipeline/data/xauusd_m1.csv --out train_pipeline/data/xauusd_m1_synmicro.csv --vp-window 240 --bin-size 0.10 --ofi-window 20
```

### Step 3: Triple-barrier labels (long-only, 2:1 RR)
```powershell
python train_pipeline/triple_barrier_labels.py --data train_pipeline/data/xauusd_m1_synmicro.csv --out train_pipeline/data/xauusd_m1_tb.csv --pt-atr 2.0 --sl-atr 1.0 --max-hold 30
```

### Step 4: Train binary ensemble
```powershell
python train_pipeline/train_ensemble_gpu.py --data train_pipeline/data/xauusd_m1_tb.csv --label-col tb_label --expanded-features --microstructure-features --use-gpu --gpu-backend auto --out-dir train_pipeline/models_gpu
```

### Step 5: Live trading
```powershell
python live_ensemble_trading.py --dry-run
python live_ensemble_trading.py                    # real orders
```

---

## PATH B — PatchTST SOTA (transformer, higher ceiling)

### One-shot pipeline (XAUUSD):
```powershell
python train_pipeline/synthetic_microstructure.py --m1 train_pipeline/data/xauusd_m1.csv --out train_pipeline/data/xauusd_m1_synmicro.csv --vp-window 240 --bin-size 0.10 --ofi-window 20

python train_pipeline/triple_barrier_labels.py --data train_pipeline/data/xauusd_m1_synmicro.csv --out train_pipeline/data/xauusd_m1_synmicro_tb.csv --pt-atr 2.0 --sl-atr 1.0 --max-hold 30

python train_pipeline/sota_signal_generator.py --data train_pipeline/data/xauusd_m1_synmicro_tb.csv --out-dir train_pipeline/models_sota --seq-len 120 --patch-len 12 --epochs 20 --gpu
```

Or use the shell script:
```bash
bash train_pipeline/run_sota_pipeline.sh
```

### One-shot pipeline (GBPUSD):
```bash
bash train_pipeline/run_gbpusd_pipeline.sh
```
```powershell
.\train_pipeline\run_gbpusd_pipeline.ps1
```

### Live SOTA trading:
```powershell
python live_sota_trading.py --dry-run --min-prob 0.55
python live_sota_trading.py --min-prob 0.55          # real orders
```

---

## PATH C — Scalper (London-NY overlap, high-frequency)

```powershell
python live_scalper_trading.py --dry-run --min-prob 0.52
python live_scalper_trading.py --min-prob 0.52       # real orders

# 24h mode:
python live_scalper_trading.py --session-start-utc 0 --session-end-utc 24 --min-prob 0.52
```

---

## Key CLI Parameters

| Flag | Default | Purpose |
|------|---------|---------|
| `--dry-run` | off | Log decisions without placing orders |
| `--min-prob` | 0.55 (SOTA) / 0.52 (scalper) | Minimum calibrated long probability |
| `--pt-atr` | 2.0 | Take profit multiplier on ATR |
| `--sl-atr` | 1.0 | Stop loss multiplier on ATR |
| `--max-hold` | 30 | Max bars before time-based exit |
| `--use-gpu` | off | Enable GPU training (OpenCL) |
| `--interval` | 10 | Seconds between live scans |

---

## What Changed (vs old bidirectional system)

| Component | Old | New |
|-----------|-----|-----|
| Labels | -1=SELL, 0=HOLD, +1=BUY | 0=WAIT, 1=LONG |
| Model output | 3-class probabilities | Binary p_long |
| Action space | Continuous [-1, 1] | Discrete {0, 1} |
| TP/SL ratio | Variable (1.8-3.5) | Fixed 2:1 |
| Exit logic | Trailing + partials + breakeven | Fixed levels at entry |
| Short trades | Allowed | Never opened |

---

## GPU Notes (AMD RX 6700 XT)

- LightGBM uses OpenCL — works out of the box on Windows with `--use-gpu`
- PatchTST falls back through CUDA -> MPS -> DirectML -> CPU
- For transformer training, prefer larger batch sizes and `float32`
- On Linux/WSL with ROCm, the transformer path becomes much faster
