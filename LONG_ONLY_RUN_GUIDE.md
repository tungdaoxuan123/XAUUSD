# XAUUSD Dual-Direction Training Guide — AMD RX 6700 XT

Train **two independent binary models** — one for long entries, one for short entries.
Both use fixed 2:1 TP/SL ratio with ATR-derived dynamic pip distances.

---

## Data Source

**Use Dukascopy** (522k+ bars) — MT5 is capped at ~100k.

```powershell
# Already downloaded: train_pipeline/data/xauusd_m1_1y_dukas.csv
# 522,720 rows, Apr 2025 – Apr 2026

# To fetch more (requires internet):
python train_pipeline/fetch_data_dukascopy.py --symbol XAUUSD --from 2015-01-01 --to 2026-04-10 --out train_pipeline/data/xauusd_m1_dukas.csv
```

---

## Full Training Commands (LightGBM GPU)

```powershell
$DATA = "train_pipeline/data/xauusd_m1_1y_dukas.csv"

# STEP 1: Synthetic microstructure features
python train_pipeline/synthetic_microstructure.py --m1 $DATA --out train_pipeline/data/xauusd_m1_synmicro.csv --vp-window 240 --bin-size 0.10 --ofi-window 20

# STEP 2: Binary labels — LONG (0=wait, 1=enter long)
python train_pipeline/triple_barrier_labels.py --data train_pipeline/data/xauusd_m1_synmicro.csv --out train_pipeline/data/xauusd_m1_tb_long.csv --side long --pt-atr 2.0 --sl-atr 1.0 --max-hold 30

# STEP 3: Binary labels — SHORT (0=wait, 1=enter short)
python train_pipeline/triple_barrier_labels.py --data train_pipeline/data/xauusd_m1_synmicro.csv --out train_pipeline/data/xauusd_m1_tb_short.csv --side short --pt-atr 2.0 --sl-atr 1.0 --max-hold 30

# STEP 4: Train LONG model
python train_pipeline/train_ensemble_gpu.py --data train_pipeline/data/xauusd_m1_tb_long.csv --label-col tb_label --expanded-features --microstructure-features --use-gpu --gpu-backend auto --out-dir train_pipeline/models_gpu_long

# STEP 5: Train SHORT model
python train_pipeline/train_ensemble_gpu.py --data train_pipeline/data/xauusd_m1_tb_short.csv --label-col tb_label --expanded-features --microstructure-features --use-gpu --gpu-backend auto --out-dir train_pipeline/models_gpu_short
```

---

## Full Training Commands (PatchTST SOTA)

```powershell
# Steps 1-3 same as above, then:

# Train PatchTST LONG
python train_pipeline/sota_signal_generator.py --data train_pipeline/data/xauusd_m1_tb_long.csv --out-dir train_pipeline/models_sota_long --seq-len 120 --patch-len 12 --epochs 20 --gpu

# Train PatchTST SHORT
python train_pipeline/sota_signal_generator.py --data train_pipeline/data/xauusd_m1_tb_short.csv --out-dir train_pipeline/models_sota_short --seq-len 120 --patch-len 12 --epochs 20 --gpu
```

---

## Running Both Bots Simultaneously

```powershell
# Terminal 1 — LONG bot
python live_ensemble_trading.py --model train_pipeline/models_gpu_long --side long --dry-run

# Terminal 2 — SHORT bot
python live_ensemble_trading.py --model train_pipeline/models_gpu_short --side short --dry-run
```

Remove `--dry-run` when ready for real orders.

---

## Running with SOTA models

```powershell
python live_sota_trading.py --model train_pipeline/models_sota_long/patchtst_primary.pt --config train_pipeline/models_sota_long/sota_config.json --side long --dry-run
python live_sota_trading.py --model train_pipeline/models_sota_short/patchtst_primary.pt --config train_pipeline/models_sota_short/sota_config.json --side short --dry-run
```

---

## GBPUSD (same commands, different bin-size)

```powershell
# Fetch from Dukascopy
python train_pipeline/fetch_data_dukascopy.py --symbol GBPUSD --from 2020-01-01 --to 2026-04-10 --out train_pipeline/data/gbpusd_m1_dukas.csv

# Synthetic micro (bin-size 0.0001 for forex pips)
python train_pipeline/synthetic_microstructure.py --m1 train_pipeline/data/gbpusd_m1_dukas.csv --out train_pipeline/data/gbpusd_m1_synmicro.csv --vp-window 240 --bin-size 0.0001 --ofi-window 20

# Labels and training — same as XAUUSD, just change paths
```

---

## Key Notes

- Both models use **identical binary architecture** — only the training labels differ
- Labels have the same structure (`0=wait, 1=enter`), the `--side` flag controls which direction the barriers are placed
- Each bot opens at most 1 position at a time (FTMO rule)
- `--dry-run` logs everything without placing real orders
- Use `--use-gpu` for AMD OpenCL acceleration; omit for CPU-only
