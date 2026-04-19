# XAUUSD 3-Model GPU Ensemble - Full Usage Guide

## Architecture Overview

```
fetch_data_mt5.py        <- Bulk MT5 data fetcher (batched, up to 2M bars)
train_ensemble_gpu.py    <- GPU-accelerated LightGBM training (AMD OpenCL)
ensemble_gpu.py          <- Live inference wrapper (soft voting)
```

---

## Step 1 — Install GPU Dependencies

```powershell
# Install LightGBM with GPU (OpenCL) support
pip install lightgbm

# Verify AMD GPU OpenCL is available
# Your AMD Adrenalin driver installs OpenCL automatically.
# Check: C:\Windows\System32\OpenCL.dll should exist.
```

> **AMD GPU Note**: LightGBM uses OpenCL for GPU acceleration.
> Your **RX 6700 XT** is supported. No ROCm install required.
> AMD Adrenalin Software (any recent version) provides the OpenCL runtime.

---

## Step 2 — Fetch Bulk Data (up to 2M bars)

```powershell
python train_pipeline/fetch_data_mt5.py ^
    --symbol XAUUSD ^
    --timeframe M1 ^
    --total-bars 2000000 ^
    --batch-size 100000 ^
    --out train_pipeline/data/xauusd_m1_2m.csv
```

**Important**: FTMO Demo typically stores only ~100,000 M1 bars (~70 days).
The script will stop gracefully when the broker's history limit is reached and save whatever it collected.

---

## Step 3A — GPU Training (AMD RX 6700 XT via OpenCL)

```powershell
python train_pipeline/train_ensemble_gpu.py ^
    --data "train_pipeline/data/xauusd_m1_2m.csv" ^
    --horizon 5 ^
    --buy-threshold 0.0005 ^
    --sell-threshold 0.0005 ^
    --expanded-features ^
    --use-gpu ^
    --gpu-backend auto ^
    --out-dir train_pipeline/models_gpu
```

The script will:
1. Auto-probe whether LightGBM GPU (OpenCL) is working.
2. Train 3 models: `trend`, `structure`, `regime`.
3. Fall back to CPU automatically if GPU probe fails.
4. Save 9 artifact files + `ensemble_metadata.json`.

---

## Step 3B — CPU Fallback Training

```powershell
python train_pipeline/train_ensemble_gpu.py ^
    --data "train_pipeline/data/xauusd_m1_2m.csv" ^
    --horizon 5 ^
    --buy-threshold 0.0005 ^
    --sell-threshold 0.0005 ^
    --out-dir train_pipeline/models_gpu
```

*(No `--use-gpu` flag = always uses CPU LightGBM.)*

---

## Step 4 — Live Bot Integration

In `live_ensemble_trading.py`:

```python
from train_pipeline.ensemble_gpu import EnsembleGPU, build_obs_from_rates

# In __init__:
self.gpu_signal = EnsembleGPU.load("train_pipeline/models_gpu")

# In run_live_trading loop (confluence confirmation):
obs_df = build_obs_from_rates(rates, expanded=False)
gp_action, gp_confidence = self.gpu_signal.predict(obs_df)

# Combine with RL ensemble signal:
if abs(action) >= 0.5 and gp_action != 0 and np.sign(action) == np.sign(gp_action):
    # RL + LightGBM ensemble agree — high-confidence entry
    ...
```

---

## Output Artifacts

After training, `train_pipeline/models_gpu/` contains:

| File | Purpose |
| :--- | :--- |
| `lgbm_trend_default.txt` | Trend model (LightGBM booster) |
| `lgbm_structure_default.txt` | Structure model |
| `lgbm_regime_default.txt` | Regime model |
| `features_trend_default.json` | Feature names + label map |
| `summary_trend_default.json` | Fold metrics |
| `ensemble_metadata.json` | Backend, GPU flag, all model paths |

---

## LightGBM GPU Architecture (3 Models)

| Model | Role | Key Hyperparams |
| :--- | :--- | :--- |
| **Trend** | Directional bias (linear) | Low leaves, high regularization |
| **Structure** | Nonlinear indicator patterns | Deep trees, feature sampling |
| **Regime** | Volatility/regime detection | Many trees, ATR/BB features |

---

## Troubleshooting

| Issue | Fix |
| :--- | :--- |
| `GPU probe failed` | LightGBM falls back to CPU automatically |
| `OpenCL.dll not found` | Reinstall AMD Adrenalin Software |
| FTMO only returns 100k bars | Increase "Max bars in chart" in MT5 Options > Charts |
| Low Macro F1 | Lower `--buy-threshold` to 0.0003 or increase `--horizon` to 10 |
