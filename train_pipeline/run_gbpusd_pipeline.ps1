# Run the full SOTA signal-generation pipeline end-to-end for GBPUSD.
#
# Assumes you already have Dukascopy M1 OHLCV at:
#   train_pipeline\data\gbpusd_m1_dukascopy.csv
#
# Output artifacts:
#   train_pipeline\data\gbpusd_m1_synmicro.csv
#   train_pipeline\data\gbpusd_m1_synmicro_tb.csv
#   train_pipeline\models_sota_gbpusd\patchtst_primary.pt

$ErrorActionPreference = "Stop"

$DATA_DIR = "train_pipeline\data"
$OUT_DIR = "train_pipeline\models_sota_gbpusd"
$RAW = "$DATA_DIR\gbpusd_m1_dukascopy.csv"
$SM = "$DATA_DIR\gbpusd_m1_synmicro.csv"
$TB = "$DATA_DIR\gbpusd_m1_synmicro_tb.csv"

if (!(Test-Path $OUT_DIR)) {
    New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
}

Write-Host "[1/3] Synthetic microstructure features" -ForegroundColor Cyan
# For Forex, bin-size must be much smaller than Gold (0.0001 = 1 pip)
python train_pipeline\synthetic_microstructure.py --m1 "$RAW" --out "$SM" --vp-window 240 --bin-size 0.0001 --ofi-window 20

Write-Host "[2/3] Triple-barrier labels (LONG-ONLY, 2:1 RR)" -ForegroundColor Cyan
# ATR scales automatically regardless of symbol value
python train_pipeline\triple_barrier_labels.py --data "$SM" --out "$TB" --pt-atr 2.0 --sl-atr 1.0 --max-hold 30

Write-Host "[3/3] PatchTST-lite primary + LightGBM meta filter" -ForegroundColor Cyan
python train_pipeline\sota_signal_generator.py --data "$TB" --out-dir "$OUT_DIR" --seq-len 120 --patch-len 12 --epochs 20 --gpu

Write-Host ""
Write-Host "Done. To generate signals on new data:" -ForegroundColor Green
Write-Host "  python train_pipeline\sota_signal_generator.py --predict --data $TB --out-dir $OUT_DIR"
