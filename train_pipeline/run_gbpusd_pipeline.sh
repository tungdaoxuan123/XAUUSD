#!/usr/bin/env bash
# Run the full SOTA signal-generation pipeline end-to-end for GBPUSD.
#
# Assumes you already have Dukascopy M1 OHLCV at:
#   train_pipeline/data/gbpusd_m1_dukascopy.csv
#
# Output artifacts:
#   train_pipeline/data/gbpusd_m1_synmicro.csv       # OHLCV + synthetic microstructure
#   train_pipeline/data/gbpusd_m1_synmicro_tb.csv    # + triple-barrier labels
#   train_pipeline/models_sota_gbpusd/patchtst_primary.pt   # primary model
#   train_pipeline/models_sota_gbpusd/meta_filter.txt       # meta filter
#   train_pipeline/models_sota_gbpusd/sota_config.json
set -euo pipefail

DATA_DIR=${DATA_DIR:-train_pipeline/data}
OUT_DIR=${OUT_DIR:-train_pipeline/models_sota_gbpusd}
RAW=${RAW:-$DATA_DIR/gbpusd_m1_dukascopy.csv}
SM=$DATA_DIR/gbpusd_m1_synmicro.csv
TB=$DATA_DIR/gbpusd_m1_synmicro_tb.csv

mkdir -p "$OUT_DIR"

echo "[1/3] Synthetic microstructure features"
# For Forex, bin-size must be much smaller than Gold (0.0001 = 1 pip)
python train_pipeline/synthetic_microstructure.py \
    --m1 "$RAW" --out "$SM" --vp-window 240 --bin-size 0.0001 --ofi-window 20

echo "[2/3] Triple-barrier labels"
# ATR scales automatically regardless of symbol value
python train_pipeline/triple_barrier_labels.py \
    --data "$SM" --out "$TB" \
    --pt-atr 1.5 --sl-atr 1.0 --max-hold 30

echo "[3/3] PatchTST-lite primary + LightGBM meta filter"
python train_pipeline/sota_signal_generator.py \
    --data "$TB" --out-dir "$OUT_DIR" \
    --seq-len 120 --patch-len 12 --epochs 20 --gpu

echo
echo "Done. To generate signals on new data:"
echo "  python train_pipeline/sota_signal_generator.py --predict --data $TB --out-dir $OUT_DIR"
