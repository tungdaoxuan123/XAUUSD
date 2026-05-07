Step 1 — Fix train_ensemble_gpu.py (All 3 Problems Live Here)
Tell your coding agent to make these changes in this order:

1a. Remove current_position and current_balance from features
These are live-state variables. Delete them from build_feature_matrix() entirely.

1b. Replace close_lag_* with normalized returns

python
# Delete this pattern:
close_lag_{n} = close.shift(n)

# Replace with:
return_lag_{n} = (close.shift(n) - close.shift(n+1)) / atr.shift(n)
1c. Wire in the synmicro columns
In build_feature_matrix(), explicitly append these columns from the loaded dataframe if they exist:

python
MICRO_COLS = [
    'tick_imbalance', 'ofi_window', 'cs_spread',
    'kyle_lambda', 'amihud', 'vprof_poc_dist',
    'jump_flag', 'regime_flag'
]
Fill NaNs with 0 before appending (already partially null per the synmicro log).

1d. Split models into 3 genuinely different configs

python
MODEL_CONFIGS = {
    "trend": {
        "features": ["return_lag_*", "RSI", "MACD", "atr_norm", "ema_ratio"],
        "num_leaves": 63, "max_depth": 6, "min_child_samples": 300,
        "scale_pos_weight": 1.3, "lambda_l1": 0.1, "lambda_l2": 0.1
    },
    "structure": {
        "features": ["ofi_window", "tick_imbalance", "cs_spread",
                     "kyle_lambda", "vprof_poc_dist"],
        "num_leaves": 127, "max_depth": 8, "min_child_samples": 100,
        "scale_pos_weight": 1.3, "lambda_l1": 0.05, "lambda_l2": 0.05
    },
    "regime": {
        "features": ["atr_pct", "amihud", "jump_flag",
                     "regime_flag", "vol_zscore"],
        "num_leaves": 31, "max_depth": 4, "min_child_samples": 500,
        "scale_pos_weight": 1.2, "lambda_l1": 0.2, "lambda_l2": 0.2
    }
}
Step 2 — Re-run Training (Same Commands, No Changes Needed)
Once the code is fixed, re-run exactly the same pipeline commands you already ran. No need to re-run synthetic_microstructure.py or triple_barrier_labels.py — those outputs are already correct and saved.

powershell
python train_pipeline/train_ensemble_gpu.py `
    --data train_pipeline/data/xauusd_m1_tb_long.csv `
    --label-col tb_label --expanded-features --microstructure-features `
    --use-gpu --out-dir train_pipeline/models_gpu_long

python train_pipeline/train_ensemble_gpu.py `
    --data train_pipeline/data/xauusd_m1_tb_short.csv `
    --label-col tb_label --expanded-features --microstructure-features `
    --use-gpu --out-dir train_pipeline/models_gpu_short
Step 3 — Validate Before Proceeding
After the retrain, check these 3 things in the logs before doing anything else:

Check	Pass condition	Fail = do this
Three different F1 scores	trend F1 ≠ structure F1 ≠ regime F1	1d not applied correctly
WAIT recall > 0.45	Not stuck predicting always-LONG	Lower scale_pos_weight further to 1.1
Micro features in feature list	tick_imbalance, ofi_window visible in feature log	1c not wired in
Only move to PatchTST / sota_signal_generator.py training after all 3 checks pass.