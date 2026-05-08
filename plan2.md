Rewrite train_ensemble_gpu.py with these changes:

Remove all return_lag_N features from MODEL_CONFIGS and build_features(). Replace with: rsi_14, macd_hist, atr_norm, ema_ratio, close_minus_vwap_norm (=(close-VWAP)/ATR), bb_position (=(close-lower_band)/bb_width), vol_zscore, candle_body (=(close-open)/ATR), upper_wick (=(high-max(open,close))/ATR), lower_wick (=(min(open,close)-low)/ATR), range_vs_atr (=(high-low)/ATR). All features must already exist or be computed in add_technical_indicators() — add the missing ones there.

In build_labels(), add a loud warning if --label-col is not found in the CSV: print a red error and call sys.exit(1). Force the user to always provide triple-barrier labels. Remove the fallback naive labeler entirely.

In walk_forward_lgbm(), after the fold loop ends, add a final retrain step on 100% of X_role + y_enc using the same params and num_boost_round = int(mean of best_iteration across all folds). Return this full-data model, not last_model.

Add a CLI flag --side with values long (default) or short. When --side short, flip the label interpretation: in the triple-barrier CSV, treat tb_label == -1 as 1 (ENTER SHORT) and tb_label == 1 as 0 (WAIT). Save to --out-dir as given. This enables running two training commands for two independent binary models.