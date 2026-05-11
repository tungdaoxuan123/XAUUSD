 Pre-warm z-score buffers on startup (10 min)
python
# Before entering the main loop, fetch historical data and warm the buffers:
warmup_rates = self.interface.get_rates(count=600)
if warmup_rates is not None:
    warmup_df = pd.DataFrame(warmup_rates)
    warmup_df = self._compute_indicators(warmup_df)
    for i in range(len(warmup_df)):
        feat = self._compute_live_features(warmup_df.iloc[:i+1])
        if feat:
            for zk in ["atr_norm", "kyle_lambda", "vprof_poc_dist", "ofi_window", "tick_imbalance"]:
                self._zscore(zk, feat.get(zk, 0.0))
logger.info("Z-score buffers warmed up.")