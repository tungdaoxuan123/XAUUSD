What's Still Broken ⚠️
1. build_labels() still uses fixed-horizon return — unchanged.
This is the same issue from the last review. (close[fi] - close[i]) / close[i] > long_threshold ignores path. This only matters when someone runs without a pre-labeled CSV. If you always pass --label-col tb_label with a triple-barrier labeled file, this never executes. Recommendation: add a hard warning log when build_labels() is called so it's never used silently in production.

2. sample_weight removed from walk_forward_lgbm().
The previous version computed compute_sample_weights() per fold and passed it to lgb.Dataset(..., weight=sample_weight). The new version removed this — lgb.Dataset(X_train, label=y_train_enc) has no weight arg. The scale_pos_weight in MODEL_CONFIGS partially compensates, but per-sample uniqueness weighting from triple_barrier_labels.py is no longer used at all. For XAUUSD M1 where many consecutive bars have the same label (overlapping triple-barrier windows), this matters.

3. feature_names undefined in sklearn fallback path.
In main(), the sklearn branch calls save_sklearn_model(..., feature_names, ...) but feature_names is never defined in scope. This will crash with NameError: name 'feature_names' is not defined if LightGBM is unavailable. Should be all_feature_names or get_all_feature_names().

4. ensemble_metadata.json still missing "side" tag.
save_ensemble_metadata() writes classification: "binary" and class_names: ["WAIT", "LONG"] but has no "side": "long" field. When you build the SELL model mirror, the dispatcher loading both binaries cannot distinguish them from metadata alone.

5. structure model features depend entirely on synmicro columns.
MODEL_CONFIGS["structure"]["features"] is ["ofi_window", "tick_imbalance", "cs_spread", "kyle_lambda", "vprof_poc_dist", "amihud"] — all six are synmicro columns. If someone runs without synmicro data, all six will be 0.0 (the fallback fill), and the structure model will train on a constant feature matrix. There's a logger.warning for missing features, but no hard fail. This should at minimum log a clear ERROR if more than 50% of a role's features are zero-filled.