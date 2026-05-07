Root Cause 1: Microstructure Features Are Not Entering the Model
Your log shows synthetic_microstructure.py computed 5 feature groups:

text
tick_imbalance         0 nulls
ofi_window             0 nulls
spread_mean            1 null
vprof_poc_dist    627,222 nulls
kyle_lambda       682,774 nulls
But the training step only sees 37 features: ['close_lag_9', ..., 'RSI', 'MACD', 'Signal_Line', 'current_position', 'current_balance'].

The microstructure features are in the CSV but never make it into the feature matrix. build_features() in train_ensemble_gpu.py hardcodes its own feature set and ignores the microstructure columns from the input CSV. The model is trained purely on lagged closes + RSI + MACD + position state — features that have almost zero predictive power for 1-minute directional moves.

Root Cause 2: Naïve Class Weighting Destroys Calibration
The log shows class_weight = {1: ~1.5, 0: ~0.75} — this is raw inverse-frequency weighting. On imbalanced binary data, this forces the model to maximize recall on the minority class (LONG) at the expense of everything else. The result is a model that fires on every bar just to avoid missing any true positives.

Root Cause 3: The Labels Themselves Are Noisy
With 2:1 TP/SL on 1-minute bars over 4 years, ~22.6% of bars are labeled as LONG. But many of these are not real "setups" — they're just bars where price happened to wobble up 2 ATR before wobbling down 1 ATR due to random walk. The signal-to-noise ratio in the labels is extremely low, which amplifies every other problem.

The Fixes (Ordered by Impact)
Fix 1: Inject Microstructure Features into build_features()
In train_ensemble_gpu.py, modify the feature matrix builder to read all columns from the input CSV except the reserved label/timestamp columns. The current build_features() creates its own 37 columns and ignores everything else. You need to append the microstructure features.

The columns with massive nulls (vprof_poc_dist, kyle_lambda) should be dropped or forward-filled. Keep tick_imbalance, ofi_window, spread_mean, cs_spread, ar_spread, tick_rule, volume_imbalance, jump_flag, regime_flag — these have near-complete coverage and carry genuine predictive signal.

Fix 2: Replace Inverse-Frequency with scale_pos_weight
Change from:

python
class_weight = {1: n_neg/n_pos, 0: 1.0}  # current
To:

python
scale_pos_weight = np.sqrt(n_neg / n_pos)  # ~1.87 for your data
This is a standard heuristic in imbalanced binary classification. It gives the minority class a boost without completely destroying calibration. LightGBM uses scale_pos_weight directly in its objective function, which handles it more gracefully than per-sample weights.

Fix 3: Stronger LightGBM Regularization
Your current params (not fully visible in log but implied by the overfitting) need tightening:

max_depth=4 (down from default 6)

num_leaves=16 (down from 31)

min_child_samples=200 (up from default 20)

feature_fraction=0.7 (enable column subsampling)

bagging_fraction=0.8 (enable row subsampling)

bagging_freq=5

reg_alpha=0.1, reg_lambda=1.0

With ~1.5M rows, the model has enough data to support aggressive regularization. The goal is to force the model to find genuinely robust patterns rather than memorizing noise.

Fix 4: Change Evaluation Metric from F1 to Average Precision
F1 is unstable on imbalanced data because it depends on a fixed threshold (default 0.5). Average Precision (AP) measures the area under the precision-recall curve and is the standard metric for rare-event detection. LightGBM supports metric="average_precision".

Fix 5: Add Feature Selection / Importance Pruning
After training, drop features with zero or near-zero importance. Many of your 37 features (especially current_position, current_balance which are always 0 in the training data) are pure noise. LightGBM's built-in importance + a feature_fraction of 0.7 will handle this automatically.

Fix 6: Label Quality — Consider Direction-Conditioned Filtering
After Fix 1–5, if performance is still poor, the issue is the labels themselves. Options:

Meta-labeling: Train a primary model to detect "trend direction" (e.g., price > EMA), then train the binary barrier model only on bars where the primary agrees. This is what sota_signal_generator.py does.

Confidence filtering: Only keep labels where the barrier was hit cleanly (not timeout) and within a reasonable number of bars.

Higher ATR multiplier: Try pt_atr=3.0, sl_atr=1.5 (still 2:1 but wider absolute barriers) to reduce noise from micro-wiggles.