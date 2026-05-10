Step 1 — Filter training data to last 12 months only (do this first, ~10 min)

In train_pipeline/train_ensemble_gpu.py, after loading the CSV and dropping timeouts, add:

python
if args.train_months:
    cutoff = df['timestamp'].max() - pd.DateOffset(months=args.train_months)
    df = df[df['timestamp'] >= cutoff]
    logger.info(f"Filtered to last {args.train_months} months: {len(df)} rows")
Add --train-months 12 as a CLI argument. Then retrain both models:

text
python train_pipeline/train_ensemble_gpu.py --data train_pipeline/data/events_long_events_long_labeled.csv --label-col tb_label --use-gpu --out-dir train_pipeline/models_gpu_long --side long --recency-weight --train-months 12

python train_pipeline/train_ensemble_gpu.py --data train_pipeline/data/events_short_events_short_labeled.csv --label-col tb_label --use-gpu --out-dir train_pipeline/models_gpu_short --side short --zscore-window 250 --recency-weight --train-months 12
Step 2 — Add EMA200 regime feature (do alongside Step 1, same commit)

In the feature engineering section, add one column before z-scoring:

python
ema200 = close.ewm(span=200, adjust=False).mean()
df['regime'] = (df['close'] > ema200).astype(int)
Add regime to the features list. Do NOT z-score it — it's already binary 0/1.

Step 3 — Report back these specific numbers

After retraining, post:

New row count after 12-month filter (expect ~9,000–10,000 LONG, ~11,000 SHORT)

Confusion matrix for both models

Val F1 for both models

Threshold table at 0.50 and 0.55

Success condition: confusion matrix must have non-zero values in column 1 on both models before proceeding to walk-forward backtest.