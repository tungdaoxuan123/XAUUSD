# Operation Plan - XAUUSD High-Frequency Trading

---
## [2026-04-23 10:00] Task: Train an algorithm for frequent trading (5-6 trades/hour)

### Goal
Adjust the training pipeline to produce a model that trades approximately 5-6 times per hour by using M1 data, path-aware triple-barrier labeling with tight profit/stop targets, and short holding periods.

### Plan
- [x] Step 1: Research and verify the current data resolution (confirmed M1 is available).
- [x] Step 2: Configure `triple_barrier_labels.py` for high-frequency targets:
    - Set `max_hold` to 10-15 bars (10-15 minutes).
    - Set `pt_atr` and `sl_atr` to tight values (e.g., 0.5 to 1.0) to encourage faster exits.
- [x] Step 3: Generate the scalping-optimized labels using `triple_barrier_labels.py`.
- [x] Step 4: Update the `TradingEnv` or create a `ScalperTradingEnv` with matching parameters for validation.
- [x] Step 5: Train a model (PatchTST or LightGBM Ensemble) using the new frequent-trade labels.
- [x] Step 6: Validate the trade frequency in a backtest.

### Notes
- Assumptions:
  - M1 data provides enough signal for 10-minute trades.
  - XAUUSD has enough volatility at the M1 level to hit tight ATR targets frequently.
- Risks / Unknowns:
  - Transaction costs (spread + commission) might eat up small profits from high-frequency trades.
  - Tighter stops might lead to lower win rates.
- Related files / modules:
  - `train_pipeline/triple_barrier_labels.py`
  - `train_pipeline/sota_signal_generator.py`
  - `trading_env.py`
  - `train_model.py`
---
