All 5 concerns are valid. Direct resolutions:

***

**1. `bars_since_ema_cross` — already dropped, non-issue**

This feature was removed two exchanges ago and replaced with `ema_gap` and `ema_gap_delta`. The O(n²) problem does not exist in the current spec. No action needed.

***

**2. SHAP — already dropped, non-issue**

Also removed. The spec now uses `feature_importance(importance_type='gain')` natively from LightGBM. No SHAP import, no CPU overhead. No action needed.

***

**3. Self-healing loop — add diagnostic logging before abort**

Add this before the abort in `triple_barrier_labels.py`:

```python
# Log WHY events are timing out before aborting
timeout_rows = labeled_df[labeled_df['tb_label'] == 1]
logger.info(f"Timeout diagnostic ({len(timeout_rows)} events):")
logger.info(f"  Median return at timeout bar: {timeout_rows['timeout_return'].median():.5f}")
logger.info(f"  Median ATR at event bar:      {timeout_rows['atr_at_entry'].mean():.5f}")
logger.info(f"  Median bars held:             {timeout_rows['bars_held'].median():.0f}")
logger.info(f"  Hint: if median return ≈ 0 → market is ranging, tighten ATR filter")
logger.info(f"  Hint: if median ATR is high → pt_atr/sl_atr may be too wide for this regime")
logger.info("ABORT: clean_pct never reached 30% — fix primary setup before training")
sys.exit(1)
```

Requires adding `timeout_return` (close at timeout bar minus close at entry, normalized by ATR), `atr_at_entry`, and `bars_held` columns during the labeling loop — all trivially available at label time.

***

**4. Simultaneous signal — skip the bar**

Add this check in `primary_signal_generator.py` immediately after computing both setup conditions:

```python
both_fire = setup_a_condition & setup_b_condition
if both_fire.any():
    logger.info(f"Skipping {both_fire.sum()} bars where both Setup A and B fired simultaneously (ambiguous)")

# Apply mutual exclusion
setup_a_condition = setup_a_condition & ~both_fire
setup_b_condition = setup_b_condition & ~both_fire
```

This is a one-liner vectorized mask. No loop needed.

***

**5. Velocity feature lookback — the ATR rolling(50) already solves this**

The primary signal generator already requires `ATR(14).rolling(50).mean()` to fire any event. The first event cannot occur before bar 50 (at minimum). Since the longest velocity feature needs bar `i-10`, and `i >= 50` always, all velocity lookbacks are guaranteed to have valid data. The `MIN_CONTEXT_BARS = 15` guard in `build_features()` is a redundant safety net on top of this — keep it, but it will never actually trigger in practice given the ATR rolling(50) requirement upstream. No additional action needed beyond what is already in the spec.