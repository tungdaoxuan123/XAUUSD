Bug 1 — Lookback not wired into vectorized feature builder
In build_features(), the shift loop must use the lookback parameter:

python
# Wrong — hardcoded range:
for i in range(10):
    X[f"return_lag_{i}"] = ...

# Correct — use the passed lookback:
for i in range(lookback):
    X[f"return_lag_{i}"] = (
        (df["close"].shift(i) - df["close"].shift(i + 1)) / df["ATR"].shift(i)
    ).clip(-10, 10)
After this fix, with --lookback 60 you should see return_lag_0 through return_lag_59 — 60 features just from price lags, plus the 16 indicator/micro features = ~76 total.

Bug 2 — scale_pos_weight is now too low (or zero)
The correct value for a 2:1 class imbalance (1M WAIT vs 512K LONG) is not 1.0 and not 2.0. It is 1.3 — a gentle partial rebalance. Set this explicitly in get_lgbm_params() for all three roles and make absolutely sure nothing overrides it afterward:

python
# In get_lgbm_params(), add to base dict:
"scale_pos_weight": 1.3,

# In walk_forward_lgbm() — DELETE these lines entirely:
n_neg = (y_train_enc == 0).sum()
n_pos = max((y_train_enc == 1).sum(), 1)
params["scale_pos_weight"] = n_neg / n_pos   # ← DELETE

# In main() refit block — DELETE these lines entirely:
n_neg_full = (y_enc == 0).sum()
n_pos_full = max((y_enc == 1).sum(), 1)
params["scale_pos_weight"] = n_neg_full / n_pos_full  # ← DELETE
Also remove weight=sample_weight from both lgb.Dataset() calls. Using both scale_pos_weight AND sample_weight simultaneously is the root cause of the oscillation between always-LONG and always-WAIT.

Expected After Both Fixes
text
Features: 76  ← 60 return lags + 16 indicators/micro
[trend]     Fold 1 | Acc: ~0.61  F1: ~0.57  LONG recall: ~0.58
[structure] Fold 1 | Acc: ~0.58  F1: ~0.54  LONG recall: ~0.55
[regime]    Fold 1 | Acc: ~0.55  F1: ~0.51  LONG recall: ~0.52