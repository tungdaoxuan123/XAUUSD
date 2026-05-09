\The Logic
Roll a fixed training window forward across the 4-year dataset, month by month. At each step, train on 12 months, test on the next 1 month. Record trades fired and win rate for each test month. Do this for all 36 forward steps (months 13–48 of your data).

Exact Parameters
text
Training window : 12 months (fixed, rolling)
Test window     : 1 month (out-of-sample)
Step size       : 1 month
Total steps     : ~36 (covering 2023-01 to 2026-05)
Threshold       : 0.55 (LONG), 0.55 (SHORT)
Min trades/month: log if < 10 (regime silence flag)
What to Log Per Step
text
month | n_train_events | n_test_events | fires | win_rate | expectancy | avg_confidence
Success Criteria
Median fires per month ≥ 30

Median win rate ≥ 60% across all 36 months

No more than 4 consecutive months with fires < 10 (regime silence)

Expectancy positive in at least 28 of 36 months

If those pass, the model is genuinely validated and ready for live deployment. If they fail, you see exactly which months break and why.