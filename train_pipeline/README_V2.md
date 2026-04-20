# v2 upgrade: calibration, entry gating, dynamic exits

Three targeted fixes for the production issues you hit:

| Issue | Root cause | Fix |
|---|---|---|
| macro-F1 = 0.39 | Focal loss + inverse-frequency class weights push softmax to corners; features at inference don't match training; no augmentation or purged CV | `train_sota_v2.py` — label-smoothed CE (ε=0.1) + class-balanced weights (Cui 2019) + purged walk-forward split + sequence mixup + OneCycle schedule + early stop; session features added |
| Confidence always 1 → "when do I enter?" | Uncalibrated probabilities from a saturated softmax | Temperature scaling (Guo 2017) fitted on val; saved into checkpoint; applied at inference in `live_sota_trading.py` |
| Fixed 2-ATR SL / 3-ATR TP | No quant-style risk management | `dynamic_exits.py` — vol-targeted risk, half-Kelly sizing from calibrated p, confidence+regime+session-scaled RR, chandelier trail + breakeven at 1R + time stop, spread & news gates |

Plus a silent but critical fix: `live_features.py` rebuilds features using the **same functions as training** (Corwin-Schultz, Lee-Ready, Kyle-λ, volume profile). Your old `prepare_sota_data` stubbed many of these with placeholders (`vprof_in_value_area = 1`, etc.), causing covariate shift that also hurts F1 in production.

---

## How to run

```bash
# 1. (unchanged) produce synmicro + triple-barrier labels
bash train_pipeline/run_sota_pipeline.sh   # or the individual python calls

# 2. Train v2
python train_pipeline/train_sota_v2.py \
    --data train_pipeline/data/xauusd_m1_synmicro_tb.csv \
    --out-dir train_pipeline/models_sota_v2 \
    --seq-len 120 --patch-len 12 \
    --epochs 40 --batch-size 256 --gpu

# 3. Point live trader at v2 artifacts (via env vars or config.py)
export SOTA_MODEL_PATH=train_pipeline/models_sota_v2/patchtst_primary.pt
export SOTA_CONFIG_PATH=train_pipeline/models_sota_v2/sota_config.json
python live_sota_trading.py --dry-run --min-prob 0.58
```

---

## Why these specific choices

### Calibration

Expected Calibration Error (ECE) tells you how honest your probabilities are. On financial M1 with focal loss, ECE typically lands 0.15–0.25 (probabilities are 90% when the model is only 65% accurate). Temperature scaling fits a single scalar `T` on val so `softmax(logits/T)` minimizes NLL — [Guo et al. 2017](https://arxiv.org/abs/1706.04599) showed this single parameter almost always beats more complex calibrators on deep nets.

Post-calibration, `p_buy=0.72` means "72% hit rate historically", which makes entry thresholds meaningful and enables Kelly sizing.

### Label smoothing vs focal

Focal loss was designed for object detection where 99% of anchors are background. With 3 classes at roughly 40/20/40 split and path-aware labels, focal's `(1-p)^γ` term actively encourages the model to push confident examples to the corners — producing the "conf=1" you observed. Class-balanced CE ([Cui 2019](https://arxiv.org/abs/1901.05555)) weights the loss by effective sample count rather than raw inverse frequency, and label smoothing bounds the target distribution so no example can ever demand p=1.

### Dynamic RR

Empirically on XAUUSD M1 the optimal hold-till-barrier RR depends on:
- **Conviction**: high-p trades can hold wider TPs because their edge is larger.
- **Regime**: in trending (high-vol) regimes, runs extend; in chop, price mean-reverts fast and you want tighter TP.
- **Session**: London-NY overlap has 2-3× the directional follow-through of Asian.

`ExitPlanner._dynamic_rr` encodes all three with clipping at `[rr_min, rr_max]`.

### Half-Kelly + cap

Full Kelly maximizes log-growth but has ~50% drawdowns. Half-Kelly gives ~75% of the growth at ~25% of the DD. Practitioner standard; also friendly to FTMO's 10% max DD rule.

### In-trade management

- **BE at 1R**: once you're up one stop-distance, slide SL to entry. Keeps the expected R ≥ 0 for the remaining trade.
- **Chandelier trail**: industry-standard volatility trailing from LeBeau; uses highest high - k·ATR (or lowest low + k·ATR for shorts).
- **Time stop at `max_hold_bars`**: matches the triple-barrier horizon you trained on — the model has zero information about what happens beyond that bar, so neither should you.
