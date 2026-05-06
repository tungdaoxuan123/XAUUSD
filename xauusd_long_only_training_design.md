# BUY/LONG-Only Training Design for XAUUSD and GBPUSD on AMD RX 6700 XT

## Overview

The current repository is built around both long and short trading logic. The environment accepts negative actions that open shorts, the ensemble trainer uses three labels (`SELL`, `HOLD`, `BUY`), and the existing reward and exit logic mixes stop loss, take profit, trailing stop, partial take profit, and time-based exits in the same policy loop.[cite:5][cite:6] For a pure BUY/LONG system with fixed 2:1 take-profit to stop-loss, the cleanest design is to convert the entire stack into a long-only decision engine where every sample answers a simpler question: enter long now, or wait.[cite:5][cite:6]

The right way to make pip distance “depend” on market conditions is to tie stop distance to volatility, then define take profit as exactly twice that stop distance. In this repository, the natural volatility anchor is ATR, which is already computed in both the ensemble and triple-barrier pipelines.[cite:5][cite:6] That means the system can preserve a fixed 2:1 risk-reward ratio while allowing actual pip distance to expand and contract with XAUUSD or GBPUSD regime changes.[cite:6][cite:7]

## Current repository behavior

The current GPU ensemble trainer uses LightGBM with an AMD-compatible OpenCL path and falls back to CPU if GPU is unavailable.[cite:5] It trains three models named trend, structure, and regime, and all of them use the same three-class label scheme based on `-1`, `0`, and `1`, mapped internally to LightGBM-compatible non-negative class indices.[cite:5] This setup is solid for a directional classifier, but it is not aligned with a long-only execution policy because the models are explicitly rewarded for learning short setups as a separate positive class.[cite:5]

The current triple-barrier implementation is already much better than the old fixed-horizon labeler because it simulates actual path-dependent outcomes using upper barrier, lower barrier, and vertical timeout.[cite:6] It also includes spread and commission deterioration directly in the barrier math, which makes the labels more realistic than a simple close-to-close return threshold.[cite:6] However, the default formulation still supports both sides, and that means short-side information continues flowing into the training target instead of being collapsed into “do not buy.”[cite:6]

The GBPUSD shell pipeline already shows a more advanced architecture than the older ensemble-only flow. It runs synthetic microstructure feature generation, then triple-barrier labels, then a PatchTST-lite primary model followed by a LightGBM meta filter.[cite:7][cite:8] That pipeline is the best structural base for a long-only redesign because it already separates raw directional prediction from trade-quality filtering.[cite:7][cite:8]

## Label redesign

The highest-impact change is to convert the target from three-class directional prediction into long-only opportunity detection. The triple-barrier file already supports ATR-scaled upper and lower barriers, and its interface exposes `--pt-atr`, `--sl-atr`, and `--max-hold`, which makes it straightforward to encode a fixed 2:1 ratio by setting profit target at `2.0 × ATR` and stop loss at `1.0 × ATR`.[cite:6] This should become the default long-only labeling rule for both XAUUSD and GBPUSD.[cite:6][cite:7]

For long-only training, every bar should be evaluated as a potential long entry only. The short-side branch in the triple-barrier scan should be removed from the label-generation path, or equivalently, `side` should be fixed to `+1` for all candidate entries.[cite:6] Once that is done, any outcome that would currently be labeled `-1` should be remapped to `0`, meaning “wait” rather than “sell,” and only successful long opportunities should remain as `1`.[cite:6]

This turns the learning problem into binary classification: `0 = no long trade`, `1 = valid long trade`. That is a much better fit for a BUY-only live strategy because the model no longer spends capacity distinguishing between bearish continuation and neutral chop; both simply become “do nothing.”[cite:5][cite:6] This usually improves class meaning, reduces confusion around weak `SELL` patterns, and makes downstream execution logic far easier to control.[cite:6]

## Fixed 2:1 with dynamic pip distance

A fixed risk-reward ratio does not require a fixed pip target. In this design, stop distance is defined by current volatility and take profit is derived mechanically as twice that value, so the ratio remains 2:1 while the price distance adapts to the instrument and regime.[cite:6] On quiet sessions, ATR contracts and the system uses tighter stops and smaller targets; during high-volatility moves, ATR expands and the trade gets more room automatically.[cite:6]

In practical terms, a long entry should lock four values at entry time: deteriorated entry price, stop distance, stop price, and target price. The triple-barrier code already computes effective spread as spread plus commission and shifts entry and exit checks to worse executable prices, so that same logic should be reused in live execution to keep training and trading aligned.[cite:6] The important principle is that these levels are fixed when the trade opens, not moved later by trailing logic, partial exits, or reward hacks.[cite:6]

This means the live bot should read the model signal, compute `sl_dist = ATR × sl_multiplier`, compute `tp_dist = 2 × sl_dist`, and place the order with those levels immediately. Because the repository already computes ATR and stores barrier-driven outcomes, this is conceptually consistent with the labeling layer and avoids a train-live mismatch.[cite:5][cite:6]

## Environment redesign

The reinforcement-learning environment should be simplified aggressively. A long-only policy does not need a continuous action in the range `[-1, 1]`; it only needs to decide whether to enter long, stay flat, hold an existing long, or optionally force exit.[cite:2] The best redesign is a discrete action space where action `0` means wait or close, and action `1` means enter or continue holding a long.[cite:2]

The existing environment mixes multiple exit mechanisms including stop loss, take profit, trailing stop, partial take profit, and time-based closure.[cite:2] That produces noisy reward attribution because the policy is being trained against several overlapping objectives at once. For a 2:1 long-only system, reward should be normalized to R-multiples: `+2` for target hit, `-1` for stop hit, and near-zero for passive waiting, with only a very small penalty for endless inactivity if needed.[cite:2]

This makes the policy objective clean: it should enter only when expected value is positive under a hard 2:1 framework. That is much easier for PPO or any actor-critic method to learn than the current environment, where payout multipliers and trailing mechanics can dominate reward variance.[cite:2] Even if reinforcement learning is kept only as a fine-tuning layer on top of supervised signals, the environment should match the exact execution assumptions used during label generation.[cite:2][cite:6]

## Model architecture recommendations

For the ensemble path, the current LightGBM setup can be preserved structurally but switched from multiclass to binary. The trend, structure, and regime split is still useful because it gives three different views of the same long opportunity, but each model should output the probability of a successful long trade instead of probabilities for sell, hold, and buy.[cite:5] On AMD hardware, LightGBM with `device_type='gpu'` remains the best low-friction GPU path because it already works through OpenCL in the repository and is explicitly designed for the RX 6700 XT workflow.[cite:5]

For the SOTA path, the existing PatchTST-lite pipeline is the strongest base for future work. It already supports long sequence windows, transformer-style patch encoding, weighted training, and a downstream LightGBM meta filter.[cite:8] For long-only training, the classifier head should become binary, the primary output should be interpreted as long-entry probability, and the meta filter should estimate whether that long signal is likely to achieve TP before SL or timeout.[cite:6][cite:8]

A strong production design is a two-stage system. Stage one predicts `p_long_entry`, and stage two predicts `p_tp_given_entry`; final action quality is the product or joint rule derived from both scores.[cite:6][cite:8] This is cleaner than a single monolithic model because the first model learns timing, while the second learns trade quality conditional on the first model already wanting to enter.[cite:8]

## GBPUSD pipeline adaptation

The GBPUSD shell pipeline already uses a good ordering: synthetic microstructure first, triple-barrier labeling second, and PatchTST-lite plus LightGBM meta filter third.[cite:7][cite:8] To make it long-only, the barrier settings should be changed from `--pt-atr 1.5 --sl-atr 1.0` to `--pt-atr 2.0 --sl-atr 1.0`, and the generated labels should be remapped so that adverse outcomes become wait labels instead of short labels.[cite:6][cite:7]

The same idea applies to XAUUSD even though the shell wrapper shown in the repository is for GBPUSD. The triple-barrier and SOTA components are instrument-agnostic because they work from ATR, OHLC, spread estimate, and synthetic microstructure context rather than hardcoded symbol assumptions.[cite:6][cite:8] That means a unified long-only training philosophy can be shared across both instruments while still allowing ATR to produce different pip distances on gold versus cable.[cite:6][cite:7]

One practical difference is that GBPUSD pip granularity is much smaller than XAUUSD price movement, and the existing shell script already reflects that with a smaller `--bin-size` for synthetic microstructure feature generation.[cite:7] That is exactly the right pattern: keep feature-extraction parameters symbol-aware, but keep the risk logic universal through ATR-scaled 2:1 barriers.[cite:6][cite:7]

## AMD RX 6700 XT strategy

The repository already contains the two main AMD-friendly compute paths. LightGBM uses OpenCL for GPU acceleration on Windows, and the PatchTST pipeline attempts CUDA first, then MPS, then `torch_directml` before falling back to CPU.[cite:5][cite:8] For the RX 6700 XT, this means tree models are already in a good place, while transformer training is workable through DirectML but still more constrained than NVIDIA CUDA.[cite:5][cite:8]

For classical tabular or hybrid feature models, LightGBM on AMD is the most practical high-speed option in this codebase. It is stable, fast enough for repeated walk-forward experiments, and already integrated with GPU probing and CPU fallback.[cite:5] For transformer experiments, PatchTST-lite is light enough that the 6700 XT can still train it, but the workflow should favor larger batch sizes, plain float32, and conservative DataLoader settings because DirectML is less forgiving than CUDA.[cite:8]

A sensible development plan is to use LightGBM as the fast iteration engine and PatchTST-lite as the higher-ceiling experiment layer. That gives quick feedback for feature and label changes while preserving a more advanced sequence model path for deeper research.[cite:5][cite:8] If training later moves to Linux or WSL with ROCm, the transformer path can become much more competitive without changing the overall modeling philosophy.[cite:8]

## Recommended end-state design

The most robust long-only architecture for this repository is a four-layer stack. First, generate path-aware labels with long-only triple barriers using ATR-scaled `1R` stop and `2R` take profit.[cite:6] Second, train a binary primary model that predicts whether a long entry should be taken now.[cite:5][cite:8] Third, train a binary confidence or meta model that predicts whether the proposed long has enough quality to keep.[cite:6][cite:8] Fourth, execute through a simplified environment or live bot that always uses fixed 2:1 ATR-derived TP/SL and never opens shorts.[cite:2][cite:6]

This architecture aligns labels, model outputs, environment logic, and live trade handling around the same objective. That alignment is the real edge in the redesign, more than any single algorithm choice, because it removes contradictions between what the trainer rewards and what the bot actually does.[cite:2][cite:5][cite:6] The result is a much cleaner research loop for both XAUUSD and GBPUSD on the AMD hardware already targeted by the repository.[cite:5][cite:7][cite:8]
