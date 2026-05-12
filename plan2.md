# MT5 Real-Time Data Streaming & Multi-Strategy Ensemble Trading Bot
## Theory & Architecture Document — XAUUSD Focus

***

## Executive Summary

This document defines the complete conceptual architecture for a trading system that streams live tick and bar data from a MetaTrader 5 (MT5) terminal into a Python-based intelligence layer, runs multiple independent EA-style sub-strategies in parallel, combines their signals through a weighted voting engine, and routes final trade decisions back to MT5 for execution.

The core insight is that no single EA strategy is universally correct — each one excels in a specific market regime (trending, ranging, volatile, exhaustion). By running all strategies simultaneously and letting a voting engine arbitrate, the system reduces the false-signal rate dramatically while still capturing a high percentage of genuine opportunities. This is the same concept as the Ensemble Consensus System used in professional Pine Script indicators, adapted for a full Python ML pipeline connected to a live broker feed.[1]

This document contains zero implementation code. It is a theory and logic document intended to be handed to a coding agent.

***

## Part 1 — The MT5 Data Bridge

### 1.1 Why Stream from MT5 Instead of Using Historical Files

Your existing pipeline is built on `xauusd_m1_4y_dukas.csv` — a static historical file. A live system must replace this with a continuous data bridge that receives each new tick and assembles it into OHLCV bar structures in real time.

MetaTrader 5 exposes a native Python API (the `MetaTrader5` package) that communicates directly with a locally installed and running MT5 terminal. The Python process and the MT5 terminal must live on the same machine. The MT5 terminal handles the broker connection, order routing, and tick subscription — Python merely reads from it and sends trade requests.[2]

There are two core data functions available for live use:[3][4]

- `copy_ticks_from(symbol, datetime_from, count, COPY_TICKS_ALL)` — pulls the last N ticks from a specific timestamp as a `numpy.ndarray` with columns `time`, `bid`, `ask`, `last`, `volume`, `flags`
- `copy_rates_from(symbol, TIMEFRAME_M1, datetime_from, count)` — pulls fully-formed M1 OHLCV bars from a timestamp[4]

The tick function gives maximum granularity. The rates function gives pre-built bars, which is more convenient for indicator computation. The choice between them is architectural.

### 1.2 The Polling Loop — Core Design

The Python bridge runs a continuous polling loop. On each cycle, it calls the MT5 API, retrieves the latest bars, computes indicators, evaluates all strategies, and passes signals to the voting engine. This is not event-driven (MT5's Python API does not push data; you pull it).[3]

**The Polling Loop Cycle:**

1. **Tick Pull**: Call `symbol_info_tick(symbol)` to get the absolute latest bid/ask. This is lightweight and can run at 100ms–500ms intervals.
2. **Bar Pull**: Once per completed bar (on M1, that is once per minute), call `copy_rates_from` to fetch the last N completed bars. This is the "heartbeat" of the strategy layer.
3. **Staleness Check**: Before acting on bar data, verify that the most recently fetched bar's timestamp equals `floor(current_time / 60) * 60 — 60` (i.e., the last *completed* bar, not the currently forming one). An incomplete bar will produce misleading indicator values. Never run strategy logic on an open/forming bar.[5]
4. **Data Assembly**: Convert the numpy array from MT5 into a pandas DataFrame with columns `time, open, high, low, close, tick_volume`.
5. **Indicator Computation**: Compute all required indicators (see Part 2) on the assembled DataFrame.
6. **Strategy Evaluation**: Pass the DataFrame to each strategy module. Each returns a vote.
7. **Voting Engine**: Aggregate votes and determine if a signal threshold is met.
8. **Order Routing**: If threshold is met, send a trade request back to MT5 via `order_send()`.
9. **Sleep**: Sleep until the next bar close.

### 1.3 Higher-Timeframe (HTF) Data Within the Loop

Your existing pipeline exposed the critical problem of timeframe confusion — an EMA200 on M1 data only looks back 3.3 hours, which is useless as a macro trend filter. The solution is to request multiple timeframes within the same loop cycle.

On each heartbeat, the system fetches not just M1 bars but also:
- **M5 bars** (last 250): For short-term momentum context
- **M15 bars** (last 200): For intraday structure
- **H1 bars** (last 250): For macro trend — H1 EMA200 is a genuine multi-week trend filter
- **H4 bars** (last 100): For session-level bias
- **D1 bars** (last 50): For daily structure and weekly open levels

Each timeframe DataFrame is computed independently. H1 indicators are not approximated from M1 data — they are real H1 bars pulled from the same MT5 terminal. This eliminates the span multiplication problem entirely.

### 1.4 Latency Considerations

MT5's Python API is not a low-latency HFT feed. Typical round-trip latency between the Python polling call and data return is 5–50ms on local machine communication. For M1 strategies this is entirely acceptable — you have up to 60 seconds between bars. The system is designed around bar-close execution, not tick-level execution, so latency is not a constraint.[2]

The one latency risk is the bar-close race condition: if you poll at exactly the second a new bar closes, the MT5 terminal may not have finalized the bar yet. The mitigation is to poll 2–3 seconds *after* the expected bar close timestamp, not at exactly `time % 60 == 0`.

### 1.5 Connection Architecture

```
[MT5 Terminal] ←——— Broker Feed (TCP) ——→ [Broker Server]
      ↑ ↓
 MT5 Python API (local IPC)
      ↑ ↓
[Python Intelligence Layer]
   ├── Data Assembler
   ├── Indicator Engine
   ├── Strategy Modules (N strategies)
   ├── Voting Engine
   ├── Risk Manager
   └── Order Router → back to MT5 terminal → broker
```

The Python process is the "brain" and MT5 is the "hands." MT5 handles all order execution, SL/TP management, and broker communication. Python handles all signal generation and decision-making.[6]

***

## Part 2 — Indicator Engine

The Indicator Engine is a shared computation layer that calculates all technical indicators once per bar, stores them in memory, and makes them available to all strategy modules without redundant recalculation.

### 2.1 Indicators Computed at Each Timeframe

**On M1 bars:**
- EMA5, EMA20 (existing pipeline indicators)
- ATR(14), ATR(50) rolling mean
- VWAP (session-anchored, reset at London open each day)
- Bollinger Bands: 20-period SMA ± 2 standard deviations; band width; band width rolling 20-period minimum (squeeze detector)
- RSI(14)
- MACD: fast EMA(12) – slow EMA(26), signal EMA(9), histogram
- Tick volume normalized: current bar's tick volume divided by 50-bar rolling average (volume ratio)

**On H1 bars:**
- EMA50, EMA200 (macro trend filter — these are real H1 bars, not approximated M1 spans)
- EMA50 slope: difference of current EMA50 minus EMA50 ten bars ago, divided by ATR — normalized slope
- Market structure: swing highs and swing lows using a lookback of 20 bars
- VWAP (daily)
- RSI(14) for divergence detection

**On H4 bars:**
- EMA50, EMA200 (session bias)
- ADX(14): trend strength filter — below 20 means ranging, above 25 means trending[7]

**On D1 bars:**
- Weekly open level (stored from Monday's open)
- Previous day's high and low (PDH/PDL)
- Daily ATR

### 2.2 The Regime Detector

Before any strategy runs, the Regime Detector classifies the current market state. This is critical because every strategy performs very differently across regimes.[1]

**Regime Classification Logic:**

| Regime | Conditions | Strategy Implication |
|--------|-----------|---------------------|
| **Bull Trend** | H1 close > H1 EMA200, H1 EMA50 slope > 0, H4 ADX > 25 | Favor LONG signals. SHORT only on exhaustion/blow-off. |
| **Bear Trend** | H1 close < H1 EMA200, H1 EMA50 slope < 0, H4 ADX > 25 | Favor SHORT signals. LONG only on deep fib retracements. |
| **Ranging** | H4 ADX < 20, price oscillating between PDH and PDL | Favor mean-reversion strategies. Disable trend-following. |
| **Volatile** | M1 ATR > 1.5× ATR50_mean, large news event window | Reduce position size. Increase SL distance. May disable entry. |

The current regime is stamped onto every signal so the voting engine can weight votes differently by regime.

***

## Part 3 — The Strategy Modules

Each strategy module is an independent, self-contained unit that:
1. Receives the multi-timeframe indicator DataFrame
2. Applies its own entry logic
3. Returns a standardized signal object: `{side: LONG/SHORT/FLAT, confidence: 0.0–1.0, strategy_id: string}`

No strategy module places orders. It only votes. This is the key architectural separation.

### Strategy 1 — EMA Crossover + MACD Trend Following

**Category:** Trend Following  
**Best Regime:** Bull Trend, Bear Trend  
**Worst Regime:** Ranging  

**LONG entry logic:**
- EMA5 crosses above EMA20 on M1 (golden cross)
- MACD line crosses above signal line *simultaneously or within 3 bars*
- H1 close is above H1 EMA200 (macro regime confirmation)
- H4 ADX > 20 (trend is real, not noise)
- Price is above M1 VWAP

**SHORT entry logic (mirror):**
- EMA5 crosses below EMA20
- MACD line crosses below signal line
- H1 close is below H1 EMA200
- H4 ADX > 20

**Confidence scoring:**
- Each additional confirming condition adds to the confidence score
- Maximum confidence requires all 5 conditions: 1.0
- Base (crossover only): 0.4
- Crossover + MACD: 0.6
- Crossover + MACD + H1 regime: 0.8
- All 5 conditions: 1.0

This strategy implements the classic EMA-MACD stack described widely across EA implementations. The dual confirmation from both the crossover and MACD histogram prevents trading in the wrong direction when price is chopping.[8][9]

***

### Strategy 2 — Bollinger Band Squeeze Breakout

**Category:** Volatility Breakout  
**Best Regime:** Post-squeeze, any macro direction  
**Worst Regime:** Already-expanded bands, trending  

**Core logic:**
The Bollinger Band Squeeze identifies periods of low volatility (consolidation) that historically precede significant directional moves. When volatility contracts — shown by the bands narrowing toward each other — energy builds. The breakout direction after the squeeze determines the trade direction.[10][11]

**Squeeze Detection:**
- Calculate band width = (upper band – lower band) / middle band
- Calculate the 20-period minimum band width
- A "squeeze" is active when current band width equals the 20-bar minimum (i.e., the bands are at their narrowest point in 20 bars)
- Cross-validate with the Keltner Channel: if the Bollinger Bands are *inside* the Keltner Channel, squeeze is confirmed[12]

**LONG entry logic:**
- Squeeze was active within the last 3 bars
- Price closes *above* the upper Bollinger Band (breakout)
- RSI is rising (not overbought — avoid above 70 at entry)
- Volume ratio > 1.5 (volume expanding on breakout)
- H1 EMA50 slope is flat or positive (macro bias neutral-to-bullish)

**SHORT entry logic:**
- Squeeze was active within the last 3 bars
- Price closes *below* the lower Bollinger Band
- RSI is falling (not oversold — avoid below 30 at entry)
- Volume ratio > 1.5
- H1 EMA50 slope flat or negative

**Confidence scoring:**
- Squeeze active + breakout only: 0.5
- + volume confirmation: 0.7
- + RSI direction: 0.85
- + H1 regime alignment: 1.0

***

### Strategy 3 — ICT Smart Money Concepts (Order Block + FVG + Market Structure)

**Category:** Institutional Price Action  
**Best Regime:** Trending with clear structural shifts  
**Worst Regime:** Choppy, news-driven spikes  

**Theoretical Basis:**
ICT methodology — originally developed by Michael Huddleston — models the behavior of institutional (smart money) participants who place large orders in defined zones. These zones, called Order Blocks, are the last opposing candle before a significant institutional move. Price frequently returns to these zones to fill remaining institutional orders before continuing the primary move.[13][14]

**Core Concepts Implemented:**

*Order Block Detection:*
An Order Block (OB) is identified on the H1 chart when:
- There is a series of same-directional candles (at least 3 consecutive bullish or bearish candles)
- Immediately before this run, there is one candle in the opposite direction
- That opposing candle's body-to-range ratio exceeds 0.5 (it is a real-bodied candle, not a doji)
- The run that followed this candle broke a significant swing high or low
The entire body of that opposing candle defines the Order Block zone[15][16]

*Fair Value Gap (FVG) Detection:*
An FVG exists when a 3-candle pattern creates an unfilled price gap:[17]
- Bullish FVG: `candle[n-1].high < candle[n+1].low` (gap between the high of bar n-1 and the low of bar n+1)
- Bearish FVG: `candle[n-1].low > candle[n+1].high`
FVGs represent areas where price moved too fast and institutions will seek to rebalance

*Break of Structure (BoS) and Change of Character (CHoCH):*
- BoS: Price makes a new swing high (in uptrend) or new swing low (in downtrend) — trend continuation confirmation[13]
- CHoCH: Price breaks the most recent opposing swing — potential trend reversal signal

**LONG entry logic (OB + FVG confluence):**
1. On H1, identify a Bullish Order Block
2. Confirm that a subsequent BoS occurred (price broke above the swing high before the OB)
3. On M1, wait for price to retrace *into* the OB zone
4. Look for a Bullish FVG forming inside or near the OB zone during the retracement
5. Entry is triggered when price taps the FVG and shows a reversal M1 candle (bullish engulfing or pin bar)
6. Weekly open level or PDH/PDL proximity adds confidence
7. H1 EMA200 must be below price (regime alignment)

**SHORT entry logic (mirror):**
1. Bearish Order Block identified on H1
2. Price breaks below the swing low (BoS confirmation)
3. Price retraces into the OB zone
4. Bearish FVG forming inside the OB zone
5. Reversal M1 candle at the FVG tap

**Confidence scoring:**
- OB tap alone: 0.4
- OB + FVG confluence: 0.65
- OB + FVG + BoS confirmed: 0.8
- All above + weekly open / PDH confluence: 1.0

This strategy is the most theoretically robust for XAUUSD specifically because gold's price action is heavily dominated by institutional (central bank, hedge fund) order flow that reliably creates and respects these zones.[14][15]

***

### Strategy 4 — RSI Divergence Reversal

**Category:** Momentum Reversal  
**Best Regime:** Late-trend, exhaustion conditions, ranging  
**Worst Regime:** Early strong trend  

**Theoretical Basis:**
RSI divergence occurs when price and the RSI oscillator disagree about the direction of momentum. When price prints a higher high but RSI prints a lower high, the market is showing internal weakness — buyers are getting exhausted. When price prints a lower low but RSI prints a higher low, the market is showing internal strength — sellers are running out of energy.[18][19]

**Types of Divergence Detected:**

| Type | Price Pattern | RSI Pattern | Signal |
|------|--------------|------------|--------|
| Regular Bearish | Higher High | Lower High | SHORT |
| Regular Bullish | Lower Low | Higher Low | LONG |
| Hidden Bearish | Lower High | Higher High | SHORT (trend continuation) |
| Hidden Bullish | Higher Low | Lower Low | LONG (trend continuation) |

**Detection Logic:**
1. Identify swing highs and swing lows on the M15 chart using a 10-bar lookback on each side (the pivot must be the highest/lowest of the 10 bars before and after it)[19]
2. For each pair of consecutive swing highs: compare price direction to RSI direction at those same pivots
3. Divergence is confirmed only if the two swing points are separated by at least 5 bars (prevents detecting divergence on a single candle's noise)
4. Divergence is *active* for a maximum of 20 bars — if price has not reversed within 20 bars of the divergence signal, the signal is invalidated
5. Entry trigger: after divergence is detected, wait for the first M1 candle that closes *against* the prior trend (e.g., after bearish divergence, first M1 close that is lower than its open, with RSI crossing below 50)

**LONG entry logic (Regular Bullish Divergence):**
- M15 shows price lower low, RSI higher low
- RSI(14) is between 30 and 50 at the second pivot (oversold recovery area)
- M1 entry candle: bullish close after the divergence confirms
- H1 price is not in a violent downtrend (H1 ADX < 35 — avoid fighting a high-conviction institutional trend)

**SHORT entry logic (Regular Bearish Divergence):**
- M15 shows price higher high, RSI lower high
- RSI(14) is between 50 and 70 at the second pivot
- M1 entry candle: bearish close after divergence confirms
- H1 not in violent uptrend

**Confidence scoring:**
- Divergence detected: 0.5
- + RSI in confirming zone (30-50 for bullish, 50-70 for bearish): 0.7
- + Volume contraction at the second pivot (sellers/buyers exhausting): 0.85
- + H1 Order Block or FVG nearby (confluence with Strategy 3): 1.0

***

### Strategy 5 — Exhaustion SHORT (Blow-Off Top Reversal)

**Category:** Trend Exhaustion / Climax Reversal  
**Best Regime:** Bull Trend (for shorting the climax)  
**Worst Regime:** Ranging, Bear Trend  

**Theoretical Basis:**
In a structural multi-year bull market (which XAUUSD has been since 2022), shorting pullbacks or crossdowns is extremely dangerous — the macro trend absorbs and reverses those entries. The only time sellers win consistently is when buyers have temporarily run out of liquidity after a parabolic vertical spike. This is the "rubber band snapback" — after a near-vertical move, price mean-reverts to find the next support area before the bull trend resumes.

This strategy was specifically designed for XAUUSD's current regime and is the primary SHORT strategy in the pipeline, replacing the original EMA20 crossdown signal.

**Entry Logic:**
- Price is more than 2.5×ATR above the M1 VWAP (significantly extended)
- There have been 5 consecutive bullish M1 candles (climax exhaustion run)
- Volume ratio > 2.0 (volume spike — the "blow-off" phase)
- M1 ATR > 0.8 × ATR50_mean (volatility is elevated, move is real)
- RSI(14) > 72 on M15 (momentum is overbought at multiple timeframes)
- No new significant catalyst in the last 30 minutes (avoid entering against a high-impact news release)

**Confidence scoring:**
- Base (VWAP extension + 5 green bars): 0.5
- + Volume spike: 0.7
- + RSI overbought on M15: 0.85
- + H1 Bearish Order Block or FVG overhead (resistance zone confluence): 1.0

**Important:** This strategy only votes SHORT. It never votes LONG.

***

### Strategy 6 — VWAP Mean Reversion (LONG Only)

**Category:** Mean Reversion  
**Best Regime:** Bull Trend pullbacks, Ranging  
**Worst Regime:** Violently trending, post-news spike  

**Theoretical Basis:**
The VWAP (Volume Weighted Average Price) represents the "fair value" of the instrument for the current session, weighted by the volume at each price level. Institutional traders frequently use VWAP as a benchmark — they buy below VWAP and sell above it to ensure good execution quality. This creates a mean-reversion tendency around VWAP.[8]

This strategy specifically applies to LONG entries in a bull regime — it captures the "buy the dip back to VWAP" behavior that characterizes institutional accumulation during uptrends.

**Entry Logic:**
- H1 close is above H1 EMA200 (macro bull regime — this is the regime gate missing from the original pipeline)
- H1 EMA50 slope is positive over the last 5 bars
- On M1, price has pulled back to within 0.5×ATR of the M1 VWAP from above
- RSI(14) on M15 is between 40 and 55 (neutralizing from oversold without being extended)
- EMA5 is beginning to recover above EMA20 (micro-structure turning)
- No bearish Order Block directly overhead (check Strategy 3's OB inventory)

**Confidence scoring:**
- H1 regime + VWAP touch: 0.5
- + RSI neutralizing: 0.65
- + EMA5 recovery: 0.8
- + No overhead OB resistance: 1.0

**Important:** This strategy only votes LONG. It never votes SHORT. The original LONG pipeline signal from `primary_signal_generator.py` maps approximately to this strategy.

***

## Part 4 — The Voting Engine

The Voting Engine is the heart of the system. It receives the output from all 6 strategy modules and determines whether a trade should be executed, on which side, and with what confidence level.

### 4.1 The Vote Object

Each strategy returns a standardized vote:

```
{
  strategy_id: "EMA_MACD_TREND",
  side: "LONG" | "SHORT" | "FLAT",
  confidence: 0.0 to 1.0,
  regime_alignment: True | False
}
```

A `FLAT` vote means the strategy sees no valid signal — it abstains rather than voting against the other side.

### 4.2 The Weighted Vote Aggregation

Not all strategies are equal. The Voting Engine applies regime-dependent weights to each strategy's vote:[1]

| Strategy | Bull Regime Weight | Bear Regime Weight | Ranging Weight |
|----------|-------------------|-------------------|----------------|
| EMA + MACD Trend | 1.5 | 1.5 | 0.5 |
| BB Squeeze Breakout | 1.0 | 1.0 | 0.8 |
| ICT Order Block | 1.8 | 1.8 | 1.2 |
| RSI Divergence | 0.8 | 0.8 | 1.5 |
| Exhaustion SHORT | 0.0 (no LONG) | 2.0 | 0.5 |
| VWAP Mean Reversion | 2.0 | 0.0 (no SHORT) | 1.2 |

**Aggregation formula:**

For LONG votes:
\[ \text{LONG\_score} = \sum_{i=1}^{N} w_i \times c_i \times \mathbf{1}[\text{side}_i = \text{LONG}] \]

For SHORT votes:
\[ \text{SHORT\_score} = \sum_{i=1}^{N} w_i \times c_i \times \mathbf{1}[\text{side}_i = \text{SHORT}] \]

Where \(w_i\) is the regime-adjusted weight, \(c_i\) is the confidence score, and the indicator function \(\mathbf{1}\) selects only votes of the matching side.

### 4.3 The Dynamic Threshold Gate

The minimum score required to trigger a trade is not fixed — it adapts to market conditions:[1]

| Volatility Regime | ATR/Close Ratio | Min Score to Trade |
|------------------|-----------------|--------------------|
| Low Volatility | < 0.3% | 2.5 |
| Normal Market | 0.3% – 0.8% | 3.0 |
| High Volatility | > 0.8% | 4.0 |

This prevents the system from overtrading during news events (when ATR spikes and the threshold rises) and maintains responsiveness during normal conditions.

### 4.4 Hard Block Conditions

Even if the vote score exceeds the threshold, the following conditions completely block trade entry:

1. **Existing position in same direction**: If a LONG is already open and the score votes LONG again, no new position is opened. The system waits for the existing position to close.
2. **Daily loss limit**: If the day's P&L is below –2R (two times the risk per trade), all new entries are blocked until the next trading day.
3. **High-impact news within 15 minutes**: Detected by checking a pre-loaded economic calendar. XAUUSD is highly sensitive to NFP, CPI, Fed rate decisions, and geopolitical headlines.
4. **Weekend/session gap**: No new positions during the 30 minutes before market close on Friday.
5. **Spread too wide**: If current spread > 3×normal spread, execution quality is poor — block entry.

### 4.5 Signal Direction Arbitration

If both LONG\_score and SHORT\_score exceed their respective thresholds simultaneously (a "conflicted" market), the system does not trade. Conflicted signals indicate regime ambiguity and increased risk of whipsaw. The engine logs the conflict and waits for the next bar.

***

## Part 5 — Risk Manager

The Risk Manager runs between the Voting Engine and the Order Router. It translates the abstract voting decision into a concrete trade specification with precise position size, stop-loss level, and take-profit level.

### 5.1 Position Sizing

Position size is calculated using fixed fractional risk: a defined percentage of account equity is risked on each trade, adjusted by the distance from entry to stop-loss in dollar terms.

\[ \text{Lot Size} = \frac{\text{Account Equity} \times \text{Risk Percent}}{(\text{Stop Distance in pips}) \times \text{Pip Value per Lot}} \]

Default risk per trade: 1.0% of equity. This is adjustable. The system never exceeds 2% risk regardless of signal confidence — this is a hard ceiling.

Position size is further reduced (by 0.5×) when:
- Current volatility regime is "High Volatility"
- The winning strategy module has a confidence below 0.7
- The vote score is between the threshold and 1.5× threshold (borderline signal)

### 5.2 Stop-Loss Placement

Stop-loss distance is determined dynamically based on ATR, not a fixed pip value:

**LONG stops:** Placed at `entry – (SL_multiplier × ATR14_M1)`, where SL_multiplier defaults to 1.0.
**SHORT stops:** Placed at `entry + (SL_multiplier × ATR14_M1)`.

The stop is further adjusted to respect structural levels — it is moved to just below the nearest swing low (for LONG) or just above the nearest swing high (for SHORT) if the structural level is within 1.5×ATR of the ATR-calculated stop. This avoids placing stops in "obvious" locations where liquidity pools accumulate.[14]

### 5.3 Take-Profit Targets

The system uses a multi-target exit structure rather than a single take-profit:

- **TP1** (partial exit, 50% of position): `entry + 1.5 × ATR14_M1` for LONG. This locks in profit quickly.
- **TP2** (remainder): At the nearest significant resistance/support zone from Strategy 3's OB inventory, OR `entry + 3.0 × ATR14_M1`, whichever is closer.
- **Trailing Stop** (after TP1 hit): Move stop to breakeven + 0.5×ATR. Then trail the stop at 1.0×ATR below the running high (for LONG).

This structure produces a minimum R/trade of +0.75R when TP1 is hit and trailing stop is eventually activated, and a full 3R when TP2 is reached.

### 5.4 Maximum Drawdown Circuit Breaker

If the account equity draws down more than 5% from its recent peak (calculated on a rolling 14-day basis), the system enters "Circuit Breaker" mode:
- All new trade entries are halted
- Existing positions are managed normally
- The system resumes when equity recovers above the –5% threshold OR after 48 hours, whichever comes first

***

## Part 6 — LightGBM ML Overlay (Meta-Layer)

The 6 strategy modules and voting engine described above constitute the "rules-based" layer. On top of this, an optional LightGBM ML layer acts as a meta-filter — it does not generate signals, it only validates or rejects signals from the voting engine.

### 6.1 What the ML Layer Does

The ML model is trained on historical data using your existing `train_pipeline/`. It receives the same feature set as the strategy modules plus the voting engine's raw scores as additional features. Its output is a probability of trade success.

If the voting engine approves a trade AND the LightGBM model produces a probability ≥ 0.55, the trade is executed at full size. If the voting engine approves but LightGBM gives a probability between 0.45 and 0.55, the trade is executed at half size. If LightGBM gives < 0.45, the trade is blocked regardless of the voting score.

### 6.2 Training Data for the ML Layer

The ML model is trained using the Triple Barrier Method for labeling:[20][21]
- **Label = 1** (profit): Price hit the TP barrier first
- **Label = -1** (loss): Price hit the SL barrier first
- **Label = 0** (neutral): Neither barrier hit within the max holding period

The features fed to LightGBM include:
- All indicator values at the bar of signal (EMA positions, VWAP distance, ATR ratio, RSI, MACD histogram, Bollinger Band width)
- The voting engine's LONG\_score and SHORT\_score
- The top-contributing strategy's confidence
- Current regime (encoded as integer)
- Time features: hour of day, day of week, minutes to next session open
- The HTF EMA200 distance on H1 (price relative to macro trend)

### 6.3 Walk-Forward Validation

The ML model is not trained once and forgotten. It uses expanding-window walk-forward training: the model is retrained monthly on all data up to the current month, and its performance on the next month's out-of-sample data is tracked. If the model's out-of-sample win rate drops below 48% for two consecutive months, the ML layer is disabled and the system reverts to pure rules-based voting until a new model can be validated.

***

## Part 7 — Session and Calendar Awareness

XAUUSD has well-defined session behavior that every EA must respect.

### 7.1 Trading Sessions

| Session | Time (UTC) | Characteristics | Strategy Preference |
|---------|-----------|----------------|---------------------|
| Asian | 00:00 – 08:00 | Low volume, tight ranges | BB Squeeze, RSI Divergence |
| London | 08:00 – 12:00 | High volume, strong directional moves | ICT OB, EMA Trend, Exhaustion |
| New York | 13:00 – 17:00 | High volume, continuation or reversal | All strategies active |
| Overlap | 12:00 – 14:00 | Peak liquidity | ICT OB, Trend Following |
| NY Close | 21:00 – 22:00 | Session close, position squaring | Avoid new entries |

The system applies session-based strategy weighting — during Asian session, high-momentum strategies (EMA Trend, Exhaustion) are down-weighted and mean-reversion strategies are up-weighted.

### 7.2 Economic Calendar Integration

The system maintains a pre-loaded weekly economic calendar (sourced from a public JSON API, refreshed each Sunday). Before each trade:
- If a Tier-1 event (NFP, FOMC, CPI) is within 30 minutes: block all entries
- If a Tier-2 event (ADP, ISM) is within 15 minutes: reduce position size by 50%
- Immediately after a Tier-1 event: allow trades but increase SL distance to 1.5×ATR (volatility expanded)

***

## Part 8 — Logging, Monitoring, and Performance Tracking

A live trading system without comprehensive logging is a black box that will eventually lose money without explanation.

### 8.1 Per-Signal Log Entry

Every time the voting engine evaluates a bar, regardless of whether a trade is taken, log:
- Timestamp
- Current regime
- Each strategy's vote and confidence
- Final LONG\_score and SHORT\_score
- Trade decision (taken / blocked / conflicted)
- If blocked: reason code

### 8.2 Per-Trade Performance Metrics

For every closed trade:
- Entry time, exit time, holding period
- Entry price, exit price, gross P&L in pips
- P&L in R-multiples (relative to initial risk)
- Which strategies voted correctly vs. incorrectly
- ML model probability at entry
- Final exit reason: TP1/TP2/trailing stop/manual close/circuit breaker

### 8.3 Rolling Performance Dashboard

Maintained daily:
- Running win rate (last 50 trades)
- Average R/trade (last 50 trades)
- Maximum drawdown (current streak and historical peak)
- Strategy-level contribution: which strategy module's votes most reliably predicted profitable outcomes
- Regime-level win rate: does the system perform better in Bull or Ranging regimes?

Strategy-level attribution allows the voting weights to be tuned over time — if ICT Order Block votes have a 65% win rate and EMA Trend votes only 51%, increasing ICT's weight and reducing EMA Trend's weight will improve overall system performance.

***

## Part 9 — Data Pipeline Architecture (End-to-End Summary)

```
MT5 Terminal (XAUUSD live feed)
        ↓ (copy_rates_from + copy_ticks_from)
Data Bridge / Polling Loop (every M1 close)
        ↓
Multi-Timeframe DataFrame Builder
[M1, M5, M15, H1, H4, D1]
        ↓
Indicator Engine
[EMAs, ATR, VWAP, BB, RSI, MACD, Volume Ratio, Swing Points]
        ↓
Regime Detector → Regime Tag (Bull/Bear/Ranging/Volatile)
        ↓
Strategy Modules (parallel evaluation)
[1.EMA+MACD | 2.BB Squeeze | 3.ICT OB | 4.RSI Div | 5.Exhaustion | 6.VWAP MR]
        ↓
Vote Aggregation (weighted, regime-adjusted scores)
        ↓
Hard Block Gate (news, drawdown, spread, time)
        ↓
Dynamic Threshold Check
        ↓  [if threshold met]
LightGBM Meta-Filter (probability ≥ 0.55)
        ↓  [if approved]
Risk Manager (lot size, SL, TP1, TP2, trailing stop)
        ↓
Order Router → MT5 order_send()
        ↓
MT5 Terminal executes with broker
        ↓
Position Monitor (trailing stop, partial TP management)
        ↓
Performance Logger
```

***

## Part 10 — Key Design Principles and Trade-Offs

### 10.1 Why Ensemble Voting Instead of a Single Strategy

A single EA strategy has a "regime problem" — it outperforms in one regime and underperforms in another. The 2022–2026 XAUUSD bull market exposed this when the original EMA20 crossdown SHORT signal stopped working because the macro trend dominated every short-term signal. An ensemble of strategies with different characters (trend-following, mean-reversion, exhaustion, institutional price action) means the system always has at least two or three relevant strategies producing valid signals regardless of regime, while strategies that do not fit the current regime either abstain or have low weight.[22][1]

### 10.2 Why Not MQL5 Native EA for the Full Logic

A native MQL5 EA written in MetaQuotes Language runs inside the MT5 terminal directly. It is faster for order execution and does not require a running Python process. However, integrating LightGBM, numpy, pandas, or any ML library into MQL5 is impractical — MQL5 is not a general-purpose scientific computing language. The hybrid architecture (MT5 as broker bridge + Python as intelligence layer) is the standard approach for ML-enhanced trading systems.[6][2]

The risk of the hybrid approach is that if the Python process crashes, all open positions are managed by MT5's built-in SL/TP levels (which are set at entry). The system never opens a trade without hard SL/TP in place precisely for this failsafe.

### 10.3 Overfitting Risk in the ML Layer

The greatest technical risk in any ML trading system is overfitting — a model that memorizes historical patterns that do not repeat. Mitigations in this architecture:
- Walk-forward validation forces the model to prove out-of-sample performance before going live
- The rules-based voting engine provides a floor — the ML layer is additive, not replacement
- Feature set is limited to indicators with proven theoretical grounding, not raw price columns
- Monthly retraining keeps the model current with regime changes[20]

### 10.4 The Data Starvation Warning (Applies to ML Layer)

As identified in the pipeline analysis, the 2022–2026 XAUUSD dataset is predominantly bull-regime data. If the ML layer is trained on this full dataset and then used to filter SHORT signals, the model will be systematically biased against short trades — it has rarely seen a short trade succeed in training. The mitigation is to train **separate ML models per trade side**: one LightGBM model for LONG signals (using the 2022–2026 data which is rich in LONG examples) and one LightGBM model for SHORT signals trained only on Exhaustion SHORT events (using data from the last 12 months where the strategy has been defined).

***

## Appendix A — Strategy Signal Summary Table

| Strategy | Trade Side | Core Trigger | Regime Gate | Best Timeframe |
|----------|-----------|--------------|-------------|----------------|
| EMA + MACD Trend | BOTH | EMA5/20 crossover + MACD cross | H1 EMA200 side | M1 signal, H1 bias |
| BB Squeeze Breakout | BOTH | Band width minimum + close outside bands | Any | M1 squeeze, M15 confirmation |
| ICT Order Block | BOTH | OB tap + FVG + BoS | H1 EMA200 side | H1 OB, M1 entry |
| RSI Divergence | BOTH | Price/RSI divergence at swing points | ADX < 35 | M15 divergence, M1 entry |
| Exhaustion SHORT | SHORT ONLY | VWAP extension + 5 green bars + volume spike | Bull regime (ironically) | M1 climax, M15 overbought |
| VWAP Mean Reversion | LONG ONLY | Price at VWAP in bull regime + EMA recovery | H1 Bull regime | M1 entry, H1 gate |

***

## Appendix B — Glossary

- **ATR**: Average True Range — measures volatility over N bars
- **BoS**: Break of Structure — price exceeds a prior swing high/low, confirming trend direction
- **CHoCH**: Change of Character — price breaks the opposing swing, signaling possible trend reversal
- **FVG**: Fair Value Gap — unfilled price gap between three consecutive candles, left by institutional speed
- **HTF**: Higher Time Frame — a chart timeframe longer than the primary execution timeframe
- **OB**: Order Block — last opposing candle before a significant institutional move; defines a supply/demand zone
- **R**: Risk unit — 1R = the dollar value risked per trade; 2R means twice the risk unit in profit
- **VWAP**: Volume Weighted Average Price — session's average price weighted by volume at each level
- **Walk-Forward Validation**: Out-of-sample testing methodology where the model trains on past data and is tested on the immediately following period, rolled forward sequentially