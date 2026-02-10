# TradingMamba — System Build Plan

## Vision

Build an automated SMC/ICT pattern detection and signal generation system that:
1. Reads live (and historical) market candlestick data
2. Detects ICT patterns using rules extracted from 23 Hindi SMC training videos
3. Scores pattern confluence and probability
4. Generates actionable trading signals with entry, SL, and TP levels

---

## Phase 1: Mathematical Formalization (Knowledge → Code)

### Goal
Translate the `key_rules` from all 23 knowledge bases into deterministic detection algorithms. This is the **core engine** — everything else depends on it.

### Why This Is Phase 1
Our 23 KBs contain human-language rules like "BOS requires inducement taken + candle body close above previous high's wick." These need to become functions that take OHLCV data as input and output detected patterns with coordinates.

### 1.1 — Candle & Swing Point Detection (Foundation Layer)

**What it does:** Takes raw OHLCV candle data and identifies the structural building blocks — swing highs, swing lows, and their relationships.

**Rules from V01 (Structure Mapping):**
- Identify local swing highs and swing lows from price action
- Classify swing relationships: HH, HL, LH, LL
- Use the **SMC definition**, not retail: a Higher High is only valid when inducement is taken + candle closes above the high AND below the inducement
- A swing without any inducement is NOT a valid structural swing
- Impulse candles (no pullback) can act as inducement (V04 rule)

**Input:** Array of OHLCV candles for a given timeframe
**Output:** Array of labeled swing points: `{ index, price, type: "swing_high" | "swing_low", classification: "HH" | "HL" | "LH" | "LL" | "unclassified" }`

### 1.2 — Inducement (IDM) Detection

**What it does:** For each swing, finds the valid inducement point — the first pullback on the left side of the current high/low.

**Rules from V02-V04 (Liquidity & Inducement):**
- Inducement = first valid pullback on the LEFT side of the current swing high
- In bullish trend: look at HIGH's left side for the first pullback low
- In bearish trend: look at LOW's left side for the first pullback high
- If no inducement exists in the current swing → inducement TRANSFERS from the previous swing
- Multiple pullbacks = multiple IDM points; first (most recent) is primary
- Internal candles (inside previous candle's range) do NOT provide inducement
- Taking inducement: wick sweep is sufficient (body close NOT required)

**Input:** Swing points array + OHLCV candles
**Output:** Array of inducement levels: `{ swing_index, idm_price, idm_candle_index, status: "active" | "taken" | "transferred" }`

### 1.3 — Liquidity Pool Identification

**What it does:** Maps where liquidity accumulates — equal highs, equal lows, trendline clusters, and previous swing extremes.

**Rules from V02 (Liquidity):**
- Equal Highs (double top) = buy-side liquidity pool ABOVE
- Equal Lows (double bottom) = sell-side liquidity pool BELOW
- Trendlines create liquidity on the opposite side
- Every pending order cluster, stop loss cluster, and breakeven level = liquidity
- Liquidity sweep = wick sweeps but doesn't close beyond (returns)
- Liquidity grab = body closes beyond, traps breakout traders, then reverses

**Input:** OHLCV candles + swing points
**Output:** Array of liquidity pools: `{ price_level, type: "buy_side" | "sell_side", source: "equal_highs" | "equal_lows" | "swing_extreme" | "trendline", swept: boolean }`

### 1.4 — Break of Structure (BOS) Detection

**What it does:** Identifies valid trend continuation signals where price breaks previous structure.

**Rules from V05 (BOS):**
- **Rule 1:** The price swing MUST have taken inducement from the previous swing
- **Rule 2:** Candle body MUST close ABOVE the previous highest candle's WICK (bullish) or BELOW the previous lowest candle's WICK (bearish)
- If Rule 1 fails (no inducement taken) → BOS is automatically INVALID
- If the first sweeping candle doesn't close beyond the wick → wait for the next candle
- If the next candle's body closes above the sweeping candle's high → BOS is VALID
- If neither closes → it's a liquidity sweep, NOT BOS
- Both rules must be satisfied simultaneously

**Input:** Swing points + IDM status + OHLCV candles
**Output:** Array of BOS events: `{ candle_index, direction: "bullish" | "bearish", broken_swing_index, valid: boolean, invalidation_reason?: string }`

### 1.5 — Change of Character (CHoCH) Detection

**What it does:** Detects potential trend reversals where price breaks major swing points.

**Rules from V07, V09, V10 (CHoCH + Fake CHoCH + Confirmation):**
- CHoCH = break of the current MAJOR swing Higher Low (bullish→bearish) or Lower High (bearish→bullish)
- Candle body MUST close beyond the level (wick alone is insufficient)
- Distinguish from inducement: what retail calls "CHoCH" is often just inducement being taken
- **Fake CHoCH filters (V09):** 3 conditions that invalidate apparent CHoCH
- **CHoCH confirmation (V10):** requires follow-through — new LH after bullish CHoCH, new HL after bearish CHoCH
- CHoCH is timeframe-dependent: smaller TF CHoCH occurs first, larger TF follows
- Climax detection (V23): largest single move in trend = warning signal; CHoCH after climax = strong reversal signal

**Input:** Swing points + BOS events + OHLCV candles
**Output:** Array of CHoCH events: `{ candle_index, direction: "bullish_to_bearish" | "bearish_to_bullish", broken_swing_index, confidence: number, has_climax_confluence: boolean }`

### 1.6 — Fair Value Gap (FVG) Detection

**What it does:** Finds price imbalances (3-candle gaps) that act as magnetic zones for future price.

**Rules from V13 (FVG):**
- FVG = gap measured WICK to WICK: candle[i-1] high to candle[i+1] low (bullish) or candle[i-1] low to candle[i+1] high (bearish)
- Candle color does NOT matter — only the wick-to-wick gap
- **CRITICAL:** In sell trend, FVG must be drawn from the HIGHEST candle's wick → any other = INVALID
- In buy trend, FVG must be drawn from the LOWEST candle's wick → any other = INVALID
- Only trade FVG in trend direction (buy-trend FVG in buy, sell-trend FVG in sell)
- Only trade FVG AFTER a liquidity sweep or grab event
- FVGs coupled with Order Blocks = highest probability
- Target 50% of FVG zone for entry
- After first valid FVG from extreme candle: subsequent FVGs in same swing are also valid

**Input:** OHLCV candles + swing points + trend direction
**Output:** Array of FVGs: `{ start_candle, end_candle, upper_price, lower_price, direction: "bullish" | "bearish", valid: boolean, from_extreme_candle: boolean }`

### 1.7 — Order Block (OB) Detection

**What it does:** Identifies institutional order placement zones — the refined form of institutional order flow (IOF).

**Rules from V14 (Order Blocks):**
- Bearish OB = the bullish candle(s) immediately before a sell move
- Bullish OB = the bearish candle(s) immediately before a buy move
- **Rule 1:** OB candle MUST have swept liquidity of the previous candle or previous high/low
- **Rule 2:** Below (or above) that candle's wick, a price imbalance (FVG) MUST exist
- Both rules required simultaneously — one without the other = INVALID
- 50% rule: when price returns to OB, candle body should NOT close beyond OB's 50% level
- OB in main trend direction = HIGH PROBABILITY
- Bullish trend: OB below inducement = high probability
- Bearish trend: OB above inducement = high probability
- OB that is part of inducement/engineered liquidity = SMART MONEY TRAP → invalid
- OB from candles inside previous candle's range (invalid pullback) = INVALID

**Input:** OHLCV candles + swing points + FVGs + liquidity events
**Output:** Array of OBs: `{ candle_range: [start, end], upper_price, lower_price, direction: "bullish" | "bearish", valid: boolean, has_fvg: boolean, swept_liquidity: boolean, is_trap: boolean }`

### 1.8 — Premium & Discount Zone Calculator

**What it does:** Determines whether current price is in premium (expensive, sell zone) or discount (cheap, buy zone) relative to the current range.

**Rules from V12 (Premium/Discount):**
- Range = current swing high to current swing low
- 50% = equilibrium line
- Above 50% = premium zone (look for sells in bearish trend)
- Below 50% = discount zone (look for buys in bullish trend)
- OBs/FVGs in discount during buy trend = high probability
- OBs/FVGs in premium during sell trend = high probability
- Entry at or beyond 50% improves risk:reward significantly

**Input:** Current swing high and low prices + current price
**Output:** `{ equilibrium: number, zone: "premium" | "discount" | "equilibrium", depth_percentage: number }`

### 1.9 — Session & Kill Zone Awareness

**What it does:** Adds time-of-day context — when institutional activity is highest.

**Rules from V16 (Sessions):**
- Asian session: low volatility, range building, liquidity accumulation
- London Open (kill zone): first major sweep/displacement
- NY Open (kill zone): continuation or counter-move
- London Close: reversal risk, profit-taking
- Best setups: London/NY kill zone entries when structure aligns
- Avoid trading during news events without clear structure

**Input:** Current timestamp (UTC)
**Output:** `{ session: "asian" | "london" | "ny" | "london_close", is_kill_zone: boolean, volatility_expectation: "low" | "medium" | "high" }`

### 1.10 — Entry Signal Generator (The Master Checklist)

**What it does:** Combines ALL detectors above into the final signal using the V23 Master Trading Checklist.

**The V23 Master Checklist (adapted for W1→D1→H4→M15):**
```
1. IDENTIFY TREND → W1 swing point classifier (1.1) → HH/HL = buy bias, LH/LL = sell bias
2. MARK STRUCTURE → D1 BOS/CHoCH, swing points, Higher Lows / Lower Highs
3. FIND IDM → D1/H4: after each BOS, locate new inducement (1.2)
4. FIND ZONE → H4: below IDM → OB that swept liquidity + has FVG = sell/buy zone (1.6 + 1.7)
5. CHECK CONTEXT → premium/discount on D1 range (1.8) + session awareness (1.9)
6. WAIT FOR TAP → price must reach the H4 identified zone
7. CONFIRM ENTRY on M15:
   - MSS (Market Structure Shift) after zone tap
   - SCOB: only at POI zone tap or after liquidity sweep
   - Valid pullback break after zone tap
8. SET SL → 2-4 pips beyond the H4 sell/buy zone (never at immediate extreme)
9. SET TP → previous H4 high/low from which market took inducement
10. MONITOR FOR CLIMAX → if largest move in D1 trend → flag as caution
11. DETECT CHoCH → if D1 HL/LH breaks → direction switch protocol
12. COUNTER-TREND → only when D1 BOS + IDM close + FVG all present → TP first opposing zone
```

**Input:** All detector outputs + current price + timeframe
**Output:** Trading signal: `{ direction, entry_price, stop_loss, take_profit, risk_reward_ratio, confidence_score, confluences: string[], timeframe, pattern_type }`

---

## Phase 2: Historical Validation (Does It Actually Work?)

### Goal
Run the Phase 1 detectors on historical candlestick data and verify they find patterns that match what the videos teach. This is **NOT backtesting profitability yet** — it's validating detection accuracy.

### Why This Is Phase 2
Before connecting live data or measuring profit/loss, we need to confirm the detectors work correctly. A BOS detector that fires on every candle is useless. An FVG detector that misses valid FVGs from extreme candles is broken.

### 2.1 — Acquire Historical Data

**What:** Download historical OHLCV candlestick data for major forex pairs and crypto.

**Pairs to test:** BTCUSDT (primary). Later: ETHUSDT, other crypto, forex, indices.

**Timeframes:** M15, H4, D1, W1 (our confirmed TF hierarchy).

**Data range:** Minimum 6 months for each pair/timeframe combination.

**Source:** Binance REST API provides free historical klines data (crypto). For forex, free sources like OANDA practice API or CSV downloads.

### 2.2 — Detection Rate Analysis

**What:** Run each detector individually and measure detection frequency.

**Metrics per detector:**
- **Detection rate:** How many patterns found per 1000 candles?
- **Trend alignment:** What percentage of detected patterns align with the current trend direction?
- **Zone distribution:** Are detected OBs/FVGs distributed in premium/discount correctly?
- **False positive indicators:** How many detected BOS events are immediately reversed? (Indicates fake BOS that wasn't filtered)

**Expected ranges (sanity checks):**
- BOS should fire less frequently than raw "price breaks high/low" — our rules filter aggressively
- FVGs should only appear from extreme candles — not on every 3-candle sequence
- OBs require BOTH liquidity sweep AND FVG — double filter means fewer but higher quality

### 2.3 — Visual Validation

**What:** Generate charts that overlay detected patterns on candlestick data. Compare with examples from the training videos.

**For each pattern type, generate:**
- 10 "detected" examples — do they look like what the videos taught?
- 10 "rejected" examples — patterns that almost qualified but were filtered out. Were they correctly rejected?

**Comparison method:** Our knowledge summaries contain real chart examples (EUR/USD, GBP/USD, AUD/USD, Gold). The visual validation should show similar-looking patterns.

### 2.4 — Multi-Timeframe Alignment Check

**What:** Verify that the hierarchical structure works — H4 trend direction should constrain M15 entries, M15 zone taps should trigger M5 entry checks.

**Test:**
- Run H4 trend detection → does it correctly identify major swing direction?
- When W1 is bullish, do D1 BOS events predominantly confirm bullish continuation?
- When D1/H4 identifies a zone, do M15 structure shifts occur at zone taps?

### 2.5 — Detection Accuracy Report

**Deliverable:** A quantitative report showing:
- Each detector's hit rate, trend alignment, and rejection rate
- Visual examples of correct detections and correct rejections
- Any detectors that need tuning (thresholds, lookback periods, etc.)
- Confidence that the rule engine matches the training material

---

## Phase 3: Backtesting (Does It Make Money?)

### Goal
For every signal the system generates on historical data, measure what happened next. Build a success/failure dataset that teaches the system **selectivity** — which setups to prioritize and which to skip.

### Why This Is Phase 3
Phase 2 validates "does the detector find the right patterns?" Phase 3 asks "do those patterns lead to profitable trades?" This is where the system goes from pattern recognition to signal quality scoring.

### 3.1 — Signal Generation on Historical Data

**What:** Run the full Phase 1 pipeline (all detectors + Master Checklist) on 6+ months of historical data across all pairs/timeframes.

**For each generated signal, record:**
- Entry price, SL, TP (as calculated by the system)
- Direction (buy/sell)
- All confluences present (BOS, IDM, FVG, OB, premium/discount, session, climax warning, etc.)
- Timeframe combination (e.g., H4 trend + M15 zone + M5 entry)
- Timestamp and pair

### 3.2 — Outcome Labeling

**What:** For each historical signal, track what ACTUALLY happened.

**Outcomes to measure:**
- Did price hit TP first or SL first?
- How far did price move in the intended direction before reversing?
- How long did the trade take (time to TP or SL)?
- Maximum adverse excursion (MAE) — how far against you before hitting TP?
- Maximum favorable excursion (MFE) — how far in your favor?

**Labels:**
- **WIN:** TP hit before SL
- **LOSS:** SL hit before TP
- **PARTIAL:** Price moved favorably but didn't reach TP, then reversed
- **TIMEOUT:** Neither TP nor SL hit within reasonable time window

### 3.3 — Confluence vs Outcome Analysis (Contrastive Learning Dataset)

**What:** The core insight — which combinations of confluences produce wins vs losses?

**Analysis dimensions:**
- **Single-factor:** Win rate for signals with FVG vs without FVG. Win rate with OB vs without. Etc.
- **Pair-factor:** Win rate for FVG+OB vs FVG alone vs OB alone
- **Context-factor:** Win rate in premium/discount alignment. Win rate during kill zones vs off-hours.
- **Multi-TF factor:** Win rate for W1+D1+H4 alignment vs H4 alone
- **Climax factor:** Win rate when climax warning is active vs not

**Expected discoveries:**
- Signals with more confluences should have higher win rates
- Kill zone entries should outperform off-hours entries
- Multi-TF aligned signals should have better R:R than single-TF
- Counter-trend signals should have lower win rate but the wins should still hit first opposing zone

### 3.4 — Confidence Scoring Model

**What:** Build a scoring function that predicts signal quality based on confluences.

**Approach:**
- Each confluence adds weight: BOS (base), +FVG, +OB, +premium/discount alignment, +kill zone, +multi-TF, etc.
- Weights derived from the Phase 3.3 analysis (data-driven, not guessed)
- Climax warning reduces confidence
- Counter-trend signals capped at lower max confidence

**Output:** Each signal gets a confidence score (0-100) and a grade (A/B/C/D)
- A-grade: 3+ confluences, trend-aligned, kill zone, multi-TF confirmation
- B-grade: 2 confluences, trend-aligned
- C-grade: 1 confluence or weak alignment
- D-grade: counter-trend or climax warning active

### 3.5 — Backtest Performance Report

**Deliverable:**
- Overall win rate, average R:R, profit factor
- Win rate by grade (A/B/C/D)
- Win rate by pair, timeframe, session
- Best and worst performing confluence combinations
- Recommended minimum grade for live trading (likely A or B only)
- Equity curve visualization

---

## Phase 4: Live Data Integration (Real-Time)

### Goal
Connect to live market data via Binance WebSocket, run the validated detectors in real-time, and generate signals in shadow mode (detect and log, no actual trading).

### Why This Is Phase 4
Only after historical validation (Phase 2) and backtesting (Phase 3) confirm the system works should we add the complexity of real-time data. Live data introduces latency, partial candles, connection handling, and data integrity challenges.

### 4.1 — WebSocket Connection & Candle Aggregation

**What:** Connect to Binance WebSocket and build real-time OHLCV candle streams for multiple timeframes.

**Data streams needed:**
- Kline/candlestick streams for BTCUSDT: `btcusdt@kline_15m`, `btcusdt@kline_4h`, `btcusdt@kline_1d`, `btcusdt@kline_1w`
- Each stream provides: open time, OHLCV, close time, number of trades
- Multiple timeframes run simultaneously for multi-TF analysis (W1→D1→H4→M15)

**Architecture considerations:**
- WebSocket reconnection handling (Binance drops connections after 24h)
- Candle completion detection (only process CLOSED candles, ignore partial)
- Data buffer: keep rolling window of last N candles per timeframe for detector context
- Heartbeat/ping-pong to maintain connection

### 4.2 — Real-Time Detection Pipeline

**What:** As each candle closes, run the Phase 1 detection pipeline on the updated candle buffer.

**Pipeline flow (per candle close):**
```
Candle closes on M15
  → Update M15 buffer
  → Run swing detection on M15
  → Run IDM detection on M15
  → Check if any D1/H4 zone is near current price
  → If zone tap detected:
      → Run BOS/CHoCH check on M15
      → Run FVG/OB validation
      → Run entry signal generator (Master Checklist)
      → If signal generated → log + notify

Higher TF candle closes (H4/D1/W1)
  → Update respective buffer
  → Re-run structure analysis on that TF
  → Update active zones, trend direction, IDM levels
```

**Performance requirement:** Full pipeline must complete before next candle opens. For M15, that's well within the 15-minute window.

### 4.3 — Shadow Mode (Log & Learn)

**What:** The system detects patterns and generates signals but does NOT execute trades. Instead, it logs every signal and tracks what happens.

**Logged for each shadow signal:**
- All signal details (entry, SL, TP, confluences, confidence score)
- Timestamp of detection
- What price did after detection (tracked until TP or SL would have been hit)
- Outcome: would the trade have won or lost?

**Duration:** Run shadow mode for minimum 2-4 weeks before considering any live execution.

**Success criteria:** Shadow mode win rate and R:R should be consistent with Phase 3 backtest results. If shadow mode performs significantly worse than backtest, there's a bug in the live pipeline.

### 4.4 — Alerting System

**What:** When a high-confidence signal is generated, send a notification.

**Channels (options):**
- Console/terminal output
- Telegram bot notification
- Web dashboard (future)

**Alert content:**
- Pair, timeframe, direction
- Entry, SL, TP prices
- Confidence score and grade
- Confluences present
- Screenshot/chart reference (optional)

---

## Phase 5: ML Enhancement (Learning From Experience)

### Goal
Once the rule-based system has generated sufficient labeled data (signals + outcomes), train ML models that learn **selectivity** — predicting which setups have the highest probability of success.

### Why This Is Phase 5 (Last)
ML needs labeled data to learn from. Phases 1-4 generate that data. Training ML on insufficient or unlabeled data produces garbage. This is why ChatGPT's "Week 3: use ML as filter" is premature — you need thousands of labeled signals first.

### 5.1 — Feature Engineering

**What:** Convert each signal into a feature vector that ML can process.

**Features per signal:**
- **Structural:** Trend strength (how many consecutive HH/HL), distance from last CHoCH, BOS count in current trend
- **Zone quality:** FVG width relative to ATR, OB size relative to ATR, distance from premium/discount equilibrium
- **Confluence count:** Binary flags for each confluence type present
- **Temporal:** Session (Asian/London/NY), day of week, time since last signal
- **Volatility:** ATR value, ATR percentile (is current vol high or low vs recent history)
- **Multi-TF alignment:** Binary flags for W1 trend match, D1 zone active, H4 refined zone, M15 entry structure shift
- **Liquidity context:** Distance to nearest liquidity pool, number of unswept pools nearby
- **Climax indicator:** Is climax warning active? How large was the climax move vs avg?

### 5.2 — Contrastive Model Training

**What:** Train a model that learns WHY some signals succeed and others fail.

**Approach:** Binary classification — given a feature vector, predict WIN or LOSS.

**Training data:** All signals from Phase 3 (backtest) + Phase 4 (shadow mode), each labeled with outcome.

**Model types to evaluate:**
- Gradient boosted trees (XGBoost/LightGBM) — interpretable, works well with structured features
- Logistic regression — baseline, fully interpretable
- Small neural network — if dataset is large enough (10K+ signals)

**Key insight:** The model doesn't replace the rule engine. It learns which COMBINATIONS of features predict success that simple rules can't capture. For example: "FVG+OB in London kill zone with H4 trend alignment after 2 BOS events" might have 80% win rate, while "FVG alone in Asian session against H4 trend" might have 30%.

### 5.3 — Confidence Score Refinement

**What:** Replace the rule-based confidence score (Phase 3.4) with the ML model's probability output.

**Before ML:** Confidence = weighted sum of confluences (hand-tuned weights)
**After ML:** Confidence = model's predicted win probability (data-driven weights)

This should improve signal selectivity — the system recommends fewer but better trades.

### 5.4 — Continuous Learning Loop

**What:** As the live system generates more signals and outcomes, periodically retrain the ML model.

**Loop:**
```
Live signal generated → Outcome tracked →
  Added to training dataset →
  Model retrained periodically (weekly/monthly) →
  Updated confidence scores deployed
```

**Guard rails:**
- Never auto-deploy a retrained model without validation on held-out data
- Track model drift: if live performance diverges from training performance, flag for review
- Keep human oversight: the rule engine (Phase 1) always runs; ML only adjusts confidence scores

---

## What Stays Constant Across All Phases

These are ICT methodology constants that don't change:

| Constant | Value | Source |
|----------|-------|--------|
| Fibonacci retracement levels | 0.5, 0.618, 0.705, 0.786 | V12, V15 (Premium/Discount + MDS) |
| OB 50% rule | Price body shouldn't close beyond OB's 50% | V14 (Order Blocks) |
| SL placement | 2-4 pips beyond zone (not at immediate extreme) | V23 (Master Checklist) |
| Counter-trend TP | First opposing zone only | V21 (Counter-Trend) |
| Multi-TF pairs | W1→D1→H4→M15 | V20 (Multi-TF), adapted for crypto |
| Entry types | MSS, SBC, SCOB, valid pullback break | V17-V19 (Entry Techniques) |
| SCOB validity | Only at POI zone tap or after liquidity sweep | V19, V23 |
| Kill zones | London Open, NY Open | V16 (Sessions) |

---

## Dependency Graph

```
Phase 1: Mathematical Formalization
│   1.1 Swing Points ──────────────────┐
│   1.2 Inducement ─────────────────┐  │
│   1.3 Liquidity Pools ────────┐   │  │
│   1.4 BOS ◄───────────────────┤───┤──┤
│   1.5 CHoCH ◄─────────────────┤───┤──┤
│   1.6 FVG ◄───────────────────┤───┘  │
│   1.7 Order Block ◄───────────┤──────┘
│   1.8 Premium/Discount ◄──────┘
│   1.9 Session/Kill Zone (independent)
│   1.10 Master Checklist ◄── ALL of above
│
▼
Phase 2: Historical Validation
│   2.1 Acquire data (independent)
│   2.2 Detection rate analysis ◄── Phase 1 complete
│   2.3 Visual validation ◄── Phase 2.2
│   2.4 Multi-TF alignment check ◄── Phase 2.2
│   2.5 Report ◄── Phase 2.2-2.4
│
▼
Phase 3: Backtesting
│   3.1 Signal generation ◄── Phase 2 validated
│   3.2 Outcome labeling ◄── Phase 3.1
│   3.3 Confluence analysis ◄── Phase 3.2
│   3.4 Confidence scoring ◄── Phase 3.3
│   3.5 Report ◄── Phase 3.1-3.4
│
▼
Phase 4: Live Data Integration
│   4.1 WebSocket connection ◄── Phase 3 profitable
│   4.2 Real-time pipeline ◄── Phase 1 + 4.1
│   4.3 Shadow mode ◄── Phase 4.2
│   4.4 Alerting ◄── Phase 4.3
│
▼
Phase 5: ML Enhancement
│   5.1 Feature engineering ◄── Phase 3+4 data
│   5.2 Model training ◄── Phase 5.1
│   5.3 Confidence refinement ◄── Phase 5.2
│   5.4 Continuous learning ◄── Phase 5.3
```

---

## Success Metrics

| Phase | Success Criteria |
|-------|-----------------|
| Phase 1 | All 10 detectors produce output on sample data. No crashes. |
| Phase 2 | Detection rates within expected ranges. Visual spot-checks pass. Multi-TF alignment works. |
| Phase 3 | Positive profit factor (>1.0) on 6-month backtest. A-grade signals win rate >60%. Average R:R >1:2. |
| Phase 4 | Shadow mode results within 10% of backtest results after 2+ weeks. No missed signals, no data gaps. |
| Phase 5 | ML confidence score improves signal selectivity — higher win rate when filtering by ML score vs rule-based score. |

---

## Decisions (Confirmed)

| Question | Decision |
|----------|----------|
| **Pairs** | BTCUSDT initially → expand to more crypto, forex, indices later |
| **Timeframes** | W1 (macro trend) → D1 (structure + zones) → H4 (refined zones + IDM) → M15 (entry) |
| **Tech Stack** | Python 3.11+ / FastAPI (backend :8000) + Next.js / TradingView lightweight-charts (frontend :3000) |
| **Execution** | Alerts + paper trading → eventually live after rigorous backtesting |
| **Deployment** | Localhost only (localhost:3000 + localhost:8000) → domain hosting far future |
| **Frontend** | Dashboard with candlestick charts + signal overlays on localhost:3000 |
| **Scalability** | Knowledge base grows with more training videos → detectors improve without restructuring |

### Tech Stack Detail

**Backend (localhost:8000):**
- Python 3.11+ — data/ML ecosystem, existing scripts
- FastAPI — async, WebSocket support, auto-docs at /docs
- pandas + numpy — financial data processing
- python-binance — official Binance WebSocket + REST SDK
- SQLite — zero-setup local DB (upgrade to PostgreSQL when deploying to domain)
- Custom backtesting engine — ICT rules too specific for generic frameworks

**Frontend (localhost:3000):**
- Next.js (React) — production-ready when domain time comes
- TradingView lightweight-charts — candlestick charts with pattern overlay markers
- Tailwind CSS — fast responsive styling
- shadcn/ui — component library
- WebSocket client — real-time signal updates from backend

### Crypto-Specific Adaptations

**Session/Kill Zone replacement for 24/7 crypto:**
- No traditional London/NY forex sessions
- Crypto high-activity windows: US equity open (~13:30 UTC), US close (~20:00 UTC), weekly candle close (Sunday midnight UTC), CME BTC futures open/close
- Asian hours = lower volatility (similar to forex)
- All structural ICT concepts (BOS, CHoCH, FVG, OB, IDM, liquidity) are fractal and instrument-agnostic

### Multi-Timeframe Hierarchy

| Role | Timeframe | What It Does |
|------|-----------|-------------|
| Macro trend direction | W1 (Weekly) | HH/HL = bullish bias, LH/LL = bearish bias |
| Structure + key zones | D1 (Daily) | BOS/CHoCH, major OBs/FVGs, swing points |
| Refined zones + IDM | H4 (4-Hour) | Inducement levels, refined entry zones, premium/discount |
| Entry confirmation | M15 (15-Min) | MSS, SCOB, valid pullback break, precise SL/TP |

### Scalability Design

The system is designed so adding more training videos later:
- New `*_knowledge_base.json` files add to the knowledge foundation
- Detector rules can be refined/expanded based on new teachings
- Confidence scoring model retrains with more data
- No structural changes needed — just knowledge enrichment
