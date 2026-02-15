# TradingMamba — System Architecture (Quant Edition)

> **Version:** 2.0 (Quant Mode)
> **Branch:** TradingMambaHindi
> **Codebase:** ~22,000 lines (13,200 backend Python + 8,800 frontend TypeScript)
> **Last Updated:** February 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Knowledge Extraction Pipeline](#2-knowledge-extraction-pipeline)
3. [Backend Architecture](#3-backend-architecture)
4. [Detection Engine — 11 SMC Detectors](#4-detection-engine--11-smc-detectors)
5. [Signal Generation — The Master Checklist](#5-signal-generation--the-master-checklist)
6. [Quant Scoring Engine — 4-Layer System](#6-quant-scoring-engine--4-layer-system)
7. [Data Services & External APIs](#7-data-services--external-apis)
8. [API Endpoints](#8-api-endpoints)
9. [Frontend Architecture](#9-frontend-architecture)
10. [Database Schema](#10-database-schema)
11. [Data Flow Diagrams](#11-data-flow-diagrams)
12. [Configuration Reference](#12-configuration-reference)

---

## 1. System Overview

TradingMamba is a **dual-mode institutional trading system** that detects Smart Money Concepts (SMC/ICT) patterns on live BTCUSDT data and generates graded trading signals. It operates in two modes:

- **SMC Mode:** Pure structural analysis using 11 detectors derived from 23 Hindi trading education videos
- **QUANT Mode:** Same SMC pipeline + a 4-layer quantitative scoring overlay powered by 10+ institutional data sources

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        KNOWLEDGE FOUNDATION                             │
│   23 Hindi YouTube Videos → whisper.cpp → Claude Code Expert Analysis   │
│   → 23 Structured Knowledge Bases (JSON) → Detection Rules             │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           BACKEND (Python)                              │
│  ┌───────────────┐  ┌──────────────────┐  ┌─────────────────────────┐  │
│  │ Data Fetcher  │  │  Detection Engine │  │   Quant Engine          │  │
│  │ (Binance API) │→ │  (11 Detectors)  │→ │  (4-Layer Scoring)      │  │
│  │ + Futures     │  │  7 Timeframes    │  │  + 10 Data Sources      │  │
│  └───────────────┘  └──────────────────┘  └─────────────────────────┘  │
│  ┌───────────────┐  ┌──────────────────┐  ┌─────────────────────────┐  │
│  │ Signal Store  │  │  Demo Account    │  │   Telegram Bot          │  │
│  │ (Lifecycle)   │  │  (SQLite)        │  │   + Signal Monitor      │  │
│  └───────────────┘  └──────────────────┘  └─────────────────────────┘  │
│                                                                         │
│  FastAPI Server (port 8000) — 20+ REST API Endpoints                   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ HTTP + WebSocket
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (TypeScript)                           │
│  Next.js 14 + TailwindCSS + lightweight-charts                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ 7 Tabs: Live | Backtest | Performance | Signals | Demo |          │ │
│  │         Market Intel | Quant Analysis (quant mode only)           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│  Candlestick Chart with 13+ Overlay Types                              │
│  Dark Glassmorphism UI — Inter Font — Auto-Refresh Polling             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Python 3.11+, FastAPI, uvicorn | REST API server, async data fetching |
| **Frontend** | Next.js 14, TypeScript, TailwindCSS | Reactive UI with server-side rendering |
| **Charts** | lightweight-charts (TradingView) | Candlestick chart with custom overlays |
| **Database** | SQLite (aiosqlite) | Demo account, trade history, signals |
| **Data** | Binance REST + WS, Deribit, CFTC, etc. | Market data from 10+ free APIs |
| **Transcription** | whisper.cpp (Metal GPU) | Hindi audio → text transcription |
| **Knowledge** | Claude Code Expert Analysis | Structured concept extraction |
| **Alerts** | python-telegram-bot | Real-time signal notifications |

---

## 2. Knowledge Extraction Pipeline

The entire detection engine is built from domain knowledge extracted from 23 Hindi SMC/ICT YouTube videos. This section documents how raw video content becomes actionable trading rules.

### Pipeline Stages

```
┌──────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  YouTube     │    │  whisper.cpp  │    │  Claude Code   │    │  Knowledge   │
│  Video (Hi)  │ →  │  Transcribe   │ →  │  Expert        │ →  │  Base (JSON) │
│  .wav audio  │    │  Hindi → Text │    │  Analysis      │    │  + Summary   │
└──────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
       │                   │                    │                     │
       ▼                   ▼                    ▼                     ▼
  data/audio/        data/transcripts/    data/video_frames/    data/audio_first_training/
  {id}.wav           {id}.json            {id}/*.png           {id}_knowledge_base.json
                                                               {id}_knowledge_summary.md
```

### Stage 1: Audio Download

- **Tool:** `yt-dlp` via `scripts/download_audio.py`
- **Format:** WAV (16kHz mono for Whisper compatibility)
- **Storage:** `data/audio/{video_id}.wav`

### Stage 2: Frame Extraction

- **Tool:** FFmpeg via `scripts/extract_frames.py`
- **Method:** Extract key frames at teaching transitions (scene changes, whiteboard updates)
- **Storage:** `data/video_frames/{video_id}/frame_*.png`

### Stage 3: Transcription

- **Tool:** `whisper.cpp` with Metal GPU acceleration
- **Model:** `ggml-large-v3-turbo.bin` (quantized for Apple Silicon)
- **Language:** Hindi (`TRANSCRIPTION_LANGUAGE=hi`)
- **Binary:** `/opt/homebrew/bin/whisper-cli`
- **Output:** `data/transcripts/{video_id}.json` — timestamped segments
- **Critical:** Never run 2+ whisper.cpp instances simultaneously on Metal GPU

### Stage 4: Claude Code Expert Analysis

- **Input:** Transcript + video frames
- **Process:** Claude Code reads Hindi transcript natively, views key frames with vision, identifies ICT/SMC concepts with expert-level understanding
- **Output per video:**
  - `{video_id}_knowledge_base.json` — structured concepts with `llm_summary`, `key_rules`, `visual_evidence`, `statistics`
  - `{video_id}_knowledge_summary.md` — human-readable summary

### Stage 5: Rule Compilation

Knowledge bases are compiled into detection rules implemented across 11 detectors. Each detector references specific video concepts:

| Video | Key Concept | Detector |
|-------|------------|----------|
| V01 | Structure Mapping (HH/HL/LH/LL) | `swing_detector.py` |
| V02 | Liquidity & Inducement | `liquidity.py`, `inducement.py` |
| V03-V04 | Valid Pullback, IDM Traps | `inducement.py` |
| V05-V06 | Break of Structure Rules | `bos_detector.py` |
| V07, V09-V10 | CHoCH, Fake CHoCH, Confirmation | `choch_detector.py` |
| V08 | High Probability IDM | `inducement.py` |
| V11-V12 | Price Cycles, Premium/Discount | `price_cycle_detector.py`, `premium_discount.py` |
| V13 | Fair Value Gap | `fvg_detector.py` |
| V14 | Valid Order Blocks | `order_block.py` |
| V15 | Million Dollar Setup (MSS) | `choch_detector.py`, `signal_generator.py` |
| V16 | Sessions & Kill Zones | `session.py` |
| V17 | AMD Pattern, Entry Technique | `amd_detector.py`, `signal_generator.py` |
| V18 | QML & POI Zones | `signal_generator.py` |
| V19 | SCOB Entry | `signal_generator.py` |
| V20 | Multi-Timeframe Analysis | `engine.py` |
| V21 | Counter-Trend Trading | `signal_generator.py` |
| V22 | Wyckoff/SBC + VSA | `vsa_detector.py`, `signal_generator.py` |
| V23 | Master Trading Checklist | `signal_generator.py` |

### Knowledge Base Schema

```json
{
  "generation_method": "Claude Code expert analysis",
  "metadata": {
    "training_type": "claude_code_expert",
    "source_language": "hi",
    "video_id": "UIRBfCT1kI4",
    "title": "Structure Mapping"
  },
  "concepts": [
    {
      "name": "Swing Classification",
      "llm_summary": "Definition: ... Rules: ... Identification: ...",
      "key_rules": ["Rule 1: ...", "Rule 2: ..."],
      "visual_evidence": ["frame_001.png: shows HH/HL sequence"],
      "statistics": {
        "teaching_time_seconds": 180,
        "word_count": 450
      },
      "teaching_types": ["diagram", "live_chart", "annotation"]
    }
  ]
}
```

---

## 3. Backend Architecture

### Directory Structure

```
backend/
├── app/
│   ├── config.py                      # All constants, thresholds, API keys
│   ├── main.py                        # FastAPI app, 20+ endpoints, lifespan
│   ├── models.py                      # 25+ data models (dataclasses + enums)
│   │
│   ├── core/                          # SMC Detection Engine (11 detectors)
│   │   ├── engine.py                  # Multi-TF orchestrator
│   │   ├── swing_detector.py          # V01: HH/HL/LH/LL
│   │   ├── inducement.py             # V02-V04, V08: IDM detection
│   │   ├── bos_detector.py            # V05-V06: Break of Structure
│   │   ├── choch_detector.py          # V07, V09-V10, V15: CHoCH + MSS
│   │   ├── liquidity.py              # V02: Liquidity pools
│   │   ├── fvg_detector.py            # V13: Fair Value Gaps
│   │   ├── order_block.py             # V14: Order Blocks
│   │   ├── premium_discount.py        # V12: Premium/Discount zones
│   │   ├── session.py                 # V16: Kill zones
│   │   ├── vsa_detector.py            # V22: Volume Spread Analysis
│   │   ├── amd_detector.py            # V17: AMD pattern
│   │   ├── price_cycle_detector.py    # V11: Price delivery cycles
│   │   ├── signal_generator.py        # V17-V23: Master checklist (1,100 lines)
│   │   ├── signal_store.py            # Signal lifecycle tracking
│   │   ├── deep_analysis.py           # Multi-TF signal explanation
│   │   └── futures_confluence.py      # OI/funding as signal confluence
│   │
│   ├── quant/                         # Quant Scoring Engine (4 layers)
│   │   ├── engine.py                  # Orchestrator: apply_quant_layer_sync()
│   │   ├── alpha_model.py             # Layer 1: 11-component conviction score
│   │   ├── microstructure.py          # Layer 2: VPIN, liquidations, flow
│   │   ├── risk_model.py              # Layer 3: ATR, volatility, position sizing
│   │   ├── execution_model.py         # Layer 4: Session timing, VWAP, entry quality
│   │   └── scoring.py                # Combined scorer, grade modification
│   │
│   └── services/                      # Data & Infrastructure
│       ├── data_fetcher.py            # Binance klines (spot)
│       ├── futures_data.py            # Binance Futures (OI, funding, ratios)
│       ├── quant_data.py              # TIER 2+3 data (8 sources, Black-Scholes)
│       ├── liquidation_ws.py          # Binance forceOrder WebSocket
│       ├── database.py                # SQLite: accounts, trades, signals
│       ├── history_db.py              # Historical data SQLite cache
│       ├── backtester.py              # Sliding-window backtest engine
│       ├── signal_monitor.py          # Auto-create pending trades from signals
│       ├── position_monitor.py        # Auto-close at SL/TP via price check
│       └── telegram_bot.py            # Telegram alert bot
```

### Application Lifecycle

```python
# main.py lifespan (startup → shutdown)

async def lifespan(app):
    # ── STARTUP ──
    1. init_db("smc")          # Initialize SMC demo account (SQLite)
    2. init_db("quant")        # Initialize QUANT demo account (SQLite)
    3. TelegramAlertBot.start() # Start Telegram bot (if token configured)
    4. SignalMonitor.start()    # Poll for new signals → create pending trades (30s)
    5. PositionMonitor.start()  # Check SL/TP hits for open trades (10s)
    6. LiquidationWSManager.start()  # Connect Binance forceOrder WebSocket

    yield  # ── APP RUNNING ──

    # ── SHUTDOWN ──
    7. LiquidationWSManager.stop()
    8. SignalMonitor.stop()
    9. PositionMonitor.stop()
    10. TelegramAlertBot.stop()
```

### Data Models (25+)

All models are Python `dataclass` objects in `models.py`:

```
Enums:
  Direction (bullish/bearish), SwingType, SwingClassification (HH/HL/LH/LL),
  TrendState, IDMStatus, LiquidityType, LiquiditySource, LiquidityEvent,
  ZoneType, EntryMethod (MSS/SCOB/SBC/PULLBACK), SignalGrade (A/B/C/D),
  MSSGrade (none/standard/a_plus/a_plus_plus), VolatilityRegime,
  AMDPhase, PricePhase, TradeOutcome

Core Models:
  Candle, SwingPoint, Inducement, LiquidityPool, BOS, CHoCH,
  FVG, OrderBlock, PremiumDiscount, Session, AMDPattern,
  PriceCycleEvent, TradingSignal

Quant Models:
  QuantScore, QuantContext

Validation/Backtest:
  DetectorMetrics, ValidationReport, TradeRecord
```

---

## 4. Detection Engine — 11 SMC Detectors

The engine runs detectors in strict sequence — each builds on the previous. This is the core analysis pipeline executed per timeframe.

### Pipeline Flow

```
Candles (OHLCV)
    │
    ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 1: SWING DETECTION                                          │
│ detect_and_classify(candles, lookback)                            │
│ → SwingPoint[] (HH/HL/LH/LL) + TrendState                      │
│ Rules: Fractal detection (N left + N right), alternation enforce  │
│ Source: V01 — Structure Mapping                                   │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 2: INDUCEMENT (IDM)                                         │
│ detect_and_validate(candles, swings)                              │
│ → Inducement[] + validated SwingPoint[]                          │
│ Rules: First pullback on LEFT of swing; major vs minor;           │
│        transferred IDM = impulse; body-close vs wick sweep        │
│ Source: V02-V04, V08                                              │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 3: LIQUIDITY POOLS                                          │
│ detect_all_liquidity(candles, swings, inducements)                │
│ → LiquidityPool[] (equal highs/lows, swing extremes, IDM levels) │
│ Rules: 0.1% price tolerance; sweep (wick) vs grab (body close)   │
│ Source: V02                                                       │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 4: BREAK OF STRUCTURE (BOS)                                 │
│ detect_bos(candles, swings, inducements)                          │
│ → BOS[] (trend continuation events)                              │
│ Rules: IDM MUST be taken before break; body close beyond wick     │
│ Source: V05-V06                                                   │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 5a: VSA ABSORPTION                                          │
│ detect_vsa_absorptions(candles)                                   │
│ → VsaAbsorption[] (ultra-high volume institutional flow)         │
│ Source: V22 — Wyckoff SBC                                         │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 5b: CHANGE OF CHARACTER (CHoCH)                             │
│ detect_choch() + filter_fake_choch()                              │
│ → CHoCH[] (trend reversal events)                                │
│ Rules: 3 fake filters (weak swing, equal H/L, transferred IDM);   │
│        4-rule swing confirmation; 2-rule sweep confirmation       │
│ Source: V07, V09-V10                                              │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 6: FAIR VALUE GAPS (FVG)                                    │
│ detect_all_fvgs(candles, swings, trend)                           │
│ → FVG[] (price imbalance zones)                                  │
│ Rules: Wick-to-wick gap; extreme candle rule for validity;        │
│        mitigation = price fills entire gap                        │
│ Source: V13                                                       │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 5c: MSS CLASSIFICATION                                     │
│ classify_mss(choch_events, liquidity_pools, candles, fvgs)        │
│ → Updates CHoCH[] with is_mss, mss_grade                         │
│ Rules: Liquidity swept + body expansion + V25 grading             │
│        (Standard / A+ / A++)                                      │
│ Source: V15, V25                                                  │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 7: ORDER BLOCKS (OB)                                        │
│ detect_all_order_blocks(candles, swings, fvgs, inducements, trend)│
│ → OrderBlock[] (institutional order zones)                       │
│ Rules: Must sweep previous candle liquidity + adjacent FVG;       │
│        50% midpoint mitigation rule                               │
│ Source: V14                                                       │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 8: PREMIUM / DISCOUNT                                       │
│ calculate_premium_discount(swings, price, bos, choch, trend)      │
│ → PremiumDiscount (zone + Fibonacci levels)                      │
│ Rules: Anchored to BOS/CHoCH structural leg; Fib qualification    │
│ Source: V12                                                       │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 9: SESSION / KILL ZONE                                      │
│ get_current_session(timestamp)                                    │
│ → Session (name + is_kill_zone + volatility_expectation)         │
│ Source: V16                                                       │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 10: AMD PATTERN                                             │
│ detect_amd_patterns(candles, swings, inducements, trend)          │
│ → AMDPattern[] (Accumulation → Manipulation → Distribution)      │
│ Rules: IDM NOT taken; internal liquidity sweep → MSS entry        │
│ Source: V17                                                       │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ Step 11: PRICE DELIVERY CYCLE                                    │
│ detect_price_cycles(candles, bos, choch, fvgs, swings)            │
│ → PriceCycleEvent[] + current PricePhase                         │
│ Phases: Consolidation → Expansion → Retracement → Reversal       │
│ Source: V11                                                       │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
                    AnalysisResult
```

### Multi-Timeframe Orchestration

The engine runs the above pipeline on **7 timeframes simultaneously**:

```
                   ┌─── 1M  (Monthly)  — 120 candles  ~10 years
                   ├─── W1  (Weekly)   — 500 candles  ~9.5 years
                   ├─── D1  (Daily)    — 365 candles  ~1 year
run_multi_tf() ────├─── H4  (4-Hour)   — 500 candles  ~83 days
                   ├─── H1  (1-Hour)   — 500 candles  ~20 days
                   ├─── M15 (15-Min)   — 1000 candles ~10 days
                   └─── M5  (5-Min)    — 1000 candles ~3.5 days
```

After individual analysis, cross-TF processing occurs:

1. **Cross-TF Fake CHoCH (V09 Rule 4):** Lower TF CHoCH marked fake if broken price matches a higher TF active IDM
2. **Multi-Style Signal Generation:** 4 trading styles × (bias_tf → setup_tf → entry_tf):
   - Positional: 1M → W1 → D1
   - Swing: W1 → D1 → H4
   - Short-Term: D1 → H4 → H1
   - Intraday: H4 → H1 → M15
3. **Quant Post-Processing:** If mode="quant", apply 4-layer scoring (see Section 6)
4. **Signal Store:** Cross-style deduplication + lifecycle tracking

### Lookback Configuration (per timeframe)

```python
SWING_LOOKBACK = {
    "1M": 2, "W1": 3, "D1": 3,
    "H4": 5, "H1": 5, "M15": 5, "M5": 5,
}
```

---

## 5. Signal Generation — The Master Checklist

**File:** `backend/app/core/signal_generator.py` (1,100 lines)
**Source:** V17-V23 (6 videos combined into unified checklist)

### V23 Master Trading Checklist (13 Steps)

```
 Step 1:  IDENTIFY TREND      → W1 swing classifier (HH+HL = buy, LH+LL = sell)
 Step 2:  MARK STRUCTURE       → D1 BOS/CHoCH + swing points
 Step 3:  FIND IDM             → D1/H4 inducement after structure break
 Step 4:  FIND ZONE            → H4 OB + FVG below/above IDM
 Step 5:  CHECK CONTEXT        → Premium/Discount on D1 range + session
 Step 6:  WAIT FOR TAP         → Price must reach the H4 zone midpoint
 Step 7:  CONFIRM ENTRY (M15)  → MSS / SCOB / SBC / valid pullback break
 Step 8:  SET STOP LOSS        → Beyond the H4 zone + SL_BUFFER_PCT (0.1%)
 Step 9:  SET TAKE PROFIT      → Previous swing high/low from IDM origin
 Step 10: VSA ABSORPTION       → Ultra-high volume = institutional confirmation
 Step 11: DETECT CHoCH          → D1 direction switch for bias update
 Step 12: COUNTER-TREND        → D1 BOS + IDM close + FVG → TP first opposing zone
 Step 13: SBC ENTRY            → Sweep (wick) + body close opposite = standalone
```

### Entry Methods

| Method | Description | Min R:R | Source |
|--------|------------|---------|--------|
| **MSS** | Liquidity swept + expansion + body close (Million Dollar Setup) | 1.5:1 | V15 |
| **SCOB** | Sweep-based Change of Bias (smart candle OB) | 1.5:1 | V19 |
| **SBC** | Sweep Based Change of Character (standalone) | 3.0:1 | V22 |
| **Pullback Break** | Valid pullback after zone tap + candle confirmation | 1.5:1 | V17 |

### Signal Grading

```
Grade A:  3+ confluences, trend-aligned, kill zone, multi-TF confirmed
Grade B:  2+ confluences, trend-aligned
Grade C:  1 confluence or weak alignment
Grade D:  Counter-trend signal (higher risk, higher reward)
```

### MSS Quality Grading (V25)

```
NONE:          No FVG in MSS shift leg            → Weak (~0% WR expected)
STANDARD:      FVG present in shift leg           → ~50% WR
A_PLUS:        FVG + iFVG (inverted FVG)          → ~65% WR
A_PLUS_PLUS:   FVG + BPR (balanced price range)   → ~75-85% WR
```

### Confluence Factors

Signals accumulate confluence points from:
- OB + FVG combo at same level
- Premium/discount alignment (buy in discount, sell in premium)
- Kill zone timing (London/NY session)
- Fibonacci level proximity (within 3% of key Fib)
- VSA absorption confirmation
- MSS quality grade (A+ / A++)
- Futures data (OI divergence, crowded funding)
- HTF zone overlap (H4/D1 zone supporting M15 entry)

---

## 6. Quant Scoring Engine — 4-Layer System

The quant engine is a **post-processing layer** that applies institutional-grade quantitative scoring on top of SMC signals. It does NOT replace the SMC pipeline — it enhances it.

### Architecture

```
SMC Signals (Grade A/B/C/D)
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│                    QUANT ENGINE ORCHESTRATOR                    │
│                 apply_quant_layer_sync()                        │
│                                                                │
│  ┌────────────────────┐  ┌────────────────────────────────────┐│
│  │ TIER 1 (Computed)  │  │ TIER 2+3 (External APIs)          ││
│  │ • ATR (14/100)     │  │ • Deribit Options (P/C, GEX, skew)││
│  │ • DVOL             │  │ • Cross-Exchange Funding           ││
│  │ • VPIN             │  │ • CME COT Data                    ││
│  │ • Vol Regime       │  │ • On-Chain Exchange Flows          ││
│  │ • Liquidations     │  │ • Fear & Greed Index              ││
│  └────────────────────┘  │ • L2 Order Book Depth             ││
│                          │ • Whale Transactions               ││
│                          └────────────────────────────────────┘│
│                                                                │
│  For each signal, compute 4 layers:                            │
│                                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌───────────┐│
│  │ Alpha Model  │ │ Microstructure│ │ Risk     │ │ Execution ││
│  │    55%       │ │    20%       │ │   15%    │ │   10%     ││
│  │ -100 to +100 │ │ -100 to +100│ │ 0 to 100 │ │-100 to+100││
│  └──────────────┘ └──────────────┘ └──────────┘ └───────────┘│
│         │                │               │            │       │
│         └────────────────┴───────────────┴────────────┘       │
│                          │                                     │
│                    Combined Score                              │
│                   -100 to +100                                 │
│                          │                                     │
│              ┌───────────┴───────────┐                        │
│              │ Grade Modification    │                        │
│              │ > +40 → promote tier  │                        │
│              │ < -40 → demote tier   │                        │
│              │ ATR SL/TP replacement │                        │
│              │ Position sizing       │                        │
│              │ Signal suppression    │                        │
│              └───────────────────────┘                        │
└────────────────────────────────────────────────────────────────┘
         │
         ▼
Modified Signals + QuantContext (for frontend)
```

### Layer 1: Alpha Model (55% weight)

**File:** `backend/app/quant/alpha_model.py`
**Function:** `compute_alpha_score(quant_data, futures_data, direction, regime, phase) → (float, dict)`
**Range:** -100 to +100 (directional conviction)

11 scoring components, each returns -10 to +10:

| # | Component | Source | Weight | Bullish Logic |
|---|-----------|--------|--------|---------------|
| 1 | OI-Price Divergence | Binance Futures | 3 | OI rising + price falling = accumulation (+6) |
| 2 | Funding Rate | Binance Futures | 2 | High funding opposing signal = contrarian (+7) |
| 3 | VPIN | Taker Volume | 2 | VPIN > 0.7 + aligned taker flow (+8) |
| 4 | Liquidation Cascade | Binance WS | 2 | Opposite-side cascade = fuel (+10) |
| 5 | Volatility Regime | ATR + DVOL | 1.5 | Low vol = cleaner signals (+5) |
| 6 | Price Cycle Phase | Engine | 1 | Expansion = momentum (+5) |
| 7 | Options Positioning | Deribit | 1.5 | P/C > 1.3, max pain magnet, GEX, skew |
| 8 | Funding Arbitrage | Bybit + OKX | 1 | Cross-exchange divergence (+4) |
| 9 | COT Net Positioning | CFTC | 1 | Percentile < 20 = contrarian bullish (+6) |
| 10 | On-Chain Exchange Flow | CoinMetrics | 1 | Outflow > 5000 BTC = accumulation (+6) |
| 11 | Fear & Greed | Alternative.me | 0.5 | Extreme fear < 20 = contrarian buy (+7) |

### Layer 2: Microstructure Filter (20% weight)

**File:** `backend/app/quant/microstructure.py`
**Function:** `compute_microstructure_score(quant_data, futures_data, candles, direction) → (float, dict)`
**Range:** -100 to +100 (market quality)

| # | Component | Weight | What It Measures |
|---|-----------|--------|-----------------|
| 1 | VPIN Flow Toxicity | 3 | Informed vs uninformed trading (0-1 scale) |
| 2 | OI Delta + Divergence | 3 | Open interest accumulation/distribution |
| 3 | Liquidation Heatmap | 2 | Opposite-side cascade = exhaustion signal |
| 4 | Taker Flow Imbalance | 2 | Buy/sell ratio extremes |
| 5 | Cross-Exchange Funding | 2 | Dispersion between exchanges |
| 6 | Options Skew + GEX | 2 | Dealer gamma positioning |

**VPIN Algorithm:**
```
For each taker volume bucket:
  imbalance = |buy_vol - sell_vol| / (buy_vol + sell_vol)
VPIN = rolling average of imbalances over N buckets
> 0.7 = high probability of informed flow
```

**Liquidation Proximity Check:**
```
check_liquidation_proximity(liquidations, sl_price, tp_price, tolerance=0.5%)
→ {near_sl_usd, near_tp_usd, sl_risk_elevated (>$10M), tp_catalyst}
```

### Layer 3: Risk Model (15% weight)

**File:** `backend/app/quant/risk_model.py`
**Functions:**
- `compute_atr(candles, period=14) → float`
- `detect_volatility_regime(candles, dvol) → VolatilityRegime`
- `compute_atr_sl_tp(signal, atr, regime) → (sl, tp)`
- `compute_position_size(entry, sl, balance, regime) → pct`
- `compute_risk_tradability(regime) → float (0-100)`

**Volatility Regime Detection:**
```
ATR_ratio = ATR(14) / ATR(100)

LOW:     ratio < 0.7  OR  DVOL < 40   → tradability=85, SL_mult=2.0x
NORMAL:  0.7 ≤ ratio ≤ 1.3           → tradability=100, SL_mult=2.5x
HIGH:    ratio > 1.3  OR  DVOL > 65   → tradability=50, SL_mult=3.5x
EXTREME: ratio > 2.0  OR  DVOL > 100  → tradability=15, SL_mult=5.0x → SUPPRESS
```

**ATR-Based SL/TP:**
```python
sl = entry ± (ATR × regime_multiplier)   # Dynamic based on current volatility
tp = entry ± (sl_distance × RR_target)    # Maintains risk-reward ratio
```

**Position Sizing:**
```python
risk_amount = balance × base_risk_pct × vol_scale
# vol_scale: LOW=1.2x, NORMAL=1.0x, HIGH=0.6x, EXTREME=0.3x
position_size = risk_amount / abs(entry - sl)
```

### Layer 4: Execution Model (10% weight)

**File:** `backend/app/quant/execution_model.py`
**Function:** `compute_execution_score(session, candles, direction, entry_method, mss_grade, l2_depth) → (float, dict)`
**Range:** -100 to +100

| Component | Weight | Scoring |
|-----------|--------|---------|
| Session Timing | 3 | Kill zone = +7, NY = +5, London = +4, Asian = -3 |
| Spread/Slippage | 2 | Tight < 0.01% = +5, Wide > 0.1% = -5 |
| VWAP Distance | 2 | Entry below VWAP on bull = discount (+5) |
| Micro-Trend (5-EMA) | 2 | Slope aligned with direction = +5 |
| Entry Method Quality | 1 | A++ MSS = +10, A+ = +7, SBC = +5, Pullback = +2 |

### Combined Scoring & Grade Modification

**File:** `backend/app/quant/scoring.py`

```python
combined = alpha×0.55 + micro×0.20 + risk_normalized×0.15 + exec×0.10

Mapping:
  combined > +40  → 3 confirmations → PROMOTE grade (C→B, B→A)
  combined > +20  → 2 confirmations → confidence +10
  combined > +5   → 1 confirmation  → confidence +5
  -5 to +5        → neutral
  combined < -5   → 1 contradiction → confidence -5
  combined < -20  → 2 contradictions → confidence -10
  combined < -40  → 3 contradictions → DEMOTE grade (A→B, B→C)
```

**Signal Suppression:**
When `VolatilityRegime = EXTREME` and `risk_tradability < 20`:
- Signal marked `suppressed = True`
- Frontend shows it greyed out with warning
- Not sent to Telegram

### Graceful Degradation

Each external API is wrapped in try/except with TTL cache. If sources fail, weights redistribute:

| Scenario | Alpha | Micro | Risk | Exec |
|----------|-------|-------|------|------|
| All available | 55% | 20% | 15% | 10% |
| No TIER 2 APIs | 35% | 25% | 30% | 10% |
| TIER 1 only | 30% | 25% | 35% | 10% |

The system **never crashes** — it falls back gracefully to fewer data sources.

---

## 7. Data Services & External APIs

### Data Tier Classification

```
TIER 1 — ALWAYS AVAILABLE (computed from existing data, no external calls)
  • ATR (14/100-period)         — computed from candle data
  • VPIN                        — computed from taker volume
  • Volatility Regime           — computed from ATR ratio
  • Liquidation events          — Binance WebSocket (free, real-time)

TIER 2 — FREE PUBLIC APIs (with in-memory TTL cache)
  • Deribit Options             — P/C ratio, max pain, GEX, 25Δ skew
  • Cross-Exchange Funding      — Bybit + OKX rates
  • CME COT Data                — CFTC institutional positioning
  • On-Chain Exchange Flows     — CoinMetrics community API
  • DVOL (implied volatility)   — Deribit volatility index

TIER 3 — SUPPLEMENTARY (nice-to-have)
  • Fear & Greed Index          — Alternative.me
  • L2 Order Book Depth         — Binance depth
  • Whale Transactions          — Blockchain.info
```

### Complete API Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL API CONNECTIONS                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  BINANCE (FREE, no key required)                                       │
│  ├── REST: api.binance.com                                             │
│  │   ├── GET /api/v3/klines          → OHLCV candles (spot)            │
│  │   ├── GET /api/v3/ticker/24hr     → Live price ticker (2s poll)     │
│  │   ├── GET /api/v3/ticker/price    → Current price (for trade close) │
│  │   └── GET /api/v3/depth?limit=100 → L2 order book (5s TTL)         │
│  │                                                                      │
│  ├── REST: fapi.binance.com (Futures)                                  │
│  │   ├── GET /fapi/v1/openInterest                → Current OI         │
│  │   ├── GET /futures/data/openInterestHist (1h)   → 48 OI snapshots   │
│  │   ├── GET /futures/data/openInterestHist (4h)   → 500 OI (83 days)  │
│  │   ├── GET /fapi/v1/fundingRate?limit=30         → Funding history   │
│  │   ├── GET /fapi/v1/premiumIndex                 → Mark vs index     │
│  │   ├── GET /futures/data/topLongShortPositionRatio → Top trader L/S  │
│  │   ├── GET /futures/data/globalLongShortAccountRatio → Retail L/S    │
│  │   └── GET /futures/data/takerlongshortRatio     → Taker buy/sell    │
│  │                                                                      │
│  └── WebSocket: fstream.binance.com                                    │
│      └── WS /ws/btcusdt@forceOrder   → Real-time liquidations          │
│          Buffer: deque(maxlen=2000), auto-reconnect with backoff        │
│                                                                         │
│  DERIBIT (FREE, no key required)                                       │
│  ├── GET /api/v2/public/get_volatility_index_data                      │
│  │   → DVOL (BTC implied volatility index), 60s TTL                    │
│  │                                                                      │
│  └── GET /api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option
│      → Full options chain, 120s TTL                                     │
│      Computed: P/C ratio, max pain, GEX (Black-Scholes), 25Δ skew     │
│                                                                         │
│  BYBIT (FREE, no key required)                                         │
│  └── GET api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT│
│      → Current funding rate, 60s TTL                                    │
│                                                                         │
│  OKX (FREE, no key required)                                           │
│  └── GET okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP      │
│      → Current funding rate, 60s TTL                                    │
│                                                                         │
│  CFTC (FREE, no key required)                                          │
│  └── GET publicreporting.cftc.gov/resource/gpe5-46if.json              │
│      → CME Bitcoin futures COT data (weekly), 7-day TTL                 │
│      Extracted: leveraged long/short, net positioning, percentile       │
│                                                                         │
│  COINMETRICS (FREE community tier)                                     │
│  └── GET community-api.coinmetrics.io/v4/timeseries/asset-metrics      │
│      → Exchange inflows/outflows (BTC), 1-hour TTL                     │
│      Metrics: FlowInExNtv, FlowOutExNtv                               │
│                                                                         │
│  ALTERNATIVE.ME (FREE, no key required)                                │
│  └── GET api.alternative.me/fng/?limit=1                               │
│      → Fear & Greed Index (0-100), 5-minute TTL                        │
│                                                                         │
│  BLOCKCHAIN.INFO (FREE, no key required)                               │
│  └── GET blockchain.info/unconfirmed-transactions?format=json           │
│      → Large BTC transactions (>50 BTC), 5-minute TTL                  │
│                                                                         │
│  OPTIONAL (requires env var API keys):                                  │
│  ├── CRYPTOQUANT_API_KEY → Enhanced on-chain analytics                 │
│  └── COINGLASS_API_KEY   → Historical liquidation data                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### In-Memory TTL Cache

All external API data is cached in-memory with configurable TTLs:

```python
CACHE_TTL_FUNDING  = 60      # Cross-exchange funding: 1 minute
CACHE_TTL_OPTIONS  = 600     # Deribit options: 10 minutes
CACHE_TTL_COT      = 604800  # CME COT: 7 days (weekly data)
CACHE_TTL_ONCHAIN  = 3600    # On-chain flows: 1 hour
CACHE_TTL_FNG      = 3600    # Fear & Greed: 1 hour
CACHE_TTL_L2       = 5       # Order book depth: 5 seconds
CACHE_TTL_DVOL     = 60      # DVOL: 1 minute
```

### Black-Scholes Implementation

The options analytics in `quant_data.py` implement Black-Scholes Greeks for GEX computation:

```python
def _bs_d1(spot, strike, iv, t_years):
    """Black-Scholes d1 parameter."""
    return (ln(spot/strike) + 0.5 * iv² * t) / (iv * sqrt(t))

def _bs_gamma(spot, strike, iv, t_years):
    """Option gamma using Black-Scholes."""
    return norm_pdf(d1) / (spot * iv * sqrt(t))

def compute_net_gex(instruments, spot):
    """Net Gamma Exposure from all BTC options."""
    # Positive GEX = dealers hedge by dampening moves
    # Negative GEX = dealers amplify moves
```

### Liquidation WebSocket

**File:** `backend/app/services/liquidation_ws.py`

```python
class LiquidationWSManager:
    URL = "wss://fstream.binance.com/ws/btcusdt@forceOrder"

    # Stores up to 2000 recent events
    events: deque(maxlen=2000)

    # Event format:
    {
        "timestamp": 1707123456000,  # ms
        "side": "sell",              # "sell" = long liquidated
        "price": 97250.0,
        "qty": 0.5,                  # BTC
        "qty_usd": 48625.0           # Notional USD
    }

    # Methods:
    get_recent(window_minutes=30)    # Last 30 min liquidations
    get_all()                        # All buffered events
```

---

## 8. API Endpoints

### Complete Endpoint Map

| Method | Endpoint | Purpose | Mode |
|--------|----------|---------|------|
| **Core Analysis** ||||
| GET | `/api/health` | Health check + engine version | - |
| GET | `/api/analyze/{timeframe}` | Single TF analysis + M15 signals | smc/quant |
| GET | `/api/analyze` | Full multi-TF analysis | smc/quant |
| GET | `/api/signals` | Current trading signals | smc/quant |
| GET | `/api/signals/detailed` | Signals + V24/V23 checklists + context | smc/quant |
| GET | `/api/signals/{id}/deep-analysis` | Multi-TF explanation for a signal | smc/quant |
| GET | `/api/signals/resolved` | SL/TP hit history + win rate stats | smc/quant |
| POST | `/api/signals/resolved/clear` | Clear resolved signal history | smc/quant |
| **Backtest** ||||
| POST | `/api/backtest` | Run backtest (blocking, 5-15 min) | smc/quant |
| GET | `/api/backtest/latest` | Cached backtest results | - |
| **Demo Account** ||||
| GET | `/api/demo/account` | Account balance, P&L, win rate | smc/quant |
| POST | `/api/demo/account/reset` | Reset to $10,000 initial balance | smc/quant |
| PUT | `/api/demo/account/settings` | Update risk per trade % | smc/quant |
| GET | `/api/demo/trades` | List trades (filterable by status) | smc/quant |
| POST | `/api/demo/trades/{id}/take` | Open a pending trade | smc/quant |
| POST | `/api/demo/trades/{id}/skip` | Skip a pending trade | smc/quant |
| PUT | `/api/demo/trades/{id}/sl-tp` | Modify SL/TP of open trade | smc/quant |
| POST | `/api/demo/trades/{id}/close` | Close at current market price | smc/quant |
| POST | `/api/demo/trades/manual` | Place manual trade at market | smc/quant |
| GET | `/api/demo/equity` | Equity curve data points | smc/quant |
| **Telegram** ||||
| GET | `/api/telegram/status` | Bot connection status | - |
| POST | `/api/telegram/test` | Send test message | - |
| **Market Intelligence** ||||
| GET | `/api/market-intel` | Binance Futures data (OI, funding, ratios) | - |
| GET | `/api/quant-intel` | TIER 2+3 quant data + liquidation WS status | - |

### Mode Parameter

All analysis endpoints accept `?mode=smc` or `?mode=quant`:
- **SMC:** Pure structural detection, zone-based SL/TP
- **QUANT:** Same detection + 4-layer scoring, ATR-based SL/TP, position sizing, grade modification

Each mode has **isolated** signal stores and demo accounts (separate SQLite databases).

---

## 9. Frontend Architecture

### Tech Stack

```
Next.js 14         — React framework with App Router
TypeScript         — Type safety (578 lines of types matching backend models)
TailwindCSS        — Utility-first CSS + custom dark theme
lightweight-charts  — TradingView-quality candlestick charts
lucide-react       — Icon library
Inter font         — Clean UI typography
```

### Component Tree

```
app/
├── layout.tsx          — Root HTML, Inter font, global CSS
└── page.tsx            — Main dashboard (700+ lines)
    │
    ├── Header
    │   ├── Logo + Brand
    │   ├── Engine Mode Toggle (SMC ↔ QUANT)
    │   ├── Tab Navigation (7 tabs)
    │   ├── Timeframe Selector (1M, W1, D1, H4, H1, M15, M5)
    │   ├── Detector Visibility Toggles (8 toggles)
    │   └── Status Indicators (backend, price)
    │
    ├── Tab: Live Charts
    │   ├── Chart.tsx (1,400 lines)
    │   │   ├── Candlestick + Volume bars
    │   │   ├── 13 overlay types (swings, IDM, BOS, CHoCH, FVG, OB,
    │   │   │   P/D, kill zones, funding tint, OI histogram,
    │   │   │   positions, signal markers, ATR bands)
    │   │   └── Click detection → radial picker for overlaps
    │   ├── DetectionPanel.tsx — Right sidebar detector counts
    │   ├── SignalCard.tsx — Trading signal display
    │   ├── QuantIntelPanel.tsx — Quant sidebar (quant mode only)
    │   ├── AnalysisPanel.tsx — Click-to-analyze element details
    │   ├── DeepAnalysisPanel.tsx — Multi-TF signal explanation
    │   └── ElementPicker.tsx — Radial picker for overlapping elements
    │
    ├── Tab: Backtest
    │   └── BacktestTab.tsx — Date range, run button, equity curve, trade list
    │
    ├── Tab: Performance
    │   └── PerformanceTab.tsx — Grade breakdown, confluence edges, sessions
    │
    ├── Tab: Signals
    │   └── SignalsTab.tsx — Multi-style signals, V24/V23 checklists, outcomes
    │
    ├── Tab: Demo
    │   └── DemoTab.tsx — Paper trading, take/skip/close, equity, Telegram
    │
    ├── Tab: Market Intel
    │   └── MarketIntelTab.tsx — Binance Futures dashboards with charts
    │
    └── Tab: Quant Analysis (quant mode only)
        └── QuantAnalysisTab.tsx — 8 quant data source displays
            ├── Fear & Greed Gauge
            ├── Deribit Options (P/C, Max Pain, GEX, Skew)
            ├── CME COT Positioning
            ├── On-Chain Exchange Flows
            ├── Cross-Exchange Funding Rates
            ├── L2 Order Book Depth
            ├── Liquidation Monitor (WebSocket status + events)
            └── Whale Activity
```

### Auto-Refresh Intervals

```
Live Price Ticker:      2 seconds  (Binance REST)
M5 Analysis:            5 seconds
M15 Analysis:          10 seconds
H1 Analysis:           15 seconds
H4+ Analysis:       30-120 seconds
Demo Trades:            5 seconds
Market Intel:          30 seconds
Quant Analysis:        60 seconds
Signals Tab:           30 seconds
```

### Chart Overlay Types (13)

```
 1. Candlestick bars (OHLC) + Volume histogram
 2. Swing markers (HH/HL/LH/LL — amber, blue, gray, yellow-selected)
 3. IDM rays (dashed horizontal, "IDM" label)
 4. BOS rays (cyan dashed, "BOS"/"xBOS" label)
 5. CHoCH rays (gold=MSS, pink=confirmed, gray=fake)
 6. FVG zones (green=bullish, red=bearish boxes, faded=mitigated)
 7. OB zones (blue=bullish, orange=bearish boxes)
 8. Kill Zone bands (London=blue, NY=yellow background)
 9. Premium/Discount (yellow equilibrium, red premium, green discount)
10. Funding Rate tint (red=crowded longs, green=crowded shorts)
11. OI Delta histogram (green=rising, red=falling)
12. Position display (entry, SL, TP lines for open demo trades)
13. Signal markers (LONG/SHORT arrows at entry point)
```

### Design System

```css
/* Dark glassmorphism theme */
--bg-primary:    #0a0a0f
--bg-secondary:  #12121a
--bg-tertiary:   #1a1a2e
--border-primary: rgba(255, 255, 255, 0.06)
--text-primary:  #e4e4e7
--text-secondary: #a1a1aa
--text-muted:    #52525b

/* Glass card effect */
.glass-card {
  background: var(--bg-secondary);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-primary);
}
```

---

## 10. Database Schema

### SQLite Databases

Two separate SQLite databases for mode isolation:
- `data/demo_smc.db` — SMC mode demo account
- `data/demo_quant.db` — QUANT mode demo account

### Tables

#### `account` (singleton)

```sql
CREATE TABLE account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    balance REAL NOT NULL DEFAULT 10000.0,
    initial_balance REAL NOT NULL DEFAULT 10000.0,
    risk_per_trade_pct REAL NOT NULL DEFAULT 1.0,
    total_trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
```

#### `signals`

```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_hash TEXT UNIQUE NOT NULL,     -- Dedup key
    direction TEXT NOT NULL,              -- "bullish" / "bearish"
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    risk_reward_ratio REAL NOT NULL,
    grade TEXT NOT NULL,                  -- A, B, C, D
    confidence_score REAL NOT NULL,
    confluences TEXT NOT NULL,            -- JSON array
    entry_method TEXT,
    pattern_type TEXT,
    timeframe TEXT NOT NULL,
    is_counter_trend INTEGER DEFAULT 0,
    vsa_absorption INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    telegram_sent INTEGER DEFAULT 0,
    telegram_message_id INTEGER
);
```

#### `trades`

```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL REFERENCES signals(id),
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, open, closed, skipped
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    risk_reward_ratio REAL NOT NULL,
    grade TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    confluences TEXT NOT NULL,
    entry_method TEXT,
    pattern_type TEXT,
    position_size_usd REAL,
    position_size_btc REAL,
    exit_price REAL,
    pnl_usd REAL,
    pnl_pct REAL,
    outcome TEXT,                          -- win, loss
    created_at TEXT DEFAULT (datetime('now')),
    opened_at TEXT,
    closed_at TEXT,
    action_source TEXT DEFAULT 'web',     -- web, telegram, timeout, monitor
    bars_monitored INTEGER DEFAULT 0,
    trade_source TEXT DEFAULT 'signal'    -- signal, manual
);
```

#### `equity_snapshots`

```sql
CREATE TABLE equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    balance REAL NOT NULL,
    timestamp TEXT DEFAULT (datetime('now'))
);
```

### Signal Lifecycle

```
Signal Generated (by engine)
    │
    ▼
SignalStore.update()                    ← In-memory lifecycle manager
    ├── New signal → assign signal_id, created_at
    ├── Existing signal → increment bars_active
    ├── SL hit → resolve as "sl_hit"
    ├── TP hit → resolve as "tp_hit"
    └── Expired (20+ bars) → resolve as "expired"
    │
    ▼
SignalMonitor (30s polling)             ← Creates pending trades
    ├── Grade A/B signals → INSERT into trades (status="pending")
    └── Telegram notification sent
    │
    ▼
User Action (web/telegram)
    ├── TAKE → status="open", position_size calculated
    ├── SKIP → status="skipped"
    └── TIMEOUT (5 min) → auto-skip
    │
    ▼
PositionMonitor (10s polling)           ← Checks price vs SL/TP
    ├── Price hits SL → close trade (loss)
    ├── Price hits TP → close trade (win)
    └── Manual close → close at market price
```

---

## 11. Data Flow Diagrams

### End-to-End Data Flow (Single Analysis Cycle)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     1. DATA ACQUISITION                                 │
│                                                                         │
│  Binance REST ──→ 7× fetch_klines()  ──→ candles_by_tf                 │
│  (api.binance.com)   (1M, W1, D1,         {tf: [Candle, ...]}         │
│                       H4, H1, M15, M5)                                  │
│                                                                         │
│  Binance Futures ──→ fetch_market_intel() ──→ futures_data             │
│  (fapi.binance.com)   (OI, funding, ratios,    {open_interest: ...,    │
│                        taker volume)             funding_rate: ...,     │
│                                                  taker_volume: ...}     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     2. DETECTION ENGINE                                 │
│                                                                         │
│  for each TF in [1M, W1, D1, H4, H1, M15, M5]:                        │
│      analyze_timeframe(candles, tf) → AnalysisResult                   │
│      (runs 11-step pipeline: Swings → IDM → Liquidity → BOS →         │
│       VSA → CHoCH → FVG → MSS → OB → P/D → Session → AMD → Cycle)    │
│                                                                         │
│  Cross-TF processing:                                                   │
│      _apply_cross_tf_fake_choch(results)                               │
│                                                                         │
│  results = {                                                            │
│      "1M": AnalysisResult, "W1": AnalysisResult, ...                   │
│  }                                                                      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   3. SIGNAL GENERATION                                  │
│                                                                         │
│  for style in [positional, swing, short_term, intraday]:               │
│      _generate_style_signals(style, results, candles_by_tf)            │
│      → applies V23 Master Checklist on entry_tf using bias from HTF    │
│                                                                         │
│  all_style_signals = [TradingSignal, TradingSignal, ...]               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
              mode="smc"            mode="quant"
                    │                     │
                    │                     ▼
                    │  ┌──────────────────────────────────────────────────┐
                    │  │              4. QUANT SCORING                    │
                    │  │                                                  │
                    │  │  TIER 1: compute ATR, VPIN, vol regime, liqs    │
                    │  │  TIER 2: fetch options, funding, COT, on-chain  │
                    │  │  TIER 3: fetch Fear&Greed, L2, whales           │
                    │  │                                                  │
                    │  │  For each signal:                                │
                    │  │    alpha  = compute_alpha_score()      (55%)    │
                    │  │    micro  = compute_microstructure()    (20%)    │
                    │  │    risk   = compute_risk_tradability()  (15%)    │
                    │  │    exec   = compute_execution_score()   (10%)    │
                    │  │    combined → grade modify, ATR SL/TP, pos size │
                    │  │                                                  │
                    │  │  Output: (QuantContext, modified_signals)       │
                    │  └──────────────────────┬───────────────────────────┘
                    │                         │
                    └───────────┬─────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  5. SIGNAL LIFECYCLE                                    │
│                                                                         │
│  SignalStore.update(signals, candles_by_tf)                             │
│    ├── Cross-style deduplication (same entry → merge styles)           │
│    ├── SL/TP hit check against latest candles                          │
│    ├── Expiration (20+ bars without resolution)                        │
│    └── Assign signal_id, track bars_active                             │
│                                                                         │
│  active_signals = store.get_active_as_trading_signals()                │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    6. API RESPONSE                                      │
│                                                                         │
│  _serialize_result(result, candles, trade_bias, htf_zones)             │
│  → JSON with all detections, signals, futures_context, quant_context   │
│                                                                         │
│  → HTTP Response to Frontend                                           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    7. FRONTEND RENDERING                                │
│                                                                         │
│  page.tsx receives AnalysisResult → updates state                      │
│  Chart.tsx renders candles + 13 overlay types                          │
│  SignalCard renders trading signals with quant scores                  │
│  QuantIntelPanel shows quant context sidebar                           │
│  Auto-refresh triggers next cycle (2-120s depending on TF)             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Backtest Data Flow

```
User selects date range (e.g., 2024-06-01 to 2025-01-01)
    │
    ▼
POST /api/backtest?start_date=...&end_date=...&mode=smc|quant
    │
    ▼
backtester.py:
  1. Fetch ALL candles for date range (Binance REST, paginated)
  2. Cache in JSON: data/historical/{symbol}_M15_{date}.json
  3. Sliding window: step through 96 candles at a time (1 day)
  4. At each step:
     a. Extract candles[start:start+1000] as context window
     b. Run full run_multi_tf_analysis() → signals
     c. If mode="quant", TIER 1 scoring applied (no TIER 2/3 for history)
     d. For each signal: check if SL or TP hit in future candles
     e. Record TradeRecord with outcome (win/loss/timeout)
  5. Aggregate stats: win_rate, profit_factor, by_grade, by_confluence
  6. Cache result: data/backtest/latest.json
    │
    ▼
BacktestTab.tsx renders:
  - Stat cards (trades, win rate, P&L, drawdown)
  - Equity curve chart
  - Trade list table
  - PerformanceTab: grade breakdown, confluence edges
```

### Demo Trade Flow

```
SignalMonitor (30s polling)
    │
    ├── New A/B signal detected
    │   └── INSERT into signals table
    │       └── INSERT into trades table (status="pending")
    │           └── Telegram notification sent (if configured)
    │
    ▼
User sees pending trade in DemoTab
    │
    ├── TAKE → status="open"
    │   ├── Position size = balance × risk% / |entry - SL|
    │   └── PositionMonitor starts tracking (10s check)
    │       ├── Price hits SL → close (loss), update balance
    │       ├── Price hits TP → close (win), update balance
    │       └── User clicks Close → close at market price
    │
    └── SKIP → status="skipped"
        └── No position opened
```

---

## 12. Configuration Reference

### Backend Constants (`config.py`)

```python
# ── Primary ──
SYMBOL = "BTCUSDT"
TIMEFRAMES = {"W1": "1w", "D1": "1d", "H4": "4h", "M15": "15m"}

# ── Trading Styles (V20) ──
TRADING_STYLES = {
    "positional":  {"bias_tf": "1M", "setup_tf": "W1",  "entry_tf": "D1"},
    "swing":       {"bias_tf": "W1", "setup_tf": "D1",  "entry_tf": "H4"},
    "short_term":  {"bias_tf": "D1", "setup_tf": "H4",  "entry_tf": "H1"},
    "intraday":    {"bias_tf": "H4", "setup_tf": "H1",  "entry_tf": "M15"},
}

# ── Signal Quality ──
SL_BUFFER_PCT = 0.001          # 0.1% beyond zone
MIN_RISK_REWARD = 1.5          # Minimum R:R for signals
SBC_MIN_RISK_REWARD = 3.0      # SBC standalone minimum R:R

# ── Sessions (UTC) ──
CRYPTO_SESSIONS = {
    "asian":  {"start": 0,  "end": 8},
    "london": {"start": 8,  "end": 13},
    "ny":     {"start": 13, "end": 21},
    "late":   {"start": 21, "end": 24},
}
KILL_ZONES = {
    "us_equity_open":  {"start": 13, "end": 15},
    "us_equity_close": {"start": 20, "end": 21},
    "cme_btc_open":    {"start": 13, "end": 14},
}

# ── Demo Account ──
DEMO_INITIAL_BALANCE = 10_000.0
DEMO_RISK_PER_TRADE_PCT = 1.0
DEMO_PENDING_TIMEOUT_SECONDS = 300
DEMO_MIN_SIGNAL_GRADE = "B"

# ── Quant Engine ──
ATR_PERIOD = 14
ATR_LONG_PERIOD = 100
VPIN_BUCKET_COUNT = 50
VPIN_THRESHOLD = 0.7
QUANT_WEIGHTS = {"alpha": 0.55, "micro": 0.20, "risk": 0.15, "exec": 0.10}
VOL_REGIME_LOW = 0.7
VOL_REGIME_HIGH = 1.3
VOL_REGIME_EXTREME = 2.0
DVOL_HIGH = 80
DVOL_EXTREME = 100
ATR_SL_MULT = {"low": 1.5, "normal": 2.0, "high": 2.5, "extreme": 3.0}
MAX_DRAWDOWN_PCT = 0.10
SUPPRESS_ON_EXTREME_VOL = True

# ── Cache TTLs (seconds) ──
CACHE_TTL_FUNDING = 60
CACHE_TTL_OPTIONS = 600
CACHE_TTL_COT = 604800
CACHE_TTL_ONCHAIN = 3600
CACHE_TTL_FNG = 3600
CACHE_TTL_L2 = 5
CACHE_TTL_DVOL = 60

# ── Optional API Keys ──
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CRYPTOQUANT_API_KEY = os.environ.get("CRYPTOQUANT_API_KEY", "")
COINGLASS_API_KEY = os.environ.get("COINGLASS_API_KEY", "")
```

### Swing Lookback (per timeframe)

```python
SWING_LOOKBACK = {
    "1M": 2, "W1": 3, "D1": 3,
    "H4": 5, "H1": 5, "M15": 5, "M5": 5,
}
```

### Candle Buffer Sizes

```python
CANDLE_BUFFER_SIZE = {
    "1M": 120,    # ~10 years
    "W1": 500,    # ~9.5 years
    "D1": 365,    # ~1 year
    "H4": 500,    # ~83 days
    "H1": 500,    # ~20 days
    "M15": 1000,  # ~10 days
    "M5": 1000,   # ~3.5 days
}
```

### Frontend TypeScript Types (matching backend)

```typescript
// 578 lines in types.ts — exact mirror of backend models
export type Direction = "bullish" | "bearish";
export type SwingClassification = "HH" | "HL" | "LH" | "LL" | "unclassified";
export type SignalGrade = "A" | "B" | "C" | "D" | "M";
export type MSSGrade = "none" | "standard" | "a_plus" | "a_plus_plus";
export type VolatilityRegime = "low" | "normal" | "high" | "extreme";
export type AppTab = "live" | "backtest" | "performance" | "signals"
                   | "demo" | "market_intel" | "quant_analysis";
// ... 25+ interfaces
```

---

## Running the System

```bash
# Terminal 1: Backend (port 8000)
cd backend && python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend (port 3000)
cd frontend && npm run dev

# Open: http://localhost:3000

# CLI Backtest (for long date ranges):
python3 scripts/run_backtest.py --start 2024-06-01 --end 2025-01-01

# Historical Validation:
python3 scripts/run_validation.py
```

---

*Built from 23 Hindi SMC/ICT videos • 11 SMC detectors • 4-layer quant scoring • 10+ free data APIs • dual-mode architecture*
