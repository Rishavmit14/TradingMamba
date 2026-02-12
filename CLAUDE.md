# TradingMamba - Claude Code Project Instructions

## Project Purpose

Build a **trading system** powered by domain knowledge extracted from 23 Hindi SMC/ICT YouTube videos (Forex Minions "Complete Logical SMC" playlist). The system applies Smart Money Concepts against live market data to detect patterns and generate trading signals, with a web-based dashboard for live analysis, backtesting, and performance analytics.

## Branch: TradingMambaHindi (Hindi Audio Videos)

This branch processes **Hindi audio** YouTube videos (`TRANSCRIPTION_LANGUAGE=hi`).

## Knowledge Foundation (COMPLETED - 23/23 Videos)

All 23 videos have been trained using Claude Code Expert Analysis. The knowledge is stored as structured English knowledge bases derived from Hindi audio source material.

### Playlist: Complete Logical SMC (PLLxESps7ndeWtSZowIo2v0fMmYj4Yiy-a)

| # | Video ID | Title | Status |
|---|----------|-------|--------|
| 01 | UIRBfCT1kI4 | Structure Mapping | TRAINED |
| 02 | DkOv2qG9Cn0 | Liquidity & Inducement | TRAINED |
| 03 | Ovd5QzZutsw | Pullback & Valid Inducement | TRAINED |
| 04 | PYHUHpvKIYo | Inducement Shift & Traps | TRAINED |
| 05 | fTf3pO7F5T8 | Break of Structure | TRAINED |
| 06 | dknxgbZrN3g | BOS vs Liquidity Sweep | TRAINED |
| 07 | VH882sl1pl8 | CHoCH & Structure Mapping | TRAINED |
| 08 | lIbO4JrL_2I | High Prob Inducement | TRAINED |
| 09 | u-LM2DjDd7M | Fake CHoCH | TRAINED |
| 10 | SaTvah4OSpA | CHoCH Confirmation | TRAINED |
| 11 | 3-NsLSV6huY | Price Cycle Theory | TRAINED |
| 12 | anpETX7ahKo | Premium & Discount Zones | TRAINED |
| 13 | 5c14eElh42k | Fair Value Gap | TRAINED |
| 14 | -20GrDt-Aws | Valid Order Blocks | TRAINED |
| 15 | cCugrbID3wI | Million Dollar Setup | TRAINED |
| 16 | 9BfwzGhkOFE | Candlestick & Sessions | TRAINED |
| 17 | exF1jB8qCxo | Complete SMC Guideline (Entry) | TRAINED |
| 18 | PxShy_00yR8 | QML & POI Zones | TRAINED |
| 19 | Kc74F908K-8 | SCOB Entry Technique | TRAINED |
| 20 | JaYMVwyauKA | Multi-Timeframe Analysis | TRAINED |
| 21 | sRXIoNPq85Q | Counter-Trend Trading | TRAINED |
| 22 | newJA7qOzpE | Liquidity Sweep Module (Wyckoff) | TRAINED |
| 23 | vuZqh_oKBp8 | Putting All Together - SMC The End | TRAINED |

### Knowledge Base Location

- **Knowledge bases**: `data/audio_first_training/{video_id}_knowledge_base.json` (23 files)
- **Summaries**: `data/audio_first_training/{video_id}_knowledge_summary.md` (23 files)
- **Transcripts**: `data/transcripts/{video_id}.json` (23 files)
- **Video frames**: `data/video_frames/{video_id}/` (23 directories)
- **Audio files**: `data/audio/` (downloaded .wav files)
- **Playlist data**: `data/playlists/`

### Knowledge Base Schema

Each `*_knowledge_base.json` contains:
- `generation_method`: `"Claude Code expert analysis"`
- `metadata.training_type`: `"claude_code_expert"`
- `metadata.source_language`: `"hi"`
- `concepts`: Array of concepts, each with:
  - `llm_summary`: Structured English explanation (Definition, Rules, Identification)
  - `statistics`: Teaching time, word count, frame count
  - `teaching_types`: Array of teaching methods used
  - `key_rules`: Array of actionable trading rules
  - `visual_evidence`: References to specific analyzed frames

### Core SMC Concepts Covered

The 23 videos progressively build a complete SMC trading system:
1. **Structure**: HH/HL/LH/LL, swing points, trend identification (V01)
2. **Liquidity**: Buy/sell liquidity pools, inducement, traps (V02-V04, V08)
3. **BOS**: Break of Structure rules, BOS vs liquidity sweep (V05-V06)
4. **CHoCH**: Change of Character, fake CHoCH, confirmation (V07, V09-V10)
5. **Market Context**: Price cycle theory, premium/discount zones (V11-V12)
6. **Zones**: FVG, Order Blocks, IOF identification (V13-V14)
7. **Setups**: Million Dollar Setup, sessions/kill zones (V15-V16)
8. **Entries**: MSS/SBC/CHoCH entries, QML, SCOB technique (V17-V19)
9. **Multi-TF**: Higher TF direction + lower TF entry (V20)
10. **Advanced**: Counter-trend rules, Wyckoff/SBC module, full integration (V21-V23)

## System Architecture

### Phase 1: Detection Engine (COMPLETE)

10 ICT/SMC detectors running on live Binance BTCUSDT data across 7 timeframes (1M, W1, D1, H4, H1, M15, M5):

| Detector | File | Description |
|----------|------|-------------|
| Swing Detector | `backend/app/core/swing_detector.py` | HH/HL/LH/LL structural swings with 3-condition ICT validation |
| Inducement | `backend/app/core/inducement.py` | IDM detection with body-close confirmation, major/minor classification |
| BOS | `backend/app/core/bos_detector.py` | Break of Structure with IDM body-close tier system |
| CHoCH | `backend/app/core/choch_detector.py` | Change of Character with V10 composite scoring, fake CHoCH filter |
| Liquidity | `backend/app/core/liquidity.py` | Equal highs/lows, swing extremes, sweep/grab detection |
| FVG | `backend/app/core/fvg_detector.py` | Fair Value Gap detection with extreme candle rule |
| Order Block | `backend/app/core/order_block.py` | OB with FVG + liquidity sweep validation |
| Premium/Discount | `backend/app/core/premium_discount.py` | Zone calculation with Fibonacci qualification |
| Session | `backend/app/core/session.py` | Kill zone detection (London, NY, Asian sessions) |
| Signal Generator | `backend/app/core/signal_generator.py` | Master checklist combining all detectors into graded signals (A/B/C/D) |

Additional core files:
- `backend/app/core/engine.py` — Multi-TF analysis orchestrator
- `backend/app/core/amd_detector.py` — AMD (Accumulation-Manipulation-Distribution) pattern
- `backend/app/core/price_cycle_detector.py` — Price delivery cycle phases

### Phase 2: Historical Validation (COMPLETE)

Validation pipeline that runs all detectors on 6+ months of BTCUSDT historical data to verify detection rates and sanity:
- Script: `scripts/run_validation.py`
- Results: `data/validation/` (JSON reports)
- All 6 sanity checks PASSED on 20,545 M15 candles

### Phase 3: Backtesting + Tabbed Dashboard (COMPLETE)

Sliding-window backtester + 3-tab web UI:

**Backend:**
- `backend/app/services/backtester.py` — Sliding window backtest engine (96 M15-candle steps, outcome evaluation, statistics)
- `backend/app/services/data_fetcher.py` — Binance data fetching with pagination + local JSON caching
- API: `POST /api/backtest` (run backtest), `GET /api/backtest/latest` (cached results)
- CLI: `scripts/run_backtest.py` — Terminal backtest runner

**Frontend (3 tabs):**
- **Live Charts** — Candlestick chart with all detector overlays, click-to-analyze, radial element picker, signal cards, live price ticker
- **Backtest** — Date range config, run button, stat cards (trades/win rate/P&L/drawdown), equity curve chart, trade list table
- **Performance** — Grade breakdown (TRADE/SKIP), confluence edge analysis, entry method comparison, session analytics

**Note:** The backtest API runs synchronously and blocks the server during execution. For long backtests (6+ months), use the CLI script instead: `python3 scripts/run_backtest.py --start 2024-06-01 --end 2025-01-01`

## Project Structure

```
TradingMambaHindi/
├── CLAUDE.md                              # This file
├── .env.example                           # Environment variables template
├── backend/
│   ├── app/
│   │   ├── config.py                      # Symbol, timeframes, intervals config
│   │   ├── main.py                        # FastAPI app with all API endpoints
│   │   ├── models.py                      # All data models (Candle, SwingPoint, BOS, etc.)
│   │   ├── core/                          # Detection engine
│   │   │   ├── engine.py                  # Multi-TF analysis orchestrator
│   │   │   ├── swing_detector.py          # HH/HL/LH/LL swing detection
│   │   │   ├── inducement.py              # IDM detection
│   │   │   ├── bos_detector.py            # Break of Structure
│   │   │   ├── choch_detector.py          # Change of Character
│   │   │   ├── liquidity.py               # Liquidity pool detection
│   │   │   ├── fvg_detector.py            # Fair Value Gap
│   │   │   ├── order_block.py             # Order Block detection
│   │   │   ├── premium_discount.py        # Premium/Discount zones
│   │   │   ├── session.py                 # Session/kill zone detection
│   │   │   ├── signal_generator.py        # Signal grading (A/B/C/D)
│   │   │   ├── amd_detector.py            # AMD pattern detection
│   │   │   └── price_cycle_detector.py    # Price cycle phases
│   │   └── services/
│   │       ├── data_fetcher.py            # Binance klines + caching
│   │       └── backtester.py              # Phase 3 backtest engine
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx                 # Root layout
│   │   │   └── page.tsx                   # Main dashboard (3-tab navigation)
│   │   ├── components/
│   │   │   ├── Chart.tsx                  # Candlestick chart with detector overlays
│   │   │   ├── DetectionPanel.tsx         # Right sidebar detection counts
│   │   │   ├── SignalCard.tsx             # Trading signal display card
│   │   │   ├── AnalysisPanel.tsx          # Click-to-analyze element details
│   │   │   ├── ElementPicker.tsx          # Radial picker for overlapping elements
│   │   │   ├── BacktestTab.tsx            # Backtest config + results + equity curve
│   │   │   └── PerformanceTab.tsx         # Grade/confluence/method/session analytics
│   │   └── lib/
│   │       ├── api.ts                     # API client (analyze, backtest, Binance ticker)
│   │       └── types.ts                   # TypeScript types matching backend models
├── scripts/
│   ├── run_validation.py                  # Phase 2 historical validation
│   ├── run_backtest.py                    # Phase 3 CLI backtest runner
│   ├── transcribe_local.py               # whisper.cpp transcription
│   └── ...                                # Other transcription/processing utilities
├── data/
│   ├── audio_first_training/              # 23 knowledge bases + summaries
│   ├── transcripts/                       # 23 Hindi transcripts
│   ├── video_frames/                      # Extracted video frames
│   ├── historical/                        # Cached Binance candle data (JSON)
│   ├── validation/                        # Phase 2 validation reports
│   ├── backtest/                          # Phase 3 backtest results
│   └── playlists/                         # Playlist metadata
└── models/                                # Whisper model files
```

## Running the System

### Start Backend + Frontend
```bash
# Terminal 1: Backend (port 8000)
cd backend && python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend (port 3000)
cd frontend && npm run dev
```

### Run Backtest via CLI (recommended for long backtests)
```bash
python3 scripts/run_backtest.py --start 2024-06-01 --end 2025-01-01
```

### Run Validation
```bash
python3 scripts/run_validation.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/analyze/{timeframe}` | Run analysis on single timeframe |
| GET | `/api/analyze` | Run multi-TF analysis (all timeframes) |
| POST | `/api/backtest?start_date=&end_date=&symbol=&step_size=` | Run backtest |
| GET | `/api/backtest/latest` | Get most recent cached backtest results |

## Transcription Tools

- **whisper.cpp**: `/opt/homebrew/bin/whisper-cli` with Metal GPU acceleration
- **Model**: `models/ggml-large-v3-turbo.bin`
- **Script**: `scripts/transcribe_local.py --video VIDEO_ID`
- **CRITICAL**: Never run 2+ whisper.cpp instances simultaneously on Metal GPU

## Video Training Workflow (for adding new videos)

When training ANY video, follow the Claude Code Expert Training path:

**Phase 0-3 (Automated):** Run `prepare_for_claude_code()` to download, extract frames, transcribe, and detect teaching units.

```python
from backend.app.ml.audio_first_learning import AudioFirstTrainer
trainer = AudioFirstTrainer(data_dir='data')
result = trainer.prepare_for_claude_code(video_id)
```

**Phase 4-5 (Claude Code Expert):**
1. Read the transcript at `data/transcripts/{video_id}.json`
2. View key frames at `data/video_frames/{video_id}/`
3. Analyze ICT/SMC concepts with expert-level vision analysis
4. Write knowledge base to `data/audio_first_training/{video_id}_knowledge_base.json`
5. Write summary to `data/audio_first_training/{video_id}_knowledge_summary.md`

### Quality Checklist
- `generation_method` = `"Claude Code expert analysis"`
- Transcript read and analyzed (Hindi understood natively)
- Key frames viewed (vision analysis by Claude Code)
- All ICT/SMC concepts captured with structured `llm_summary`
- `visual_evidence` references actual frames
- `metadata.source_language` = `"hi"`
- Output in English
