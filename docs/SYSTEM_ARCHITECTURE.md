# TradingMamba - System Architecture

> **AI-Powered Trading Signal System Based on Smart Money (ICT) Methodology**

This document provides a comprehensive overview of the TradingMamba system architecture, including all components, data flows, and ML training pipelines.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [ML Training Pipeline](#ml-training-pipeline)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [API Endpoints](#api-endpoints)
7. [Directory Structure](#directory-structure)
8. [Technology Stack](#technology-stack)
9. [Performance Optimizations](#performance-optimizations)

---

## System Overview

TradingMamba is an intelligent trading system that learns ICT (Inner Circle Trader) methodology from YouTube videos and applies that knowledge to generate trading signals. The system uses a unique **Audio-First Learning** approach where audio/transcript is the primary source of knowledge, and video frames provide visual evidence.

### Key Principles

1. **ML Learns HOW, Not WHAT**: The ML learns how to identify patterns (like Fair Value Gaps), not specific instances
2. **Audio-First**: Audio captures 100% of teaching vs ~70% from frames alone
3. **Hedge Fund Level**: Patterns are graded A+ to F, only high-quality setups generate signals
4. **Zero Paid APIs**: Uses free alternatives (yfinance, local models) for all functionality

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              TradingMamba System                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   YouTube       │    │   ML Training   │    │   Knowledge     │          │
│  │   Videos        │───▶│   Pipeline      │───▶│   Base          │          │
│  │   (ICT)         │    │   (Audio-First) │    │   (JSON)        │          │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘          │
│                                                          │                   │
│                                                          ▼                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   Market Data   │    │   Smart Money   │    │   Signal        │          │
│  │   (yfinance)    │───▶│   Analyzer      │───▶│   Generator     │          │
│  │                 │    │   (ML-Powered)  │    │   (Hedge Fund)  │          │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘          │
│                                                          │                   │
│                                                          ▼                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │   FastAPI       │    │   React         │    │   Telegram      │          │
│  │   Backend       │◀──▶│   Frontend      │    │   Notifications │          │
│  │   (uvloop)      │    │   (Vite)        │    │   (Optional)    │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## ML Training Pipeline

### End-to-End Training Flow

The system provides a complete end-to-end pipeline. Just provide a YouTube URL and everything happens automatically.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE END-TO-END TRAINING PIPELINE                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  INPUT: YouTube URL or Video ID                                               │
│         (Single video or entire playlist)                                     │
│                                                                               │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                               │
│  PHASE 0: PREREQUISITES (~2-5 min)                   [video_preprocessor.py] │
│  ├── 0.1 Download Audio                                                       │
│  │   └── pytubefix (handles YouTube auth issues)                             │
│  │   └── Output: data/audio/{video_id}.mp3                                   │
│  │                                                                            │
│  ├── 0.2 Extract Frames                                                       │
│  │   └── ffmpeg (every 3 seconds by default)                                 │
│  │   └── Output: data/video_frames/{video_id}/frame_*.jpg                    │
│  │                                                                            │
│  └── 0.3 Transcribe Audio                                                     │
│      └── faster-whisper (GPU-accelerated)                                    │
│      └── Output: data/transcripts/{video_id}.json                            │
│                                                                               │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                               │
│  PHASE 1-3: TRAINING (~1 second)                   [audio_first_learning.py] │
│  ├── Phase 1: Transcript Loading                                              │
│  │   └── Parse transcript into segments with timestamps                      │
│  │                                                                            │
│  ├── Phase 2: Teaching Unit Detection                                         │
│  │   ├── Group segments into logical teaching units                          │
│  │   ├── Detect ICT concepts (FVG, Order Block, Liquidity, etc.)            │
│  │   ├── Find deictic references ("look here", "this candle")               │
│  │   └── Classify teaching type (definition, example, annotation, summary)  │
│  │                                                                            │
│  └── Phase 3: Smart Frame Selection                                           │
│      ├── Select frames at teaching unit starts                               │
│      ├── Select frames at deictic references                                 │
│      ├── Select frames at visual changes                                     │
│      └── Result: 359 frames → ~80 selected (77% reduction)                   │
│                                                                               │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                               │
│  PHASE 4: VISION ANALYSIS (~35 min)                [video_vision_analyzer.py] │
│  ├── For each selected frame:                                                 │
│  │   ├── Build context-enriched prompt (includes audio context)             │
│  │   ├── Run vision model (MLX-VLM on Apple Silicon)                        │
│  │   └── Extract: patterns, annotations, price levels, structure            │
│  │                                                                            │
│  └── Output: data/audio_first_training/{video_id}_vision_analysis.json       │
│                                                                               │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                               │
│  PHASE 5: KNOWLEDGE SYNTHESIS (~3 min)             [audio_first_learning.py] │
│  ├── Group vision analyses by detected concept                               │
│  ├── Generate LLM summary per concept                                        │
│  │   ├── What is it?                                                         │
│  │   ├── How to identify it?                                                 │
│  │   └── How to trade it?                                                    │
│  │                                                                            │
│  └── Output:                                                                  │
│      ├── data/audio_first_training/{video_id}_knowledge_base.json           │
│      └── data/audio_first_training/{video_id}_knowledge_summary.md          │
│                                                                               │
│  ═══════════════════════════════════════════════════════════════════════════ │
│                                                                               │
│  TOTAL TIME: ~40-45 minutes for 18-minute video                              │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Usage Examples

```python
# Simplest usage - single video (full end-to-end)
from backend.app.ml.audio_first_learning import train_from_url
result = train_from_url('https://youtube.com/watch?v=VIDEO_ID')

# Train entire playlist
from backend.app.ml.audio_first_learning import train_playlist
results = train_playlist('https://youtube.com/playlist?list=PLxxx')

# More control with trainer instance
from backend.app.ml.audio_first_learning import AudioFirstTrainer
trainer = AudioFirstTrainer()
result = trainer.train_from_url('VIDEO_URL', force_preprocess=True)

# CLI usage
python -m backend.app.ml.audio_first_learning 'https://youtube.com/watch?v=FgacYSN9QEo'
python -m backend.app.ml.audio_first_learning 'https://youtube.com/playlist?list=PLxxx' --playlist
```

### Audio-First Innovation

Traditional approach analyzes every frame and adds audio context (~70% coverage).
Audio-First flips this: audio is PRIMARY, frames are SECONDARY evidence.

| Metric | Traditional | Audio-First |
|--------|-------------|-------------|
| Audio Coverage | ~70% | 97.7% |
| Frames Analyzed | 359 | 81 (77% reduction) |
| Teaching Capture | Partial | Complete |
| Processing Time | ~60 min | ~40 min |

---

## Core Components

### Backend Modules

#### ML Layer (`backend/app/ml/`)

| Module | Description |
|--------|-------------|
| `audio_first_learning.py` | Main training pipeline - orchestrates all phases |
| `video_preprocessor.py` | Phase 0 - downloads, extracts frames, transcribes |
| `video_vision_analyzer.py` | Vision model integration, frame analysis |
| `ml_pattern_engine.py` | Pattern detection using learned knowledge |
| `hedge_fund_ml.py` | Pattern grading (A+ to F), edge tracking |
| `training_pipeline.py` | Legacy training pipeline |
| `synchronized_learning.py` | Synchronized audio-visual learning |
| `pattern_recognition.py` | Pattern matching algorithms |
| `technical_analysis.py` | Technical indicators (RSI, MACD, etc.) |
| `kill_zones.py` | ICT Kill Zone detection (London, NY, etc.) |
| `signal_fusion.py` | Multi-signal combination logic |
| `feature_extractor.py` | Feature extraction for ML models |
| `price_predictor.py` | Price prediction models |

#### Services Layer (`backend/app/services/`)

| Module | Description |
|--------|-------------|
| `smart_money_analyzer.py` | Core ICT analysis engine (ML-powered) |
| `signal_generator.py` | Trading signal generation (hedge fund level) |
| `market_data.py` | Market data fetching |
| `free_market_data.py` | Free market data via yfinance |
| `backtest_engine.py` | Strategy backtesting |
| `paper_trading.py` | Paper trading simulation |
| `risk_metrics.py` | Risk calculations (Sharpe, Sortino, etc.) |
| `telegram_notifier.py` | Telegram notifications |
| `chart_generator.py` | Chart image generation |
| `concept_extractor.py` | ICT concept extraction from text |

#### Utilities (`backend/app/utils/`)

| Module | Description |
|--------|-------------|
| `json_utils.py` | Fast JSON operations (orjson-based, 6x faster) |
| `audio_downloader.py` | YouTube audio download (pytubefix) |

### ICT Concepts Detected

The system learns and detects these ICT concepts:

```python
ICT_CONCEPTS = {
    # Core Concepts
    "fair value gap": ["fair value gap", "fvg", "fair value gaps"],
    "order block": ["order block", "order blocks", "ob"],
    "liquidity": ["liquidity", "liquidity pool", "buy side liquidity", "sell side liquidity"],
    "displacement": ["displacement", "displaced"],
    "imbalance": ["imbalance", "imbalances"],
    "breaker": ["breaker", "breaker block"],
    "mitigation": ["mitigation", "mitigate", "mitigated"],

    # Price Action
    "buy stops": ["buy stops", "buy stop"],
    "sell stops": ["sell stops", "sell stop"],
    "stop hunt": ["stop hunt", "stop run"],
    "turtle soup": ["turtle soup"],
    "equal highs": ["equal highs"],
    "equal lows": ["equal lows"],

    # Market Structure
    "market structure": ["market structure", "structure"],
    "swing high": ["swing high"],
    "swing low": ["swing low"],
    "higher high": ["higher high"],
    "lower low": ["lower low"],

    # Time & Sessions
    "kill zone": ["kill zone", "kill zones"],
    "power of three": ["power of three"],
    "accumulation": ["accumulation"],
    "manipulation": ["manipulation"],
    "distribution": ["distribution"],

    # Entries
    "optimal trade entry": ["optimal trade entry", "ote"],
    "fibonacci": ["fibonacci", "fib"],
}
```

### Pattern Grading System

Patterns are graded like a hedge fund:

| Grade | Meaning | Action |
|-------|---------|--------|
| A+ | Perfect setup - institutional, kill zone, HTF confluence | Trade aggressively |
| A | Excellent - clear pattern, good location | Trade normally |
| B | Good - valid pattern but minor issues | Trade with caution |
| C | Average - pattern exists but weak | No trade |
| D | Poor - questionable validity | Avoid |
| F | Invalid - misidentified | Ignore |

**Only A+, A, and B grades generate trading signals.**

---

## Data Flow

### Training Data Flow

```
YouTube Video
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 0: VideoPreprocessor             │
│  ├── Audio (pytubefix) → .mp3          │
│  ├── Frames (ffmpeg) → .jpg files      │
│  └── Transcript (whisper) → .json      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 1-3: AudioFirstTrainer           │
│  ├── Transcript Segments               │
│  ├── Teaching Units                    │
│  └── Selected Frames                   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 4: VisionAnalyzer                │
│  └── Frame Analysis Results            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 5: Knowledge Synthesis           │
│  └── Knowledge Base + Summary          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  MLPatternEngine                        │
│  └── Patterns ML Can Detect            │
└─────────────────────────────────────────┘
```

### Signal Generation Flow

```
Market Data (yfinance)
    │
    ▼
┌─────────────────────────────────────────┐
│  SmartMoneyAnalyzer (ML-Powered)        │
│  ├── Market Structure Analysis         │
│  ├── Order Block Detection             │
│  ├── FVG Detection                     │
│  ├── Liquidity Detection               │
│  └── Premium/Discount Zones            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  HedgeFundML                            │
│  ├── PatternGrader (A+ to F)           │
│  ├── EdgeTracker (win rate, R:R)       │
│  ├── HistoricalValidator               │
│  └── MTF Confluence Checker            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  SignalGenerator                        │
│  ├── Filter: Only A/B grades           │
│  ├── Calculate Entry/SL/TP             │
│  ├── Risk Assessment                   │
│  └── Confidence Score                  │
└─────────────────────────────────────────┘
    │
    ▼
Trading Signal
```

---

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

### Training Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/playlists` | List all playlists |
| GET | `/playlists/{id}` | Get playlist details |
| GET | `/transcripts/{video_id}` | Get video transcript |
| POST | `/train` | Trigger ML training |

### Signal Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/signals` | Get active signals |
| GET | `/signals/{id}` | Get signal details |
| POST | `/signals/generate` | Generate new signals |

### Analysis Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Analyze market data |
| GET | `/patterns` | Get detected patterns |
| GET | `/knowledge` | Get ML knowledge base |

---

## Directory Structure

```
TradingMamba/
├── backend/
│   └── app/
│       ├── main.py                    # FastAPI application entry
│       ├── config.py                  # Configuration settings
│       ├── database.py                # Database connections
│       │
│       ├── ml/                        # Machine Learning modules
│       │   ├── audio_first_learning.py    # Main training pipeline
│       │   ├── video_preprocessor.py      # Phase 0 preprocessing
│       │   ├── video_vision_analyzer.py   # Vision analysis
│       │   ├── ml_pattern_engine.py       # Pattern detection
│       │   ├── hedge_fund_ml.py           # Pattern grading
│       │   ├── training_pipeline.py       # Legacy pipeline
│       │   ├── synchronized_learning.py   # Sync learning
│       │   ├── pattern_recognition.py     # Pattern matching
│       │   ├── technical_analysis.py      # Technical indicators
│       │   ├── kill_zones.py              # Kill zone detection
│       │   ├── signal_fusion.py           # Signal combination
│       │   └── ...
│       │
│       ├── services/                  # Business logic services
│       │   ├── smart_money_analyzer.py    # ICT analysis engine
│       │   ├── signal_generator.py        # Signal generation
│       │   ├── market_data.py             # Market data
│       │   ├── free_market_data.py        # Free data (yfinance)
│       │   ├── backtest_engine.py         # Backtesting
│       │   ├── paper_trading.py           # Paper trading
│       │   └── ...
│       │
│       ├── models/                    # Data models
│       │   ├── signal.py
│       │   ├── concept.py
│       │   ├── video.py
│       │   └── user.py
│       │
│       ├── api/                       # API routes
│       │   └── signals.py
│       │
│       └── utils/                     # Utilities
│           ├── json_utils.py              # Fast JSON (orjson)
│           └── audio_downloader.py        # YouTube download
│
├── frontend/                          # React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Signals.jsx
│   │   │   ├── Learning.jsx
│   │   │   ├── Concepts.jsx
│   │   │   ├── Performance.jsx
│   │   │   ├── HedgeFund.jsx
│   │   │   └── LiveChart.jsx
│   │   └── services/
│   │       └── api.js
│   └── ...
│
├── data/                              # Data storage
│   ├── audio/                         # Downloaded audio files
│   │   └── {video_id}.mp3
│   ├── video_frames/                  # Extracted frames
│   │   └── {video_id}/
│   │       └── frame_*.jpg
│   ├── transcripts/                   # Video transcripts
│   │   └── {video_id}.json
│   ├── playlists/                     # Playlist metadata
│   │   └── {playlist_id}.json
│   ├── audio_first_training/          # Training outputs
│   │   ├── {video_id}_teaching_units.json
│   │   ├── {video_id}_selected_frames.json
│   │   ├── {video_id}_vision_analysis.json
│   │   ├── {video_id}_knowledge_base.json
│   │   └── {video_id}_knowledge_summary.md
│   ├── charts/                        # Generated charts
│   └── tradingmamba.db               # SQLite database
│
├── scripts/                           # Utility scripts
├── docs/                              # Documentation
├── ml-training/                       # Training notebooks
│
├── requirements.txt                   # Python dependencies
├── run_api.py                         # API runner
└── README.md
```

---

## Technology Stack

### Backend

| Technology | Purpose | Why |
|------------|---------|-----|
| **FastAPI** | Web framework | Async, fast, auto-docs |
| **uvloop** | Event loop | 2-4x faster than default |
| **orjson** | JSON serialization | 6x faster than stdlib |
| **SQLite** | Database | Simple, no setup |
| **Pydantic** | Validation | Type safety |

### ML/AI

| Technology | Purpose | Why |
|------------|---------|-----|
| **faster-whisper** | Speech-to-text | Fast, accurate transcription |
| **MLX-VLM** | Vision model | 5-7x faster on Apple Silicon |
| **Qwen2-VL-2B-Instruct** | Vision LLM | Good quality, 4-bit quantized |
| **OpenCV** | Image processing | Frame deduplication |

### Data Sources

| Technology | Purpose | Why |
|------------|---------|-----|
| **pytubefix** | YouTube download | Handles auth issues |
| **yfinance** | Market data | Free, reliable |
| **ffmpeg** | Video processing | Frame extraction |

### Frontend

| Technology | Purpose |
|------------|---------|
| **React** | UI framework |
| **Vite** | Build tool |
| **TailwindCSS** | Styling |

---

## Performance Optimizations

### Applied Optimizations

| Optimization | Improvement | Location |
|--------------|-------------|----------|
| **uvloop** | 2-4x faster async | `main.py` |
| **orjson** | 6x faster JSON | `json_utils.py` |
| **MLX-VLM** | 5-7x faster vision | `video_vision_analyzer.py` |
| **faster-whisper** | 4x faster transcription | `video_preprocessor.py` |
| **Smart Frame Selection** | 77% fewer frames | `audio_first_learning.py` |
| **Frame Deduplication** | ~60% reduction | `video_vision_analyzer.py` |

### Memory Management

- Lazy loading of ML models
- Streaming for large files
- Batch processing for vision analysis
- Automatic cleanup of temp files

### Benchmarks (18-minute video)

| Phase | Time | Notes |
|-------|------|-------|
| Audio Download | ~30s | Depends on connection |
| Frame Extraction | ~60s | ~350 frames @ 3s interval |
| Transcription | ~90s | faster-whisper small model |
| Teaching Detection | <1s | Pure Python |
| Frame Selection | <1s | Reduces to ~80 frames |
| Vision Analysis | ~35min | ~24s per frame with MLX |
| Knowledge Synthesis | ~3min | LLM generation |
| **Total** | **~40-45min** | |

---

## Getting Started

### Prerequisites

```bash
# Python 3.10+
python --version

# FFmpeg (for frame extraction)
brew install ffmpeg  # macOS
apt install ffmpeg   # Ubuntu

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Start the backend
python run_api.py

# 2. Start the frontend
cd frontend && npm run dev

# 3. Train on a video
python -m backend.app.ml.audio_first_learning 'https://youtube.com/watch?v=VIDEO_ID'
```

### Environment Variables

Create a `.env` file:

```env
# Optional - for Claude/GPT vision analysis
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional - for notifications
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

---

## Future Enhancements

- [ ] Real-time signal streaming via WebSocket
- [ ] Multi-broker integration
- [ ] Advanced backtesting with Monte Carlo
- [ ] Portfolio optimization
- [ ] Mobile app (React Native)

---

*Last Updated: January 2025*
