# TradingMamba - System Architecture

**Last Updated:** 2026-02-08  
**Version:** 2.0 (Post ML Training Completion)  
**Repository:** https://github.com/Rishavmit14/TradingMamba

---

## 📊 System Overview

TradingMamba is an AI-powered trading signal system that learns ICT (Inner Circle Trader) Smart Money methodology from YouTube videos and generates real-time buy/sell signals using 100% free data sources.

### Core Statistics (Current State)
- **Backend:** 57 Python files, 30 ML modules, 13 services, 105+ API endpoints
- **Frontend:** 10 React components, 7 pages
- **ML Training:** 16 videos trained (100%), 111 concepts learned, 105 unique patterns
- **Data:** 37 transcripts, 15 video frame directories, 16 knowledge bases
- **Documentation:** 11 comprehensive markdown files

---

## 🏗️ High-Level Architecture Diagram

\`\`\`
┌─────────────────────────────────────────────────────────────────────┐
│                         TradingMamba System                          │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   DATA SOURCES   │         │  VIDEO SOURCES   │         │   ML TRAINING    │
│  (100% Free)     │         │   (YouTube)      │         │    PIPELINE      │
└────────┬─────────┘         └────────┬─────────┘         └────────┬─────────┘
         │                            │                            │
         │ yfinance                   │ yt-dlp                    │ Whisper
         │ (Yahoo Finance)            │ (video download)          │ (transcription)
         │                            │                            │
         ├────────────────────────────┼────────────────────────────┤
         │                            │                            │
         v                            v                            v
┌─────────────────────────────────────────────────────────────────────┐
│                        BACKEND (FastAPI)                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │ Market Data│  │  ML Engine │  │  Services  │  │  API Layer   │ │
│  │   Cache    │  │  (30 mods) │  │ (13 mods)  │  │ (105+ EPs)   │ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
         │                            │                            │
         │                            │                            │
         v                            v                            v
┌─────────────────────────────────────────────────────────────────────┐
│                      FRONTEND (React + Vite)                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │ LiveChart  │  │  Signals   │  │ HedgeFund  │  │  Learning    │ │
│  │  (2373L)   │  │   Page     │  │   Grading  │  │   Hub        │ │
│  └────────────┘  └────────────┘  └────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
\`\`\`

---

## 📁 Complete Directory Structure

See [full structure in documentation](#directory-structure) - Key highlights:

\`\`\`
TradingMamba/
├── backend/app/
│   ├── main.py (2500 lines, 105+ endpoints)
│   ├── ml/ (30 modules)
│   ├── services/ (13 modules)
│   └── models/ (4 models)
│
├── frontend/src/
│   ├── pages/ (7 pages, 2373 lines in LiveChart)
│   └── services/api.js (40+ functions)
│
├── data/
│   ├── audio_first_training/ (16 videos, 111 concepts)
│   ├── transcripts/ (37 videos)
│   ├── video_frames/ (15 directories)
│   └── ml_models/
│
└── Documentation/ (11 MD files, 220KB+ total)
\`\`\`

---

## 🧠 ML Training Status (COMPLETE)

**16/16 Forex Minions Videos Trained**

- Total Concepts: 111
- Unique Patterns: 105
- Teaching Units: 868
- Total Frames: 4,000+
- Method: Claude Code Expert Analysis
- Confidence: 90-95%

**10 Core Components Learned:**
1. Inducement (IDM) - 70% of market
2. Liquidity & Liquidity Sweep
3. Market Structure (HH/HL/LL/LH) - 50% of success
4. Break of Structure (BOS)
5. Change of Character (CHoCH)
6. Valid Pullback
7. Fair Value Gap (FVG) - 70-80% fill rate
8. Order Block (OB)
9. Premium/Discount Zones
10. Engineered Liquidity (ENG LIQ)

---

## 🔄 Complete System Workflows

### 1. Video Training Pipeline (Audio-First - RECOMMENDED)

\`\`\`
YouTube URL → yt-dlp download → Extract audio + frames
           ↓
    Whisper transcription (word-level timestamps)
           ↓
    Teaching unit detection (deictic references)
           ↓
    Frame selection (±3s from teaching units)
           ↓
    CREATE PENDING FILE (signal for Claude Code)
           ↓
    CLAUDE CODE EXPERT ANALYSIS ← YOU (the user)
    - Read transcript with timestamps
    - View selected frames
    - Extract ICT/SMC concepts
    - Write knowledge_base.json + summary.md
           ↓
    ML Engine auto-loads → 105 patterns learned
\`\`\`

### 2. Real-Time Signal Generation

\`\`\`
User request → API endpoint → Free market data (yfinance)
                            ↓
                     Smart Money Analyzer
                     - ML Pattern Engine (105 patterns)
                     - Pattern Validator (ICT rules)
                     - Conflict Resolver (confluences)
                     - Hedge Fund Grading (A+-F)
                            ↓
                     Feature Engineering (42 features)
                            ↓
                     ML Ensemble (RF + LR + GB)
                            ↓
                     Signal Fusion → Return to frontend
\`\`\`

---

## 🎯 Key Technologies

**Backend:**
- FastAPI, Uvicorn, uvloop (async)
- scikit-learn, hmmlearn (ML)
- yfinance (free market data)
- Whisper (transcription)
- SQLite, PyArrow/Parquet

**Frontend:**
- React 18, Vite 5
- TailwindCSS, lightweight-charts (TradingView)
- Axios, React Router

---

## 📚 Documentation Files

1. `ARCHITECTURE.md` (this file) - Complete system architecture
2. `CLAUDE.md` - Mandatory workflow instructions
3. `COMPLETE_SMC_KNOWLEDGE_MAP.md` - 16 videos narrative (37KB)
4. `COMPLETE_SMC_KNOWLEDGE_MAP_NEW.md` - Enriched reference (115KB)
5. `KNOWLEDGE_MAP_COMPARISON.md` - Detailed comparison
6. `JSON_ERROR_FIX_REPORT.md` - JSON error documentation
7. `PATTERN_CLEANUP_SUMMARY.md` - Pattern filter recommendations
8. `PATTERN_FILTER_ANALYSIS.md` - Code impact assessment
9. `README.md` - Project overview
10. `generate_complete_knowledge_map.py` - Knowledge map generator

---

**For complete details, see the expanded sections in the full architecture document or the codebase directly.**

**Last Updated:** 2026-02-08 | **Maintainer:** TradingMamba Project
