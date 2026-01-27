# TradingMamba

An AI-powered trading signal system that learns ICT (Inner Circle Trader) methodology from YouTube videos and generates buy/sell signals with chart analysis across multiple timeframes.

**100% FREE** - No paid APIs required. Uses only open-source tools and free services.

## Features

- **ICT Concept Learning** - Learns from 100+ ICT YouTube videos
- **Self-Improving ML** - Gets smarter as more videos are added
- **Real-time Signals** - Generates trading signals with confidence scores
- **Multi-Timeframe Analysis** - Analyzes H1, H4, D1 timeframes
- **ICT Pattern Detection** - Order Blocks, FVGs, Liquidity, Premium/Discount
- **Telegram Notifications** - Free signal alerts
- **Web Dashboard** - Modern React UI

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Rishavmit14/TradingMamba.git
cd TradingMamba

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install
```

### 2. Get Transcripts

```bash
# List available playlists
python scripts/get_transcripts.py --list

# Get transcripts for a playlist
python scripts/get_transcripts.py --playlist 1

# Or get all playlists
python scripts/get_transcripts.py --all
```

### 3. Train ML Models

```bash
python scripts/run_pipeline.py --train
```

### 4. Run the System

```bash
# Start backend API
python run_api.py

# Start frontend (in another terminal)
cd frontend && npm run dev
```

### 5. Access Dashboard

Open http://localhost:3000 in your browser.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/signals/analyze/{symbol}` | GET | Full ICT analysis with signal |
| `/api/signals/quick/{symbol}` | GET | Quick single-timeframe signal |
| `/api/ml/status` | GET | ML model status |
| `/api/ml/train` | POST | Trigger model training |
| `/api/market/price/{symbol}` | GET | Current price |
| `/api/market/ohlcv/{symbol}` | GET | OHLCV candle data |
| `/api/concepts` | GET | List ICT concepts |
| `/api/notifications/test` | POST | Test Telegram |

## Adding New Videos

```bash
# Add a new playlist
python scripts/add_playlist.py "https://youtube.com/playlist?list=..." --tier 2

# Add a single video
python scripts/add_playlist.py --video "https://youtube.com/watch?v=..."

# Retrain after adding videos
python scripts/run_pipeline.py --train
```

## Telegram Notifications

1. Create a bot with @BotFather on Telegram
2. Set environment variables:
```bash
export TELEGRAM_BOT_TOKEN='your_bot_token'
export TELEGRAM_CHAT_ID='your_chat_id'
```
3. Test: `curl -X POST http://localhost:8000/api/notifications/test`

## Project Structure

```
TradingMamba/
├── backend/
│   └── app/
│       ├── ml/              # ML pipeline (classifier, features, signals)
│       ├── models/          # Data models & ICT taxonomy
│       └── services/        # Market data, notifications
├── frontend/
│   └── src/
│       ├── pages/           # Dashboard, Signals, Concepts, Learning
│       └── services/        # API client
├── scripts/
│   ├── run_pipeline.py      # Main orchestrator
│   ├── get_transcripts.py   # YouTube transcript fetcher
│   └── add_playlist.py      # Add new playlists
└── data/
    ├── playlists/           # Playlist metadata
    ├── transcripts/         # Video transcripts
    └── ml_models/           # Trained models
```

## ICT Concepts Detected

- **Market Structure** - BOS, CHoCH, HH/HL/LH/LL
- **Order Blocks** - Bullish/Bearish OBs, Mitigation
- **Fair Value Gaps** - FVG, Imbalance, BPR
- **Liquidity** - BSL, SSL, Sweeps, Equal Highs/Lows
- **Premium/Discount** - Equilibrium, Optimal Zones
- **Kill Zones** - London, NY, Asian Sessions
- **Entry Models** - OTE, Silver Bullet, Power of Three

## Tech Stack

- **Backend:** FastAPI (Python)
- **Frontend:** React 18 + Vite + TailwindCSS
- **ML:** scikit-learn, pandas, numpy (100% FREE)
- **Market Data:** Yahoo Finance (FREE)
- **Transcription:** YouTube Captions + Whisper (FREE)
- **Notifications:** Telegram Bot API (FREE)

## Roadmap Progress

- [x] Phase 1: Video Learning Pipeline
- [x] Phase 2: Market Data Integration
- [x] Phase 3: ML Model Development
- [x] Phase 4: Signal Generation
- [x] Phase 5: Web Dashboard
- [x] Phase 6: Telegram Notifications
- [ ] Phase 7: More ICT Videos (ongoing)

## Documentation

- [System Plan](ict_ai_trading_system_plan.md) - Complete project blueprint
- [Technical Implementation](ict_ai_trading_implementation.md) - Implementation guide

## Disclaimer

This project is for educational and informational purposes only. This is NOT financial advice. Trading involves substantial risk of loss. AI predictions are not guaranteed. Past performance is not indicative of future results. Always do your own research and risk management.

## License

MIT
