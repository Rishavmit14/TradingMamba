# TradingMamba

An AI-powered trading signal system that learns ICT (Inner Circle Trader) methodology from YouTube videos and generates buy/sell signals with chart analysis across multiple timeframes.

## Project Overview

TradingMamba is designed to:
- Process and learn from ICT YouTube educational content
- Analyze market data using ICT concepts (Order Blocks, Fair Value Gaps, Liquidity, etc.)
- Generate trading signals with confidence scores
- Provide multi-timeframe analysis
- Send notifications via WhatsApp

## Documentation

- [System Plan](ict_ai_trading_system_plan.md) - Complete project blueprint and roadmap
- [Technical Implementation](ict_ai_trading_implementation.md) - Detailed implementation guide with code

## Core Components

1. **Video Learning Pipeline** - YouTube to Knowledge Base
2. **AI/ML Model Training** - ICT Concept Recognition
3. **Market Data Integration & Analysis**
4. **Signal Generation Engine**
5. **Web Application** - Dashboard & Charts
6. **Notification System** - WhatsApp Alerts

## Tech Stack

- **Backend:** FastAPI (Python)
- **Frontend:** Next.js 14 with shadcn/ui
- **Database:** PostgreSQL + TimescaleDB
- **Cache:** Redis
- **ML:** PyTorch, Whisper, Claude API
- **Infrastructure:** Docker, AWS/Railway

## Disclaimer

This project is for educational and informational purposes only. This is not financial advice. Trading carries risk of substantial losses. AI predictions are not guaranteed. Past performance is not indicative of future results.

## License

MIT
