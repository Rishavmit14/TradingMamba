# TradingMamba - System Architecture

```
============================================================================
                        SYSTEM ARCHITECTURE DIAGRAM
============================================================================

+---------------------------+     +---------------------------+
|     DATA SOURCES          |     |     CONTENT SOURCES       |
|  (100% Free - No API Keys)|     |  (YouTube Smart Money)    |
+---------------------------+     +---------------------------+
|                           |     |                           |
|  Yahoo Finance (yfinance) |     |  YouTube Playlists/Videos |
|  +-----------------------+|     |  +-----------------------+|
|  | Forex:  EURUSD, GBPUSD||     |  | ICT Trading Content   ||
|  | Crypto: BTC, ETH, SOL ||     |  | Smart Money Concepts  ||
|  | Indices: US30, NAS100  ||     |  | Order Flow Education  ||
|  | Metals: XAUUSD (Gold)  ||     |  +-----------------------+|
|  | Stocks: AAPL, TSLA...  ||     |                           |
|  +-----------------------+|     |  Tools:                   |
|                           |     |  - yt-dlp (download)      |
|  Timeframes:              |     |  - pytubefix (metadata)   |
|  M1 M5 M15 M30 H1 H4     |     |  - youtube-transcript-api |
|  D1 W1 MN                 |     |                           |
+-------------+-------------+     +-------------+-------------+
              |                                 |
              v                                 v
+---------------------------+     +---------------------------+
|     DATA CACHE LAYER      |     |   VIDEO PROCESSING        |
|  (data/market_cache/)     |     |   PIPELINE                |
+---------------------------+     +---------------------------+
|                           |     |                           |
|  PyArrow/Parquet storage  |     |  +-------+   +---------+ |
|  Concurrent fetch (aiohttp)|     |  | Audio |   | Visual  | |
|  Symbol mapping (Yahoo)   |     |  | Path  |   | Path    | |
|  Auto-refresh on staleness|     |  +---+---+   +----+----+ |
|                           |     |      |             |      |
+-------------+-------------+     |      v             v      |
              |                   |  +-------+   +---------+  |
              |                   |  |Whisper|   | Frame   |  |
              |                   |  |/Whis- |   | Extract |  |
              |                   |  |perX   |   | + Dedup |  |
              |                   |  +---+---+   +----+----+  |
              |                   |      |             |      |
              |                   |      v             v      |
              |                   |  +-------+   +---------+  |
              |                   |  |Trans- |   | MLX VLM |  |
              |                   |  |cripts |   | (Apple  |  |
              |                   |  |(word- |   | Silicon)|  |
              |                   |  | level)|   | or CPU  |  |
              |                   |  +---+---+   +----+----+  |
              |                   |      |             |      |
              |                   |      +------+------+      |
              |                   |             |             |
              |                   |             v             |
              |                   |  +---------------------+  |
              |                   |  | Synchronized        |  |
              |                   |  | Learning Pipeline   |  |
              |                   |  | (audio-visual       |  |
              |                   |  |  alignment +        |  |
              |                   |  |  verification)      |  |
              |                   |  +----------+----------+  |
              |                   +-------------|-------------+
              |                                 |
              |                                 v
              |                   +---------------------------+
              |                   |   ML KNOWLEDGE BASE       |
              |                   |   (data/ml_models/)       |
              |                   +---------------------------+
              |                   |                           |
              |                   |  Learned Patterns (JSON)  |
              |                   |  - Pattern frequency      |
              |                   |  - Confidence scores      |
              |                   |  - Teaching contexts      |
              |                   |  - Visual examples        |
              |                   |                           |
              |                   |  Trained Models (joblib)  |
              |                   |  - Concept classifiers    |
              |                   |  - Signal predictors      |
              |                   |  - Pattern recognizers    |
              |                   |  - Price predictors       |
              |                   |                           |
              |                   +-------------+-------------+
              |                                 |
              v                                 v
+---------------------------------------------------------------+
|                                                               |
|                   FASTAPI BACKEND (Port 8000)                 |
|                   Python 3.11+ | Uvicorn | uvloop             |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|  +----------------------------------------------------------+ |
|  |              6-TIER ML ENGINE                             | |
|  +----------------------------------------------------------+ |
|  |                                                          | |
|  |  TIER 1: Pattern Detection          TIER 2: Parameter    | |
|  |  +------------------------+         Optimization         | |
|  |  | smart_money_analyzer   |         +------------------+ | |
|  |  | - Order Blocks         |         | parameter_       | | |
|  |  | - Fair Value Gaps      |         | optimizer        | | |
|  |  | - Liquidity Levels     |         | - Grid search    | | |
|  |  | - Market Structure     |         | - Walk-forward   | | |
|  |  | - BOS/CHOCH            |         | - Validation     | | |
|  |  | - OTE Zones            |         +------------------+ | |
|  |  | - Breaker Blocks       |                              | |
|  |  | - Kill Zones           |         TIER 4: Hedge Fund   | |
|  |  +------------------------+         +------------------+ | |
|  |                                     | hedge_fund_ml    | | |
|  |  TIER 3: ML Classifiers            | - Pattern grading | | |
|  |  +------------------------+         |   (A+ to F)     | | |
|  |  | ml_models              |         | - Edge tracking  | | |
|  |  | - HistGradientBoosting |         | - Confluence     | | |
|  |  | - ExtraTreesClassifier |         | - MTF analysis   | | |
|  |  | - scikit-learn         |         +------------------+ | |
|  |  +------------------------+                              | |
|  |                                     TIER 6: Price        | |
|  |  TIER 5: Quant Engine              Predictor (Genuine ML)| |
|  |  +------------------------+         +------------------+ | |
|  |  | quant_engine           |         | price_predictor  | | |
|  |  | - Regime detection     |         | - HistGBT (45%)  | | |
|  |  |   (HMM - hmmlearn)    |         | - ExtraTrees(35%)| | |
|  |  | - Correlation matrix   |         | - Ridge (20%)    | | |
|  |  | - Risk metrics         |         | - Calibrated     | | |
|  |  | - Signal fusion        |         |   probabilities  | | |
|  |  +------------------------+         | - Walk-forward   | | |
|  |                                     |   validation     | | |
|  |  42 FEATURES (feature_engineering)  | - 3 horizons     | | |
|  |  - Price action (OHLCV ratios)      |   (5/10/20 bars) | | |
|  |  - Volatility (ATR, Bollinger)      | - Auto-retrain   | | |
|  |  - Momentum (RSI, MACD, Stoch)      +------------------+ | |
|  |  - Volume analysis                                       | |
|  |  - Trend (EMA crossovers)                                | |
|  |  - Candlestick patterns                                  | |
|  +----------------------------------------------------------+ |
|                                                               |
|  +----------------------------------------------------------+ |
|  |              SERVICES LAYER                               | |
|  +----------------------------------------------------------+ |
|  |                                                          | |
|  |  signal_generator     - Trading signal generation        | |
|  |  signal_fusion        - Multi-timeframe combination      | |
|  |  backtest_engine      - Historical pattern testing       | |
|  |  paper_trading        - Simulated trading                | |
|  |  risk_metrics         - Position sizing & risk           | |
|  |  chart_generator      - Annotated chart creation         | |
|  |  signal_scheduler     - Periodic signal generation       | |
|  |  telegram_notifier    - Alert notifications              | |
|  |  free_market_data     - Yahoo Finance integration        | |
|  |  video_processor      - YouTube video/playlist handling  | |
|  |  concept_extractor    - Smart Money concept extraction   | |
|  |                                                          | |
|  +----------------------------------------------------------+ |
|                                                               |
|  +----------------------------------------------------------+ |
|  |              API ENDPOINTS (85+)                          | |
|  +----------------------------------------------------------+ |
|  |                                                          | |
|  |  /api/signals/*          Signal generation & analysis    | |
|  |  /api/market/*           Price & OHLCV data              | |
|  |  /api/live/*             Live market data                | |
|  |  /api/ml/*               ML training, prediction, status| |
|  |  /api/quant/*            Quant engine (regimes, corr)    | |
|  |  /api/backtest/*         Backtesting                     | |
|  |  /api/optimize/*         Parameter optimization          | |
|  |  /api/hedge-fund/*       Pattern grading & confluence    | |
|  |  /api/performance/*      Signal performance tracking     | |
|  |  /api/playlists/*        Playlist management             | |
|  |  /api/transcripts/*      Transcript access               | |
|  |  /api/concepts/*         Smart Money concepts            | |
|  |  /api/training/*         Training database               | |
|  |  /api/scheduler/*        Signal scheduler                | |
|  |  /api/notifications/*    Telegram alerts                 | |
|  |  /api/session/*          Kill zones & sessions           | |
|  |  /api/db/*               Database stats                  | |
|  |  /api/data-cache/*       Cache management                | |
|  |                                                          | |
|  +----------------------------------------------------------+ |
|                                                               |
|  +----------------------------------------------------------+ |
|  |              MIDDLEWARE & PERFORMANCE                      | |
|  +----------------------------------------------------------+ |
|  |  CORS: allow_origins=["*"]                               | |
|  |  JSON: orjson (6x faster serialization)                  | |
|  |  Async: uvloop (2-4x faster event loop)                  | |
|  |  Data: Polars (10-100x faster than pandas, optional)     | |
|  |  Cache: PyArrow/Parquet for market data                  | |
|  +----------------------------------------------------------+ |
|                                                               |
+-------------------------------+-------------------------------+
                                |
                                | HTTP/REST (JSON)
                                | SSE (Server-Sent Events for streaming)
                                |
+-------------------------------v-------------------------------+
|                                                               |
|                   SQLITE DATABASE                             |
|                   (data/tradingmamba.db)                       |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | videos           |  | signals          |  | performance  | |
|  | - youtube_id     |  | - symbol         |  | - signal_id  | |
|  | - title          |  | - direction      |  | - pnl_pips   | |
|  | - playlist_id    |  | - confidence     |  | - outcome    | |
|  | - status         |  | - entry/SL/TP    |  | - concepts   | |
|  +------------------+  +------------------+  +--------------+ |
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | transcripts      |  | backtest_results |  | optimized_   | |
|  | - video_id       |  | - pattern_type   |  |   params     | |
|  | - text           |  | - win_rate       |  | - param_name | |
|  | - timestamps     |  | - profit_factor  |  | - param_value| |
|  +------------------+  +------------------+  +--------------+ |
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | ict_concepts     |  | price_predictions|  | predictor_   | |
|  | - name           |  | - direction      |  |   metrics    | |
|  | - category       |  | - confidence     |  | - accuracy   | |
|  | - trading_rules  |  | - probabilities  |  | - f1_macro   | |
|  | - keywords       |  | - was_correct    |  | - dir_acc    | |
|  +------------------+  +------------------+  +--------------+ |
|                                                               |
|  +------------------+  +------------------+                   |
|  | concept_mentions |  | ml_model_metrics |                   |
|  | training_history |  | users            |                   |
|  +------------------+  +------------------+                   |
|                                                               |
+---------------------------------------------------------------+
                                |
                                | HTTP/REST (Axios)
                                |
+-------------------------------v-------------------------------+
|                                                               |
|                   REACT FRONTEND (Port 3000)                  |
|                   Vite 5 | React 18 | TailwindCSS 3.3        |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|  +----------------------------------------------------------+ |
|  |              ROUTING (React Router v6)                    | |
|  +----------------------------------------------------------+ |
|  |  /              -> Dashboard.jsx    (1,732 lines)        | |
|  |  /live-chart    -> LiveChart.jsx    (1,643 lines)        | |
|  |  /signals       -> Signals.jsx      (791 lines)         | |
|  |  /hedge-fund    -> HedgeFund.jsx    (631 lines)         | |
|  |  /learning      -> Learning.jsx     (603 lines)         | |
|  |  /performance   -> Performance.jsx  (568 lines)         | |
|  |  /concepts      -> Concepts.jsx     (504 lines)         | |
|  +----------------------------------------------------------+ |
|                                                               |
|  +----------------------------------------------------------+ |
|  |              KEY UI COMPONENTS                            | |
|  +----------------------------------------------------------+ |
|  |                                                          | |
|  |  Dashboard:                                              | |
|  |  - PricePredictionPanel (ML predictions per symbol)      | |
|  |  - QuickAnalysis (real-time signal generation)           | |
|  |  - 6-Tier Quant Engine status                            | |
|  |  - Training hub (playlist processing)                    | |
|  |                                                          | |
|  |  LiveChart:                                              | |
|  |  - TradingView lightweight-charts (candlestick)          | |
|  |  - Pattern overlays (OB, FVG, BOS, liquidity)            | |
|  |  - Signal panel with ML prediction badge                 | |
|  |  - Multi-timeframe selector                              | |
|  |                                                          | |
|  |  Signals:                                                | |
|  |  - Signal analysis with confidence scores                | |
|  |  - ML Price Prediction (3-horizon cards)                 | |
|  |  - Pattern detection summary                             | |
|  |  - Quant engine insights                                 | |
|  |                                                          | |
|  +----------------------------------------------------------+ |
|                                                               |
|  +----------------------------------------------------------+ |
|  |              API CLIENT (services/api.js)                 | |
|  +----------------------------------------------------------+ |
|  |  Base URL: http://localhost:8000/api                      | |
|  |  70+ exported functions                                  | |
|  |  Timeout: 120s (default), 600s (chart generation)        | |
|  |  Error handling with fallback mechanisms                  | |
|  +----------------------------------------------------------+ |
|                                                               |
+---------------------------------------------------------------+

============================================================================
                        SIGNAL GENERATION FLOW
============================================================================

  Market Data (Yahoo Finance)
         |
         v
  +------+------+
  | OHLCV Fetch |  D1 timeframe (primary)
  | + Caching   |  H4, H1 (confirmation)
  +------+------+
         |
         v
  +------+------+     +-------------+     +-------------+
  | 42 Feature  |---->| ML Pattern  |---->| Signal      |
  | Extraction  |     | Engine      |     | Fusion      |
  +-------------+     | (learned    |     | (MTF combo) |
                      |  patterns)  |     +------+------+
                      +-------------+            |
                                                 v
  +-------------+     +-------------+     +------+------+
  | Quant Engine|---->| Hedge Fund  |---->| Signal      |
  | (regime,    |     | ML (grade,  |     | Generator   |
  |  correlation)|    |  confluence)|     +------+------+
  +-------------+     +-------------+            |
                                                 v
  +-------------+     +-------------+     +------+------+
  | Price       |---->| Risk Metrics|---->| Final Signal|
  | Predictor   |     | (position   |     | - Direction |
  | (Tier 6)    |     |  sizing)    |     | - Entry/SL  |
  | 3 horizons  |     +-------------+     | - TP 1/2/3  |
  +-------------+                         | - Confidence|
                                          +------+------+
                                                 |
                                    +------------+------------+
                                    |            |            |
                                    v            v            v
                              +----------+ +---------+ +----------+
                              | Database | | Frontend| | Telegram |
                              | Storage  | | Display | | Alert    |
                              +----------+ +---------+ +----------+

============================================================================
                        TRAINING PIPELINE FLOW
============================================================================

  YouTube Playlist URL
         |
         v
  +------+------+
  | Video       |  yt-dlp / pytubefix
  | Discovery   |  Extract metadata
  +------+------+
         |
    +----+----+
    |         |
    v         v
  +---+     +---+
  |Audio|   |Visual|
  |Path |   |Path  |
  +--+--+   +--+---+
     |         |
     v         v
  +------+  +------+
  |Whisper|  |Frame |  faster-whisper (large-v3)
  |/Whis- |  |Extract|  MLX VLM (Apple Silicon)
  |perX   |  |+Dedup |  or CPU fallback
  +--+---+  +--+---+
     |         |
     +----+----+
          |
          v
  +-------+--------+
  | Synchronized   |  Audio-visual alignment
  | Learning       |  Cross-modal verification
  | Pipeline       |  Contamination prevention
  +-------+--------+
          |
          v
  +-------+--------+
  | Concept        |  Smart Money concept extraction
  | Extraction     |  Pattern characteristic learning
  | & Rule         |  Trading rule extraction
  | Learning       |  Teaching context accumulation
  +-------+--------+
          |
          v
  +-------+--------+
  | Model Training |  scikit-learn classifiers
  | & Storage      |  joblib serialization
  |                |  Walk-forward validation
  +----------------+

============================================================================
                        TECH STACK SUMMARY
============================================================================

  FRONTEND                    BACKEND                     DATA/ML
  --------                    -------                     -------
  React 18                    FastAPI                     scikit-learn
  Vite 5                      Uvicorn                     NumPy / SciPy
  TailwindCSS 3.3             uvloop                      Pandas / Polars
  React Router 6              Python 3.11+                pandas-ta
  Axios                       orjson                      hmmlearn (HMM)
  lightweight-charts          aiohttp                     joblib
  lucide-react                Pydantic                    PyArrow

  MARKET DATA                 VIDEO/AUDIO                 STORAGE
  -----------                 -----------                 -------
  yfinance (FREE)             yt-dlp                      SQLite
  Yahoo Finance API           faster-whisper              Parquet cache
  No API keys needed          WhisperX                    JSON knowledge
                              mlx-vlm (Apple M1/M2/M3)   joblib models
                              pytubefix
                              youtube-transcript-api

  OPTIONAL                    NOTIFICATIONS
  --------                    -------------
  Telegram Bot API            Telegram Bot
  (alerts only)               (TELEGRAM_BOT_TOKEN)
```
