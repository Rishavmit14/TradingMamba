"""TradingMamba configuration."""

import os

# Primary trading pair
SYMBOL = "BTCUSDT"

# Multi-TF Hierarchy: W1 → D1 → H4 → M15
TIMEFRAMES = {
    "W1": "1w",     # Macro trend direction
    "D1": "1d",     # Structure + key zones
    "H4": "4h",     # Refined zones + IDM
    "M15": "15m",   # Entry confirmation
}

# Binance kline interval mapping (all viewable timeframes)
BINANCE_INTERVALS = {
    "1M": "1M",
    "W1": "1w",
    "D1": "1d",
    "H4": "4h",
    "H1": "1h",
    "M15": "15m",
    "M5": "5m",
}

# How many candles to keep in rolling buffer per timeframe
CANDLE_BUFFER_SIZE = {
    "1M": 120,    # ~10 years of monthly
    "W1": 500,    # ~9.5 years of weekly
    "D1": 365,    # ~1 year of daily
    "H4": 500,    # ~83 days of 4h
    "H1": 500,    # ~20 days of 1h
    "M15": 1000,  # ~10 days of 15m
    "M5": 1000,   # ~3.5 days of 5m
}

# V20: Trading Style TF Hierarchy (Bias → Setup → Entry)
TRADING_STYLES = {
    "positional":  {"label": "Positional",  "bias_tf": "1M", "setup_tf": "W1",  "entry_tf": "D1"},
    "swing":       {"label": "Swing",       "bias_tf": "W1", "setup_tf": "D1",  "entry_tf": "H4"},
    "short_term":  {"label": "Short-Term",  "bias_tf": "D1", "setup_tf": "H4",  "entry_tf": "H1"},
    "intraday":    {"label": "Intraday",    "bias_tf": "H4", "setup_tf": "H1",  "entry_tf": "M15"},
}

# Swing detection: minimum candles on each side to confirm a swing point
SWING_LOOKBACK = {
    "1M": 2,
    "W1": 3,
    "D1": 3,
    "H4": 5,
    "H1": 5,
    "M15": 5,
    "M5": 5,
}

# SL buffer beyond zone (in price percentage for crypto, replaces "2-4 pips")
SL_BUFFER_PCT = 0.001  # 0.1% beyond zone

# Minimum R:R to consider a signal valid
MIN_RISK_REWARD = 1.5

# V22 SBC (Sweep Based Change of Character) standalone entry parameters
SBC_MIN_RISK_REWARD = 3.0       # V22: minimum 1:3 R:R for SBC entries
SBC_RECENCY_WINDOW = 10         # Only sweeps within last N candles qualify
SBC_CONFIRM_WINDOW = 3          # Candles after sweep to look for body close confirmation

# Crypto session windows (UTC hours)
CRYPTO_SESSIONS = {
    "asian": {"start": 0, "end": 8},        # 00:00-08:00 UTC (low vol)
    "london": {"start": 8, "end": 13},       # 08:00-13:00 UTC
    "ny": {"start": 13, "end": 21},           # 13:00-21:00 UTC (highest vol)
    "late": {"start": 21, "end": 24},         # 21:00-00:00 UTC
}

KILL_ZONES = {
    "us_equity_open": {"start": 13, "end": 15},   # 13:30 UTC
    "us_equity_close": {"start": 20, "end": 21},   # 20:00 UTC
    "cme_btc_open": {"start": 13, "end": 14},      # CME futures
}

# ── Phase 5: Demo Account + Telegram ──

DEMO_INITIAL_BALANCE = 10_000.0        # Starting paper balance (USD)
DEMO_RISK_PER_TRADE_PCT = 1.0          # Risk 1% of balance per trade
DEMO_PENDING_TIMEOUT_SECONDS = 300     # Auto-skip after 5 minutes
DEMO_MIN_SIGNAL_GRADE = "B"            # Only alert on A and B grades

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Quant Mode Configuration ──

# ATR
ATR_PERIOD = 14              # Standard ATR lookback
ATR_LONG_PERIOD = 100        # Long-term ATR for regime detection

# VPIN (Volume-Synchronized Probability of Informed Trading)
VPIN_BUCKET_COUNT = 50       # Number of volume buckets
VPIN_THRESHOLD = 0.7         # VPIN above this = high informed flow probability

# 4-Layer Quant Scoring Weights
QUANT_WEIGHTS = {"alpha": 0.55, "micro": 0.20, "risk": 0.15, "exec": 0.10}

# Volatility Regime Thresholds (ATR14/ATR100 ratio)
VOL_REGIME_LOW = 0.7         # Below this = low volatility
VOL_REGIME_HIGH = 1.3        # Above this = high volatility
VOL_REGIME_EXTREME = 2.0     # Above this = extreme — suppress signals

# Deribit DVOL Implied Volatility Thresholds
DVOL_HIGH = 80               # DVOL above this = HIGH regime override
DVOL_EXTREME = 100           # DVOL above this = EXTREME regime override

# ATR-Based SL Multipliers per Volatility Regime
ATR_SL_MULT = {"low": 1.5, "normal": 2.0, "high": 2.5, "extreme": 3.0}

# Risk Model
MAX_DRAWDOWN_PCT = 0.10      # Halt trading if drawdown exceeds 10%
SUPPRESS_ON_EXTREME_VOL = True  # Suppress signals during extreme volatility

# Cache TTLs (seconds)
CACHE_TTL_FUNDING = 60       # Cross-exchange funding: 1 minute
CACHE_TTL_OPTIONS = 600      # Deribit options: 10 minutes
CACHE_TTL_COT = 604800       # CME COT: 7 days
CACHE_TTL_ONCHAIN = 3600     # On-chain flows: 1 hour
CACHE_TTL_FNG = 3600         # Fear & Greed: 1 hour
CACHE_TTL_L2 = 5             # Order book depth: 5 seconds
L2_RECORD_INTERVAL = 10      # L2 depth recorder: seconds between snapshots
CACHE_TTL_DVOL = 60          # DVOL: 1 minute

# Optional API Keys (from environment)
CRYPTOQUANT_API_KEY = os.environ.get("CRYPTOQUANT_API_KEY", "")
COINGLASS_API_KEY = os.environ.get("COINGLASS_API_KEY", "")
