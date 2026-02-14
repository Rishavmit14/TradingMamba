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
