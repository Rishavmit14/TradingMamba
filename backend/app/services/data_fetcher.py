"""Binance data fetcher — historical klines via REST API.

Downloads OHLCV candlestick data for backtesting and initial analysis.
Uses the public Binance API (no API key needed for market data).
"""

from __future__ import annotations
import httpx
from app.models import Candle
from app.config import SYMBOL, BINANCE_INTERVALS

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


async def fetch_klines(
    symbol: str = SYMBOL,
    timeframe: str = "H4",
    limit: int = 500,
) -> list[Candle]:
    """Fetch historical klines from Binance REST API.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Our timeframe key ("W1", "D1", "H4", "M15")
        limit: Number of candles to fetch (max 1000 per Binance)

    Returns:
        List of Candle objects, oldest first
    """
    interval = BINANCE_INTERVALS.get(timeframe)
    if not interval:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(3):
            try:
                response = await client.get(BINANCE_KLINES_URL, params=params)
                response.raise_for_status()
                raw_klines = response.json()
                break
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt == 2:
                    raise
                import asyncio
                await asyncio.sleep(1)

    candles = []
    for i, kline in enumerate(raw_klines):
        candles.append(Candle(
            timestamp=int(kline[0]),
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[5]),
            index=i,
        ))

    return candles


async def fetch_all_timeframes(
    symbol: str = SYMBOL,
) -> dict[str, list[Candle]]:
    """Fetch candles for all timeframes in our hierarchy.

    Returns dict keyed by timeframe: {"W1": [...], "D1": [...], "H4": [...], "M15": [...]}
    """
    from app.config import CANDLE_BUFFER_SIZE

    result = {}
    for tf, limit in CANDLE_BUFFER_SIZE.items():
        result[tf] = await fetch_klines(symbol, tf, limit)

    return result
