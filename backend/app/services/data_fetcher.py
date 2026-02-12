"""Binance data fetcher — historical klines via REST API.

Downloads OHLCV candlestick data for backtesting and initial analysis.
Uses the public Binance API (no API key needed for market data).
"""

from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from app.models import Candle
from app.config import SYMBOL, BINANCE_INTERVALS

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Cache directory for historical data
HISTORICAL_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "historical"


def _parse_klines(raw_klines: list, start_index: int = 0) -> list[Candle]:
    """Convert raw Binance kline arrays to Candle objects."""
    candles = []
    for i, kline in enumerate(raw_klines):
        candles.append(Candle(
            timestamp=int(kline[0]),
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[5]),
            index=start_index + i,
        ))
    return candles


def _date_to_ms(date_str: str) -> int:
    """Convert YYYY-MM-DD string to Unix milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


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
            except (httpx.ConnectError, httpx.TimeoutException):
                if attempt == 2:
                    raise
                await asyncio.sleep(1)

    return _parse_klines(raw_klines)


async def fetch_historical_klines(
    symbol: str = SYMBOL,
    timeframe: str = "H4",
    start_date: str = "2024-06-01",
    end_date: str | None = None,
) -> list[Candle]:
    """Fetch historical klines with pagination for large date ranges.

    Pages through Binance API using startTime/endTime params,
    fetching up to 1000 candles per request.

    Args:
        symbol: Trading pair
        timeframe: Our timeframe key
        start_date: Start date as YYYY-MM-DD
        end_date: End date as YYYY-MM-DD (defaults to now)

    Returns:
        List of Candle objects covering the full date range, oldest first
    """
    interval = BINANCE_INTERVALS.get(timeframe)
    if not interval:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    start_ms = _date_to_ms(start_date)
    end_ms = _date_to_ms(end_date) if end_date else int(datetime.now(timezone.utc).timestamp() * 1000)

    all_klines: list = []
    current_start = start_ms

    async with httpx.AsyncClient(timeout=30.0) as client:
        while current_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": 1000,
            }

            for attempt in range(3):
                try:
                    response = await client.get(BINANCE_KLINES_URL, params=params)
                    response.raise_for_status()
                    batch = response.json()
                    break
                except (httpx.ConnectError, httpx.TimeoutException):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1)

            if not batch:
                break

            all_klines.extend(batch)

            # Next page starts after the last candle's open time
            last_timestamp = int(batch[-1][0])
            if last_timestamp <= current_start:
                break  # No progress — avoid infinite loop
            current_start = last_timestamp + 1

            if len(batch) < 1000:
                break  # Last page

            # Rate limit: 200ms between requests
            await asyncio.sleep(0.2)

    return _parse_klines(all_klines)


async def fetch_or_cache_historical(
    symbol: str = SYMBOL,
    timeframe: str = "H4",
    start_date: str = "2024-06-01",
    end_date: str | None = None,
) -> list[Candle]:
    """Fetch historical klines with local JSON cache.

    If cached data exists for this symbol/timeframe/date range, loads from disk.
    Otherwise fetches from Binance and caches for future use.
    """
    effective_end = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache_key = f"{symbol}_{timeframe}_{start_date}_{effective_end}"
    cache_path = HISTORICAL_CACHE_DIR / f"{cache_key}.json"

    # Try loading from cache
    if cache_path.exists():
        with open(cache_path, "r") as f:
            cached = json.load(f)
        candles = []
        for i, c in enumerate(cached):
            candles.append(Candle(
                timestamp=c["timestamp"],
                open=c["open"],
                high=c["high"],
                low=c["low"],
                close=c["close"],
                volume=c["volume"],
                index=i,
            ))
        return candles

    # Fetch from Binance
    candles = await fetch_historical_klines(symbol, timeframe, start_date, end_date)

    # Save to cache
    os.makedirs(HISTORICAL_CACHE_DIR, exist_ok=True)
    serializable = [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]
    with open(cache_path, "w") as f:
        json.dump(serializable, f)

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
