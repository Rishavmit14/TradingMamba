"""Fetch historical candle + futures data into SQLite.

One-time fetch + incremental sync. First run fetches everything from --start
to now. Subsequent runs only fetch data since the last synced timestamp.

Usage:
    python3 scripts/fetch_history.py --start 2020-01-01
    python3 scripts/fetch_history.py                      # incremental sync
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import BINANCE_INTERVALS
from app.services.history_db import (
    DB_PATH,
    CANDLE_TFS,
    open_db,
    get_last_timestamp,
    get_sync_summary,
    upsert_candles,
    upsert_oi_history,
    upsert_funding_rate,
    upsert_taker_volume,
    upsert_top_trader_ratio,
    upsert_global_ratio,
)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
FAPI_BASE = "https://fapi.binance.com"

# Skip M5 by default (not used by any trading style, and is 54% of all requests)
DEFAULT_TFS = ["1M", "W1", "D1", "H4", "H1", "M15"]


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ── Candle fetcher ──


async def _fetch_candle_page(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list:
    """Fetch one page of up to 1000 candles."""
    for attempt in range(3):
        try:
            resp = await client.get(
                BINANCE_KLINES_URL,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            resp.raise_for_status()
            return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt == 2:
                print(f"    ERROR: {e}")
                return []
            await asyncio.sleep(1)
    return []


async def fetch_candles_for_tf(
    client: httpx.AsyncClient,
    conn,
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
) -> int:
    """Fetch all candles for a single TF from start_ms to end_ms, paginated."""
    interval = BINANCE_INTERVALS.get(tf)
    if not interval:
        print(f"  Skipping unknown TF: {tf}")
        return 0

    # Check if we have existing data — start from there
    table = f"candles_{tf}"
    last_ts = get_last_timestamp(conn, table)
    if last_ts and last_ts >= start_ms:
        actual_start = last_ts + 1  # Incremental: start after last known
        print(f"  {tf}: Incremental from {_ms_to_date(actual_start)}")
    else:
        actual_start = start_ms
        print(f"  {tf}: Full fetch from {_ms_to_date(actual_start)}")

    total_rows = 0
    current_start = actual_start

    while current_start < end_ms:
        batch = await _fetch_candle_page(client, symbol, interval, current_start, end_ms)
        if not batch:
            break

        rows = [
            {
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            for k in batch
        ]
        upsert_candles(conn, tf, rows)
        total_rows += len(rows)

        last_timestamp = int(batch[-1][0])
        if last_timestamp <= current_start:
            break
        current_start = last_timestamp + 1

        if len(batch) < 1000:
            break

        await asyncio.sleep(0.2)  # Rate limit

    print(f"  {tf}: {total_rows} candles synced")
    return total_rows


# ── Futures fetchers ──


async def _fetch_futures_page(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
) -> list:
    """Fetch one page of futures data."""
    for attempt in range(3):
        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (400, 404):
                # Endpoint doesn't support this date range
                return []
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt == 2:
                print(f"    ERROR: {e}")
                return []
            await asyncio.sleep(1)
    return []


async def fetch_oi_history(
    client: httpx.AsyncClient,
    conn,
    symbol: str,
) -> int:
    """Fetch OI history — last ~30 days only (Binance /futures/data/ limit).

    These endpoints don't support startTime/endTime pagination.
    We fetch at multiple periods to maximize coverage.
    """
    total = 0

    # Fetch hourly (500 rows = ~20.8 days) for fine granularity
    for period in ["1h", "4h", "1d"]:
        data = await _fetch_futures_page(
            client,
            f"{FAPI_BASE}/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": 500},
        )
        if data:
            rows = [
                {
                    "timestamp": int(d["timestamp"]),
                    "oi": float(d["sumOpenInterest"]),
                    "oi_usd": float(d["sumOpenInterestValue"]),
                }
                for d in data
            ]
            upsert_oi_history(conn, rows)
            total += len(rows)
            await asyncio.sleep(0.2)

    print(f"  OI History: {total} rows synced (~30 days, API limit)")
    return total


async def fetch_funding_rate(
    client: httpx.AsyncClient,
    conn,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> int:
    """Fetch funding rate (8h periods, limit 1000)."""
    table = "funding_rate"
    last_ts = get_last_timestamp(conn, table)
    actual_start = (last_ts + 1) if (last_ts and last_ts >= start_ms) else start_ms

    total = 0
    current = actual_start

    while current < end_ms:
        data = await _fetch_futures_page(
            client,
            f"{FAPI_BASE}/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": 1000, "startTime": current, "endTime": end_ms},
        )
        if not data:
            break

        rows = [
            {
                "timestamp": int(d["fundingTime"]),
                "rate": float(d["fundingRate"]),
                "mark_price": float(d.get("markPrice") or 0),
            }
            for d in data
        ]
        upsert_funding_rate(conn, rows)
        total += len(rows)

        last = int(data[-1]["fundingTime"])
        if last <= current:
            break
        current = last + 1
        if len(data) < 1000:
            break
        await asyncio.sleep(0.2)

    print(f"  Funding Rate: {total} rows synced")
    return total


async def fetch_taker_volume(
    client: httpx.AsyncClient,
    conn,
    symbol: str,
) -> int:
    """Fetch taker buy/sell volume — last ~30 days (Binance /futures/data/ limit)."""
    total = 0
    for period in ["1h", "4h", "1d"]:
        data = await _fetch_futures_page(
            client,
            f"{FAPI_BASE}/futures/data/takerlongshortRatio",
            {"symbol": symbol, "period": period, "limit": 500},
        )
        if data:
            rows = [
                {
                    "timestamp": int(d["timestamp"]),
                    "buy_vol": float(d["buyVol"]),
                    "sell_vol": float(d["sellVol"]),
                    "ratio": float(d["buySellRatio"]),
                }
                for d in data
            ]
            upsert_taker_volume(conn, rows)
            total += len(rows)
            await asyncio.sleep(0.2)

    print(f"  Taker Volume: {total} rows synced (~30 days, API limit)")
    return total


async def fetch_top_trader_ratio(
    client: httpx.AsyncClient,
    conn,
    symbol: str,
) -> int:
    """Fetch top trader L/S ratio — last ~30 days (Binance /futures/data/ limit)."""
    total = 0
    for period in ["1h", "4h", "1d"]:
        data = await _fetch_futures_page(
            client,
            f"{FAPI_BASE}/futures/data/topLongShortPositionRatio",
            {"symbol": symbol, "period": period, "limit": 500},
        )
        if data:
            rows = [
                {
                    "timestamp": int(d["timestamp"]),
                    "long_pct": float(d["longAccount"]),
                    "short_pct": float(d["shortAccount"]),
                    "ratio": float(d["longShortRatio"]),
                }
                for d in data
            ]
            upsert_top_trader_ratio(conn, rows)
            total += len(rows)
            await asyncio.sleep(0.2)

    print(f"  Top Trader L/S: {total} rows synced (~30 days, API limit)")
    return total


async def fetch_global_ratio(
    client: httpx.AsyncClient,
    conn,
    symbol: str,
) -> int:
    """Fetch global L/S ratio — last ~30 days (Binance /futures/data/ limit)."""
    total = 0
    for period in ["1h", "4h", "1d"]:
        data = await _fetch_futures_page(
            client,
            f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": period, "limit": 500},
        )
        if data:
            rows = [
                {
                    "timestamp": int(d["timestamp"]),
                    "long_pct": float(d["longAccount"]),
                    "short_pct": float(d["shortAccount"]),
                    "ratio": float(d["longShortRatio"]),
                }
                for d in data
            ]
            upsert_global_ratio(conn, rows)
            total += len(rows)
            await asyncio.sleep(0.2)

    print(f"  Global L/S: {total} rows synced (~30 days, API limit)")
    return total


# ── Main ──


async def main(
    start_date: str,
    symbol: str,
    timeframes: list[str],
    include_m5: bool,
    funding_only: bool = False,
) -> None:
    start_ms = _date_to_ms(start_date)
    end_ms = _now_ms()

    tfs = list(timeframes)
    if include_m5 and "M5" not in tfs:
        tfs.append("M5")

    print(f"\nTradingMamba Historical Data Fetch")
    print(f"  Symbol:     {symbol}")
    print(f"  Range:      {start_date} -> now")
    if funding_only:
        print(f"  Mode:       Funding rate only")
    else:
        print(f"  Timeframes: {', '.join(tfs)}")
    print(f"  DB:         {DB_PATH}")
    print(f"{'=' * 55}\n")

    conn = open_db()
    t0 = time.time()

    async with httpx.AsyncClient(timeout=30.0) as client:
        if not funding_only:
            # ── Candles ──
            print("CANDLES:")
            for tf in tfs:
                await fetch_candles_for_tf(client, conn, symbol, tf, start_ms, end_ms)

        # ── Futures ──
        # Funding rate: supports startTime/endTime, full history from 2019
        # OI/taker/L-S: no pagination, only last ~30 days from Binance
        print("\nFUTURES:")
        if not funding_only:
            await fetch_oi_history(client, conn, symbol)
        await fetch_funding_rate(client, conn, symbol, start_ms, end_ms)
        if not funding_only:
            await fetch_taker_volume(client, conn, symbol)
            await fetch_top_trader_ratio(client, conn, symbol)
            await fetch_global_ratio(client, conn, symbol)

    elapsed = time.time() - t0

    # Print summary
    summary = get_sync_summary(conn)
    conn.close()

    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024) if DB_PATH.exists() else 0

    print(f"\n{'=' * 55}")
    print(f"SYNC COMPLETE in {elapsed:.1f}s")
    print(f"\n  {'Table':<22} {'Rows':>10}  {'Last Sync':>18}")
    print(f"  {'─' * 22} {'─' * 10}  {'─' * 18}")
    for table, info in sorted(summary.items()):
        last = _ms_to_date(info["last_timestamp"]) if info["last_timestamp"] else "—"
        print(f"  {table:<22} {info['row_count']:>10,}  {last:>18}")
    print(f"\n  DB size: {db_size_mb:.1f} MB")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical data into SQLite")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD (default: 2020-01-01)")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--include-m5", action="store_true", help="Also fetch M5 candles (slow, +884 requests)")
    parser.add_argument("--funding-only", action="store_true", help="Only fetch funding rate history (skip candles + other futures)")
    args = parser.parse_args()

    asyncio.run(main(args.start, args.symbol, DEFAULT_TFS, args.include_m5, args.funding_only))
