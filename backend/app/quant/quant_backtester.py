"""Quant Algo Bias Backtester — Historical Replay of Alert Threshold Crossings.

Replays 9/10 institutional algorithms over historical data using multi-source
free data: Binance futures klines (taker buy/sell), Coinalyze (OI, liquidations,
L/S ratios), and Deribit history (options trades → P/C, IV, skew).

At each step, assembles raw_data, runs run_algo_bias(), detects alert conditions
(same thresholds as live dashboard), and evaluates price outcomes at T+1h/4h/24h.

Only OFI remains excluded (requires L2 order book depth — no free historical source).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable, Optional

import httpx

from app.quant.algo_bias import run_algo_bias, CompositeBias
from app.quant.alert_detector import detect_alerts, QuantAlert
from app.quant.external_data import (
    get_coinalyze_key,
    fetch_coinalyze_oi,
    fetch_coinalyze_liquidations,
    fetch_coinalyze_ls_ratio,
    fetch_deribit_options_hourly,
    derive_taker_from_klines,
    synthesize_liquidation_events,
    lookup_options_data,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
QUANT_CACHE_DIR = DATA_DIR / "historical" / "quant"
QUANT_BACKTEST_DIR = DATA_DIR / "backtest" / "quant"

BINANCE_BASE = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_FUTURES_DATA = "https://fapi.binance.com"

# ---------------------------------------------------------------------------
# Alert record
# ---------------------------------------------------------------------------

@dataclass
class AlertRecord:
    """A triggered alert with outcome evaluation."""
    alert_id: str
    severity: str
    title: str
    direction: str
    direction_reason: str
    combo: Optional[str]
    trigger_timestamp: int      # ms
    trigger_price: float
    composite_score: float
    composite_confidence: float
    composite_direction: str
    metrics: dict = field(default_factory=dict)

    # Outcome
    price_1h: float = 0.0
    price_4h: float = 0.0
    price_24h: float = 0.0
    move_1h_pct: float = 0.0
    move_4h_pct: float = 0.0
    move_24h_pct: float = 0.0
    correct_1h: Optional[bool] = None
    correct_4h: Optional[bool] = None
    correct_24h: Optional[bool] = None
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Historical data download + cache
# ---------------------------------------------------------------------------

async def _fetch_paginated(
    client: httpx.AsyncClient,
    url: str,
    params: dict,
    start_ms: int,
    end_ms: int,
    limit: int = 500,
    ts_key: str = "timestamp",
) -> list[dict]:
    """Generic paginated fetcher for Binance data endpoints."""
    all_data = []
    current_start = start_ms

    while current_start < end_ms:
        p = {**params, "startTime": current_start, "endTime": end_ms, "limit": limit}
        resp = await client.get(url, params=p)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_data.extend(batch)

        # Get last timestamp for pagination
        last_item = batch[-1]
        if isinstance(last_item, list):
            last_ts = int(last_item[0])
        elif isinstance(last_item, dict):
            # fundingTime for funding rate, timestamp for others
            last_ts = int(last_item.get(ts_key, last_item.get("fundingTime", 0)))
        else:
            break

        if last_ts <= current_start:
            break
        current_start = last_ts + 1

        if len(batch) < limit:
            break

        await asyncio.sleep(0.15)  # Rate limiting

    return all_data


async def _fetch_klines_historical(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    """Fetch klines with pagination."""
    now_ms = int(time.time() * 1000)
    end_ms = min(end_ms, now_ms)
    all_klines = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        # Use futures API for futures-specific taker volume data
        resp = await client.get(f"{BINANCE_FAPI}/fapi/v1/klines", params=params)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        for k in batch:
            all_klines.append({
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "taker_buy_vol": float(k[9]),        # Taker buy base asset volume (k[8] is trade count)
                "taker_buy_quote_vol": float(k[10]),  # Taker buy quote asset volume
            })

        last_ts = int(batch[-1][0])
        if last_ts <= current_start:
            break
        current_start = last_ts + 1

        if len(batch) < 1000:
            break

        await asyncio.sleep(0.15)

    return all_klines


async def _fetch_funding_rate_history(
    client: httpx.AsyncClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    """Fetch funding rate history (8h intervals)."""
    now_ms = int(time.time() * 1000)
    end_ms = min(end_ms, now_ms)
    all_rates = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = await client.get(f"{BINANCE_FAPI}/fapi/v1/fundingRate", params=params)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        for r in batch:
            all_rates.append({
                "timestamp": int(r["fundingTime"]),
                "rate": float(r["fundingRate"]),
                "mark_price": float(r.get("markPrice") or 0),
            })

        last_ts = int(batch[-1]["fundingTime"])
        if last_ts <= current_start:
            break
        current_start = last_ts + 1

        if len(batch) < 1000:
            break

        await asyncio.sleep(0.15)

    return all_rates


async def _fetch_futures_data(
    client: httpx.AsyncClient,
    endpoint: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    period: str = "1h",
    limit: int = 500,
) -> list[dict]:
    """Fetch Binance futures data (OI, taker volume, L/S ratios).

    NOTE: These Binance endpoints do NOT support startTime for historical lookback.
    They only return the most recent `limit` records (max 500 at 1h = ~21 days).
    We fetch the maximum available and the replay engine uses whatever overlaps
    with its time window.
    """
    url = f"{BINANCE_FUTURES_DATA}{endpoint}"
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit,
    }
    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPStatusError, httpx.ConnectError) as e:
        logger.warning(f"Futures data fetch error for {endpoint}: {e}")
        return []


async def _fetch_fng_history() -> list[dict]:
    """Fetch Fear & Greed Index history (daily, all available)."""
    cache_path = QUANT_CACHE_DIR / "fng_history.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get("https://api.alternative.me/fng/", params={"limit": 0})
        resp.raise_for_status()
        data = resp.json().get("data", [])

    # Convert to {timestamp_ms, value}
    result = []
    for d in data:
        try:
            ts = int(d.get("timestamp", 0)) * 1000  # API returns seconds
            result.append({"timestamp": ts, "value": int(d["value"]), "classification": d.get("value_classification", "")})
        except (ValueError, KeyError):
            continue

    result.sort(key=lambda x: x["timestamp"])

    os.makedirs(QUANT_CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(result, f)

    return result


def _cache_key(data_type: str, symbol: str, start: str, end: str) -> Path:
    return QUANT_CACHE_DIR / f"{data_type}_{symbol}_{start}_{end}.json"


async def fetch_all_historical(
    symbol: str,
    start_date: str,
    end_date: str,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Download and cache all historical data needed for quant backtest.

    Returns dict with keys: candles_h1, candles_h4, candles_d1,
    taker_volume, funding_rates, oi_history, top_trader, global_ratio, fng_history.
    """
    os.makedirs(QUANT_CACHE_DIR, exist_ok=True)

    start_ms = _date_to_ms(start_date)
    end_ms = _date_to_ms(end_date)

    # Check cache for each data type
    data_types = {
        "candles_h1": None,
        "candles_h4": None,
        "candles_d1": None,
        "taker_volume": None,
        "funding_rates": None,
        "oi_history": None,
        "top_trader": None,
        "global_ratio": None,
    }

    cached = {}
    to_fetch = []
    for dtype in data_types:
        cache_path = _cache_key(dtype, symbol, start_date, end_date)
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            # Check if cached candle data has taker fields (v2 format)
            if dtype.startswith("candles_") and data and "taker_buy_vol" not in data[0]:
                logger.info(f"  Cache outdated (no taker fields): {dtype} — re-fetching")
                to_fetch.append(dtype)
            else:
                cached[dtype] = data
                logger.info(f"  Cache hit: {dtype} ({len(cached[dtype])} records)")
        else:
            to_fetch.append(dtype)

    if to_fetch:
        logger.info(f"  Fetching from Binance: {to_fetch}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            for dtype in to_fetch:
                if progress_callback:
                    progress_callback(f"Downloading {dtype}...")

                if dtype == "candles_h1":
                    data = await _fetch_klines_historical(client, symbol, "1h", start_ms, end_ms)
                elif dtype == "candles_h4":
                    data = await _fetch_klines_historical(client, symbol, "4h", start_ms, end_ms)
                elif dtype == "candles_d1":
                    data = await _fetch_klines_historical(client, symbol, "1d", start_ms, end_ms)
                elif dtype == "taker_volume":
                    raw = await _fetch_futures_data(
                        client, "/futures/data/takerlongshortRatio", symbol, start_ms, end_ms)
                    data = []
                    for r in raw:
                        try:
                            data.append({"timestamp": int(r["timestamp"]),
                                         "buy_vol": float(r.get("buyVol", 0)),
                                         "sell_vol": float(r.get("sellVol", 0)),
                                         "ratio": float(r.get("buySellRatio", 1))})
                        except (KeyError, ValueError):
                            continue
                elif dtype == "funding_rates":
                    data = await _fetch_funding_rate_history(client, symbol, start_ms, end_ms)
                elif dtype == "oi_history":
                    raw = await _fetch_futures_data(
                        client, "/futures/data/openInterestHist", symbol, start_ms, end_ms)
                    data = []
                    for r in raw:
                        try:
                            data.append({"timestamp": int(r["timestamp"]),
                                         "oi_usd": float(r.get("sumOpenInterestValue", 0))})
                        except (KeyError, ValueError):
                            continue
                elif dtype == "top_trader":
                    raw = await _fetch_futures_data(
                        client, "/futures/data/topLongShortPositionRatio", symbol, start_ms, end_ms)
                    data = []
                    for r in raw:
                        try:
                            data.append({"timestamp": int(r["timestamp"]),
                                         "long_pct": float(r.get("longAccount", 0.5)) * 100,
                                         "short_pct": float(r.get("shortAccount", 0.5)) * 100,
                                         "ratio": float(r.get("longShortRatio", 1))})
                        except (KeyError, ValueError):
                            continue
                elif dtype == "global_ratio":
                    raw = await _fetch_futures_data(
                        client, "/futures/data/globalLongShortAccountRatio", symbol, start_ms, end_ms)
                    data = []
                    for r in raw:
                        try:
                            data.append({"timestamp": int(r["timestamp"]),
                                         "long_pct": float(r.get("longAccount", 0.5)) * 100,
                                         "short_pct": float(r.get("shortAccount", 0.5)) * 100,
                                         "ratio": float(r.get("longShortRatio", 1))})
                        except (KeyError, ValueError):
                            continue
                else:
                    data = []

                data.sort(key=lambda x: x.get("timestamp", 0))
                cached[dtype] = data
                logger.info(f"  Fetched {dtype}: {len(data)} records")

                # Cache to disk
                with open(_cache_key(dtype, symbol, start_date, end_date), "w") as f:
                    json.dump(data, f)

    # Fetch FnG separately (global cache, not date-range specific)
    fng = await _fetch_fng_history()
    cached["fng_history"] = fng

    # ── External data sources (Coinalyze + Deribit) ──

    # Coinalyze: OI, liquidations, L/S ratio (4h resolution, ~8-11 months)
    coinalyze_key = get_coinalyze_key()
    if coinalyze_key:
        try:
            if progress_callback:
                progress_callback("Downloading Coinalyze data (OI, liquidations, L/S)...")

            coinalyze_oi = await fetch_coinalyze_oi(symbol, start_date, end_date, coinalyze_key)
            coinalyze_liq = await fetch_coinalyze_liquidations(symbol, start_date, end_date, coinalyze_key)
            coinalyze_ls = await fetch_coinalyze_ls_ratio(symbol, start_date, end_date, coinalyze_key)

            cached["coinalyze_oi"] = coinalyze_oi
            cached["coinalyze_liq"] = coinalyze_liq
            cached["coinalyze_ls"] = coinalyze_ls
            logger.info(f"  Coinalyze: OI={len(coinalyze_oi)}, Liq={len(coinalyze_liq)}, "
                        f"L/S={len(coinalyze_ls)}")
        except Exception as e:
            logger.warning(f"  Coinalyze fetch failed: {e}")
            cached.setdefault("coinalyze_oi", [])
            cached.setdefault("coinalyze_liq", [])
            cached.setdefault("coinalyze_ls", [])
    else:
        logger.info("  No COINALYZE_API_KEY — skipping Coinalyze data")
        cached["coinalyze_oi"] = []
        cached["coinalyze_liq"] = []
        cached["coinalyze_ls"] = []

    # Deribit: option trades → hourly P/C ratio, ATM IV, skew (years of history)
    try:
        if progress_callback:
            progress_callback("Downloading Deribit options data (may take 1-2 min first time)...")

        deribit_options = await fetch_deribit_options_hourly(
            start_date, end_date, progress_callback=progress_callback,
        )
        cached["deribit_options"] = deribit_options
        logger.info(f"  Deribit: {len(deribit_options)} hourly option records")
    except Exception as e:
        logger.warning(f"  Deribit fetch failed: {e}")
        cached["deribit_options"] = []

    return cached


# ---------------------------------------------------------------------------
# Replay engine helpers
# ---------------------------------------------------------------------------

def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _slice_by_time(data: list[dict], start_ms: int, end_ms: int) -> list[dict]:
    """Binary search slice of sorted timestamp data."""
    if not data:
        return []
    timestamps = [d["timestamp"] for d in data]
    lo = bisect_left(timestamps, start_ms)
    hi = bisect_right(timestamps, end_ms)
    return data[lo:hi]


def _get_price_at_ts(candles: list[dict], ts: int) -> float:
    """Get the close price at or just before the given timestamp."""
    if not candles:
        return 0.0
    timestamps = [c["timestamp"] for c in candles]
    idx = bisect_right(timestamps, ts) - 1
    if idx < 0:
        return candles[0]["close"] if candles else 0.0
    return candles[min(idx, len(candles) - 1)]["close"]


def _lookup_fng(fng_data: list[dict], ts: int) -> int:
    """Find the nearest Fear & Greed value for a timestamp."""
    if not fng_data:
        return 50  # neutral default
    timestamps = [d["timestamp"] for d in fng_data]
    idx = bisect_right(timestamps, ts) - 1
    if idx < 0:
        return fng_data[0]["value"] if fng_data else 50
    return fng_data[min(idx, len(fng_data) - 1)]["value"]


def _fng_classification(value: int) -> str:
    if value <= 25:
        return "Extreme Fear"
    if value <= 40:
        return "Fear"
    if value <= 60:
        return "Neutral"
    if value <= 75:
        return "Greed"
    return "Extreme Greed"


def _synthesize_smart_money_proxy(coinalyze_ls: list[dict]) -> list[dict]:
    """Create a 'smart money' proxy from Coinalyze L/S using 24h moving average.

    Logic: 24h MA of L/S ratios smooths out noise → represents institutional/smart
    money positioning. The raw Coinalyze data (already used as global_ratio) acts
    as 'retail'. The smoothed version as 'top_trader' creates a meaningful
    divergence when smart money moves before retail.

    Coinalyze data is 4h resolution, so 24h = 6 data points for the MA window.
    """
    if len(coinalyze_ls) < 6:
        return coinalyze_ls

    result = []
    for i in range(len(coinalyze_ls)):
        # 24h MA window (6 x 4h candles)
        window_start = max(0, i - 5)
        window = coinalyze_ls[window_start:i + 1]
        avg_long = sum(e.get("long_pct", 50) for e in window) / len(window)
        avg_short = sum(e.get("short_pct", 50) for e in window) / len(window)
        result.append({
            "timestamp": coinalyze_ls[i]["timestamp"],
            "long_pct": round(avg_long, 2),
            "short_pct": round(avg_short, 2),
            "ratio": round(avg_long / max(avg_short, 0.01), 4),
        })

    return result


def _assemble_raw_data(
    current_ts: int,
    h1_candles: list[dict],
    h4_candles: list[dict],
    d1_candles: list[dict],
    taker_volume: list[dict],
    funding_rates: list[dict],
    oi_history: list[dict],
    top_trader: list[dict],
    global_ratio: list[dict],
    fng_history: list[dict],
    coinalyze_oi: list[dict] | None = None,
    coinalyze_liq: list[dict] | None = None,
    coinalyze_ls: list[dict] | None = None,
    deribit_options: list[dict] | None = None,
) -> dict:
    """Build the raw_data dict for run_algo_bias() from historical data at timestamp.

    Multi-source data assembly:
    - Taker buy/sell: derived from futures klines (field[8]) — 6.4 years
    - OI: Coinalyze (4h, ~8-11mo) merged with Binance (1h, ~21 days)
    - L/S ratios: Coinalyze global (4h) merged with Binance (1h, ~21 days)
    - Liquidations: Coinalyze aggregated (4h) → synthetic events
    - Options: Deribit history trades → P/C ratio, ATM IV, skew
    - DVOL: Deribit ATM IV as proxy
    """
    HOUR = 3600_000
    lookback_48h = 48 * HOUR

    # ── Slice base datasets ──
    h1_window = _slice_by_time(h1_candles, current_ts - lookback_48h, current_ts)
    h4_window = _slice_by_time(h4_candles, current_ts - 200 * 4 * HOUR, current_ts)
    d1_window = _slice_by_time(d1_candles, current_ts - 30 * 24 * HOUR, current_ts)
    funding_window = _slice_by_time(funding_rates, current_ts - 30 * 8 * HOUR, current_ts)

    # Current values
    current_funding = funding_window[-1]["rate"] if funding_window else 0
    mark_price = h1_window[-1]["close"] if h1_window else 0
    funding_mark = funding_window[-1].get("mark_price", mark_price) if funding_window else mark_price

    # ── Taker volume: derive from futures klines (FULL HISTORY) ──
    # Primary: extract taker buy/sell from kline field[8] (6.4 years)
    # Fallback: Binance futures data endpoint (21 days only)
    taker_from_klines = derive_taker_from_klines(h1_window)
    binance_taker_window = _slice_by_time(taker_volume, current_ts - lookback_48h, current_ts)
    taker_data = taker_from_klines if len(taker_from_klines) >= 6 else binance_taker_window

    # ── OI: merge Coinalyze (4h, months) with Binance (1h, 21 days) ──
    binance_oi = _slice_by_time(oi_history, current_ts - lookback_48h, current_ts)
    coinalyze_oi_window = _slice_by_time(coinalyze_oi or [], current_ts - lookback_48h, current_ts)
    # Prefer Binance (higher resolution) when available, fill gaps with Coinalyze
    if len(binance_oi) >= 6:
        oi_data = binance_oi
    elif coinalyze_oi_window:
        oi_data = coinalyze_oi_window
    else:
        oi_data = binance_oi

    # ── L/S ratios: merge Coinalyze global (4h) with Binance (1h) ──
    binance_top = _slice_by_time(top_trader, current_ts - lookback_48h, current_ts)
    binance_global = _slice_by_time(global_ratio, current_ts - lookback_48h, current_ts)
    coinalyze_ls_window = _slice_by_time(coinalyze_ls or [], current_ts - lookback_48h, current_ts)
    # For global ratio: prefer Binance, fall back to Coinalyze
    global_data = binance_global if len(binance_global) >= 6 else coinalyze_ls_window
    # For top trader: Binance when available, otherwise synthesize "smart money"
    # proxy from Coinalyze L/S using 24h moving average (smoothed = institutional)
    if len(binance_top) >= 6:
        top_data = binance_top
    elif len(coinalyze_ls_window) >= 6:
        top_data = _synthesize_smart_money_proxy(coinalyze_ls_window)
    else:
        top_data = binance_top

    # ── Liquidations: Coinalyze aggregated → synthetic events ──
    liq_events = synthesize_liquidation_events(
        coinalyze_liq or [], current_ts, mark_price,
    )

    # ── Options: Deribit historical trades → P/C ratio, ATM IV, skew ──
    options_dict, dvol = lookup_options_data(
        deribit_options or [], current_ts, mark_price,
    )

    # FnG lookup
    fng_value = _lookup_fng(fng_history, current_ts)

    # Premium approximation (futures - spot spread)
    premium_pct = current_funding * 3 * 100 if current_funding else 0

    return {
        # Multi-source replayable fields
        "oi_history": oi_data,
        "funding_current": current_funding,
        "funding_history": funding_window,
        "mark_price": funding_mark if funding_mark > 0 else mark_price,
        "index_price": mark_price,
        "top_trader_ratio": top_data,
        "global_ratio": global_data,
        "taker_volume": taker_data,
        "premium_pct": premium_pct,
        "liquidations": liq_events,
        "options": options_dict,
        "dvol": dvol,

        # Still stubbed (no free source)
        "bybit_rate": None,
        "okx_rate": None,
        "funding_dispersion": 0,
        "fear_greed": {
            "value": fng_value,
            "classification": _fng_classification(fng_value),
            "timestamp": current_ts,
        },
        "cot": {},
        "onchain": {},
        "whales": {},
        "l2": {},  # OFI stays excluded — needs L2 order book depth

        # Candles
        "candles_h1": h1_window,
        "candles_h4": h4_window,
        "candles_d1": d1_window,

        # Backtest timestamp (used by liquidation algo instead of time.time())
        "current_timestamp": current_ts,
    }


# ---------------------------------------------------------------------------
# Outcome evaluation
# ---------------------------------------------------------------------------

def _evaluate_outcome(
    record: AlertRecord,
    h1_candles: list[dict],
    directional_threshold: float = 0.5,
    non_directional_threshold: float = 1.0,
):
    """Evaluate price outcome at T+1h, T+4h, T+24h after alert trigger."""
    HOUR = 3600_000
    ts = record.trigger_timestamp

    record.price_1h = _get_price_at_ts(h1_candles, ts + 1 * HOUR)
    record.price_4h = _get_price_at_ts(h1_candles, ts + 4 * HOUR)
    record.price_24h = _get_price_at_ts(h1_candles, ts + 24 * HOUR)

    if record.trigger_price <= 0:
        return

    # Percentage moves
    record.move_1h_pct = (record.price_1h - record.trigger_price) / record.trigger_price * 100
    record.move_4h_pct = (record.price_4h - record.trigger_price) / record.trigger_price * 100
    record.move_24h_pct = (record.price_24h - record.trigger_price) / record.trigger_price * 100

    # Max favorable/adverse within 24h
    window = _slice_by_time(h1_candles, ts, ts + 24 * HOUR)
    for c in window:
        move = (c["close"] - record.trigger_price) / record.trigger_price * 100
        if record.direction == "bullish":
            record.max_favorable_pct = max(record.max_favorable_pct, move)
            record.max_adverse_pct = max(record.max_adverse_pct, -move)
        elif record.direction == "bearish":
            record.max_favorable_pct = max(record.max_favorable_pct, -move)
            record.max_adverse_pct = max(record.max_adverse_pct, move)
        else:
            record.max_favorable_pct = max(record.max_favorable_pct, abs(move))

    # Correctness evaluation
    for move_attr, correct_attr in [
        ("move_1h_pct", "correct_1h"),
        ("move_4h_pct", "correct_4h"),
        ("move_24h_pct", "correct_24h"),
    ]:
        move = getattr(record, move_attr)
        if record.direction in ("bullish", "bearish"):
            if record.direction == "bullish":
                setattr(record, correct_attr, move >= directional_threshold)
            else:
                setattr(record, correct_attr, move <= -directional_threshold)
        else:
            setattr(record, correct_attr, abs(move) >= non_directional_threshold)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _safe_pct(records: list[AlertRecord], attr: str) -> float:
    """Compute percentage of records where attr is True."""
    if not records:
        return 0.0
    total = sum(1 for r in records if getattr(r, attr) is not None)
    if total == 0:
        return 0.0
    correct = sum(1 for r in records if getattr(r, attr) is True)
    return round(correct / total * 100, 1)


def _compute_statistics(records: list[AlertRecord]) -> dict:
    """Compute comprehensive statistics from alert records."""
    if not records:
        return {
            "total_alerts": 0, "directional_alerts": 0, "non_directional_alerts": 0,
            "hit_rate_1h": 0, "hit_rate_4h": 0, "hit_rate_24h": 0,
            "reliability_score": 0, "by_alert_type": {}, "by_severity": {},
            "by_direction": {}, "by_combo": {}, "timeline": [],
        }

    directional = [r for r in records if r.direction in ("bullish", "bearish")]
    non_directional = [r for r in records if r.direction == "neutral"]

    # Overall hit rates (directional only)
    hit_1h = _safe_pct(directional, "correct_1h")
    hit_4h = _safe_pct(directional, "correct_4h")
    hit_24h = _safe_pct(directional, "correct_24h")

    # Average absolute moves
    avg_move_1h = round(mean([abs(r.move_1h_pct) for r in records]), 3) if records else 0
    avg_move_4h = round(mean([abs(r.move_4h_pct) for r in records]), 3) if records else 0
    avg_move_24h = round(mean([abs(r.move_24h_pct) for r in records]), 3) if records else 0

    # ── By Alert Type ──
    by_alert = {}
    alert_ids = sorted(set(r.alert_id for r in records))
    for aid in alert_ids:
        subset = [r for r in records if r.alert_id == aid]
        dir_subset = [r for r in subset if r.direction in ("bullish", "bearish")]
        by_alert[aid] = {
            "title": subset[0].title,
            "count": len(subset),
            "severity": subset[0].severity,
            "hit_rate_1h": _safe_pct(dir_subset, "correct_1h"),
            "hit_rate_4h": _safe_pct(dir_subset, "correct_4h"),
            "hit_rate_24h": _safe_pct(dir_subset, "correct_24h"),
            "avg_move_24h": round(mean([abs(r.move_24h_pct) for r in subset]), 3) if subset else 0,
            "avg_favorable": round(mean([r.max_favorable_pct for r in subset]), 3) if subset else 0,
            "avg_adverse": round(mean([r.max_adverse_pct for r in subset]), 3) if subset else 0,
        }

    # ── By Severity ──
    by_severity = {}
    for sev in ["critical", "warning", "info"]:
        subset = [r for r in records if r.severity == sev]
        dir_subset = [r for r in subset if r.direction in ("bullish", "bearish")]
        by_severity[sev] = {
            "count": len(subset),
            "hit_rate_4h": _safe_pct(dir_subset, "correct_4h"),
            "hit_rate_24h": _safe_pct(dir_subset, "correct_24h"),
            "avg_move_24h": round(mean([abs(r.move_24h_pct) for r in subset]), 3) if subset else 0,
        }

    # ── By Direction ──
    by_direction = {}
    for d in ["bullish", "bearish", "neutral"]:
        subset = [r for r in records if r.direction == d]
        by_direction[d] = {
            "count": len(subset),
            "hit_rate_4h": _safe_pct(subset, "correct_4h"),
            "hit_rate_24h": _safe_pct(subset, "correct_24h"),
            "avg_move_24h": round(mean([r.move_24h_pct for r in subset]), 3) if subset else 0,
        }

    # ── By Combo ──
    by_combo = {}
    combo_records = [r for r in records if r.combo]
    combo_ids = sorted(set(r.combo for r in combo_records))
    for cid in combo_ids:
        subset = [r for r in combo_records if r.combo == cid]
        dir_subset = [r for r in subset if r.direction in ("bullish", "bearish")]
        by_combo[cid] = {
            "title": subset[0].title,
            "count": len(subset),
            "hit_rate_4h": _safe_pct(dir_subset, "correct_4h"),
            "hit_rate_24h": _safe_pct(dir_subset, "correct_24h"),
            "avg_favorable": round(mean([r.max_favorable_pct for r in subset]), 3) if subset else 0,
        }

    # ── Reliability Score ──
    # Weighted: 24h accuracy (50%) + 4h accuracy (30%) + false positive penalty (20%)
    fp_rate = 0.0
    if directional:
        false_pos = sum(1 for r in directional if r.correct_24h is False)
        fp_rate = false_pos / len(directional) * 100
    reliability = round(hit_24h * 0.5 + hit_4h * 0.3 + (100 - fp_rate) * 0.2, 1) if directional else 0

    # ── Timeline ──
    timeline = [
        {
            "timestamp": r.trigger_timestamp,
            "alert_id": r.alert_id,
            "title": r.title,
            "severity": r.severity,
            "direction": r.direction,
            "trigger_price": round(r.trigger_price, 2),
            "move_1h_pct": round(r.move_1h_pct, 3),
            "move_4h_pct": round(r.move_4h_pct, 3),
            "move_24h_pct": round(r.move_24h_pct, 3),
            "correct_4h": r.correct_4h,
            "correct_24h": r.correct_24h,
            "combo": r.combo,
        }
        for r in sorted(records, key=lambda x: x.trigger_timestamp)
    ]

    return {
        "total_alerts": len(records),
        "directional_alerts": len(directional),
        "non_directional_alerts": len(non_directional),
        "hit_rate_1h": hit_1h,
        "hit_rate_4h": hit_4h,
        "hit_rate_24h": hit_24h,
        "avg_abs_move_1h": avg_move_1h,
        "avg_abs_move_4h": avg_move_4h,
        "avg_abs_move_24h": avg_move_24h,
        "reliability_score": reliability,
        "by_alert_type": by_alert,
        "by_severity": by_severity,
        "by_direction": by_direction,
        "by_combo": by_combo,
        "timeline": timeline,
    }


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

async def run_quant_backtest(
    symbol: str = "BTCUSDT",
    start_date: str = "2025-11-01",
    end_date: str = "2026-02-01",
    step_hours: int = 1,
    cooldown_hours: int = 4,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Run the quant algo bias backtest.

    Steps hourly through historical data, runs run_algo_bias() at each step,
    detects alert threshold crossings, and evaluates price outcomes.

    Args:
        symbol: Trading pair
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        step_hours: Step size in hours (default 1)
        cooldown_hours: Alert deduplication cooldown
        progress_callback: Optional (step, total) or (message: str) callback

    Returns:
        Complete backtest result dict with statistics
    """
    t_start = time.time()
    HOUR = 3600_000

    logger.info(f"Quant Backtest: {symbol} from {start_date} to {end_date}, step={step_hours}h")

    # Step 1: Download and cache all historical data
    if progress_callback:
        progress_callback("Downloading historical data...")

    historical = await fetch_all_historical(symbol, start_date, end_date, progress_callback)

    h1_candles = historical["candles_h1"]
    h4_candles = historical["candles_h4"]
    d1_candles = historical["candles_d1"]
    taker_volume = historical["taker_volume"]
    funding_rates = historical["funding_rates"]
    oi_history = historical["oi_history"]
    top_trader = historical["top_trader"]
    global_ratio = historical["global_ratio"]
    fng_history = historical["fng_history"]
    coinalyze_oi = historical.get("coinalyze_oi", [])
    coinalyze_liq = historical.get("coinalyze_liq", [])
    coinalyze_ls = historical.get("coinalyze_ls", [])
    deribit_options = historical.get("deribit_options", [])

    logger.info(f"  Data loaded: H1={len(h1_candles)}, H4={len(h4_candles)}, D1={len(d1_candles)}, "
                f"Taker={len(taker_volume)}, Funding={len(funding_rates)}, OI={len(oi_history)}, "
                f"TopTrader={len(top_trader)}, GlobalRatio={len(global_ratio)}, FnG={len(fng_history)}")
    logger.info(f"  External: Coinalyze OI={len(coinalyze_oi)}, Liq={len(coinalyze_liq)}, "
                f"L/S={len(coinalyze_ls)}, Deribit Options={len(deribit_options)}")

    # Step 2: Determine step range
    start_ms = _date_to_ms(start_date)
    end_ms = _date_to_ms(end_date)
    step_start_ms = start_ms + 48 * HOUR  # Need 48h lookback
    step_end_ms = end_ms - 24 * HOUR      # Need 24h forward for outcome eval
    step_size_ms = step_hours * HOUR

    if step_start_ms >= step_end_ms:
        return {"error": "Date range too short. Need at least 72h (48h lookback + 24h forward)."}

    total_steps = (step_end_ms - step_start_ms) // step_size_ms

    logger.info(f"  Replay: {total_steps} steps from {_ms_to_str(step_start_ms)} to {_ms_to_str(step_end_ms)}")

    # Step 3: Sliding window replay
    all_records: list[AlertRecord] = []
    recent_alerts: list[tuple[str, int]] = []  # (alert_id, timestamp) for dedup
    cooldown_ms = cooldown_hours * HOUR
    errors = 0

    for step_idx, current_ts in enumerate(range(step_start_ms, step_end_ms, step_size_ms)):
        if progress_callback:
            if callable(progress_callback):
                try:
                    progress_callback(step_idx, total_steps)
                except TypeError:
                    progress_callback(f"Step {step_idx}/{total_steps}")

        try:
            # Assemble raw_data (multi-source)
            raw_data = _assemble_raw_data(
                current_ts, h1_candles, h4_candles, d1_candles,
                taker_volume, funding_rates, oi_history,
                top_trader, global_ratio, fng_history,
                coinalyze_oi, coinalyze_liq, coinalyze_ls, deribit_options,
            )

            # Run algo bias engine
            composite = run_algo_bias(raw_data)

            # Detect alerts
            alerts = detect_alerts(composite)

            # Process each alert
            for alert in alerts:
                # Check cooldown deduplication
                is_dup = False
                for prev_id, prev_ts in recent_alerts:
                    if prev_id == alert.id and (current_ts - prev_ts) < cooldown_ms:
                        is_dup = True
                        break

                if is_dup:
                    continue

                # Record alert
                trigger_price = _get_price_at_ts(h1_candles, current_ts)
                record = AlertRecord(
                    alert_id=alert.id,
                    severity=alert.severity,
                    title=alert.title,
                    direction=alert.direction,
                    direction_reason=alert.direction_reason,
                    combo=alert.combo,
                    trigger_timestamp=current_ts,
                    trigger_price=trigger_price,
                    composite_score=composite.score,
                    composite_confidence=composite.confidence,
                    composite_direction=composite.direction,
                    metrics=alert.metrics,
                )

                # Evaluate outcome
                _evaluate_outcome(record, h1_candles)

                all_records.append(record)
                recent_alerts.append((alert.id, current_ts))

                # Clean old entries from recent_alerts (keep last 24h)
                cutoff = current_ts - 24 * HOUR
                recent_alerts = [(aid, ats) for aid, ats in recent_alerts if ats > cutoff]

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"  Step {step_idx} error at {_ms_to_str(current_ts)}: {e}")

    elapsed = time.time() - t_start
    logger.info(f"  Replay complete: {len(all_records)} alerts in {elapsed:.1f}s ({errors} errors)")

    # Step 4: Compute statistics
    stats = _compute_statistics(all_records)

    # Step 5: Build result
    result = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(elapsed, 1),
        "total_hours": total_steps,
        "step_hours": step_hours,
        "cooldown_hours": cooldown_hours,
        "directional_threshold_pct": 0.5,
        "non_directional_threshold_pct": 1.0,
        "data_source": "binance+coinalyze+deribit",
        "algos_active": 9,
        "algos_excluded": ["ofi"],
        "data_sources": {
            "binance_klines": f"H1={len(h1_candles)}, H4={len(h4_candles)}, D1={len(d1_candles)}",
            "binance_funding": f"{len(funding_rates)} records",
            "coinalyze_oi": f"{len(coinalyze_oi)} records",
            "coinalyze_liquidations": f"{len(coinalyze_liq)} records",
            "coinalyze_ls_ratio": f"{len(coinalyze_ls)} records",
            "deribit_options": f"{len(deribit_options)} hourly records",
            "fear_greed": f"{len(fng_history)} daily records",
        },
        "errors": errors,
        **stats,
    }

    # Step 6: Save to disk
    os.makedirs(QUANT_BACKTEST_DIR, exist_ok=True)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = QUANT_BACKTEST_DIR / f"quant_backtest_{timestamp_str}.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"  Report saved: {report_path}")

    return result


def get_latest_quant_backtest() -> Optional[dict]:
    """Load the most recent quant backtest results from cache."""
    if not QUANT_BACKTEST_DIR.exists():
        return None

    files = sorted(QUANT_BACKTEST_DIR.glob("quant_backtest_*.json"), reverse=True)
    if not files:
        return None

    with open(files[0]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI report printer
# ---------------------------------------------------------------------------

def print_backtest_report(result: dict):
    """Pretty-print backtest results to terminal."""
    print("\n" + "=" * 70)
    print("  QUANT ALGO BIAS BACKTEST REPORT")
    print("=" * 70)
    print(f"  Symbol:      {result['symbol']}")
    print(f"  Period:      {result['start_date']} → {result['end_date']}")
    print(f"  Steps:       {result['total_hours']} hourly steps")
    print(f"  Runtime:     {result['elapsed_seconds']}s")
    print(f"  Algos:       {result['algos_active']}/10 active (excluded: {', '.join(result['algos_excluded'])})")
    print(f"  Errors:      {result['errors']}")
    print()

    print("  ── Overall ──")
    print(f"  Total Alerts:      {result['total_alerts']}")
    print(f"  Directional:       {result['directional_alerts']}")
    print(f"  Non-directional:   {result['non_directional_alerts']}")
    print(f"  Hit Rate @ 1h:     {result['hit_rate_1h']}%")
    print(f"  Hit Rate @ 4h:     {result['hit_rate_4h']}%")
    print(f"  Hit Rate @ 24h:    {result['hit_rate_24h']}%")
    print(f"  Avg |Move| @ 24h:  {result['avg_abs_move_24h']}%")
    print(f"  Reliability Score: {result['reliability_score']}/100")
    print()

    # By alert type
    by_alert = result.get("by_alert_type", {})
    if by_alert:
        print("  ── By Alert Type ──")
        print(f"  {'Alert':<30} {'Count':>6} {'Sev':>8} {'4h HR':>7} {'24h HR':>7} {'Avg Move':>9}")
        print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*9}")
        for aid, info in sorted(by_alert.items(), key=lambda x: -x[1]["count"]):
            sev = info["severity"][:4].upper()
            print(f"  {info['title']:<30} {info['count']:>6} {sev:>8} "
                  f"{info['hit_rate_4h']:>6.1f}% {info['hit_rate_24h']:>6.1f}% "
                  f"{info['avg_move_24h']:>8.3f}%")
        print()

    # By severity
    by_sev = result.get("by_severity", {})
    if by_sev:
        print("  ── By Severity ──")
        for sev in ["critical", "warning", "info"]:
            info = by_sev.get(sev, {})
            if info.get("count", 0) > 0:
                print(f"  {sev.upper():<10} {info['count']:>5} alerts | "
                      f"4h: {info['hit_rate_4h']:.1f}% | 24h: {info['hit_rate_24h']:.1f}% | "
                      f"avg move: {info['avg_move_24h']:.3f}%")
        print()

    # By combo
    by_combo = result.get("by_combo", {})
    if by_combo:
        print("  ── Perfect Storm Combos ──")
        for cid, info in by_combo.items():
            print(f"  {info.get('title', cid):<30} {info['count']:>4}x | "
                  f"4h: {info['hit_rate_4h']:.1f}% | 24h: {info['hit_rate_24h']:.1f}% | "
                  f"avg fav: {info['avg_favorable']:.3f}%")
        print()

    # By direction
    by_dir = result.get("by_direction", {})
    if by_dir:
        print("  ── By Direction ──")
        for d in ["bullish", "bearish", "neutral"]:
            info = by_dir.get(d, {})
            if info.get("count", 0) > 0:
                print(f"  {d.upper():<10} {info['count']:>5} alerts | "
                      f"4h: {info['hit_rate_4h']:.1f}% | 24h: {info['hit_rate_24h']:.1f}%")
        print()

    print("=" * 70)
