"""External data providers for quant backtester.

Integrates free APIs beyond Binance's limited futures data endpoints:
- Coinalyze (OI, liquidations, global L/S ratios) — free API key, 4h resolution, ~8-11 months
- Deribit history (option trades → P/C ratio, ATM IV, skew) — no auth, years of data
- Binance futures kline taker buy/sell extraction — free, 6.4 years of history
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
QUANT_CACHE_DIR = DATA_DIR / "historical" / "quant"

# Coinalyze
COINALYZE_BASE = "https://api.coinalyze.net/v1"

# Deribit
DERIBIT_HISTORY = "https://history.deribit.com/api/v2/public"

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def load_env():
    """Load .env file if it exists (no external dependency)."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def get_coinalyze_key() -> str:
    """Get Coinalyze API key from environment."""
    load_env()
    return os.environ.get("COINALYZE_API_KEY", "")


# ---------------------------------------------------------------------------
# Coinalyze API
# ---------------------------------------------------------------------------

def _to_coinalyze_symbol(symbol: str) -> str:
    """Convert BTCUSDT to Coinalyze format."""
    return f"{symbol}_PERP.A"


def _date_to_unix(date_str: str) -> int:
    """Convert YYYY-MM-DD to unix seconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _save_cache(path: Path, data: list):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


async def _coinalyze_get(
    client: httpx.AsyncClient,
    endpoint: str,
    symbol: str,
    start_date: str,
    end_date: str,
    api_key: str,
    interval: str = "4hour",
    extra_params: dict | None = None,
) -> list[dict]:
    """Generic Coinalyze API getter."""
    params = {
        "symbols": _to_coinalyze_symbol(symbol),
        "interval": interval,
        "from": _date_to_unix(start_date),
        "to": _date_to_unix(end_date),
        "api_key": api_key,
    }
    if extra_params:
        params.update(extra_params)

    resp = await client.get(f"{COINALYZE_BASE}/{endpoint}", params=params)
    resp.raise_for_status()
    raw = resp.json()

    # Coinalyze returns [{symbol, history: [...]}] per symbol
    if raw and isinstance(raw, list) and len(raw) > 0:
        return raw[0].get("history", [])
    return []


async def fetch_coinalyze_oi(
    symbol: str, start_date: str, end_date: str, api_key: str,
) -> list[dict]:
    """Fetch OI history from Coinalyze. Returns [{timestamp, oi_usd}, ...]."""
    cache_path = QUANT_CACHE_DIR / f"coinalyze_oi_{symbol}_{start_date}_{end_date}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        logger.info(f"  Coinalyze OI cache hit: {len(data)} records")
        return data

    async with httpx.AsyncClient(timeout=30.0) as client:
        raw = await _coinalyze_get(
            client, "open-interest-history", symbol, start_date, end_date, api_key,
            extra_params={"convert_to_usd": "true"},
        )

    result = []
    for entry in raw:
        result.append({
            "timestamp": int(entry["t"]) * 1000,
            "oi_usd": float(entry.get("c", 0)),
        })

    result.sort(key=lambda x: x["timestamp"])
    _save_cache(cache_path, result)
    logger.info(f"  Coinalyze OI: {len(result)} records")
    return result


async def fetch_coinalyze_liquidations(
    symbol: str, start_date: str, end_date: str, api_key: str,
) -> list[dict]:
    """Fetch liquidation history from Coinalyze.

    Returns [{timestamp, sell_liq_usd, buy_liq_usd}, ...].
    sell_liq_usd = longs liquidated (forced sells), buy_liq_usd = shorts liquidated (forced buys).
    """
    cache_path = QUANT_CACHE_DIR / f"coinalyze_liq_{symbol}_{start_date}_{end_date}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        logger.info(f"  Coinalyze liquidations cache hit: {len(data)} records")
        return data

    async with httpx.AsyncClient(timeout=30.0) as client:
        raw = await _coinalyze_get(
            client, "liquidation-history", symbol, start_date, end_date, api_key,
            extra_params={"convert_to_usd": "true"},
        )

    result = []
    for entry in raw:
        result.append({
            "timestamp": int(entry["t"]) * 1000,
            # Coinalyze: "l" = longs liquidation volume, "s" = shorts liquidation volume
            "sell_liq_usd": float(entry.get("l", 0)),   # longs liquidated = forced sells
            "buy_liq_usd": float(entry.get("s", 0)),    # shorts liquidated = forced buys
        })

    result.sort(key=lambda x: x["timestamp"])
    _save_cache(cache_path, result)
    logger.info(f"  Coinalyze liquidations: {len(result)} records")
    return result


async def fetch_coinalyze_ls_ratio(
    symbol: str, start_date: str, end_date: str, api_key: str,
) -> list[dict]:
    """Fetch global L/S ratio from Coinalyze.

    Returns [{timestamp, long_pct, short_pct, ratio}, ...].
    long_pct/short_pct are in 0-100 range.
    """
    cache_path = QUANT_CACHE_DIR / f"coinalyze_ls_{symbol}_{start_date}_{end_date}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        logger.info(f"  Coinalyze L/S cache hit: {len(data)} records")
        return data

    async with httpx.AsyncClient(timeout=30.0) as client:
        raw = await _coinalyze_get(
            client, "long-short-ratio-history", symbol, start_date, end_date, api_key,
        )

    result = []
    for entry in raw:
        long_pct = float(entry.get("l", 50))
        short_pct = float(entry.get("s", 50))
        ratio = float(entry.get("r", 1.0))
        result.append({
            "timestamp": int(entry["t"]) * 1000,
            "long_pct": long_pct,
            "short_pct": short_pct,
            "ratio": ratio,
        })

    result.sort(key=lambda x: x["timestamp"])
    _save_cache(cache_path, result)
    logger.info(f"  Coinalyze L/S: {len(result)} records")
    return result


# ---------------------------------------------------------------------------
# Deribit History API — Option Trades → Hourly Aggregates
# ---------------------------------------------------------------------------

async def fetch_deribit_options_hourly(
    start_date: str,
    end_date: str,
    currency: str = "BTC",
    progress_callback: Optional[Callable] = None,
) -> list[dict]:
    """Fetch BTC option trades from history.deribit.com and aggregate hourly.

    Returns [{timestamp, pc_ratio, atm_iv, skew_25d, trade_count}, ...].
    First run fetches all trades (~720K for 3 months) and caches hourly aggregates.
    """
    cache_path = QUANT_CACHE_DIR / f"deribit_options_{currency}_{start_date}_{end_date}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        logger.info(f"  Deribit options cache hit: {len(data)} hourly records")
        return data

    start_ms = _date_to_unix(start_date) * 1000
    end_ms = _date_to_unix(end_date) * 1000
    now_ms = int(time.time() * 1000)
    end_ms = min(end_ms, now_ms)

    HOUR = 3600_000
    DAY = 24 * HOUR
    all_hourly: list[dict] = []
    total_trades = 0

    total_days = (end_ms - start_ms) // DAY + 1

    async with httpx.AsyncClient(timeout=60.0) as client:
        current_day = start_ms
        day_count = 0

        while current_day < end_ms:
            day_end = min(current_day + DAY, end_ms)
            day_count += 1

            if progress_callback:
                try:
                    progress_callback(f"Deribit options: day {day_count}/{total_days}...")
                except TypeError:
                    pass

            # Fetch all option trades for this day
            trades = await _fetch_deribit_day_trades(client, currency, current_day, day_end)
            total_trades += len(trades)

            if trades:
                hourly = _aggregate_deribit_hourly(trades)
                all_hourly.extend(hourly)

            if day_count % 15 == 0:
                logger.info(f"  Deribit: {day_count}/{total_days} days, "
                            f"{total_trades} trades, {len(all_hourly)} hourly records")

            current_day = day_end
            await asyncio.sleep(0.1)

    all_hourly.sort(key=lambda x: x["timestamp"])
    _save_cache(cache_path, all_hourly)
    logger.info(f"  Deribit options: {len(all_hourly)} hourly records from "
                f"{day_count} days ({total_trades} trades)")
    return all_hourly


async def _fetch_deribit_day_trades(
    client: httpx.AsyncClient,
    currency: str,
    start_ms: int,
    end_ms: int,
) -> list[dict]:
    """Fetch all option trades for a single day from history.deribit.com."""
    trades = []
    current_start = start_ms

    for _ in range(100):  # Safety limit (100 pages × 1000 = 100K trades/day max)
        try:
            resp = await client.get(
                f"{DERIBIT_HISTORY}/get_last_trades_by_currency_and_time",
                params={
                    "currency": currency,
                    "kind": "option",
                    "start_timestamp": current_start,
                    "end_timestamp": end_ms,
                    "count": 1000,
                    "sorting": "asc",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
            logger.warning(f"  Deribit fetch error: {e}")
            await asyncio.sleep(1.0)
            break

        result = data.get("result", {})
        batch = result.get("trades", [])
        if not batch:
            break

        trades.extend(batch)

        last_ts = batch[-1].get("timestamp", 0)
        if last_ts <= current_start:
            break
        current_start = last_ts + 1

        if len(batch) < 1000:
            break

        await asyncio.sleep(0.05)

    return trades


def _aggregate_deribit_hourly(trades: list[dict]) -> list[dict]:
    """Aggregate option trades into hourly P/C ratio, ATM IV, skew."""
    HOUR = 3600_000
    hourly_buckets: dict[int, list[dict]] = defaultdict(list)

    for trade in trades:
        ts = trade.get("timestamp", 0)
        hour_ts = (ts // HOUR) * HOUR
        hourly_buckets[hour_ts].append(trade)

    result = []
    for hour_ts in sorted(hourly_buckets):
        bucket = hourly_buckets[hour_ts]
        metrics = _compute_hour_options_metrics(bucket)
        if metrics:
            metrics["timestamp"] = hour_ts
            result.append(metrics)

    return result


def _compute_hour_options_metrics(trades: list[dict]) -> Optional[dict]:
    """Compute P/C ratio, ATM IV, and 25-delta skew from an hour's option trades."""
    if len(trades) < 2:
        return None

    put_volume = 0.0
    call_volume = 0.0
    atm_ivs: list[float] = []
    put_otm_ivs: list[float] = []
    call_otm_ivs: list[float] = []

    for trade in trades:
        instrument = trade.get("instrument_name", "")
        parsed = _parse_instrument(instrument)
        if not parsed:
            continue

        option_type, strike, expiry_ts = parsed
        amount = abs(float(trade.get("amount", 0)))
        iv = float(trade.get("iv", 0))
        index_price = float(trade.get("index_price", 0))

        if iv <= 0 or index_price <= 0 or amount <= 0:
            continue

        # P/C volume ratio
        if option_type == "P":
            put_volume += amount
        else:
            call_volume += amount

        # ATM detection: within 5% of spot price
        moneyness = abs(strike - index_price) / index_price
        if moneyness < 0.05:
            atm_ivs.append(iv)

        # ~25 delta: 10-20% OTM, with 7-90 day expiry
        if 0.10 < moneyness < 0.20:
            trade_ts = trade.get("timestamp", 0)
            dte_days = (expiry_ts - trade_ts) / (86400 * 1000)
            if 7 < dte_days < 90:
                if option_type == "P" and strike < index_price:
                    put_otm_ivs.append(iv)
                elif option_type == "C" and strike > index_price:
                    call_otm_ivs.append(iv)

    total_volume = put_volume + call_volume
    if total_volume <= 0:
        return None

    pc_ratio = put_volume / call_volume if call_volume > 0 else 2.0
    atm_iv = sum(atm_ivs) / len(atm_ivs) if atm_ivs else 0

    # 25-delta skew: put IV - call IV (positive = put skew = bearish)
    avg_put_iv = sum(put_otm_ivs) / len(put_otm_ivs) if put_otm_ivs else 0
    avg_call_iv = sum(call_otm_ivs) / len(call_otm_ivs) if call_otm_ivs else 0
    skew_25d = (avg_put_iv - avg_call_iv) if (avg_put_iv > 0 and avg_call_iv > 0) else 0

    return {
        "pc_ratio": round(pc_ratio, 4),
        "atm_iv": round(atm_iv, 2),
        "skew_25d": round(skew_25d, 2),
        "trade_count": len(trades),
        "put_volume": round(put_volume, 4),
        "call_volume": round(call_volume, 4),
    }


def _parse_instrument(name: str) -> Optional[tuple[str, float, int]]:
    """Parse Deribit instrument name like 'BTC-28MAR25-100000-C'.

    Returns (option_type, strike, expiry_timestamp_ms) or None.
    """
    parts = name.split("-")
    if len(parts) != 4:
        return None

    option_type = parts[3]
    if option_type not in ("C", "P"):
        return None

    try:
        strike = float(parts[2])
    except ValueError:
        return None

    # Parse expiry: '28MAR25' or '7MAR25' (1-2 digit day)
    expiry_str = parts[1]
    try:
        # Find where alphabetic month code starts
        alpha_idx = 0
        for i, c in enumerate(expiry_str):
            if c.isalpha():
                alpha_idx = i
                break
        else:
            return None

        day = int(expiry_str[:alpha_idx])
        month_str = expiry_str[alpha_idx:alpha_idx + 3].upper()
        year_str = expiry_str[alpha_idx + 3:]

        month = MONTH_MAP.get(month_str)
        if month is None:
            return None

        year = 2000 + int(year_str)
        expiry_dt = datetime(year, month, day, 8, 0, tzinfo=timezone.utc)  # 08:00 UTC settlement
        expiry_ts = int(expiry_dt.timestamp() * 1000)
        return (option_type, strike, expiry_ts)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Kline taker volume extraction
# ---------------------------------------------------------------------------

def derive_taker_from_klines(h1_candles: list[dict]) -> list[dict]:
    """Derive taker buy/sell volume from kline data with taker_buy_vol field.

    Returns [{timestamp, buy_vol, sell_vol, ratio}, ...] compatible with VPIN algo.
    """
    result = []
    for c in h1_candles:
        taker_buy = c.get("taker_buy_vol", 0)
        total_vol = c.get("volume", 0)
        if total_vol <= 0 or taker_buy <= 0:
            continue
        taker_sell = max(0, total_vol - taker_buy)
        ratio = taker_buy / taker_sell if taker_sell > 0 else 2.0
        result.append({
            "timestamp": c["timestamp"],
            "buy_vol": taker_buy,
            "sell_vol": taker_sell,
            "ratio": round(ratio, 4),
        })
    return result


# ---------------------------------------------------------------------------
# Liquidation event synthesis from Coinalyze aggregated data
# ---------------------------------------------------------------------------

def synthesize_liquidation_events(
    coinalyze_liq: list[dict],
    current_ts: int,
    current_price: float,
    window_ms: int = 4 * 3600_000,
) -> list[dict]:
    """Convert Coinalyze aggregated liquidation data into individual events.

    The Liquidation Cascade algo expects [{timestamp, qty_usd, side, price}, ...].
    Coinalyze provides [{timestamp, sell_liq_usd, buy_liq_usd}, ...] per 4h interval.
    We create synthetic events from the most recent interval, using current price.
    """
    if not coinalyze_liq:
        return []

    timestamps = [d["timestamp"] for d in coinalyze_liq]
    idx = bisect_right(timestamps, current_ts) - 1
    if idx < 0:
        return []

    point = coinalyze_liq[idx]
    # Only use if within window of current time
    if current_ts - point["timestamp"] > window_ms:
        return []

    events = []
    sell_usd = point.get("sell_liq_usd", 0)
    buy_usd = point.get("buy_liq_usd", 0)

    if sell_usd > 0:
        events.append({
            "timestamp": current_ts,
            "qty_usd": sell_usd,
            "side": "sell",
            "price": current_price,
        })
    if buy_usd > 0:
        events.append({
            "timestamp": current_ts,
            "qty_usd": buy_usd,
            "side": "buy",
            "price": current_price,
        })

    return events


# ---------------------------------------------------------------------------
# Options data assembly from Deribit hourly aggregates
# ---------------------------------------------------------------------------

def lookup_options_data(
    deribit_hourly: list[dict],
    current_ts: int,
    current_price: float,
    max_age_ms: int = 2 * 3600_000,
) -> tuple[dict, float]:
    """Look up options metrics from Deribit hourly data.

    Returns (options_dict, dvol) for the algo_options_greeks and algo_vol_regime inputs.
    """
    if not deribit_hourly:
        return {}, 0

    timestamps = [d["timestamp"] for d in deribit_hourly]
    idx = bisect_right(timestamps, current_ts) - 1
    if idx < 0:
        return {}, 0

    point = deribit_hourly[idx]
    if current_ts - point["timestamp"] > max_age_ms:
        return {}, 0

    options = {
        "pc_ratio": point.get("pc_ratio", 0.85),
        "net_gex": 0,  # Cannot compute without OI per strike
        "max_pain_distance_pct": 0,  # Cannot compute without OI per strike
        "skew_25d": point.get("skew_25d", 0),
        "spot_price": current_price,
    }
    dvol = point.get("atm_iv", 0)

    return options, dvol
