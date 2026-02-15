"""Binance Futures API — Market Intel data fetcher.

Fetches free public data: open interest, funding rate, long/short ratios,
taker buy/sell volume, and premium index. No API key required.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

FAPI_BASE = "https://fapi.binance.com"


async def fetch_market_intel(symbol: str = "BTCUSDT") -> dict:
    """Fetch all futures market intel in parallel from Binance."""

    async with httpx.AsyncClient(timeout=15) as client:
        results = await asyncio.gather(
            client.get(f"{FAPI_BASE}/fapi/v1/openInterest", params={"symbol": symbol}),
            client.get(f"{FAPI_BASE}/futures/data/openInterestHist", params={"symbol": symbol, "period": "1h", "limit": 48}),
            # Extended OI history at 4h granularity for chart overlay (~83 days)
            client.get(f"{FAPI_BASE}/futures/data/openInterestHist", params={"symbol": symbol, "period": "4h", "limit": 500}),
            client.get(f"{FAPI_BASE}/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 30}),
            client.get(f"{FAPI_BASE}/fapi/v1/premiumIndex", params={"symbol": symbol}),
            client.get(f"{FAPI_BASE}/futures/data/topLongShortPositionRatio", params={"symbol": symbol, "period": "1h", "limit": 48}),
            client.get(f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio", params={"symbol": symbol, "period": "1h", "limit": 48}),
            client.get(f"{FAPI_BASE}/futures/data/takerlongshortRatio", params={"symbol": symbol, "period": "1h", "limit": 48}),
            return_exceptions=True,
        )

    oi_current_r, oi_hist_r, oi_chart_r, funding_r, premium_r, top_ls_r, global_ls_r, taker_r = results

    # --- Open Interest ---
    oi_current_data = _safe_json(oi_current_r)
    oi_hist_data = _safe_json(oi_hist_r) or []

    current_oi = float(oi_current_data.get("openInterest", 0)) if oi_current_data else 0
    # Estimate USD value from most recent history entry
    current_oi_usd = float(oi_hist_data[-1]["sumOpenInterestValue"]) if oi_hist_data else 0

    oi_history = []
    for entry in oi_hist_data:
        oi_history.append({
            "timestamp": entry["timestamp"],
            "oi": float(entry["sumOpenInterest"]),
            "oi_usd": float(entry["sumOpenInterestValue"]),
        })

    # Extended OI history for chart overlay (4h granularity, ~83 days)
    oi_chart_data = _safe_json(oi_chart_r) or []
    oi_chart_history = []
    for entry in oi_chart_data:
        oi_chart_history.append({
            "timestamp": entry["timestamp"],
            "oi": float(entry["sumOpenInterest"]),
            "oi_usd": float(entry["sumOpenInterestValue"]),
        })

    # --- Funding Rate ---
    funding_data = _safe_json(funding_r) or []
    premium_data = _safe_json(premium_r) or {}

    current_funding = float(premium_data.get("lastFundingRate", 0))
    mark_price = float(premium_data.get("markPrice", 0))
    index_price = float(premium_data.get("indexPrice", 0))
    next_funding_time = premium_data.get("nextFundingTime", 0)

    funding_history = []
    for entry in funding_data:
        funding_history.append({
            "timestamp": entry["fundingTime"],
            "rate": float(entry["fundingRate"]),
            "mark_price": float(entry.get("markPrice", 0)),
        })

    # --- Top Trader Long/Short Ratio ---
    top_ls_data = _safe_json(top_ls_r) or []
    top_trader_ratio = []
    for entry in top_ls_data:
        top_trader_ratio.append({
            "timestamp": entry["timestamp"],
            "long_pct": float(entry["longAccount"]),
            "short_pct": float(entry["shortAccount"]),
            "ratio": float(entry["longShortRatio"]),
        })

    # --- Global (Retail) Long/Short Ratio ---
    global_ls_data = _safe_json(global_ls_r) or []
    global_ratio = []
    for entry in global_ls_data:
        global_ratio.append({
            "timestamp": entry["timestamp"],
            "long_pct": float(entry["longAccount"]),
            "short_pct": float(entry["shortAccount"]),
            "ratio": float(entry["longShortRatio"]),
        })

    # --- Taker Buy/Sell Volume ---
    taker_data = _safe_json(taker_r) or []
    taker_volume = []
    for entry in taker_data:
        taker_volume.append({
            "timestamp": entry["timestamp"],
            "buy_vol": float(entry["buyVol"]),
            "sell_vol": float(entry["sellVol"]),
            "ratio": float(entry["buySellRatio"]),
        })

    # --- Premium Index ---
    premium_pct = ((mark_price - index_price) / index_price * 100) if index_price > 0 else 0

    return {
        "open_interest": {
            "current": current_oi,
            "current_usd": current_oi_usd,
            "history": oi_history,
            "chart_history": oi_chart_history,
        },
        "funding_rate": {
            "current": current_funding,
            "next_funding_time": next_funding_time,
            "mark_price": mark_price,
            "index_price": index_price,
            "history": funding_history,
        },
        "top_trader_ratio": top_trader_ratio,
        "global_ratio": global_ratio,
        "taker_volume": taker_volume,
        "premium_index": {
            "mark_price": mark_price,
            "index_price": index_price,
            "premium_pct": round(premium_pct, 4),
        },
    }


def _safe_json(response) -> dict | list | None:
    """Safely extract JSON from an httpx response or exception."""
    if isinstance(response, Exception):
        logger.warning("Futures API request failed: %s", response)
        return None
    try:
        if response.status_code == 200:
            return response.json()
        logger.warning("Futures API returned %d: %s", response.status_code, response.text[:200])
        return None
    except Exception as e:
        logger.warning("Failed to parse futures API response: %s", e)
        return None
