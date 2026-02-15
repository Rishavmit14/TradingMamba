"""TIER 2+3 Quant Data Fetcher — cross-exchange funding, Deribit options, COT, on-chain, Fear&Greed, L2.

All external APIs are FREE public endpoints (no API keys required for TIER 2).
Each source is independently fetched with TTL caching — if one fails, others still work.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timezone

import httpx

from app.config import (
    CACHE_TTL_FUNDING, CACHE_TTL_OPTIONS, CACHE_TTL_COT,
    CACHE_TTL_ONCHAIN, CACHE_TTL_FNG, CACHE_TTL_L2,
    CRYPTOQUANT_API_KEY, COINGLASS_API_KEY,
)

logger = logging.getLogger(__name__)

# ── In-memory TTL caches ──
_cache: dict[str, dict] = {}


def _get_cached(key: str, ttl: int) -> dict | None:
    """Get cached value if fresh enough."""
    entry = _cache.get(key)
    if entry and (time.time() - entry.get("_ts", 0)) < ttl:
        return {k: v for k, v in entry.items() if k != "_ts"}
    return None


def _set_cache(key: str, data: dict):
    """Store in cache with timestamp."""
    _cache[key] = {**data, "_ts": time.time()}


# ── Black-Scholes helpers for options analytics ──

def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun approximation, error < 1.5e-7)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    return 0.5 * (1.0 + sign * y)


def _bs_d1(spot: float, strike: float, iv: float, t_years: float) -> float:
    """Black-Scholes d1 (risk-free rate ≈ 0 for crypto)."""
    if iv <= 0 or t_years <= 0 or strike <= 0 or spot <= 0:
        return 0.0
    return (math.log(spot / strike) + 0.5 * iv * iv * t_years) / (iv * math.sqrt(t_years))


def _bs_gamma(spot: float, strike: float, iv: float, t_years: float) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if iv <= 0 or t_years <= 0 or spot <= 0:
        return 0.0
    d1 = _bs_d1(spot, strike, iv, t_years)
    return _norm_pdf(d1) / (spot * iv * math.sqrt(t_years))


def _parse_deribit_instrument(name: str) -> tuple[str, float, str] | None:
    """Parse 'BTC-27FEB26-63000-C' → (expiry_str, strike, 'C'/'P') or None."""
    parts = name.split("-")
    if len(parts) != 4:
        return None
    try:
        return (parts[1], float(parts[2]), parts[3])
    except ValueError:
        return None


def _expiry_to_dt(expiry_str: str) -> datetime:
    """Convert '27FEB26' to datetime (Deribit expiries settle at 08:00 UTC)."""
    dt = datetime.strptime(expiry_str, "%d%b%y")
    return dt.replace(hour=8, tzinfo=timezone.utc)


def compute_max_pain(instruments: list[dict], spot: float) -> tuple[float, float]:
    """Compute max pain strike for nearest expiry.

    Max pain = strike that minimizes total payout to option holders.
    For each test strike S:
      - Each call with strike K: if S > K → payout += call_OI * (S - K)
      - Each put with strike K:  if S < K → payout += put_OI * (K - S)

    Returns (max_pain_strike, distance_pct_from_spot).
    """
    now = datetime.now(timezone.utc)

    # Group by expiry, find nearest
    by_expiry: dict[str, list] = {}
    for inst in instruments:
        parsed = _parse_deribit_instrument(inst.get("instrument_name", ""))
        if not parsed:
            continue
        expiry_str, strike, opt_type = parsed
        by_expiry.setdefault(expiry_str, []).append((strike, opt_type, inst.get("open_interest", 0)))

    if not by_expiry:
        return 0, 0

    # Find nearest future expiry
    nearest_expiry = None
    nearest_dt = None
    for exp_str in by_expiry:
        try:
            exp_dt = _expiry_to_dt(exp_str)
            if exp_dt > now and (nearest_dt is None or exp_dt < nearest_dt):
                nearest_dt = exp_dt
                nearest_expiry = exp_str
        except ValueError:
            continue

    if not nearest_expiry:
        return 0, 0

    # Build strike OI map for nearest expiry
    strike_oi: dict[float, dict[str, float]] = {}
    for strike, opt_type, oi in by_expiry[nearest_expiry]:
        if strike not in strike_oi:
            strike_oi[strike] = {"call_oi": 0, "put_oi": 0}
        if opt_type == "C":
            strike_oi[strike]["call_oi"] += oi
        else:
            strike_oi[strike]["put_oi"] += oi

    strikes = sorted(strike_oi.keys())
    if not strikes:
        return 0, 0

    # Find strike with minimum total payout
    min_payout = float("inf")
    max_pain_strike = strikes[0]

    for test_strike in strikes:
        total_payout = 0.0
        for strike, oi_data in strike_oi.items():
            if test_strike > strike:
                total_payout += oi_data["call_oi"] * (test_strike - strike)
            if test_strike < strike:
                total_payout += oi_data["put_oi"] * (strike - test_strike)

        if total_payout < min_payout:
            min_payout = total_payout
            max_pain_strike = test_strike

    distance_pct = ((max_pain_strike - spot) / spot * 100) if spot > 0 else 0
    return max_pain_strike, round(distance_pct, 2)


def compute_net_gex(instruments: list[dict], spot: float) -> float:
    """Compute Net Gamma Exposure (GEX) for nearest expiry.

    GEX = Σ(call_OI × gamma - put_OI × gamma) × spot² × 0.01
    Positive GEX → dealers dampen price moves (sell rallies, buy dips).
    Negative GEX → dealers amplify moves (buy rallies, sell dips).

    Returns GEX in BTC units (positive = dampening, negative = amplifying).
    """
    now = datetime.now(timezone.utc)
    nearest_expiry = None
    nearest_dt = None

    # Find nearest expiry
    for inst in instruments:
        parsed = _parse_deribit_instrument(inst.get("instrument_name", ""))
        if not parsed:
            continue
        try:
            exp_dt = _expiry_to_dt(parsed[0])
            if exp_dt > now and (nearest_dt is None or exp_dt < nearest_dt):
                nearest_dt = exp_dt
                nearest_expiry = parsed[0]
        except ValueError:
            continue

    if not nearest_expiry or not nearest_dt:
        return 0.0

    t_years = max((nearest_dt - now).total_seconds() / (365.25 * 86400), 1 / 365.25)

    net_gex = 0.0
    for inst in instruments:
        parsed = _parse_deribit_instrument(inst.get("instrument_name", ""))
        if not parsed or parsed[0] != nearest_expiry:
            continue

        _, strike, opt_type = parsed
        oi = inst.get("open_interest", 0)
        if oi <= 0:
            continue

        # mark_iv from Deribit book_summary is in percentage (e.g., 55.0 = 55%)
        iv_raw = inst.get("mark_iv", 0) or 0
        iv = iv_raw / 100.0 if iv_raw > 5 else iv_raw
        if iv <= 0:
            continue

        gamma = _bs_gamma(spot, strike, iv, t_years)

        # Dealers are net short options:
        # Short calls → negative gamma → buy delta as price rises (dampen)
        # Short puts → positive gamma → sell delta as price drops (dampen)
        if opt_type == "C":
            net_gex += oi * gamma * spot * spot * 0.01
        else:
            net_gex -= oi * gamma * spot * spot * 0.01

    return round(net_gex, 2)


def compute_25d_skew(instruments: list[dict], spot: float) -> float | None:
    """Compute 25-delta risk reversal for nearest expiry.

    Skew = IV(25Δ put) - IV(25Δ call)
    Positive → puts more expensive → bearish hedging demand / fear.
    Negative → calls more expensive → bullish positioning / complacency.

    |skew| > 5 is significant. Returns IV difference in percentage points.
    """
    now = datetime.now(timezone.utc)
    nearest_expiry = None
    nearest_dt = None

    for inst in instruments:
        parsed = _parse_deribit_instrument(inst.get("instrument_name", ""))
        if not parsed:
            continue
        try:
            exp_dt = _expiry_to_dt(parsed[0])
            if exp_dt > now and (nearest_dt is None or exp_dt < nearest_dt):
                nearest_dt = exp_dt
                nearest_expiry = parsed[0]
        except ValueError:
            continue

    if not nearest_expiry or not nearest_dt:
        return None

    t_years = max((nearest_dt - now).total_seconds() / (365.25 * 86400), 1 / 365.25)

    calls: list[dict] = []
    puts: list[dict] = []

    for inst in instruments:
        parsed = _parse_deribit_instrument(inst.get("instrument_name", ""))
        if not parsed or parsed[0] != nearest_expiry:
            continue

        _, strike, opt_type = parsed
        iv_raw = inst.get("mark_iv", 0) or 0
        iv = iv_raw / 100.0 if iv_raw > 5 else iv_raw
        if iv <= 0:
            continue

        d1 = _bs_d1(spot, strike, iv, t_years)
        call_delta = _norm_cdf(d1)

        if opt_type == "C":
            calls.append({"strike": strike, "iv_pct": iv_raw if iv_raw > 5 else iv_raw * 100, "delta": call_delta})
        else:
            put_delta = abs(call_delta - 1)  # |delta| for puts
            puts.append({"strike": strike, "iv_pct": iv_raw if iv_raw > 5 else iv_raw * 100, "delta": put_delta})

    if not calls or not puts:
        return None

    # Find instruments closest to 25-delta
    call_25d = min(calls, key=lambda x: abs(x["delta"] - 0.25))
    put_25d = min(puts, key=lambda x: abs(x["delta"] - 0.25))

    # Only use if reasonably close to 25-delta (within 0.15)
    if abs(call_25d["delta"] - 0.25) > 0.15 or abs(put_25d["delta"] - 0.25) > 0.15:
        return None

    skew = put_25d["iv_pct"] - call_25d["iv_pct"]
    return round(skew, 2)


async def fetch_all_quant_data(symbol: str = "BTCUSDT") -> dict:
    """Fetch all TIER 2+3 quant data in parallel with graceful degradation."""
    results = await asyncio.gather(
        fetch_cross_exchange_funding(symbol),
        fetch_deribit_options(),
        fetch_fear_greed(),
        fetch_cot_data(),
        fetch_onchain_flows(),
        fetch_whale_transactions(),
        return_exceptions=True,
    )

    cross_funding, options, fng, cot, onchain, whales = results

    return {
        "cross_exchange_funding": cross_funding if isinstance(cross_funding, dict) else {},
        "options_data": options if isinstance(options, dict) else {},
        "cot_data": cot if isinstance(cot, dict) else {},
        "onchain_flow": onchain if isinstance(onchain, dict) else {},
        "fear_greed": fng if isinstance(fng, dict) else {},
        "whale_transactions": whales if isinstance(whales, dict) else {},
    }


async def fetch_cross_exchange_funding(symbol: str = "BTCUSDT") -> dict:
    """Fetch funding rates from Bybit, OKX, dYdX and compare with Binance.

    Returns {binance_rate, bybit_rate, okx_rate, dydx_rate, avg_rate, dispersion}.
    """
    cached = _get_cached("cross_funding", CACHE_TTL_FUNDING)
    if cached:
        return cached

    async with httpx.AsyncClient(timeout=10) as client:
        results = await asyncio.gather(
            # Bybit
            client.get(
                "https://api.bybit.com/v5/market/tickers",
                params={"category": "linear", "symbol": "BTCUSDT"},
            ),
            # OKX
            client.get(
                "https://www.okx.com/api/v5/public/funding-rate",
                params={"instId": "BTC-USDT-SWAP"},
            ),
            return_exceptions=True,
        )

    bybit_r, okx_r = results

    rates = {}

    # Parse Bybit
    if isinstance(bybit_r, httpx.Response) and bybit_r.status_code == 200:
        try:
            data = bybit_r.json()
            tickers = data.get("result", {}).get("list", [])
            if tickers:
                rates["bybit"] = float(tickers[0].get("fundingRate", 0))
        except Exception as e:
            logger.debug("Bybit funding parse error: %s", e)

    # Parse OKX
    if isinstance(okx_r, httpx.Response) and okx_r.status_code == 200:
        try:
            data = okx_r.json()
            entries = data.get("data", [])
            if entries:
                rates["okx"] = float(entries[0].get("fundingRate", 0))
        except Exception as e:
            logger.debug("OKX funding parse error: %s", e)

    if not rates:
        return {}

    # Compute dispersion
    all_rates = list(rates.values())
    avg_rate = sum(all_rates) / len(all_rates) if all_rates else 0
    dispersion = max(all_rates) - min(all_rates) if len(all_rates) > 1 else 0

    result = {
        "bybit_rate": rates.get("bybit", 0),
        "okx_rate": rates.get("okx", 0),
        "avg_rate": round(avg_rate, 8),
        "dispersion": round(dispersion, 8),
        "sources_available": len(rates),
    }

    _set_cache("cross_funding", result)
    return result


async def fetch_deribit_options() -> dict:
    """Fetch Deribit BTC options data: P/C ratio, max pain, GEX, 25d skew.

    Uses public API — no key required. All computations from OI + mark_iv.
    """
    cached = _get_cached("deribit_options", CACHE_TTL_OPTIONS)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                params={"currency": "BTC", "kind": "option"},
            )

            if resp.status_code != 200:
                return {}

            data = resp.json()
            instruments = data.get("result", [])
            if not instruments:
                return {}

            result = compute_options_analytics(instruments)
            _set_cache("deribit_options", result)
            return result

    except Exception as e:
        logger.warning("Deribit options fetch failed: %s", e)
        return {}


def compute_options_analytics(instruments: list[dict]) -> dict:
    """Compute all options analytics from Deribit book summary instruments.

    Shared by both async (quant_data.py) and sync (quant/engine.py) paths.
    """
    # Get spot price from any instrument's underlying_price
    spot = 0.0
    for inst in instruments:
        up = inst.get("underlying_price", 0)
        if up and up > 0:
            spot = up
            break

    # P/C ratio from aggregate OI (all expiries)
    total_put_oi = 0.0
    total_call_oi = 0.0
    for inst in instruments:
        name = inst.get("instrument_name", "")
        oi = inst.get("open_interest", 0)
        if "-P" in name:
            total_put_oi += oi
        elif "-C" in name:
            total_call_oi += oi

    pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

    # Max pain (nearest expiry)
    max_pain, max_pain_dist = compute_max_pain(instruments, spot) if spot > 0 else (0, 0)

    # Net GEX (nearest expiry)
    net_gex = compute_net_gex(instruments, spot) if spot > 0 else 0

    # 25-delta skew (nearest expiry)
    skew_25d = compute_25d_skew(instruments, spot) if spot > 0 else None

    return {
        "pc_ratio": round(pc_ratio, 3),
        "total_put_oi": total_put_oi,
        "total_call_oi": total_call_oi,
        "max_pain": max_pain,
        "net_gex": net_gex,
        "skew_25d": skew_25d,
        "max_pain_distance_pct": max_pain_dist,
        "spot_price": spot,
    }


async def fetch_fear_greed() -> dict:
    """Fetch Fear & Greed Index from alternative.me (FREE, no key)."""
    cached = _get_cached("fear_greed", CACHE_TTL_FNG)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.alternative.me/fng/", params={"limit": 1})
            if resp.status_code != 200:
                return {}

            data = resp.json()
            entries = data.get("data", [])
            if not entries:
                return {}

            entry = entries[0]
            result = {
                "value": int(entry.get("value", 50)),
                "classification": entry.get("value_classification", "Neutral"),
                "timestamp": int(entry.get("timestamp", 0)) * 1000,  # Convert to ms
            }

            _set_cache("fear_greed", result)
            return result

    except Exception as e:
        logger.warning("Fear & Greed fetch failed: %s", e)
        return {}


async def fetch_cot_data() -> dict:
    """Fetch CFTC COT data for Bitcoin futures (FREE, no key).

    Uses the Traders in Financial Futures (TFF) report from CFTC Socrata API.
    Extracts leveraged money (hedge fund) and asset manager net positioning.
    Computes percentile rank over 52-week history for contrarian signals.

    7-day TTL cache (data is published weekly on Fridays).
    """
    cached = _get_cached("cot_data", CACHE_TTL_COT)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Fetch 52 weeks of Bitcoin futures TFF data
            resp = await client.get(
                "https://publicreporting.cftc.gov/resource/gpe5-46if.json",
                params={
                    "$where": "contract_market_name like '%BITCOIN%'",
                    "$order": "report_date_as_yyyy_mm_dd DESC",
                    "$limit": "200",
                    "$select": (
                        "report_date_as_yyyy_mm_dd,contract_market_name,"
                        "cftc_contract_market_code,open_interest_all,"
                        "lev_money_positions_long,lev_money_positions_short,"
                        "asset_mgr_positions_long,asset_mgr_positions_short,"
                        "dealer_positions_long_all,dealer_positions_short_all,"
                        "change_in_lev_money_long,change_in_lev_money_short"
                    ),
                },
            )

            if resp.status_code != 200:
                return {}

            data = resp.json()
            if not data:
                return {}

            # Aggregate across all Bitcoin contracts (BTC + Micro + Nano) per date
            by_date: dict[str, dict] = {}
            for row in data:
                date = row.get("report_date_as_yyyy_mm_dd", "")[:10]
                if date not in by_date:
                    by_date[date] = {
                        "lev_long": 0, "lev_short": 0,
                        "asset_long": 0, "asset_short": 0,
                        "dealer_long": 0, "dealer_short": 0,
                    }
                d = by_date[date]
                d["lev_long"] += int(row.get("lev_money_positions_long", 0) or 0)
                d["lev_short"] += int(row.get("lev_money_positions_short", 0) or 0)
                d["asset_long"] += int(row.get("asset_mgr_positions_long", 0) or 0)
                d["asset_short"] += int(row.get("asset_mgr_positions_short", 0) or 0)
                d["dealer_long"] += int(row.get("dealer_positions_long_all", 0) or 0)
                d["dealer_short"] += int(row.get("dealer_positions_short_all", 0) or 0)

            if not by_date:
                return {}

            # Sort by date descending
            sorted_dates = sorted(by_date.keys(), reverse=True)
            latest_date = sorted_dates[0]
            latest = by_date[latest_date]

            # Net positioning
            leveraged_net = latest["lev_long"] - latest["lev_short"]
            asset_mgr_net = latest["asset_long"] - latest["asset_short"]
            dealer_net = latest["dealer_long"] - latest["dealer_short"]

            # Compute percentile rank from historical leveraged net
            historical_nets = [
                by_date[d]["lev_long"] - by_date[d]["lev_short"]
                for d in sorted_dates
            ]
            percentile = _compute_percentile(leveraged_net, historical_nets)

            # Week-over-week change
            prev_date = sorted_dates[1] if len(sorted_dates) > 1 else None
            wow_change = 0
            if prev_date:
                prev = by_date[prev_date]
                prev_net = prev["lev_long"] - prev["lev_short"]
                wow_change = leveraged_net - prev_net

            result = {
                "report_date": latest_date,
                "leveraged_net": leveraged_net,
                "asset_manager_net": asset_mgr_net,
                "dealer_net": dealer_net,
                "leveraged_long": latest["lev_long"],
                "leveraged_short": latest["lev_short"],
                "percentile": round(percentile, 1),
                "wow_change": wow_change,
                "weeks_of_data": len(sorted_dates),
                "signal": _cot_signal(percentile),
            }

            _set_cache("cot_data", result)
            return result

    except Exception as e:
        logger.warning("COT data fetch failed: %s", e)
        return {}


def _compute_percentile(value: float, history: list[float]) -> float:
    """Compute percentile rank of value within history (0-100)."""
    if not history:
        return 50.0
    below = sum(1 for v in history if v < value)
    return (below / len(history)) * 100


def _cot_signal(percentile: float) -> str:
    """Convert COT percentile to signal string."""
    if percentile > 80:
        return "extreme_long"   # Contrarian bearish
    elif percentile > 60:
        return "moderately_long"
    elif percentile < 20:
        return "extreme_short"  # Contrarian bullish
    elif percentile < 40:
        return "moderately_short"
    return "neutral"


async def fetch_onchain_flows() -> dict:
    """Fetch on-chain BTC exchange flows from CoinMetrics Community API (FREE, no key).

    Exchange inflows = BTC deposited to exchanges (potential selling pressure).
    Exchange outflows = BTC withdrawn from exchanges (accumulation/hodling).
    Net flow = inflow - outflow (positive = selling pressure, negative = accumulation).

    1-hour TTL cache.
    """
    cached = _get_cached("onchain_flow", CACHE_TTL_ONCHAIN)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
                params={
                    "assets": "btc",
                    "metrics": "FlowInExNtv,FlowOutExNtv",
                    "frequency": "1d",
                    "page_size": "14",
                },
            )

            if resp.status_code != 200:
                return {}

            data = resp.json()
            rows = data.get("data", [])
            if not rows:
                return {}

            # Latest day
            latest = rows[-1]
            inflow = float(latest.get("FlowInExNtv", 0) or 0)
            outflow = float(latest.get("FlowOutExNtv", 0) or 0)
            net_flow = inflow - outflow

            # 7-day average for comparison
            recent_7d = rows[-7:] if len(rows) >= 7 else rows
            avg_inflow = sum(float(r.get("FlowInExNtv", 0) or 0) for r in recent_7d) / len(recent_7d)
            avg_outflow = sum(float(r.get("FlowOutExNtv", 0) or 0) for r in recent_7d) / len(recent_7d)
            avg_net = avg_inflow - avg_outflow

            # Determine signal
            signal = "neutral"
            if net_flow > 5000:
                signal = "strong_inflow"   # Selling pressure
            elif net_flow > 1000:
                signal = "mild_inflow"
            elif net_flow < -5000:
                signal = "strong_outflow"  # Accumulation
            elif net_flow < -1000:
                signal = "mild_outflow"

            result = {
                "inflow_btc": round(inflow, 2),
                "outflow_btc": round(outflow, 2),
                "net_flow": round(net_flow, 2),
                "avg_7d_net": round(avg_net, 2),
                "flow_vs_avg": round(net_flow - avg_net, 2),
                "signal": signal,
                "date": latest.get("time", "")[:10],
                "source": "coinmetrics",
            }

            _set_cache("onchain_flow", result)
            return result

    except Exception as e:
        logger.warning("On-chain flow fetch failed: %s", e)
        return {}


async def fetch_whale_transactions() -> dict:
    """Monitor large BTC transactions using Blockchain.info unconfirmed tx pool (FREE).

    Scans the mempool for large pending transactions (>50 BTC).
    Also checks recent blocks for large confirmed transactions.
    Combined with exchange flow data for directional signal.
    5-minute TTL cache.
    """
    cached = _get_cached("whale_txs", 300)  # 5-minute cache
    if cached:
        return cached

    try:
        whale_txs: list[dict] = []
        total_large_btc = 0.0

        async with httpx.AsyncClient(timeout=15) as client:
            # Method 1: Blockchain.info unconfirmed transactions (real-time mempool)
            resp = await client.get(
                "https://blockchain.info/unconfirmed-transactions",
                params={"format": "json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for tx in data.get("txs", []):
                    total_out = sum(o.get("value", 0) for o in tx.get("out", []))
                    btc_amount = total_out / 1e8
                    if btc_amount >= 50:  # > 50 BTC threshold
                        whale_txs.append({
                            "txid": tx.get("hash", "")[:16],
                            "btc": round(btc_amount, 2),
                            "time": tx.get("time", 0),
                            "status": "unconfirmed",
                        })
                        total_large_btc += btc_amount

            # Method 2: Blockchain.info latest block for confirmed large txs
            resp2 = await client.get("https://blockchain.info/latestblock")
            if resp2.status_code == 200:
                latest_block = resp2.json()
                block_hash = latest_block.get("hash", "")
                if block_hash:
                    resp3 = await client.get(
                        f"https://blockchain.info/rawblock/{block_hash}",
                        timeout=30,  # Full block can be large
                    )
                    if resp3.status_code == 200:
                        block_data = resp3.json()
                        for tx in block_data.get("tx", [])[:500]:
                            total_out = sum(o.get("value", 0) for o in tx.get("out", []))
                            btc_amount = total_out / 1e8
                            if btc_amount >= 10:  # > 10 BTC confirmed
                                whale_txs.append({
                                    "txid": tx.get("hash", "")[:16],
                                    "btc": round(btc_amount, 2),
                                    "time": tx.get("time", 0),
                                    "status": "confirmed",
                                    "block_height": block_data.get("height", 0),
                                })
                                total_large_btc += btc_amount

        # Sort by size descending
        whale_txs.sort(key=lambda x: x.get("btc", 0), reverse=True)

        result = {
            "whale_tx_count": len(whale_txs),
            "total_whale_btc": round(total_large_btc, 2),
            "recent_whales": whale_txs[:15],
            "signal": "high_activity" if len(whale_txs) > 10 else "active" if whale_txs else "normal",
        }

        _set_cache("whale_txs", result)
        return result

    except Exception as e:
        logger.debug("Whale transaction fetch failed: %s", e)
        return {}


async def fetch_l2_depth(symbol: str = "BTCUSDT") -> dict:
    """Fetch Binance L2 order book depth snapshot (FREE)."""
    cached = _get_cached("l2_depth", CACHE_TTL_L2)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.binance.com/api/v3/depth",
                params={"symbol": symbol, "limit": 100},
            )
            if resp.status_code != 200:
                return {}

            data = resp.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            if not bids or not asks:
                return {}

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2

            # Compute bid/ask wall imbalance at ±1% from price
            bid_wall = sum(
                float(b[0]) * float(b[1]) for b in bids
                if float(b[0]) >= mid_price * 0.99
            )
            ask_wall = sum(
                float(a[0]) * float(a[1]) for a in asks
                if float(a[0]) <= mid_price * 1.01
            )

            imbalance = (bid_wall - ask_wall) / (bid_wall + ask_wall) if (bid_wall + ask_wall) > 0 else 0

            result = {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread_pct": round((best_ask - best_bid) / best_bid * 100, 5),
                "bid_wall_usd": round(bid_wall, 2),
                "ask_wall_usd": round(ask_wall, 2),
                "imbalance": round(imbalance, 4),  # Positive = bid dominant
            }

            _set_cache("l2_depth", result)
            return result

    except Exception as e:
        logger.debug("L2 depth fetch failed: %s", e)
        return {}
