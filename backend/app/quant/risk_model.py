"""Quant Risk Model (Layer 3) — ATR, DVOL, Volatility Regime, Dynamic SL/TP, Position Sizing.

Pure computation on candle data + optional Deribit DVOL fetch.
Weight: 15% of combined quant score.
"""

from __future__ import annotations

import logging
import time

import httpx

from app.models import Candle, Direction, TradingSignal, VolatilityRegime
from app.config import (
    ATR_PERIOD, ATR_LONG_PERIOD,
    VOL_REGIME_LOW, VOL_REGIME_HIGH, VOL_REGIME_EXTREME,
    DVOL_HIGH, DVOL_EXTREME,
    ATR_SL_MULT, MAX_DRAWDOWN_PCT, SUPPRESS_ON_EXTREME_VOL,
    CACHE_TTL_DVOL, MIN_RISK_REWARD,
)

logger = logging.getLogger(__name__)

# ── In-memory DVOL cache ──
_dvol_cache: dict = {"value": 0.0, "fetched_at": 0.0}


def compute_atr(candles: list[Candle], period: int = ATR_PERIOD) -> float:
    """Compute Average True Range from candle data.

    ATR = SMA of True Range over `period` candles.
    True Range = max(H-L, |H-prev_close|, |L-prev_close|).
    """
    if len(candles) < period + 1:
        # Fallback: use simple H-L range of available candles
        if candles:
            return sum(c.high - c.low for c in candles) / len(candles)
        return 0.0

    true_ranges: list[float] = []
    for i in range(1, len(candles)):
        h = candles[i].high
        l = candles[i].low
        prev_c = candles[i - 1].close
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        true_ranges.append(tr)

    # Use the most recent `period` true ranges
    recent = true_ranges[-period:]
    return sum(recent) / len(recent)


async def fetch_dvol() -> float:
    """Fetch Deribit DVOL (BTC implied volatility index). FREE, no key.

    Returns 0.0 on failure (graceful degradation).
    Uses 1-minute TTL cache.
    """
    now = time.time()
    if _dvol_cache["value"] > 0 and (now - _dvol_cache["fetched_at"]) < CACHE_TTL_DVOL:
        return _dvol_cache["value"]

    try:
        now_ms = int(now * 1000)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://www.deribit.com/api/v2/public/get_volatility_index_data",
                params={
                    "currency": "BTC",
                    "resolution": "3600",
                    "start_timestamp": str(now_ms - 7_200_000),
                    "end_timestamp": str(now_ms),
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                entries = data.get("result", {}).get("data", [])
                if entries:
                    dvol = entries[-1][4]  # latest close
                    _dvol_cache["value"] = dvol
                    _dvol_cache["fetched_at"] = now
                    return dvol
    except Exception as e:
        logger.warning("DVOL fetch failed: %s", e)

    return _dvol_cache["value"]  # Return stale value or 0


def detect_volatility_regime(
    candles: list[Candle],
    dvol: float = 0.0,
) -> VolatilityRegime:
    """Detect current volatility regime using ATR ratio + optional DVOL.

    Method: ATR(14) / ATR(100) ratio measures recent vs historical volatility.
    DVOL overrides when available (implied vol is forward-looking).
    """
    # DVOL override takes priority (forward-looking)
    if dvol > 0:
        if dvol >= DVOL_EXTREME:
            return VolatilityRegime.EXTREME
        if dvol >= DVOL_HIGH:
            return VolatilityRegime.HIGH

    # ATR ratio method
    atr_short = compute_atr(candles, ATR_PERIOD)
    atr_long = compute_atr(candles, ATR_LONG_PERIOD)

    if atr_long == 0:
        return VolatilityRegime.NORMAL

    ratio = atr_short / atr_long

    if ratio >= VOL_REGIME_EXTREME:
        return VolatilityRegime.EXTREME
    if ratio >= VOL_REGIME_HIGH:
        return VolatilityRegime.HIGH
    if ratio <= VOL_REGIME_LOW:
        return VolatilityRegime.LOW
    return VolatilityRegime.NORMAL


def compute_atr_sl_tp(
    signal: TradingSignal,
    atr: float,
    regime: VolatilityRegime,
) -> tuple[float, float]:
    """Compute ATR-scaled SL and TP, replacing fixed zone-based levels.

    SL = entry ± (ATR × regime multiplier)
    TP = entry ± (ATR × regime multiplier × R:R target)
    """
    mult = ATR_SL_MULT.get(regime.value, 2.0)
    sl_distance = atr * mult

    is_bull = (
        signal.direction == Direction.BULLISH
        if isinstance(signal.direction, Direction)
        else signal.direction == "bullish"
    )

    if is_bull:
        sl = signal.entry_price - sl_distance
        # TP: use original R:R ratio but with ATR-based SL distance
        rr = max(signal.risk_reward_ratio, MIN_RISK_REWARD)
        tp = signal.entry_price + sl_distance * rr
    else:
        sl = signal.entry_price + sl_distance
        rr = max(signal.risk_reward_ratio, MIN_RISK_REWARD)
        tp = signal.entry_price - sl_distance * rr

    return sl, tp


def compute_position_size(
    entry: float,
    sl: float,
    account_balance: float,
    regime: VolatilityRegime,
    base_risk_pct: float = 1.0,
) -> float:
    """Compute ATR-scaled + volatility regime adjusted position size.

    Instead of fixed 1% risk, scale by volatility regime:
    - Low vol: 1.2x base risk (more confident)
    - Normal: 1.0x base risk
    - High: 0.6x base risk (reduce exposure)
    - Extreme: 0.3x base risk (minimal)
    """
    regime_scale = {
        VolatilityRegime.LOW: 1.2,
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.HIGH: 0.6,
        VolatilityRegime.EXTREME: 0.3,
    }

    risk_pct = base_risk_pct * regime_scale.get(regime, 1.0)
    risk_amount = account_balance * (risk_pct / 100)

    sl_distance_pct = abs(entry - sl) / entry if entry > 0 else 0
    if sl_distance_pct == 0:
        return 0.0

    # Position size as % of account
    position_pct = (risk_amount / (account_balance * sl_distance_pct)) * 100
    return min(position_pct, 100.0)  # Cap at 100%


def compute_risk_tradability(regime: VolatilityRegime) -> float:
    """Compute tradability score based on volatility regime.

    Returns 0-100 where higher = more tradeable conditions.
    Extreme volatility = low tradability → may suppress signals.
    """
    scores = {
        VolatilityRegime.LOW: 85.0,     # Good for mean reversion
        VolatilityRegime.NORMAL: 100.0,  # Ideal conditions
        VolatilityRegime.HIGH: 50.0,     # Elevated risk
        VolatilityRegime.EXTREME: 15.0,  # Suppress signals
    }
    return scores.get(regime, 50.0)


def check_max_drawdown_guardrail(
    account_equity: float,
    peak_equity: float,
    threshold: float = MAX_DRAWDOWN_PCT,
) -> bool:
    """Check if drawdown exceeds threshold. Returns True if trading should halt."""
    if peak_equity <= 0:
        return False
    drawdown = (peak_equity - account_equity) / peak_equity
    return drawdown >= threshold


def should_suppress_signal(regime: VolatilityRegime, tradability: float) -> bool:
    """Determine if signal should be suppressed due to extreme conditions."""
    if not SUPPRESS_ON_EXTREME_VOL:
        return False
    return regime == VolatilityRegime.EXTREME and tradability < 20.0
