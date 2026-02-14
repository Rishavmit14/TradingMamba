"""VSA Ultra High Volume — Wyckoff Absorption Detection (V22)

Detects institutional absorption patterns using Volume Spread Analysis:
- Ultra-high volume candle (> 2x 20-period average) with opposite-direction body
- Bearish body + UHV in uptrend = selling absorption → bullish signal
- Bullish body + UHV in downtrend = buying absorption → bearish signal
- Confirmation: subsequent candle breaks the absorption candle's high/low
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from app.models import Candle, Direction


@dataclass
class VSAAbsorption:
    """A detected VSA Ultra High Volume absorption pattern."""
    candle_index: int                      # Index of the UHV candle
    direction: Direction                   # Expected move direction AFTER absorption
    volume_ratio: float                    # Actual volume / avg volume (e.g. 2.8x)
    absorbed_direction: str                # Body direction being absorbed ("bearish"/"bullish")
    confirmation: bool = False             # True if breakout candle confirms
    confirmed_at_candle: Optional[int] = None  # Index where breakout confirmed


def detect_vsa_absorptions(
    candles: list[Candle],
    lookback: int = 20,
    volume_threshold: float = 2.0,
) -> list[VSAAbsorption]:
    """Detect VSA Ultra High Volume absorption patterns.

    Algorithm:
    1. Calculate rolling average volume over `lookback` periods
    2. For each candle with volume > threshold * avg:
       - Check if candle body is opposite to the local trend direction
       - Bearish body + UHV = selling absorbed by institutions → bullish
       - Bullish body + UHV = buying absorbed by institutions → bearish
    3. Check confirmation: next candle breaks the UHV candle's high (bullish)
       or low (bearish)

    Args:
        candles: OHLCV candle array
        lookback: Period for average volume calculation
        volume_threshold: Multiplier above average to qualify as UHV (default 2.0)

    Returns:
        List of VSAAbsorption objects
    """
    if len(candles) < lookback + 1:
        return []

    absorptions: list[VSAAbsorption] = []

    for i in range(lookback, len(candles)):
        candle = candles[i]

        # Calculate average volume over prior `lookback` candles
        avg_vol = sum(c.volume for c in candles[i - lookback:i]) / lookback
        if avg_vol == 0:
            continue

        ratio = candle.volume / avg_vol
        if ratio < volume_threshold:
            continue

        # UHV detected — check if body is opposite to local trend
        # Determine local trend from the last few candles' close direction
        local_trend = _determine_local_trend(candles, i, window=10)

        absorption_dir: Direction | None = None
        absorbed_dir: str = ""

        if candle.is_bearish and local_trend == "up":
            # Bearish candle with UHV during uptrend = selling absorbed → bullish
            absorption_dir = Direction.BULLISH
            absorbed_dir = "bearish"
        elif candle.is_bullish and local_trend == "down":
            # Bullish candle with UHV during downtrend = buying absorbed → bearish
            absorption_dir = Direction.BEARISH
            absorbed_dir = "bullish"

        if absorption_dir is None:
            continue

        # Check confirmation: does the next candle break in the absorption direction?
        confirmed = False
        confirmed_at = None
        # Look at the next 3 candles for confirmation
        for j in range(i + 1, min(i + 4, len(candles))):
            next_c = candles[j]
            if absorption_dir == Direction.BULLISH:
                if next_c.close > candle.high:
                    confirmed = True
                    confirmed_at = j
                    break
            else:
                if next_c.close < candle.low:
                    confirmed = True
                    confirmed_at = j
                    break

        absorptions.append(VSAAbsorption(
            candle_index=i,
            direction=absorption_dir,
            volume_ratio=round(ratio, 2),
            absorbed_direction=absorbed_dir,
            confirmation=confirmed,
            confirmed_at_candle=confirmed_at,
        ))

    return absorptions


def _determine_local_trend(candles: list[Candle], current_idx: int, window: int = 10) -> str:
    """Determine local trend direction from recent price action.

    Compares the close of the current candle to the close `window` candles ago.
    Also checks if the majority of recent candles are bullish or bearish.

    Returns: "up", "down", or "neutral"
    """
    start = max(0, current_idx - window)
    if start >= current_idx:
        return "neutral"

    start_price = candles[start].close
    end_price = candles[current_idx - 1].close  # Use prior candle, not current UHV candle

    # Net direction from price change
    pct_change = (end_price - start_price) / start_price if start_price > 0 else 0

    # Count bullish vs bearish candles in the window
    bullish_count = sum(1 for c in candles[start:current_idx] if c.is_bullish)
    bearish_count = sum(1 for c in candles[start:current_idx] if c.is_bearish)

    if pct_change > 0.002 and bullish_count > bearish_count:
        return "up"
    elif pct_change < -0.002 and bearish_count > bullish_count:
        return "down"

    return "neutral"
