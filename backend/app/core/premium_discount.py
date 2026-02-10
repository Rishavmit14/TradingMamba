"""1.8 — Premium & Discount Zone Calculator

Determines whether current price is in premium (expensive) or discount (cheap)
relative to the current structural range.

Rules (V12):
- Range = current swing high to current swing low
- Above 50% = premium zone (look for sells in bearish trend)
- Below 50% = discount zone (look for buys in bullish trend)
- OBs/FVGs in discount during buy trend = high probability
- OBs/FVGs in premium during sell trend = high probability
"""

from __future__ import annotations
from app.models import (
    SwingPoint, PremiumDiscount, SwingType,
    ZoneType, TrendState, Direction,
)


def calculate_premium_discount(
    swings: list[SwingPoint],
    current_price: float,
) -> PremiumDiscount | None:
    """Calculate premium/discount based on the most recent swing range.

    Uses the last swing high and swing low to define the range.
    """
    if not swings:
        return None

    # Find last swing high and swing low
    last_high = None
    last_low = None
    for s in reversed(swings):
        if s.swing_type == SwingType.SWING_HIGH and last_high is None:
            last_high = s
        if s.swing_type == SwingType.SWING_LOW and last_low is None:
            last_low = s
        if last_high and last_low:
            break

    if not last_high or not last_low:
        return None

    return PremiumDiscount.calculate(last_high.price, last_low.price, current_price)


def is_zone_in_favorable_position(
    zone_price: float,
    pd: PremiumDiscount,
    trend: TrendState,
) -> bool:
    """Check if a zone (OB/FVG) is in a favorable premium/discount position.

    V12 rule:
    - Bullish trend: zones in DISCOUNT = high probability
    - Bearish trend: zones in PREMIUM = high probability
    """
    zone_pd = PremiumDiscount.calculate(pd.swing_high, pd.swing_low, zone_price)

    if trend == TrendState.BULLISH:
        return zone_pd.zone == ZoneType.DISCOUNT
    elif trend == TrendState.BEARISH:
        return zone_pd.zone == ZoneType.PREMIUM
    return False


def get_fibonacci_levels(swing_high: float, swing_low: float) -> dict[str, float]:
    """Calculate key Fibonacci retracement levels within the range.

    V12, V15: Key levels used in ICT methodology.
    """
    diff = swing_high - swing_low
    return {
        "0.0": swing_low,
        "0.236": swing_low + diff * 0.236,
        "0.382": swing_low + diff * 0.382,
        "0.5": swing_low + diff * 0.5,
        "0.618": swing_low + diff * 0.618,
        "0.705": swing_low + diff * 0.705,
        "0.786": swing_low + diff * 0.786,
        "1.0": swing_high,
    }
