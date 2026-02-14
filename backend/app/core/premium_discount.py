"""1.8 — Premium & Discount Zone Calculator

Determines whether current price is in premium (expensive) or discount (cheap)
relative to the current structural range.

Rules (V12):
- Range = structural swing high to structural swing low
- Anchored to the most recent BOS/CHoCH (structure reset)
- Above 50% = premium zone (look for sells in bearish trend)
- Below 50% = discount zone (look for buys in bullish trend)
- OBs/FVGs in discount during buy trend = high probability
- OBs/FVGs in premium during sell trend = high probability
"""

from __future__ import annotations
from app.models import (
    SwingPoint, PremiumDiscount, SwingType, SwingClassification,
    ZoneType, TrendState, Direction, BOS, CHoCH,
)


def _find_structural_range(
    swings: list[SwingPoint],
    bos_events: list[BOS],
    choch_events: list[CHoCH],
    trend: TrendState,
) -> tuple[float, float] | None:
    """Find the swing high/low that define the current structural leg.

    V12 + V11: The range resets on each new BOS or CHoCH because those events
    create a new expansion leg with new structural boundaries.

    Algorithm:
    1. Find the most recent valid BOS or confirmed CHoCH (structural anchor).
    2. The anchor's broken_swing_index tells us which swing was broken.
    3. Find the swing extreme (HH/LH for highs, LL/HL for lows) that forms
       the other boundary of the current structural leg.
    4. The range = [swing low boundary, swing high boundary].

    Falls back to the most recent HH/HL + LL/LH pair if no BOS/CHoCH exists.
    """
    if not swings:
        return None

    # Collect structural events sorted by candle_index (most recent first)
    events: list[tuple[int, str, int, float]] = []  # (candle_idx, type, broken_swing_idx, broken_price)

    for b in bos_events:
        if b.valid:
            events.append((b.candle_index, "bos", b.broken_swing_index, b.broken_price))

    for ch in choch_events:
        if ch.confirmed and not ch.is_fake:
            events.append((ch.candle_index, "choch", ch.broken_swing_index, ch.broken_price))

    events.sort(key=lambda e: e[0], reverse=True)

    # Build swing index lookup
    swing_by_idx = {s.candle_index: s for s in swings}

    if events:
        # Use the most recent structural event as anchor
        anchor_candle_idx, anchor_type, broken_swing_idx, broken_price = events[0]
        broken_swing = swing_by_idx.get(broken_swing_idx)

        if broken_swing:
            # The broken swing gives us one boundary.
            # Find the opposite-type swing extreme that completes the leg.
            #
            # For bullish BOS (broke a swing high): the range high = broken price,
            # range low = the most recent swing low BEFORE the break.
            #
            # For bearish BOS (broke a swing low): the range low = broken price,
            # range high = the most recent swing high BEFORE the break.
            #
            # For CHoCH: same logic — the broken swing defines one boundary.

            if broken_swing.swing_type == SwingType.SWING_HIGH:
                # Broken a high → need the swing low that pairs with it
                range_high = broken_swing.price
                # Find highest high AFTER the anchor (the expansion peak)
                post_anchor_highs = [
                    s for s in swings
                    if s.swing_type == SwingType.SWING_HIGH
                    and s.candle_index >= anchor_candle_idx
                ]
                if post_anchor_highs:
                    range_high = max(range_high, max(s.price for s in post_anchor_highs))

                # Find the swing low before or near the break
                range_low = None
                for s in reversed(swings):
                    if s.swing_type == SwingType.SWING_LOW and s.candle_index <= anchor_candle_idx:
                        range_low = s.price
                        break
                # Also check lows AFTER anchor (retracement may have made new low)
                post_anchor_lows = [
                    s for s in swings
                    if s.swing_type == SwingType.SWING_LOW
                    and s.candle_index > anchor_candle_idx
                ]
                if post_anchor_lows:
                    candidate = min(s.price for s in post_anchor_lows)
                    if range_low is None or candidate < range_low:
                        range_low = candidate

                if range_low is not None and range_high > range_low:
                    return (range_high, range_low)

            else:  # SWING_LOW broken
                # Broken a low → need the swing high that pairs with it
                range_low = broken_swing.price
                # Find lowest low AFTER the anchor (the expansion trough)
                post_anchor_lows = [
                    s for s in swings
                    if s.swing_type == SwingType.SWING_LOW
                    and s.candle_index >= anchor_candle_idx
                ]
                if post_anchor_lows:
                    range_low = min(range_low, min(s.price for s in post_anchor_lows))

                # Find the swing high before or near the break
                range_high = None
                for s in reversed(swings):
                    if s.swing_type == SwingType.SWING_HIGH and s.candle_index <= anchor_candle_idx:
                        range_high = s.price
                        break
                # Also check highs AFTER anchor
                post_anchor_highs = [
                    s for s in swings
                    if s.swing_type == SwingType.SWING_HIGH
                    and s.candle_index > anchor_candle_idx
                ]
                if post_anchor_highs:
                    candidate = max(s.price for s in post_anchor_highs)
                    if range_high is None or candidate > range_high:
                        range_high = candidate

                if range_high is not None and range_high > range_low:
                    return (range_high, range_low)

    # Fallback: no BOS/CHoCH — use most recent swing high + swing low
    last_high = None
    last_low = None
    for s in reversed(swings):
        if s.swing_type == SwingType.SWING_HIGH and last_high is None:
            last_high = s
        if s.swing_type == SwingType.SWING_LOW and last_low is None:
            last_low = s
        if last_high and last_low:
            break

    if last_high and last_low and last_high.price > last_low.price:
        return (last_high.price, last_low.price)

    return None


def calculate_premium_discount(
    swings: list[SwingPoint],
    current_price: float,
    bos_events: list[BOS] | None = None,
    choch_events: list[CHoCH] | None = None,
    trend: TrendState | None = None,
) -> PremiumDiscount | None:
    """Calculate premium/discount anchored to the current structural leg.

    V12 + V11: The range is defined by the most recent BOS/CHoCH structural
    event, ensuring the P/D zones reset when structure changes (new expansion
    or reversal). Falls back to the most recent swing high/low if no
    structural events exist.
    """
    if not swings:
        return None

    result = _find_structural_range(
        swings,
        bos_events or [],
        choch_events or [],
        trend or TrendState.RANGING,
    )

    if not result:
        return None

    swing_high, swing_low = result
    return PremiumDiscount.calculate(swing_high, swing_low, current_price)


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
