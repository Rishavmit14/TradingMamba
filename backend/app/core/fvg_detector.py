"""1.6 — Fair Value Gap (FVG) Detection

Finds price imbalances (3-candle gaps) that act as magnetic zones.

Core Rules (V13):
- FVG = gap measured WICK to WICK: candle[i-1].high to candle[i+1].low (bullish)
- Candle color does NOT matter — only wick-to-wick gap
- CRITICAL: In sell trend, FVG must come from the HIGHEST candle → any other = INVALID
- In buy trend, FVG must come from the LOWEST candle → any other = INVALID
- Only trade FVG in trend direction
- Only trade AFTER liquidity sweep/grab
- Target 50% of FVG zone for entry
- After first valid FVG from extreme candle: subsequent FVGs in same swing valid too
"""

from __future__ import annotations
from app.models import (
    Candle, SwingPoint, FVG,
    Direction, TrendState, SwingType,
)


def detect_raw_fvgs(candles: list[Candle]) -> list[FVG]:
    """Detect all 3-candle FVG patterns regardless of validity.

    Bullish FVG: candle[i-1].high < candle[i+1].low (gap up)
    Bearish FVG: candle[i-1].low > candle[i+1].high (gap down)
    """
    fvgs: list[FVG] = []

    if len(candles) < 3:
        return fvgs

    for i in range(1, len(candles) - 1):
        prev = candles[i - 1]
        curr = candles[i]
        nxt = candles[i + 1]

        # Bullish FVG: gap between prev high and next low
        if nxt.low > prev.high:
            fvgs.append(FVG(
                candle_index=curr.index,
                upper_price=nxt.low,
                lower_price=prev.high,
                direction=Direction.BULLISH,
            ))

        # Bearish FVG: gap between prev low and next high
        if nxt.high < prev.low:
            fvgs.append(FVG(
                candle_index=curr.index,
                upper_price=prev.low,
                lower_price=nxt.high,
                direction=Direction.BEARISH,
            ))

    return fvgs


def _find_extreme_candle_in_swing(
    candles: list[Candle],
    swing_start_idx: int,
    swing_end_idx: int,
    swing_type: SwingType,
) -> int | None:
    """Find the impulse-origin extreme candle within a swing leg (V13).

    V13: "In sell trend, FVG must come from HIGHEST candle"
    V13: "In buy trend, FVG must come from LOWEST candle"

    swing_type is the ENDING swing of the leg:
    - SWING_HIGH end = bullish leg (low→high): impulse starts at LOWEST candle
    - SWING_LOW end = bearish leg (high→low): impulse starts at HIGHEST candle
    """
    relevant = [c for c in candles if swing_start_idx <= c.index <= swing_end_idx]
    if not relevant:
        return None

    if swing_type == SwingType.SWING_HIGH:
        # Bullish leg: V13 buy trend → FVG from LOWEST candle (start of buy impulse)
        return min(relevant, key=lambda c: c.low).index
    else:
        # Bearish leg: V13 sell trend → FVG from HIGHEST candle (start of sell impulse)
        return max(relevant, key=lambda c: c.high).index


def validate_fvgs(
    fvgs: list[FVG],
    candles: list[Candle],
    swings: list[SwingPoint],
    trend: TrendState,
) -> list[FVG]:
    """Apply V13 validity rules to detected FVGs.

    V13 extreme candle rule applies only to trend-aligned FVGs:
    - Bearish swing leg: bearish FVGs must be near HIGHEST candle (sell impulse start)
    - Bullish swing leg: bullish FVGs must be near LOWEST candle (buy impulse start)

    Counter-trend FVGs (bullish in downswing, bearish in upswing) are marked valid
    for chart display — the signal generator handles trade filtering.
    """
    # Build swing ranges: pairs of consecutive swings define "swing legs"
    swing_ranges = []
    for i in range(len(swings) - 1):
        swing_ranges.append((swings[i].candle_index, swings[i + 1].candle_index, swings[i + 1].swing_type))

    first_valid_found_per_swing: dict[tuple[int, int], bool] = {}

    for fvg in fvgs:
        # Find which swing leg this FVG belongs to
        fvg_swing_range = None
        for start_idx, end_idx, s_type in swing_ranges:
            if start_idx <= fvg.candle_index <= end_idx:
                fvg_swing_range = (start_idx, end_idx, s_type)
                break

        if fvg_swing_range:
            start, end, s_type = fvg_swing_range

            # Determine if FVG is trend-aligned with its swing leg
            # SWING_HIGH end = bullish leg → bullish FVGs are trend-aligned
            # SWING_LOW end = bearish leg → bearish FVGs are trend-aligned
            is_trend_aligned = (
                (s_type == SwingType.SWING_HIGH and fvg.direction == Direction.BULLISH)
                or (s_type == SwingType.SWING_LOW and fvg.direction == Direction.BEARISH)
            )

            if not is_trend_aligned:
                # Counter-trend FVGs: valid for chart display, no extreme candle rule
                fvg.valid = True
                continue

            # Apply V13 extreme candle rule to trend-aligned FVGs
            swing_len = end - start
            tolerance = max(3, int(swing_len * 0.2))
            extreme_idx = _find_extreme_candle_in_swing(candles, start, end, s_type)

            if extreme_idx is not None:
                if abs(fvg.candle_index - extreme_idx) <= tolerance:
                    fvg.from_extreme_candle = True
                    fvg.valid = True
                    first_valid_found_per_swing[(start, end)] = True
                elif first_valid_found_per_swing.get((start, end), False):
                    # After first valid FVG: subsequent in same swing are also valid
                    fvg.valid = True
                else:
                    fvg.valid = False
            else:
                fvg.valid = False
        else:
            # FVG outside any swing range — keep as potentially valid
            fvg.valid = True

    return fvgs


def check_fvg_mitigation(fvgs: list[FVG], candles: list[Candle]) -> list[FVG]:
    """Check if price has returned to fill (mitigate) any FVGs.

    V13: "Target 50% of FVG zone for entry" — the midpoint is the ENTRY level,
    not the mitigation level. An FVG is only mitigated when the full gap is filled:

    - Bullish FVG: mitigated when price drops through the entire gap (low <= lower_price)
    - Bearish FVG: mitigated when price rises through the entire gap (high >= upper_price)

    Skip the 3-candle pattern itself (candle_index-1 through candle_index+1).
    """
    for fvg in fvgs:
        if fvg.mitigated:
            continue

        for candle in candles:
            # Skip the 3-candle FVG pattern itself (prev, middle, next)
            if candle.index <= fvg.candle_index + 1:
                continue

            if fvg.direction == Direction.BULLISH:
                # Bullish FVG mitigated when price fills the entire gap
                if candle.low <= fvg.lower_price:
                    fvg.mitigated = True
                    fvg.mitigated_at_candle = candle.index
                    break
            else:
                # Bearish FVG mitigated when price fills the entire gap
                if candle.high >= fvg.upper_price:
                    fvg.mitigated = True
                    fvg.mitigated_at_candle = candle.index
                    break

    return fvgs


def detect_all_fvgs(
    candles: list[Candle],
    swings: list[SwingPoint],
    trend: TrendState,
) -> list[FVG]:
    """Complete FVG detection pipeline: detect → validate → check mitigation."""
    raw = detect_raw_fvgs(candles)
    validated = validate_fvgs(raw, candles, swings, trend)
    return check_fvg_mitigation(validated, candles)
