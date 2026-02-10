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
    """Find the extreme candle within a swing.

    For swing high (sell trend FVG): the HIGHEST candle's index
    For swing low (buy trend FVG): the LOWEST candle's index
    """
    relevant = [c for c in candles if swing_start_idx <= c.index <= swing_end_idx]
    if not relevant:
        return None

    if swing_type == SwingType.SWING_HIGH:
        return max(relevant, key=lambda c: c.high).index
    else:
        return min(relevant, key=lambda c: c.low).index


def validate_fvgs(
    fvgs: list[FVG],
    candles: list[Candle],
    swings: list[SwingPoint],
    trend: TrendState,
) -> list[FVG]:
    """Apply V13 validity rules to detected FVGs.

    Key rule: FVG must come from the EXTREME candle of its swing.
    - Bearish trend: FVG must be from the HIGHEST candle in the swing
    - Bullish trend: FVG must be from the LOWEST candle in the swing

    Also filters FVGs against trend direction:
    - Bullish trend: only bullish FVGs valid
    - Bearish trend: only bearish FVGs valid
    """
    # Build swing ranges: pairs of consecutive swings define "swing legs"
    swing_ranges = []
    for i in range(len(swings) - 1):
        swing_ranges.append((swings[i].candle_index, swings[i + 1].candle_index, swings[i + 1].swing_type))

    first_valid_found_per_swing: dict[tuple[int, int], bool] = {}

    for fvg in fvgs:
        # Filter 1: trend direction alignment
        if trend == TrendState.BULLISH and fvg.direction != Direction.BULLISH:
            fvg.valid = False
            continue
        if trend == TrendState.BEARISH and fvg.direction != Direction.BEARISH:
            fvg.valid = False
            continue

        # Filter 2: extreme candle rule
        # Find which swing this FVG belongs to
        fvg_swing_range = None
        for start_idx, end_idx, s_type in swing_ranges:
            if start_idx <= fvg.candle_index <= end_idx:
                fvg_swing_range = (start_idx, end_idx, s_type)
                break

        if fvg_swing_range:
            start, end, s_type = fvg_swing_range
            extreme_idx = _find_extreme_candle_in_swing(candles, start, end, s_type)

            if extreme_idx is not None:
                # FVG's middle candle should be the extreme or adjacent to it
                if abs(fvg.candle_index - extreme_idx) <= 1:
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

    A bullish FVG is mitigated when price drops into the gap zone.
    A bearish FVG is mitigated when price rises into the gap zone.
    """
    for fvg in fvgs:
        if fvg.mitigated:
            continue

        for candle in candles:
            if candle.index <= fvg.candle_index:
                continue

            if fvg.direction == Direction.BULLISH:
                # Bullish FVG mitigated when price drops into the gap
                if candle.low <= fvg.upper_price:
                    fvg.mitigated = True
                    fvg.mitigated_at_candle = candle.index
                    break
            else:
                # Bearish FVG mitigated when price rises into the gap
                if candle.high >= fvg.lower_price:
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
