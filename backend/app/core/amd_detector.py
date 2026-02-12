"""1.11 — AMD (Accumulation-Manipulation-Distribution) Pattern Detection

V17: Detects AMD patterns where market ranges near a zone without taking
the expected inducement, then sweeps internal liquidity and provides MSS.

Key Rules:
- Market approaches buy/sell zone but doesn't sweep inducement
- Creates internal RANGE near the zone
- Sweeps INTERNAL liquidity (sell-side low or buy-side high)
- After sweep, market provides MSS (Market Structure Shift)
- BOS from AMD is NOT a valid BOS (left-side IDM never taken)
"""

from __future__ import annotations
from app.models import (
    Candle, SwingPoint, Inducement, FVG, OrderBlock,
    AMDPattern, AMDPhase, Direction, TrendState, IDMStatus,
)


def _find_internal_ranges(
    candles: list[Candle],
    swings: list[SwingPoint],
    min_range_candles: int = 6,
) -> list[tuple[int, int, float, float]]:
    """Find ranging (consolidation) areas between swing points.

    A range = at least min_range_candles where high-low stays
    within a narrow band (less than 50% of recent ATR).
    """
    if len(candles) < min_range_candles:
        return []

    ranges: list[tuple[int, int, float, float]] = []

    # Calculate simple ATR for threshold
    atr_sum = 0.0
    for c in candles[-20:]:
        atr_sum += c.total_range
    atr = atr_sum / min(20, len(candles)) if candles else 1.0
    range_threshold = atr * 2.5  # Range width must be less than 2.5x ATR

    # Sliding window to find ranges
    for start in range(len(candles) - min_range_candles):
        window = candles[start:start + min_range_candles]
        w_high = max(c.high for c in window)
        w_low = min(c.low for c in window)
        width = w_high - w_low

        if width < range_threshold and width > 0:
            # Extend the range as far as it holds
            end = start + min_range_candles
            while end < len(candles):
                ext_high = max(w_high, candles[end].high)
                ext_low = min(w_low, candles[end].low)
                if ext_high - ext_low > range_threshold:
                    break
                w_high = ext_high
                w_low = ext_low
                end += 1

            ranges.append((
                candles[start].index,
                candles[end - 1].index,
                w_high,
                w_low,
            ))

    # Deduplicate overlapping ranges — keep longest
    if not ranges:
        return []

    ranges.sort(key=lambda r: r[1] - r[0], reverse=True)
    used: set[int] = set()
    deduped: list[tuple[int, int, float, float]] = []
    for r in ranges:
        mid = (r[0] + r[1]) // 2
        if mid not in used:
            deduped.append(r)
            for i in range(r[0], r[1] + 1):
                used.add(i)

    return deduped


def _detect_internal_sweep(
    candles: list[Candle],
    range_start: int,
    range_end: int,
    range_high: float,
    range_low: float,
) -> tuple[int | None, Direction | None]:
    """Detect if internal liquidity was swept after the range.

    Look for a candle that wicks beyond range high/low then reverses.
    """
    idx_map = {c.index: c for c in candles}

    # Look at candles after the range
    for c in candles:
        if c.index <= range_end:
            continue
        if c.index > range_end + 10:
            break

        # Bearish sweep: wick above range high, close below
        if c.high > range_high and c.body_bottom < range_high:
            return c.index, Direction.BEARISH

        # Bullish sweep: wick below range low, close above
        if c.low < range_low and c.body_top > range_low:
            return c.index, Direction.BULLISH

    return None, None


def _detect_mss_after_sweep(
    candles: list[Candle],
    sweep_idx: int,
    sweep_direction: Direction,
) -> int | None:
    """Detect MSS (Market Structure Shift) after internal liquidity sweep.

    For bearish sweep (swept high): look for body close below range low
    For bullish sweep (swept low): look for body close above range high
    """
    idx_map = {c.index: c for c in candles}
    sweep_candle = idx_map.get(sweep_idx)
    if not sweep_candle:
        return None

    for c in candles:
        if c.index <= sweep_idx:
            continue
        if c.index > sweep_idx + 10:
            break

        if sweep_direction == Direction.BEARISH:
            # After bearish sweep: MSS = body close below sweep candle's low
            if c.body_bottom < sweep_candle.low:
                return c.index
        else:
            # After bullish sweep: MSS = body close above sweep candle's high
            if c.body_top > sweep_candle.high:
                return c.index

    return None


def detect_amd_patterns(
    candles: list[Candle],
    swings: list[SwingPoint],
    inducements: list[Inducement],
    trend: TrendState,
) -> list[AMDPattern]:
    """Complete AMD detection pipeline.

    1. Find internal ranges (Accumulation phase)
    2. Check if inducement was NOT taken (AMD key condition)
    3. Detect internal liquidity sweep (Manipulation)
    4. Detect MSS after sweep (Distribution)
    """
    if len(candles) < 10:
        return []

    patterns: list[AMDPattern] = []
    ranges = _find_internal_ranges(candles, swings)

    for r_start, r_end, r_high, r_low in ranges:
        # Key AMD condition: check if nearby IDM was NOT taken
        # If IDM was already taken, this is a normal BOS setup, not AMD
        nearby_idm_taken = any(
            idm.status == IDMStatus.TAKEN
            and abs(idm.candle_index - r_start) < 15
            for idm in inducements
        )
        if nearby_idm_taken:
            continue

        # Phase 2: Manipulation — internal liquidity sweep
        sweep_idx, sweep_dir = _detect_internal_sweep(
            candles, r_start, r_end, r_high, r_low
        )

        if sweep_idx is None or sweep_dir is None:
            # Range exists but no sweep yet — accumulation phase only
            patterns.append(AMDPattern(
                range_start_idx=r_start,
                range_end_idx=r_end,
                range_high=r_high,
                range_low=r_low,
                phase=AMDPhase.ACCUMULATION,
            ))
            continue

        # Phase 3: Distribution — MSS after sweep
        mss_idx = _detect_mss_after_sweep(candles, sweep_idx, sweep_dir)

        if mss_idx is not None:
            phase = AMDPhase.COMPLETE
        else:
            phase = AMDPhase.MANIPULATION

        patterns.append(AMDPattern(
            range_start_idx=r_start,
            range_end_idx=r_end,
            range_high=r_high,
            range_low=r_low,
            phase=phase,
            sweep_direction=sweep_dir,
            sweep_candle_idx=sweep_idx,
            mss_candle_idx=mss_idx,
            mss_confirmed=mss_idx is not None,
        ))

    return patterns
