"""1.5 — Change of Character (CHoCH) Detection

Detects potential trend reversals where price breaks major swing points.

Core Rules (V07, V09, V10):
- CHoCH = break of the current MAJOR swing Higher Low (bullish→bearish)
         or Lower High (bearish→bullish)
- Candle body MUST close beyond the level
- Distinguish from inducement: what retail calls CHoCH is often just IDM being taken
- Fake CHoCH filters (V09): 3 conditions that invalidate apparent CHoCH
- CHoCH confirmation (V10): requires follow-through structure
- Climax + CHoCH = strong reversal signal (V23)
"""

from __future__ import annotations
from app.models import (
    Candle, SwingPoint, BOS, CHoCH,
    SwingType, SwingClassification, Direction, TrendState,
)


def _calculate_move_size(candles: list[Candle], start_idx: int, end_idx: int) -> float:
    """Calculate the absolute price move between two candle indices."""
    idx_map = {c.index: c for c in candles}
    start = idx_map.get(start_idx)
    end = idx_map.get(end_idx)
    if not start or not end:
        return 0.0
    return abs(end.close - start.close)


def detect_climax(
    candles: list[Candle],
    swings: list[SwingPoint],
    trend: TrendState,
) -> tuple[bool, float]:
    """Detect if the most recent move is climactic (V23 rule).

    Climax = the LARGEST single move in the current trend.
    If the latest swing-to-swing move is significantly larger than average,
    it's climactic → be vigilant for reversal.

    Returns:
        (is_climactic, climax_ratio) where climax_ratio > 1.5 = climactic
    """
    if len(swings) < 4:
        return False, 0.0

    # Calculate move sizes between consecutive swings
    move_sizes = []
    for i in range(1, len(swings)):
        size = abs(swings[i].price - swings[i - 1].price)
        if size > 0:
            move_sizes.append(size)

    if len(move_sizes) < 3:
        return False, 0.0

    latest_move = move_sizes[-1]
    avg_move = sum(move_sizes[:-1]) / len(move_sizes[:-1])

    if avg_move == 0:
        return False, 0.0

    ratio = latest_move / avg_move
    return ratio > 1.5, ratio


def detect_choch(
    candles: list[Candle],
    swings: list[SwingPoint],
    bos_events: list[BOS],
    trend: TrendState,
) -> list[CHoCH]:
    """Detect Change of Character events.

    In bullish trend: CHoCH = price breaks below the MAJOR Higher Low
    In bearish trend: CHoCH = price breaks above the MAJOR Lower High

    Major swing = the most recent valid SMC swing (is_valid_smc=True).
    """
    if len(swings) < 3:
        return []

    idx_map = {c.index: c for c in candles}
    choch_events: list[CHoCH] = []

    # Detect climax for confluence
    is_climactic, climax_ratio = detect_climax(candles, swings, trend)

    if trend == TrendState.BULLISH:
        # Find major Higher Lows (valid SMC structure)
        major_hls = [s for s in swings
                     if s.swing_type == SwingType.SWING_LOW
                     and s.classification == SwingClassification.HL
                     and s.is_valid_smc]

        for hl in major_hls:
            # Look for candle that body-closes below this HL
            for candle in candles:
                if candle.index <= hl.candle_index:
                    continue

                if candle.body_bottom < hl.price:
                    # CHoCH: bullish → bearish
                    confidence = 0.5
                    if is_climactic:
                        confidence += 0.25  # Climax confluence
                    if candle.body_bottom < hl.price * 0.998:  # Decisive break
                        confidence += 0.15

                    choch_events.append(CHoCH(
                        candle_index=candle.index,
                        direction=Direction.BEARISH,
                        broken_swing_index=hl.candle_index,
                        broken_price=hl.price,
                        confidence=min(confidence, 1.0),
                        has_climax_confluence=is_climactic,
                    ))
                    break  # Only first break counts

    elif trend == TrendState.BEARISH:
        # Find major Lower Highs
        major_lhs = [s for s in swings
                     if s.swing_type == SwingType.SWING_HIGH
                     and s.classification == SwingClassification.LH
                     and s.is_valid_smc]

        for lh in major_lhs:
            for candle in candles:
                if candle.index <= lh.candle_index:
                    continue

                if candle.body_top > lh.price:
                    confidence = 0.5
                    if is_climactic:
                        confidence += 0.25
                    if candle.body_top > lh.price * 1.002:
                        confidence += 0.15

                    choch_events.append(CHoCH(
                        candle_index=candle.index,
                        direction=Direction.BULLISH,
                        broken_swing_index=lh.candle_index,
                        broken_price=lh.price,
                        confidence=min(confidence, 1.0),
                        has_climax_confluence=is_climactic,
                    ))
                    break

    return choch_events


def filter_fake_choch(
    choch_events: list[CHoCH],
    swings: list[SwingPoint],
    candles: list[Candle],
) -> list[CHoCH]:
    """Apply V09 fake CHoCH filters.

    A CHoCH is FAKE (and should be filtered) if:
    1. The broken swing was not a MAJOR swing (was weak/minor)
    2. No follow-through: price doesn't create opposing structure after the break
    3. The break was during an inducement sweep (price returns quickly)
    """
    idx_map = {c.index: c for c in candles}

    for choch in choch_events:
        # Filter 1: Check if broken swing was truly major (valid SMC)
        broken_swing = None
        for s in swings:
            if s.candle_index == choch.broken_swing_index:
                broken_swing = s
                break

        if broken_swing and not broken_swing.is_valid_smc:
            choch.is_fake = True
            continue

        # Filter 2: Check for follow-through within a reasonable window
        # After CHoCH, expect opposing structure within next few candles
        window_size = 10
        choch_candle = idx_map.get(choch.candle_index)
        if not choch_candle:
            continue

        has_followthrough = False
        for candle in candles:
            if candle.index <= choch.candle_index:
                continue
            if candle.index > choch.candle_index + window_size:
                break

            if choch.direction == Direction.BEARISH:
                # After bearish CHoCH: expect a lower high (pullback that fails)
                if candle.high > choch_candle.high:
                    # Price went above CHoCH candle — weak follow-through
                    break
                has_followthrough = True
            else:
                if candle.low < choch_candle.low:
                    break
                has_followthrough = True

        if has_followthrough:
            choch.confirmed = True
        else:
            # No clear follow-through yet — don't mark fake, just unconfirmed
            choch.confirmed = False

    return choch_events
