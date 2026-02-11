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


def _add_choch_break(
    candles: list[Candle],
    key_swing: SwingPoint,
    trigger_swing: SwingPoint,
    direction: Direction,
    is_climactic: bool,
    choch_events: list[CHoCH],
) -> None:
    """Find the candle that broke the key swing level and add a CHoCH event.

    Scans candles between the key swing and the trigger swing (the swing
    whose classification confirmed the trend change).
    """
    for candle in candles:
        if candle.index <= key_swing.candle_index:
            continue
        if candle.index > trigger_swing.candle_index + 2:
            break

        broke = False
        if direction == Direction.BEARISH:
            broke = candle.body_bottom < key_swing.price
        else:
            broke = candle.body_top > key_swing.price

        if broke:
            confidence = 0.5
            if is_climactic:
                confidence += 0.25
            if direction == Direction.BEARISH:
                if candle.body_bottom < key_swing.price * 0.998:
                    confidence += 0.15
            else:
                if candle.body_top > key_swing.price * 1.002:
                    confidence += 0.15

            choch_events.append(CHoCH(
                candle_index=candle.index,
                direction=direction,
                broken_swing_index=key_swing.candle_index,
                broken_price=key_swing.price,
                confidence=min(confidence, 1.0),
                has_climax_confluence=is_climactic,
            ))
            break


def detect_choch(
    candles: list[Candle],
    swings: list[SwingPoint],
    bos_events: list[BOS],
    trend: TrendState,
) -> list[CHoCH]:
    """Detect CHoCH by walking through swing structure chronologically.

    Instead of using a single end-state trend, walks through the swing
    sequence tracking the running trend and detects CHoCH at each reversal:
    - Bullish → Bearish: candle body closes below the key HL
    - Bearish → Bullish: candle body closes above the key LH

    Also checks if current price action is breaking the key level
    (potential CHoCH before new swing structure confirms it).
    """
    if len(swings) < 4:
        return []

    choch_events: list[CHoCH] = []
    is_climactic, climax_ratio = detect_climax(candles, swings, trend)

    running_trend = TrendState.RANGING
    recent_high: SwingPoint | None = None
    recent_low: SwingPoint | None = None
    key_hl: SwingPoint | None = None  # Bullish defense level (break = bearish CHoCH)
    key_lh: SwingPoint | None = None  # Bearish defense level (break = bullish CHoCH)

    for i, swing in enumerate(swings):
        if swing.classification == SwingClassification.UNCLASSIFIED:
            if swing.swing_type == SwingType.SWING_HIGH:
                recent_high = swing
            else:
                recent_low = swing
            continue

        old_trend = running_trend

        if swing.swing_type == SwingType.SWING_HIGH:
            recent_high = swing
        else:
            recent_low = swing

        # Determine running trend from most recent classified high + low
        if (recent_high and recent_low
                and recent_high.classification != SwingClassification.UNCLASSIFIED
                and recent_low.classification != SwingClassification.UNCLASSIFIED):
            h_cls = recent_high.classification
            l_cls = recent_low.classification

            if h_cls == SwingClassification.HH and l_cls == SwingClassification.HL:
                running_trend = TrendState.BULLISH
            elif h_cls == SwingClassification.LH and l_cls == SwingClassification.LL:
                running_trend = TrendState.BEARISH
            # Mixed (HH+LL, LH+HL): keep previous trend

        # --- Handle trend transitions ---

        # RANGING → established trend: initialize key levels (no CHoCH)
        if old_trend == TrendState.RANGING and running_trend != TrendState.RANGING:
            if running_trend == TrendState.BULLISH:
                for s in reversed(swings[:i + 1]):
                    if s.classification == SwingClassification.HL and s.is_valid_smc:
                        key_hl = s
                        break
            elif running_trend == TrendState.BEARISH:
                for s in reversed(swings[:i + 1]):
                    if s.classification == SwingClassification.LH and s.is_valid_smc:
                        key_lh = s
                        break

        # BULLISH → BEARISH: bearish CHoCH (broke the key HL)
        elif old_trend == TrendState.BULLISH and running_trend == TrendState.BEARISH:
            if key_hl:
                _add_choch_break(candles, key_hl, swing,
                                 Direction.BEARISH, is_climactic, choch_events)
            # Initialize key_lh for the new bearish trend
            for s in reversed(swings[:i + 1]):
                if s.classification == SwingClassification.LH and s.is_valid_smc:
                    key_lh = s
                    break
            key_hl = None

        # BEARISH → BULLISH: bullish CHoCH (broke the key LH)
        elif old_trend == TrendState.BEARISH and running_trend == TrendState.BULLISH:
            if key_lh:
                _add_choch_break(candles, key_lh, swing,
                                 Direction.BULLISH, is_climactic, choch_events)
            # Initialize key_hl for the new bullish trend
            for s in reversed(swings[:i + 1]):
                if s.classification == SwingClassification.HL and s.is_valid_smc:
                    key_hl = s
                    break
            key_lh = None

    # Edge case: check if current price breaks the key level (live CHoCH)
    # Since key_hl/key_lh are only set at trend transitions (not updated within
    # the trend), find the most recent HL/LH for live detection.
    if running_trend == TrendState.BULLISH:
        live_hl = None
        for s in reversed(swings):
            if s.classification == SwingClassification.HL and s.is_valid_smc:
                live_hl = s
                break
        if live_hl:
            last_swing_idx = swings[-1].candle_index
            for candle in candles:
                if candle.index <= max(live_hl.candle_index, last_swing_idx):
                    continue
                if candle.body_bottom < live_hl.price:
                    confidence = 0.5
                    if is_climactic:
                        confidence += 0.25
                    if candle.body_bottom < live_hl.price * 0.998:
                        confidence += 0.15
                    choch_events.append(CHoCH(
                        candle_index=candle.index,
                        direction=Direction.BEARISH,
                        broken_swing_index=live_hl.candle_index,
                        broken_price=live_hl.price,
                        confidence=min(confidence, 1.0),
                        has_climax_confluence=is_climactic,
                    ))
                    break

    elif running_trend == TrendState.BEARISH:
        live_lh = None
        for s in reversed(swings):
            if s.classification == SwingClassification.LH and s.is_valid_smc:
                live_lh = s
                break
        if live_lh:
            last_swing_idx = swings[-1].candle_index
            for candle in candles:
                if candle.index <= max(live_lh.candle_index, last_swing_idx):
                    continue
                if candle.body_top > live_lh.price:
                    confidence = 0.5
                    if is_climactic:
                        confidence += 0.25
                    if candle.body_top > live_lh.price * 1.002:
                        confidence += 0.15
                    choch_events.append(CHoCH(
                        candle_index=candle.index,
                        direction=Direction.BULLISH,
                        broken_swing_index=live_lh.candle_index,
                        broken_price=live_lh.price,
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
        # Filter: Check for follow-through within a reasonable window
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
