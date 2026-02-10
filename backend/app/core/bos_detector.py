"""1.4 — Break of Structure (BOS) Detection

Identifies valid trend continuation signals where price breaks previous structure.

Core Rules (V05):
- RULE 1: Price swing MUST have taken inducement from previous swing
- RULE 2: Candle body MUST close ABOVE previous highest candle's WICK (bullish)
         or BELOW previous lowest candle's WICK (bearish)
- If Rule 1 fails (no IDM taken) → BOS is INVALID (fake)
- If first sweeping candle doesn't close beyond wick → wait for next candle
- If next candle body closes above sweeping candle's high → BOS is VALID
- Both rules must be satisfied simultaneously
"""

from __future__ import annotations
from app.models import (
    Candle, SwingPoint, Inducement, BOS,
    SwingType, Direction, IDMStatus,
)


def detect_bos(
    candles: list[Candle],
    swings: list[SwingPoint],
    inducements: list[Inducement],
) -> list[BOS]:
    """Detect all Break of Structure events.

    For each swing high, check if a subsequent candle breaks above it.
    For each swing low, check if a subsequent candle breaks below it.
    Then validate using the two BOS rules.
    """
    if len(swings) < 2:
        return []

    idx_map = {c.index: c for c in candles}
    idm_by_swing = {idm.parent_swing_index: idm for idm in inducements}
    bos_events: list[BOS] = []

    for i, swing in enumerate(swings):
        # Find the next swing of same type to define the "current leg"
        next_same = None
        for j in range(i + 1, len(swings)):
            if swings[j].swing_type == swing.swing_type:
                next_same = swings[j]
                break

        # Define the search range for the break
        search_end = next_same.candle_index if next_same else candles[-1].index if candles else 0

        # Check RULE 1: Was IDM taken for the swing that BREAKS this level?
        # The IDM check is on the swing that forms AFTER the break attempt
        idm = idm_by_swing.get(swing.candle_index)
        idm_was_taken = idm is not None and idm.status == IDMStatus.TAKEN

        # Find the break candle
        break_candle_idx = None
        break_valid = False

        if swing.swing_type == SwingType.SWING_HIGH:
            # Bullish BOS: look for candle body closing above swing high's wick
            swing_candle = idx_map.get(swing.candle_index)
            if not swing_candle:
                continue
            break_level = swing_candle.high  # RULE 2: must close above WICK

            for candle in candles:
                if candle.index <= swing.candle_index:
                    continue
                if candle.index > search_end:
                    break

                # Check if this candle or the next achieves body close above wick
                if candle.high > swing.price:
                    # Candle went above the swing high
                    if candle.body_top > break_level:
                        # Body closed above wick → BOS confirmed
                        break_candle_idx = candle.index
                        break_valid = True
                        break
                    else:
                        # Wick only — check next candle (V05 rule)
                        next_candle = idx_map.get(candle.index + 1)
                        if next_candle and next_candle.body_top > candle.high:
                            break_candle_idx = next_candle.index
                            break_valid = True
                            break
                        # If next candle doesn't confirm → liquidity sweep, not BOS

        else:  # SWING_LOW
            # Bearish BOS: look for candle body closing below swing low's wick
            swing_candle = idx_map.get(swing.candle_index)
            if not swing_candle:
                continue
            break_level = swing_candle.low  # RULE 2: must close below WICK

            for candle in candles:
                if candle.index <= swing.candle_index:
                    continue
                if candle.index > search_end:
                    break

                if candle.low < swing.price:
                    if candle.body_bottom < break_level:
                        break_candle_idx = candle.index
                        break_valid = True
                        break
                    else:
                        next_candle = idx_map.get(candle.index + 1)
                        if next_candle and next_candle.body_bottom < candle.low:
                            break_candle_idx = next_candle.index
                            break_valid = True
                            break

        if break_candle_idx is not None:
            direction = (Direction.BULLISH
                         if swing.swing_type == SwingType.SWING_HIGH
                         else Direction.BEARISH)

            # Both rules must pass for valid BOS
            is_valid = idm_was_taken and break_valid
            reason = None
            if not idm_was_taken:
                reason = "IDM not taken (Rule 1 failed)"
            elif not break_valid:
                reason = "Body did not close beyond wick (Rule 2 failed)"

            bos_events.append(BOS(
                candle_index=break_candle_idx,
                direction=direction,
                broken_swing_index=swing.candle_index,
                broken_price=swing.price,
                valid=is_valid,
                invalidation_reason=reason,
            ))

    return bos_events
