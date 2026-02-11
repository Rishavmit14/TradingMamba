"""1.2 — Inducement (IDM) Detection

Finds the valid inducement point for each swing — the first pullback on the
left side of the current high/low.

Core Rules (V02-V04):
- IDM = first valid pullback on the LEFT side of the current swing high
- Bullish: look at HIGH's left side for the first pullback low
- Bearish: look at LOW's left side for the first pullback high
- If no IDM in current swing → IDM TRANSFERS from previous swing
- Internal candles (inside previous candle's range) do NOT provide IDM
- Taking IDM: wick sweep is sufficient (body close NOT required)
- Multiple pullbacks = multiple IDM points; first (most recent) is primary

The IDM detector also re-validates swing classifications:
- A swing is only valid SMC structure if IDM was taken before it formed
- Swings without IDM taken are marked as inducement zones (is_valid_smc=False)
"""

from __future__ import annotations
from app.models import (
    Candle, SwingPoint, Inducement, SwingType,
    SwingClassification, IDMStatus,
)


def find_pullbacks_in_range(
    candles: list[Candle],
    start_idx: int,
    end_idx: int,
    direction: SwingType,
) -> list[int]:
    """Find valid pullback candle indices between two points.

    V03 Rule: A valid pullback requires breaking the EXTREME candle's
    opposite-side level (not just the previous candle's level):
    - Bullish swing: LOW of the HIGHEST candle must be broken
    - Bearish swing: HIGH of the LOWEST candle must be broken

    After each valid pullback, the extreme candle tracking resets from
    the pullback point for the next sub-move.

    Filters out internal candles (inside previous candle's range) per V04 rule.
    """
    idx_map = {c.index: c for c in candles}
    indices = sorted([c.index for c in candles if start_idx <= c.index <= end_idx])

    if len(indices) < 3:
        return []

    pullbacks = []
    first_candle = idx_map.get(indices[0])
    if not first_candle:
        return []

    if direction == SwingType.SWING_HIGH:
        # Bullish swing: track HIGHEST candle, pullback = breaks its LOW
        highest = first_candle

        for i in range(1, len(indices)):
            curr = idx_map.get(indices[i])
            prev = idx_map.get(indices[i - 1])
            if not curr or not prev:
                continue

            # Skip internal candles (V04: inside previous candle's range = no IDM)
            if curr.high <= prev.high and curr.low >= prev.low:
                continue

            # V03: check if candle broke the HIGHEST candle's LOW
            if curr.low < highest.low:
                pullbacks.append(curr.index)
                highest = curr  # Reset tracking after valid pullback
            elif curr.high > highest.high:
                highest = curr  # Update highest candle in sub-move

    else:  # SWING_LOW
        # Bearish swing: track LOWEST candle, pullback = breaks its HIGH
        lowest = first_candle

        for i in range(1, len(indices)):
            curr = idx_map.get(indices[i])
            prev = idx_map.get(indices[i - 1])
            if not curr or not prev:
                continue

            # Skip internal candles (V04)
            if curr.high <= prev.high and curr.low >= prev.low:
                continue

            # V03: check if candle broke the LOWEST candle's HIGH
            if curr.high > lowest.high:
                pullbacks.append(curr.index)
                lowest = curr  # Reset tracking after valid pullback
            elif curr.low < lowest.low:
                lowest = curr  # Update lowest candle in sub-move

    return pullbacks


def detect_inducements(
    candles: list[Candle],
    swings: list[SwingPoint],
) -> list[Inducement]:
    """Detect inducement levels for each swing point.

    For each swing high: find the first pullback LOW on the left side
    (between this swing and the previous swing low).

    For each swing low: find the first pullback HIGH on the left side
    (between this swing and the previous swing high).

    If no pullback exists in the current swing → transfer IDM from previous swing.
    """
    if len(swings) < 2:
        return []

    idx_map = {c.index: c for c in candles}
    inducements: list[Inducement] = []
    # Track the most recent IDM per swing type for correct transfers
    # (swing highs transfer from previous swing high's IDM, not from any IDM)
    last_idm_for_type: dict[SwingType, Inducement] = {}

    for i, swing in enumerate(swings):
        # Find the previous swing of opposite type to define the range
        prev_opposite = None
        for j in range(i - 1, -1, -1):
            if swings[j].swing_type != swing.swing_type:
                prev_opposite = swings[j]
                break

        if prev_opposite is None:
            continue

        # Define search range: from previous opposite swing to current swing
        range_start = prev_opposite.candle_index
        range_end = swing.candle_index

        # Find pullbacks within this range
        pullback_indices = find_pullbacks_in_range(
            candles, range_start, range_end, swing.swing_type
        )

        if pullback_indices:
            # First valid pullback from the LEFT of the swing high/low
            # "First" = the most recent pullback before the swing
            # Sort descending to get closest to the swing first
            pullback_indices.sort(reverse=True)
            idm_candle_idx = pullback_indices[0]
            idm_candle = idx_map.get(idm_candle_idx)

            if idm_candle:
                # IDM price: for swing high, it's the pullback LOW
                # For swing low, it's the pullback HIGH
                if swing.swing_type == SwingType.SWING_HIGH:
                    idm_price = idm_candle.low
                else:
                    idm_price = idm_candle.high

                # V08: Classify major vs minor IDM
                # Major = deepest pullback (highest probability, 80-85%)
                # Single pullback → always major (V08 Rule 1)
                # Multiple → check if selected (most recent) is also deepest
                is_major = True
                if len(pullback_indices) > 1:
                    for pi in pullback_indices[1:]:
                        pc = idx_map.get(pi)
                        if not pc:
                            continue
                        if swing.swing_type == SwingType.SWING_HIGH:
                            if pc.low < idm_price:
                                is_major = False  # Deeper pullback exists
                                break
                        else:
                            if pc.high > idm_price:
                                is_major = False
                                break

                new_idm = Inducement(
                    candle_index=idm_candle_idx,
                    price=idm_price,
                    parent_swing_index=swing.candle_index,
                    status=IDMStatus.ACTIVE,
                    is_major=is_major,
                )
                inducements.append(new_idm)
                last_idm_for_type[swing.swing_type] = new_idm
        else:
            # No pullback found → impulse move (V04 Rule 3)
            # The opposite swing point (start of impulse) acts as IDM.
            # V04: "When market moves from low to high in IMPULSE SWING
            # (without pullback), LOW of swing acts as inducement."
            # TRANSFERRED status preserved for V09 Rule 3 (fake CHoCH detection).
            new_idm = Inducement(
                candle_index=prev_opposite.candle_index,
                price=prev_opposite.price,
                parent_swing_index=swing.candle_index,
                status=IDMStatus.TRANSFERRED,
            )
            inducements.append(new_idm)
            last_idm_for_type[swing.swing_type] = new_idm

    return inducements


def check_idm_taken(
    candles: list[Candle],
    inducements: list[Inducement],
    swings: list[SwingPoint],
    lookback: int = 5,
) -> list[Inducement]:
    """Check which inducements have been taken (swept) by price action.

    Rules (V02):
    - Taking IDM: wick sweep is SUFFICIENT (body close not required)
    - For bullish IDM (below swing high): price goes below IDM level
    - For bearish IDM (above swing low): price goes above IDM level

    Also updates swing validity:
    - Swings where IDM was taken before formation → is_valid_smc = True
    - Swings where IDM was NOT taken → is_valid_smc = False (inducement zone)

    The structural window accounts for lookback: a swing at index N with
    lookback L isn't confirmed until candle N+L, so IDM sweeps during
    the confirmation period still count.
    """
    swing_map = {s.candle_index: s for s in swings}

    for idm in inducements:
        parent_swing = swing_map.get(idm.parent_swing_index)
        if not parent_swing:
            continue

        # Structural window: swing isn't confirmed until candle_index + lookback
        structural_max_idx = parent_swing.candle_index + lookback

        for candle in candles:
            if candle.index <= idm.candle_index:
                continue

            if parent_swing.swing_type == SwingType.SWING_HIGH:
                # Bullish swing: IDM is a low. Taken if price wicks below.
                if candle.low <= idm.price:
                    idm.status = IDMStatus.TAKEN
                    idm.taken_at_candle = candle.index
                    # V04: body closed beyond IDM = stronger confirmation
                    if candle.body_bottom <= idm.price:
                        idm.body_closed = True
                    if candle.index <= structural_max_idx:
                        parent_swing.idm_taken = True
                    break
            else:
                # Bearish swing: IDM is a high. Taken if price wicks above.
                if candle.high >= idm.price:
                    idm.status = IDMStatus.TAKEN
                    idm.taken_at_candle = candle.index
                    # V04: body closed beyond IDM = stronger confirmation
                    if candle.body_top >= idm.price:
                        idm.body_closed = True
                    if candle.index <= structural_max_idx:
                        parent_swing.idm_taken = True
                    break

    return inducements


def validate_swings_with_idm(
    swings: list[SwingPoint],
    inducements: list[Inducement],
    candles: list[Candle] | None = None,
) -> list[SwingPoint]:
    """Re-validate swing classifications using IDM data.

    V01 Rule: A valid HH requires IDM taken + candle close conditions.
    Swings without IDM taken are marked as liquidity/inducement zones.

    Uses swing.idm_taken (set by check_idm_taken with structural window)
    as the single source of truth. This correctly handles:
    - ACTIVE IDMs that were never swept → invalid
    - TRANSFERRED IDMs that were never swept → invalid
    - IDMs swept after the structural window → invalid
    - IDMs swept within the structural window → valid

    V04/V06 additions:
    - candle_closed_properly: a candle body closed above previous swing high
      (for HH) or below previous swing low (for LL) — confirms the break
    - is_strong: IDM taken + IDM body closed + candle closed properly = Swing HH
      (V06 highest tier classification)
    """
    idm_by_swing = {}
    for idm in inducements:
        idm_by_swing[idm.parent_swing_index] = idm

    # Build previous same-type swing lookup for candle_closed_properly check
    prev_same_type: dict[int, SwingPoint] = {}
    last_high: SwingPoint | None = None
    last_low: SwingPoint | None = None
    for swing in swings:
        if swing.swing_type == SwingType.SWING_HIGH:
            if last_high is not None:
                prev_same_type[swing.candle_index] = last_high
            last_high = swing
        else:
            if last_low is not None:
                prev_same_type[swing.candle_index] = last_low
            last_low = swing

    # Build candle index map for body close checks
    idx_map = {c.index: c for c in candles} if candles else {}

    for swing in swings:
        if swing.classification == SwingClassification.UNCLASSIFIED:
            continue

        idm = idm_by_swing.get(swing.candle_index)

        if idm is None:
            swing.is_valid_smc = False
            continue

        # swing.idm_taken was set by check_idm_taken() — it accounts for
        # both the physical sweep AND the structural window constraint
        swing.is_valid_smc = swing.idm_taken

        # V01/V06: Check candle_closed_properly — did a candle's body
        # close beyond the PREVIOUS same-type swing's price?
        # For HH: body_top > previous_high.price
        # For LL: body_bottom < previous_low.price
        prev_swing = prev_same_type.get(swing.candle_index)
        if prev_swing and idx_map:
            _check_candle_closed_properly(swing, prev_swing, idx_map)

        # V06: is_strong = full Swing HH/HL tier
        # Requires: IDM taken + IDM body closed + candle closed properly
        idm_body_closed = idm.body_closed if idm else False
        swing.is_strong = (
            swing.idm_taken
            and idm_body_closed
            and swing.candle_closed_properly
        )

    return swings


def _check_candle_closed_properly(
    swing: SwingPoint,
    prev_same_type: SwingPoint,
    idx_map: dict[int, Candle],
) -> None:
    """Check if a candle body closed beyond the previous same-type swing.

    For swing HIGH (HH): any candle between prev high and this high must
    have body_top > prev_high.price (body closed above the previous high).

    For swing LOW (LL): any candle between prev low and this low must
    have body_bottom < prev_low.price (body closed below the previous low).
    """
    start = prev_same_type.candle_index + 1
    end = swing.candle_index + 1  # Include the swing candle itself

    if swing.swing_type == SwingType.SWING_HIGH:
        target_price = prev_same_type.price
        for ci in range(start, end):
            candle = idx_map.get(ci)
            if candle and candle.body_top > target_price:
                swing.candle_closed_properly = True
                return
    else:
        target_price = prev_same_type.price
        for ci in range(start, end):
            candle = idx_map.get(ci)
            if candle and candle.body_bottom < target_price:
                swing.candle_closed_properly = True
                return


def detect_and_validate(
    candles: list[Candle],
    swings: list[SwingPoint],
    lookback: int = 5,
) -> tuple[list[Inducement], list[SwingPoint]]:
    """Complete IDM pipeline: detect → check taken → validate swings.

    Args:
        lookback: Swing confirmation window — passed to check_idm_taken
                  so structural window = candle_index + lookback.

    Returns:
        (inducements, validated_swings)
    """
    inducements = detect_inducements(candles, swings)
    inducements = check_idm_taken(candles, inducements, swings, lookback)
    swings = validate_swings_with_idm(swings, inducements, candles)
    return inducements, swings
