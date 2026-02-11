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
    """Find pullback candle indices between two points.

    A pullback in a bullish swing is a candle whose low dips below the
    previous candle's low (a temporary retracement).

    A pullback in a bearish swing is a candle whose high rises above the
    previous candle's high (a temporary retracement).

    Filters out internal candles (inside previous candle's range) per V04 rule.
    """
    pullbacks = []

    # Build index map for quick lookup
    idx_map = {c.index: c for c in candles}
    indices = sorted([c.index for c in candles if start_idx <= c.index <= end_idx])

    if len(indices) < 3:
        return []

    for i in range(1, len(indices) - 1):
        curr_idx = indices[i]
        prev_idx = indices[i - 1]

        curr = idx_map.get(curr_idx)
        prev = idx_map.get(prev_idx)
        if not curr or not prev:
            continue

        # Skip internal candles (V04: inside previous candle's range = no IDM)
        if curr.high <= prev.high and curr.low >= prev.low:
            continue

        if direction == SwingType.SWING_HIGH:
            # For bullish swing: pullback = candle makes a lower low
            if curr.low < prev.low:
                pullbacks.append(curr_idx)
        else:
            # For bearish swing: pullback = candle makes a higher high
            if curr.high > prev.high:
                pullbacks.append(curr_idx)

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

                inducements.append(Inducement(
                    candle_index=idm_candle_idx,
                    price=idm_price,
                    parent_swing_index=swing.candle_index,
                    status=IDMStatus.ACTIVE,
                ))
        else:
            # No pullback found → IDM transfers from previous swing (V02 rule)
            # Find the most recent IDM that hasn't been taken
            for prev_idm in reversed(inducements):
                if prev_idm.status == IDMStatus.ACTIVE:
                    inducements.append(Inducement(
                        candle_index=prev_idm.candle_index,
                        price=prev_idm.price,
                        parent_swing_index=swing.candle_index,
                        status=IDMStatus.TRANSFERRED,
                    ))
                    break

    return inducements


def check_idm_taken(
    candles: list[Candle],
    inducements: list[Inducement],
    swings: list[SwingPoint],
) -> list[Inducement]:
    """Check which inducements have been taken (swept) by price action.

    Rules (V02):
    - Taking IDM: wick sweep is SUFFICIENT (body close not required)
    - For bullish IDM (below swing high): price goes below IDM level
    - For bearish IDM (above swing low): price goes above IDM level

    Also updates swing validity:
    - Swings where IDM was taken before formation → is_valid_smc = True
    - Swings where IDM was NOT taken → is_valid_smc = False (inducement zone)
    """
    idx_map = {c.index: c for c in candles}
    swing_map = {s.candle_index: s for s in swings}

    for idm in inducements:
        parent_swing = swing_map.get(idm.parent_swing_index)
        if not parent_swing:
            continue

        # Check candles from IDM onward (not just up to parent swing).
        # An IDM is "taken" whenever price sweeps its level, even after
        # the parent swing has formed — the level remains valid until swept.
        for candle in candles:
            if candle.index <= idm.candle_index:
                continue

            if parent_swing.swing_type == SwingType.SWING_HIGH:
                # Bullish swing: IDM is a low. Taken if price wicks below.
                if candle.low <= idm.price:
                    idm.status = IDMStatus.TAKEN
                    idm.taken_at_candle = candle.index
                    # Only mark parent swing's idm_taken if swept before swing formed
                    if candle.index <= parent_swing.candle_index:
                        parent_swing.idm_taken = True
                    break
            else:
                # Bearish swing: IDM is a high. Taken if price wicks above.
                if candle.high >= idm.price:
                    idm.status = IDMStatus.TAKEN
                    idm.taken_at_candle = candle.index
                    if candle.index <= parent_swing.candle_index:
                        parent_swing.idm_taken = True
                    break

    return inducements


def validate_swings_with_idm(
    swings: list[SwingPoint],
    inducements: list[Inducement],
) -> list[SwingPoint]:
    """Re-validate swing classifications using IDM data.

    V01 Rule: A valid HH requires IDM taken + candle close conditions.
    Swings without IDM taken are marked as liquidity/inducement zones.

    This is where SMC diverges from retail:
    - Retail sees 4 HH → SMC sees 1 HH + 3 inducement zones
    """
    idm_by_swing = {}
    for idm in inducements:
        idm_by_swing[idm.parent_swing_index] = idm

    for swing in swings:
        if swing.classification == SwingClassification.UNCLASSIFIED:
            continue

        idm = idm_by_swing.get(swing.candle_index)

        if idm is None:
            # No IDM at all → cannot be valid SMC structure
            swing.is_valid_smc = False
            continue

        if idm.status == IDMStatus.ACTIVE:
            # IDM exists but was never taken → invalid swing
            swing.is_valid_smc = False
        else:
            # IDM was taken → valid SMC swing
            swing.is_valid_smc = True
            swing.idm_taken = True

    return swings


def detect_and_validate(
    candles: list[Candle],
    swings: list[SwingPoint],
) -> tuple[list[Inducement], list[SwingPoint]]:
    """Complete IDM pipeline: detect → check taken → validate swings.

    Returns:
        (inducements, validated_swings)
    """
    inducements = detect_inducements(candles, swings)
    inducements = check_idm_taken(candles, inducements, swings)
    swings = validate_swings_with_idm(swings, inducements)
    return inducements, swings
