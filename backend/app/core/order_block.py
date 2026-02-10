"""1.7 — Order Block (OB) Detection

Identifies institutional order placement zones.

Core Rules (V14):
- Bearish OB = bullish candle(s) immediately before a sell move
- Bullish OB = bearish candle(s) immediately before a buy move
- RULE 1: OB candle MUST have swept liquidity of previous candle or prev high/low
- RULE 2: Below/above that candle's wick, a price imbalance (FVG) MUST exist
- Both rules required simultaneously — one without other = INVALID
- 50% rule: when price returns to OB, body should NOT close beyond OB's 50%
- OB in main trend direction = HIGH PROBABILITY
- OB from candles inside previous candle's range (inside bar) = INVALID
- OB that is part of inducement/engineered liquidity = TRAP → invalid
"""

from __future__ import annotations
from app.models import (
    Candle, SwingPoint, Inducement, FVG, OrderBlock, LiquidityPool,
    Direction, TrendState, SwingType, IDMStatus,
)


def _is_inside_bar(candle: Candle, prev_candle: Candle) -> bool:
    """Check if candle is inside previous candle's range (V14: no valid OB)."""
    return candle.high <= prev_candle.high and candle.low >= prev_candle.low


def _candle_swept_prev_liquidity(candle: Candle, prev_candle: Candle) -> bool:
    """Check if candle swept liquidity of the previous candle.

    Rule 1: OB candle must sweep previous candle's high or low.
    - Bullish OB (bearish candle): must sweep prev candle's LOW
    - Bearish OB (bullish candle): must sweep prev candle's HIGH
    """
    if candle.is_bearish:
        # Bearish candle forming bullish OB → should sweep prev low
        return candle.low < prev_candle.low
    else:
        # Bullish candle forming bearish OB → should sweep prev high
        return candle.high > prev_candle.high


def _has_adjacent_fvg(
    candle_index: int,
    fvgs: list[FVG],
    direction: Direction,
) -> bool:
    """Check if an FVG exists adjacent to (within 2 candles of) the OB candle.

    Rule 2: FVG must exist below (bullish OB) or above (bearish OB) the OB candle.
    """
    for fvg in fvgs:
        if abs(fvg.candle_index - candle_index) <= 2:
            # FVG should match OB direction
            if fvg.direction == direction:
                return True
    return False


def _is_in_inducement_zone(
    candle_index: int,
    inducements: list[Inducement],
) -> bool:
    """Check if the OB is part of an inducement/engineered liquidity zone.

    V14: OB that is part of inducement = SMART MONEY TRAP → invalid.
    """
    for idm in inducements:
        if idm.status == IDMStatus.ACTIVE:
            # OB candle is near an active (untaken) IDM → it's in the trap zone
            if abs(candle_index - idm.candle_index) <= 3:
                return True
    return False


def detect_order_blocks(
    candles: list[Candle],
    swings: list[SwingPoint],
    fvgs: list[FVG],
    inducements: list[Inducement],
    trend: TrendState,
) -> list[OrderBlock]:
    """Detect valid order blocks.

    Scans for candles that precede a significant move in the opposite direction
    (the last bearish candle before a bullish move, or vice versa).
    """
    if len(candles) < 5:
        return []

    order_blocks: list[OrderBlock] = []

    for i in range(1, len(candles) - 2):
        curr = candles[i]
        prev = candles[i - 1]

        # Skip inside bars (V14 rule)
        if _is_inside_bar(curr, prev):
            continue

        # Look for bullish OB: bearish candle followed by bullish move
        if curr.is_bearish:
            # Check if next candles show bullish displacement
            next1 = candles[i + 1]
            next2 = candles[i + 2] if i + 2 < len(candles) else None

            bullish_move = next1.is_bullish and next1.close > curr.high
            if next2:
                bullish_move = bullish_move or (next2.close > curr.high)

            if bullish_move:
                # Rule 1: Must sweep previous candle's liquidity
                swept = _candle_swept_prev_liquidity(curr, prev)

                # Rule 2: Must have adjacent FVG
                has_fvg = _has_adjacent_fvg(curr.index, fvgs, Direction.BULLISH)

                # Trap check
                is_trap = _is_in_inducement_zone(curr.index, inducements)

                valid = swept and has_fvg and not is_trap

                order_blocks.append(OrderBlock(
                    candle_index_start=curr.index,
                    candle_index_end=curr.index,
                    upper_price=curr.high,
                    lower_price=curr.low,
                    direction=Direction.BULLISH,
                    valid=valid,
                    has_fvg=has_fvg,
                    swept_liquidity=swept,
                    is_trap=is_trap,
                ))

        # Look for bearish OB: bullish candle followed by bearish move
        if curr.is_bullish:
            next1 = candles[i + 1]
            next2 = candles[i + 2] if i + 2 < len(candles) else None

            bearish_move = next1.is_bearish and next1.close < curr.low
            if next2:
                bearish_move = bearish_move or (next2.close < curr.low)

            if bearish_move:
                swept = _candle_swept_prev_liquidity(curr, prev)
                has_fvg = _has_adjacent_fvg(curr.index, fvgs, Direction.BEARISH)
                is_trap = _is_in_inducement_zone(curr.index, inducements)
                valid = swept and has_fvg and not is_trap

                order_blocks.append(OrderBlock(
                    candle_index_start=curr.index,
                    candle_index_end=curr.index,
                    upper_price=curr.high,
                    lower_price=curr.low,
                    direction=Direction.BEARISH,
                    valid=valid,
                    has_fvg=has_fvg,
                    swept_liquidity=swept,
                    is_trap=is_trap,
                ))

    return order_blocks


def check_ob_mitigation(
    order_blocks: list[OrderBlock],
    candles: list[Candle],
) -> list[OrderBlock]:
    """Check if price has returned to mitigate any order blocks.

    An OB is mitigated when price returns to its zone.
    50% rule: if body closes beyond OB's 50%, the OB is invalidated.
    """
    for ob in order_blocks:
        if ob.mitigated:
            continue

        for candle in candles:
            if candle.index <= ob.candle_index_end:
                continue

            if ob.direction == Direction.BULLISH:
                # Bullish OB: price drops to OB zone
                if candle.low <= ob.upper_price:
                    ob.mitigated = True
                    ob.mitigated_at_candle = candle.index

                    # 50% rule check
                    if candle.body_bottom < ob.midpoint:
                        ob.valid = False  # Body closed beyond 50%
                    break
            else:
                # Bearish OB: price rises to OB zone
                if candle.high >= ob.lower_price:
                    ob.mitigated = True
                    ob.mitigated_at_candle = candle.index

                    if candle.body_top > ob.midpoint:
                        ob.valid = False
                    break

    return order_blocks


def detect_all_order_blocks(
    candles: list[Candle],
    swings: list[SwingPoint],
    fvgs: list[FVG],
    inducements: list[Inducement],
    trend: TrendState,
) -> list[OrderBlock]:
    """Complete OB pipeline: detect → check mitigation."""
    obs = detect_order_blocks(candles, swings, fvgs, inducements, trend)
    return check_ob_mitigation(obs, candles)
