"""1.3 — Liquidity Pool Identification

Maps where liquidity accumulates — equal highs/lows, swing extremes,
and IDM levels — and detects sweep/grab events.

Core Rules (V02):
- Equal Highs (double top) = buy-side liquidity pool ABOVE
- Equal Lows (double bottom) = sell-side liquidity pool BELOW
- Every swing extreme is a liquidity target
- IDM levels are liquidity targets
- Liquidity sweep = wick sweeps but doesn't close beyond (returns)
- Liquidity grab = body closes beyond, traps traders, then reverses
"""

from __future__ import annotations
from app.models import (
    Candle, SwingPoint, Inducement, LiquidityPool,
    LiquidityType, LiquiditySource, LiquidityEvent,
    SwingType, IDMStatus,
)


# Price proximity threshold: two prices are "equal" if within this ratio
EQUAL_PRICE_TOLERANCE = 0.001  # 0.1%


def _prices_equal(a: float, b: float) -> bool:
    if a == 0 or b == 0:
        return a == b
    return abs(a - b) / max(a, b) < EQUAL_PRICE_TOLERANCE


def detect_equal_highs(swings: list[SwingPoint]) -> list[LiquidityPool]:
    """Find equal highs (double/triple tops) — buy-side liquidity above."""
    pools = []
    highs = [s for s in swings if s.swing_type == SwingType.SWING_HIGH]

    for i in range(len(highs)):
        cluster_indices = [highs[i].candle_index]
        for j in range(i + 1, len(highs)):
            if _prices_equal(highs[i].price, highs[j].price):
                cluster_indices.append(highs[j].candle_index)

        if len(cluster_indices) >= 2:
            # Check we haven't already added this cluster
            pool_level = highs[i].price
            already_exists = any(
                _prices_equal(p.price_level, pool_level)
                for p in pools if p.source == LiquiditySource.EQUAL_HIGHS
            )
            if not already_exists:
                pools.append(LiquidityPool(
                    price_level=pool_level,
                    pool_type=LiquidityType.BUY_SIDE,
                    source=LiquiditySource.EQUAL_HIGHS,
                    candle_indices=cluster_indices,
                ))

    return pools


def detect_equal_lows(swings: list[SwingPoint]) -> list[LiquidityPool]:
    """Find equal lows (double/triple bottoms) — sell-side liquidity below."""
    pools = []
    lows = [s for s in swings if s.swing_type == SwingType.SWING_LOW]

    for i in range(len(lows)):
        cluster_indices = [lows[i].candle_index]
        for j in range(i + 1, len(lows)):
            if _prices_equal(lows[i].price, lows[j].price):
                cluster_indices.append(lows[j].candle_index)

        if len(cluster_indices) >= 2:
            pool_level = lows[i].price
            already_exists = any(
                _prices_equal(p.price_level, pool_level)
                for p in pools if p.source == LiquiditySource.EQUAL_LOWS
            )
            if not already_exists:
                pools.append(LiquidityPool(
                    price_level=pool_level,
                    pool_type=LiquidityType.SELL_SIDE,
                    source=LiquiditySource.EQUAL_LOWS,
                    candle_indices=cluster_indices,
                ))

    return pools


def detect_swing_liquidity(swings: list[SwingPoint]) -> list[LiquidityPool]:
    """Every swing extreme is a liquidity target."""
    pools = []
    for swing in swings:
        pool_type = (LiquidityType.BUY_SIDE
                     if swing.swing_type == SwingType.SWING_HIGH
                     else LiquidityType.SELL_SIDE)
        pools.append(LiquidityPool(
            price_level=swing.price,
            pool_type=pool_type,
            source=LiquiditySource.SWING_EXTREME,
            candle_indices=[swing.candle_index],
        ))
    return pools


def detect_idm_liquidity(inducements: list[Inducement]) -> list[LiquidityPool]:
    """IDM levels are liquidity targets (retail stops cluster there)."""
    pools = []
    for idm in inducements:
        if idm.status != IDMStatus.TAKEN:
            pools.append(LiquidityPool(
                price_level=idm.price,
                pool_type=LiquidityType.SELL_SIDE,  # IDM is below swings typically
                source=LiquiditySource.IDM_LEVEL,
                candle_indices=[idm.candle_index],
            ))
    return pools


def check_liquidity_events(
    pools: list[LiquidityPool],
    candles: list[Candle],
    after_candle_index: int = 0,
) -> list[LiquidityPool]:
    """Check if any liquidity pools have been swept or grabbed.

    Rules (V02, V13):
    - Sweep: wick goes beyond pool level but body does NOT close beyond
    - Grab: body closes beyond pool level (traps breakout traders)
    """
    for pool in pools:
        if pool.swept:
            continue

        last_pool_candle = max(pool.candle_indices) if pool.candle_indices else 0
        start_check = max(after_candle_index, last_pool_candle + 1)

        for candle in candles:
            if candle.index < start_check:
                continue

            if pool.pool_type == LiquidityType.BUY_SIDE:
                # Buy-side liquidity is ABOVE — check if price went above
                if candle.high > pool.price_level:
                    pool.swept = True
                    pool.swept_at_candle = candle.index

                    # Grab vs Sweep: did body close above?
                    if candle.body_top > pool.price_level:
                        pool.event_type = LiquidityEvent.GRAB
                    else:
                        pool.event_type = LiquidityEvent.SWEEP
                    break

            else:  # SELL_SIDE
                # Sell-side liquidity is BELOW — check if price went below
                if candle.low < pool.price_level:
                    pool.swept = True
                    pool.swept_at_candle = candle.index

                    if candle.body_bottom < pool.price_level:
                        pool.event_type = LiquidityEvent.GRAB
                    else:
                        pool.event_type = LiquidityEvent.SWEEP
                    break

    return pools


def detect_all_liquidity(
    candles: list[Candle],
    swings: list[SwingPoint],
    inducements: list[Inducement],
) -> list[LiquidityPool]:
    """Complete liquidity detection pipeline.

    Returns all liquidity pools with sweep/grab status checked.
    """
    pools = []
    pools.extend(detect_equal_highs(swings))
    pools.extend(detect_equal_lows(swings))
    pools.extend(detect_swing_liquidity(swings))
    pools.extend(detect_idm_liquidity(inducements))
    pools = check_liquidity_events(pools, candles)
    return pools
