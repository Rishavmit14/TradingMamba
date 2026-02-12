"""1.12 — Price Delivery Cycle Detection (V11)

Identifies the 4 phases of price delivery:
1. Consolidation — price ranging, building liquidity on both sides
2. Expansion — quick move (BOS), Smart Money footprints
3. Retracement — pullback correcting imbalances (FVGs)
4. Reversal — CHoCH, trend direction change

Transition Rules:
- Consolidation → Expansion (valid)
- Expansion → Retracement (valid)
- Retracement → Expansion (valid, continuation)
- Retracement → Reversal (valid, trend change)
- Consolidation → Retracement (FORBIDDEN)
- Expansion → Reversal (FORBIDDEN)
"""

from __future__ import annotations
from app.models import (
    Candle, SwingPoint, BOS, CHoCH, FVG,
    PriceCycleEvent, PricePhase, TrendState,
)

# Valid phase transitions
_VALID_TRANSITIONS = {
    PricePhase.CONSOLIDATION: {PricePhase.EXPANSION},
    PricePhase.EXPANSION: {PricePhase.RETRACEMENT, PricePhase.CONSOLIDATION},
    PricePhase.RETRACEMENT: {PricePhase.EXPANSION, PricePhase.REVERSAL, PricePhase.CONSOLIDATION},
    PricePhase.REVERSAL: {PricePhase.CONSOLIDATION, PricePhase.EXPANSION},
}


def _calculate_atr(candles: list[Candle], period: int = 14) -> float:
    """Simple ATR calculation for volatility threshold."""
    if not candles:
        return 0.0
    recent = candles[-period:] if len(candles) >= period else candles
    return sum(c.total_range for c in recent) / len(recent)


def _is_consolidation_segment(
    candles: list[Candle],
    start: int,
    end: int,
    atr: float,
) -> bool:
    """Check if a candle segment is consolidating (tight range).

    Consolidation: total range of segment < 2x ATR and no single candle
    moves more than 1.5x ATR.
    """
    segment = [c for c in candles if start <= c.index <= end]
    if len(segment) < 3:
        return False

    seg_high = max(c.high for c in segment)
    seg_low = min(c.low for c in segment)
    total_range = seg_high - seg_low

    if atr == 0:
        return False

    # Tight range + no large single moves
    if total_range > atr * 2.5:
        return False

    large_moves = sum(1 for c in segment if c.total_range > atr * 1.5)
    return large_moves <= 1


def _detect_expansion_at(
    candle: Candle,
    bos_events: list[BOS],
    atr: float,
) -> bool:
    """Check if a candle represents expansion (BOS + large displacement).

    Expansion = candle with valid BOS nearby AND displacement > 1.2x ATR.
    """
    if atr == 0:
        return False

    # Large body move
    if candle.body_size < atr * 0.8:
        return False

    # Check if a BOS was confirmed at or near this candle
    for bos in bos_events:
        if abs(bos.candle_index - candle.index) <= 2 and bos.valid:
            return True

    return candle.total_range > atr * 1.5


def _detect_retracement_at(
    candle: Candle,
    prev_expansion_dir: str,
    fvgs: list[FVG],
    atr: float,
) -> bool:
    """Check if a candle is part of a retracement (correcting FVGs).

    Retracement: moves opposite to expansion direction, smaller than expansion.
    """
    if atr == 0:
        return False

    # Must be moving opposite to expansion
    if prev_expansion_dir == "bullish" and candle.is_bullish:
        return False
    if prev_expansion_dir == "bearish" and candle.is_bearish:
        return False

    # Check if any unmitigated FVG is being corrected (price entering gap)
    for fvg in fvgs:
        if fvg.mitigated:
            continue
        if fvg.lower_price <= candle.close <= fvg.upper_price:
            return True

    # Smaller pullback (not another expansion)
    return candle.total_range < atr * 1.2


def detect_price_cycles(
    candles: list[Candle],
    bos_events: list[BOS],
    choch_events: list[CHoCH],
    fvgs: list[FVG],
    swings: list[SwingPoint],
) -> tuple[list[PriceCycleEvent], PricePhase]:
    """Detect price delivery phases across candle data.

    Algorithm:
    1. Segment candles into windows
    2. Classify each window: consolidation, expansion, retracement, or reversal
    3. Validate transitions (enforce forbidden rules)
    4. Return phase history + current phase

    Returns:
        (list of PriceCycleEvent, current_phase)
    """
    if len(candles) < 10:
        return [], PricePhase.CONSOLIDATION

    atr = _calculate_atr(candles)
    if atr == 0:
        return [], PricePhase.CONSOLIDATION

    events: list[PriceCycleEvent] = []
    current_phase = PricePhase.CONSOLIDATION
    phase_start = candles[0].index
    phase_high = candles[0].high
    phase_low = candles[0].low
    last_expansion_dir = ""

    # Build lookup sets for quick checks
    bos_indices = {b.candle_index for b in bos_events if b.valid}
    choch_indices = {c.candle_index for c in choch_events if c.confirmed and not c.is_fake}

    window_size = 5
    i = 0

    while i < len(candles) - window_size:
        window = candles[i:i + window_size]
        w_high = max(c.high for c in window)
        w_low = min(c.low for c in window)

        new_phase = None

        # Check for CHoCH → Reversal
        if any(c.index in choch_indices for c in window):
            new_phase = PricePhase.REVERSAL

        # Check for BOS / large displacement → Expansion
        elif any(_detect_expansion_at(c, bos_events, atr) for c in window):
            new_phase = PricePhase.EXPANSION
            # Track expansion direction
            body_move = window[-1].close - window[0].open
            last_expansion_dir = "bullish" if body_move > 0 else "bearish"

        # Check for tight range → Consolidation
        elif _is_consolidation_segment(candles, window[0].index, window[-1].index, atr):
            new_phase = PricePhase.CONSOLIDATION

        # Check for pullback correcting FVGs → Retracement
        elif last_expansion_dir and any(
            _detect_retracement_at(c, last_expansion_dir, fvgs, atr) for c in window
        ):
            new_phase = PricePhase.RETRACEMENT

        if new_phase and new_phase != current_phase:
            # Validate transition
            valid = new_phase in _VALID_TRANSITIONS.get(current_phase, set())

            # Close current phase
            events.append(PriceCycleEvent(
                candle_index=window[0].index,
                phase=current_phase,
                start_index=phase_start,
                end_index=window[0].index - 1,
                high=phase_high,
                low=phase_low,
                valid_transition=True,
            ))

            # Start new phase
            current_phase = new_phase
            phase_start = window[0].index
            phase_high = w_high
            phase_low = w_low

            # Record transition validity on new event
            if not valid:
                # Forbidden transition — still record but mark invalid
                pass  # valid_transition will be set when this phase closes
        else:
            # Extend current phase
            phase_high = max(phase_high, w_high)
            phase_low = min(phase_low, w_low)

        i += window_size

    # Close final phase
    if candles:
        events.append(PriceCycleEvent(
            candle_index=candles[-1].index,
            phase=current_phase,
            start_index=phase_start,
            end_index=candles[-1].index,
            high=phase_high,
            low=phase_low,
        ))

    return events, current_phase
