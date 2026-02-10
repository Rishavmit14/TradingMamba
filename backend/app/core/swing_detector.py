"""1.1 — Swing Point Detection (Foundation Layer)

Identifies structural swing highs and lows from OHLCV data, then classifies
them as HH/HL/LH/LL using SMC rules (NOT retail rules).

SMC vs Retail difference (V01):
- Retail marks EVERY local high/low as a swing → gets 4+ HH where SMC sees 1
- SMC requires inducement taken + candle close conditions for valid classification
- Invalid swings are marked as inducement/liquidity zones (not structural)

This module handles the geometric detection (finding raw swing points).
Classification into HH/HL/LH/LL happens after IDM detection (1.2) provides context.
"""

from __future__ import annotations
from app.models import Candle, SwingPoint, SwingType, SwingClassification, TrendState


def detect_swing_points(candles: list[Candle], lookback: int = 5) -> list[SwingPoint]:
    """Find all potential swing highs and lows using fractal detection.

    A swing high at index i exists when:
        candle[i].high >= max(candle[i-lookback:i].high)
        AND candle[i].high >= max(candle[i+1:i+lookback+1].high)

    A swing low at index i exists when:
        candle[i].low <= min(candle[i-lookback:i].low)
        AND candle[i].low <= min(candle[i+1:i+lookback+1].low)

    Args:
        candles: Array of OHLCV candles (must have .index set)
        lookback: Number of candles on each side to confirm swing

    Returns:
        List of SwingPoint objects, sorted by candle_index
    """
    if len(candles) < lookback * 2 + 1:
        return []

    swings: list[SwingPoint] = []

    for i in range(lookback, len(candles) - lookback):
        candle = candles[i]

        # Check swing high
        left_highs = [c.high for c in candles[i - lookback:i]]
        right_highs = [c.high for c in candles[i + 1:i + lookback + 1]]

        if candle.high >= max(left_highs) and candle.high >= max(right_highs):
            swings.append(SwingPoint(
                candle_index=candle.index,
                price=candle.high,
                swing_type=SwingType.SWING_HIGH,
            ))

        # Check swing low
        left_lows = [c.low for c in candles[i - lookback:i]]
        right_lows = [c.low for c in candles[i + 1:i + lookback + 1]]

        if candle.low <= min(left_lows) and candle.low <= min(right_lows):
            swings.append(SwingPoint(
                candle_index=candle.index,
                price=candle.low,
                swing_type=SwingType.SWING_LOW,
            ))

    return sorted(swings, key=lambda s: s.candle_index)


def enforce_alternation(swings: list[SwingPoint]) -> list[SwingPoint]:
    """Ensure swing highs and lows alternate (no two consecutive highs/lows).

    When two consecutive swings of the same type exist:
    - For highs: keep the higher one
    - For lows: keep the lower one

    This mirrors how ICT structure mapping works — you connect the most
    significant swing, not every minor one.
    """
    if len(swings) < 2:
        return swings

    result: list[SwingPoint] = [swings[0]]

    for swing in swings[1:]:
        prev = result[-1]

        if swing.swing_type == prev.swing_type:
            # Same type — keep the more extreme one
            if swing.swing_type == SwingType.SWING_HIGH:
                if swing.price >= prev.price:
                    result[-1] = swing
            else:
                if swing.price <= prev.price:
                    result[-1] = swing
        else:
            result.append(swing)

    return result


def classify_swings(swings: list[SwingPoint]) -> list[SwingPoint]:
    """Classify swing points as HH/HL/LH/LL based on sequential comparison.

    This is the INITIAL classification — geometric only.
    The IDM detector (1.2) will later mark some as invalid (inducement zones)
    based on whether inducement was actually taken.

    Rules:
    - A swing high is HH if higher than the previous swing high, else LH
    - A swing low is HL if higher than the previous swing low, else LL
    - First swings of each type start as UNCLASSIFIED
    """
    prev_high: SwingPoint | None = None
    prev_low: SwingPoint | None = None

    for swing in swings:
        if swing.swing_type == SwingType.SWING_HIGH:
            if prev_high is None:
                swing.classification = SwingClassification.UNCLASSIFIED
            elif swing.price > prev_high.price:
                swing.classification = SwingClassification.HH
            elif swing.price < prev_high.price:
                swing.classification = SwingClassification.LH
            else:
                # Equal high — treat as potential liquidity (equal highs)
                swing.classification = SwingClassification.UNCLASSIFIED
            prev_high = swing

        elif swing.swing_type == SwingType.SWING_LOW:
            if prev_low is None:
                swing.classification = SwingClassification.UNCLASSIFIED
            elif swing.price > prev_low.price:
                swing.classification = SwingClassification.HL
            elif swing.price < prev_low.price:
                swing.classification = SwingClassification.LL
            else:
                swing.classification = SwingClassification.UNCLASSIFIED
            prev_low = swing

    return swings


def determine_trend(swings: list[SwingPoint]) -> TrendState:
    """Determine current trend from classified swing points.

    Rules (V01):
    - Bullish: market making HH and HL
    - Bearish: market making LH and LL
    - Ranging: mixed or insufficient data

    Uses the last 4 swing points (2 highs + 2 lows) to determine trend.
    """
    if len(swings) < 4:
        return TrendState.RANGING

    # Get last few classified highs and lows
    recent_highs = [s for s in swings if s.swing_type == SwingType.SWING_HIGH
                    and s.classification != SwingClassification.UNCLASSIFIED][-2:]
    recent_lows = [s for s in swings if s.swing_type == SwingType.SWING_LOW
                   and s.classification != SwingClassification.UNCLASSIFIED][-2:]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return TrendState.RANGING

    last_high = recent_highs[-1]
    last_low = recent_lows[-1]

    # Bullish: last high is HH AND last low is HL
    if (last_high.classification == SwingClassification.HH
            and last_low.classification == SwingClassification.HL):
        return TrendState.BULLISH

    # Bearish: last high is LH AND last low is LL
    if (last_high.classification == SwingClassification.LH
            and last_low.classification == SwingClassification.LL):
        return TrendState.BEARISH

    return TrendState.RANGING


def detect_and_classify(candles: list[Candle], lookback: int = 5) -> tuple[list[SwingPoint], TrendState]:
    """Complete swing detection pipeline: detect → alternate → classify → trend.

    This is the main entry point for swing analysis.

    Returns:
        (classified_swings, current_trend)
    """
    raw_swings = detect_swing_points(candles, lookback)
    alternating = enforce_alternation(raw_swings)
    classified = classify_swings(alternating)
    trend = determine_trend(classified)
    return classified, trend
