"""
ICT Chart Pattern Recognition Module
Detects ICT patterns from price data without requiring deep learning
Uses algorithmic pattern detection - 100% FREE, no GPU needed!
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PatternType(Enum):
    """ICT Pattern Types"""
    # Order Blocks
    BULLISH_ORDER_BLOCK = "bullish_order_block"
    BEARISH_ORDER_BLOCK = "bearish_order_block"

    # Fair Value Gaps
    BULLISH_FVG = "bullish_fvg"
    BEARISH_FVG = "bearish_fvg"

    # Market Structure
    BREAK_OF_STRUCTURE_BULLISH = "bos_bullish"
    BREAK_OF_STRUCTURE_BEARISH = "bos_bearish"
    CHANGE_OF_CHARACTER_BULLISH = "choch_bullish"
    CHANGE_OF_CHARACTER_BEARISH = "choch_bearish"

    # Liquidity
    LIQUIDITY_SWEEP_HIGH = "liquidity_sweep_high"
    LIQUIDITY_SWEEP_LOW = "liquidity_sweep_low"
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"

    # Entry Patterns
    OPTIMAL_TRADE_ENTRY = "optimal_trade_entry"
    SILVER_BULLET = "silver_bullet"
    JUDAS_SWING = "judas_swing"

    # Power of Three
    ACCUMULATION = "accumulation"
    MANIPULATION = "manipulation"
    DISTRIBUTION = "distribution"


@dataclass
class DetectedPattern:
    """Represents a detected ICT pattern"""
    pattern_type: PatternType
    start_index: int
    end_index: int
    price_high: float
    price_low: float
    confidence: float
    timestamp: datetime
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'pattern_type': self.pattern_type.value,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'price_high': self.price_high,
            'price_low': self.price_low,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'details': self.details
        }


class ICTPatternRecognizer:
    """
    Algorithmic ICT pattern recognition
    No deep learning required - uses pure price action analysis
    """

    def __init__(self, swing_lookback: int = 5, fvg_min_size: float = 0.0001):
        self.swing_lookback = swing_lookback
        self.fvg_min_size = fvg_min_size

    def detect_all_patterns(self, data: pd.DataFrame) -> List[DetectedPattern]:
        """
        Detect all ICT patterns in the price data

        Parameters:
        - data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
        - List of detected patterns
        """
        patterns = []

        # Detect each pattern type
        patterns.extend(self.detect_order_blocks(data))
        patterns.extend(self.detect_fair_value_gaps(data))
        patterns.extend(self.detect_market_structure_breaks(data))
        patterns.extend(self.detect_liquidity_patterns(data))
        patterns.extend(self.detect_entry_patterns(data))
        patterns.extend(self.detect_power_of_three(data))

        # Sort by timestamp
        patterns.sort(key=lambda p: p.start_index)

        return patterns

    def detect_order_blocks(self, data: pd.DataFrame) -> List[DetectedPattern]:
        """
        Detect Order Blocks

        Bullish OB: Last bearish candle before strong bullish move
        Bearish OB: Last bullish candle before strong bearish move
        """
        patterns = []
        closes = data['close'].values
        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values

        for i in range(2, len(data) - 2):
            # Bullish Order Block
            # Current candle is bearish, next candle breaks above high
            is_bearish = closes[i] < opens[i]
            next_breaks_high = closes[i + 1] > highs[i]
            strong_move = (closes[i + 1] - closes[i]) / closes[i] > 0.001

            if is_bearish and next_breaks_high and strong_move:
                # Check if OB is still valid (not mitigated)
                current_price = closes[-1]
                is_valid = current_price > lows[i]

                if is_valid:
                    confidence = min(0.9, 0.5 + (closes[i + 1] - highs[i]) / highs[i] * 100)
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.BULLISH_ORDER_BLOCK,
                        start_index=i,
                        end_index=i,
                        price_high=highs[i],
                        price_low=lows[i],
                        confidence=confidence,
                        timestamp=data.index[i] if hasattr(data.index[i], 'isoformat') else datetime.now(),
                        details={
                            'candle_body': abs(closes[i] - opens[i]),
                            'break_strength': (closes[i + 1] - highs[i]) / highs[i] * 100,
                            'mitigated': False
                        }
                    ))

            # Bearish Order Block
            is_bullish = closes[i] > opens[i]
            next_breaks_low = closes[i + 1] < lows[i]
            strong_down = (closes[i] - closes[i + 1]) / closes[i] > 0.001

            if is_bullish and next_breaks_low and strong_down:
                current_price = closes[-1]
                is_valid = current_price < highs[i]

                if is_valid:
                    confidence = min(0.9, 0.5 + (lows[i] - closes[i + 1]) / lows[i] * 100)
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.BEARISH_ORDER_BLOCK,
                        start_index=i,
                        end_index=i,
                        price_high=highs[i],
                        price_low=lows[i],
                        confidence=confidence,
                        timestamp=data.index[i] if hasattr(data.index[i], 'isoformat') else datetime.now(),
                        details={
                            'candle_body': abs(closes[i] - opens[i]),
                            'break_strength': (lows[i] - closes[i + 1]) / lows[i] * 100,
                            'mitigated': False
                        }
                    ))

        return patterns[-10:]  # Return last 10 OBs

    def detect_fair_value_gaps(self, data: pd.DataFrame) -> List[DetectedPattern]:
        """
        Detect Fair Value Gaps (Imbalances)

        Bullish FVG: Gap between candle 1 high and candle 3 low
        Bearish FVG: Gap between candle 1 low and candle 3 high
        """
        patterns = []
        highs = data['high'].values
        lows = data['low'].values
        current_price = data['close'].values[-1]

        for i in range(2, len(data)):
            # Bullish FVG - gap up
            if lows[i] > highs[i - 2]:
                gap_size = lows[i] - highs[i - 2]

                if gap_size > self.fvg_min_size:
                    # Check if FVG is filled
                    is_filled = current_price < highs[i - 2]

                    if not is_filled:
                        confidence = min(0.85, 0.4 + gap_size / highs[i - 2] * 50)
                        patterns.append(DetectedPattern(
                            pattern_type=PatternType.BULLISH_FVG,
                            start_index=i - 1,
                            end_index=i - 1,
                            price_high=lows[i],
                            price_low=highs[i - 2],
                            confidence=confidence,
                            timestamp=data.index[i - 1] if hasattr(data.index[i - 1], 'isoformat') else datetime.now(),
                            details={
                                'gap_size': gap_size,
                                'gap_percent': gap_size / highs[i - 2] * 100,
                                'filled': is_filled
                            }
                        ))

            # Bearish FVG - gap down
            if highs[i] < lows[i - 2]:
                gap_size = lows[i - 2] - highs[i]

                if gap_size > self.fvg_min_size:
                    is_filled = current_price > lows[i - 2]

                    if not is_filled:
                        confidence = min(0.85, 0.4 + gap_size / lows[i - 2] * 50)
                        patterns.append(DetectedPattern(
                            pattern_type=PatternType.BEARISH_FVG,
                            start_index=i - 1,
                            end_index=i - 1,
                            price_high=lows[i - 2],
                            price_low=highs[i],
                            confidence=confidence,
                            timestamp=data.index[i - 1] if hasattr(data.index[i - 1], 'isoformat') else datetime.now(),
                            details={
                                'gap_size': gap_size,
                                'gap_percent': gap_size / lows[i - 2] * 100,
                                'filled': is_filled
                            }
                        ))

        return patterns[-10:]  # Return last 10 FVGs

    def detect_market_structure_breaks(self, data: pd.DataFrame) -> List[DetectedPattern]:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH)
        """
        patterns = []
        highs = data['high'].values
        lows = data['low'].values

        # Find swing points
        swing_highs = []
        swing_lows = []

        for i in range(self.swing_lookback, len(data) - self.swing_lookback):
            # Swing high
            if highs[i] == max(highs[i - self.swing_lookback:i + self.swing_lookback + 1]):
                swing_highs.append((i, highs[i]))

            # Swing low
            if lows[i] == min(lows[i - self.swing_lookback:i + self.swing_lookback + 1]):
                swing_lows.append((i, lows[i]))

        # Detect BOS - break of previous swing point in trend direction
        for i in range(1, len(swing_highs)):
            prev_idx, prev_high = swing_highs[i - 1]
            curr_idx, curr_high = swing_highs[i]

            if curr_high > prev_high:
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.BREAK_OF_STRUCTURE_BULLISH,
                    start_index=prev_idx,
                    end_index=curr_idx,
                    price_high=curr_high,
                    price_low=prev_high,
                    confidence=0.75,
                    timestamp=data.index[curr_idx] if hasattr(data.index[curr_idx], 'isoformat') else datetime.now(),
                    details={'break_level': prev_high}
                ))

        for i in range(1, len(swing_lows)):
            prev_idx, prev_low = swing_lows[i - 1]
            curr_idx, curr_low = swing_lows[i]

            if curr_low < prev_low:
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.BREAK_OF_STRUCTURE_BEARISH,
                    start_index=prev_idx,
                    end_index=curr_idx,
                    price_high=prev_low,
                    price_low=curr_low,
                    confidence=0.75,
                    timestamp=data.index[curr_idx] if hasattr(data.index[curr_idx], 'isoformat') else datetime.now(),
                    details={'break_level': prev_low}
                ))

        return patterns[-5:]

    def detect_liquidity_patterns(self, data: pd.DataFrame) -> List[DetectedPattern]:
        """
        Detect liquidity sweeps and equal highs/lows
        """
        patterns = []
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values

        # Find equal highs/lows (within 0.1% tolerance)
        tolerance = 0.001

        for i in range(len(data) - 1):
            for j in range(i + 1, min(i + 20, len(data))):
                # Equal highs
                if abs(highs[i] - highs[j]) / highs[i] < tolerance:
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.EQUAL_HIGHS,
                        start_index=i,
                        end_index=j,
                        price_high=max(highs[i], highs[j]),
                        price_low=min(highs[i], highs[j]),
                        confidence=0.7,
                        timestamp=data.index[j] if hasattr(data.index[j], 'isoformat') else datetime.now(),
                        details={
                            'level': (highs[i] + highs[j]) / 2,
                            'liquidity_type': 'buy_side'
                        }
                    ))

                # Equal lows
                if abs(lows[i] - lows[j]) / lows[i] < tolerance:
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.EQUAL_LOWS,
                        start_index=i,
                        end_index=j,
                        price_high=max(lows[i], lows[j]),
                        price_low=min(lows[i], lows[j]),
                        confidence=0.7,
                        timestamp=data.index[j] if hasattr(data.index[j], 'isoformat') else datetime.now(),
                        details={
                            'level': (lows[i] + lows[j]) / 2,
                            'liquidity_type': 'sell_side'
                        }
                    ))

        # Detect liquidity sweeps (wick through level then reversal)
        for i in range(5, len(data) - 1):
            recent_high = max(highs[i - 5:i])
            recent_low = min(lows[i - 5:i])

            # Sweep high then close below
            if highs[i] > recent_high and closes[i] < recent_high:
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.LIQUIDITY_SWEEP_HIGH,
                    start_index=i,
                    end_index=i,
                    price_high=highs[i],
                    price_low=recent_high,
                    confidence=0.8,
                    timestamp=data.index[i] if hasattr(data.index[i], 'isoformat') else datetime.now(),
                    details={
                        'sweep_level': recent_high,
                        'wick_size': highs[i] - recent_high
                    }
                ))

            # Sweep low then close above
            if lows[i] < recent_low and closes[i] > recent_low:
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.LIQUIDITY_SWEEP_LOW,
                    start_index=i,
                    end_index=i,
                    price_high=recent_low,
                    price_low=lows[i],
                    confidence=0.8,
                    timestamp=data.index[i] if hasattr(data.index[i], 'isoformat') else datetime.now(),
                    details={
                        'sweep_level': recent_low,
                        'wick_size': recent_low - lows[i]
                    }
                ))

        return patterns[-10:]

    def detect_entry_patterns(self, data: pd.DataFrame) -> List[DetectedPattern]:
        """
        Detect ICT entry patterns: OTE, Silver Bullet, Judas Swing
        """
        patterns = []
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values

        # Find swing range for OTE calculation
        for i in range(20, len(data) - 5):
            range_high = max(highs[i - 20:i])
            range_low = min(lows[i - 20:i])
            range_size = range_high - range_low

            if range_size == 0:
                continue

            # OTE zone: 61.8% - 78.6% retracement
            ote_high = range_high - (range_size * 0.618)
            ote_low = range_high - (range_size * 0.786)

            current_price = closes[i]

            # Check if price is in OTE zone
            if ote_low <= current_price <= ote_high:
                # Bullish OTE - price in discount OTE zone with bullish bias
                if current_price < (range_high + range_low) / 2:
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.OPTIMAL_TRADE_ENTRY,
                        start_index=i,
                        end_index=i,
                        price_high=ote_high,
                        price_low=ote_low,
                        confidence=0.75,
                        timestamp=data.index[i] if hasattr(data.index[i], 'isoformat') else datetime.now(),
                        details={
                            'range_high': range_high,
                            'range_low': range_low,
                            'ote_zone': 'bullish',
                            'fib_level': 0.705  # Middle of 61.8-78.6
                        }
                    ))

        return patterns[-5:]

    def detect_power_of_three(self, data: pd.DataFrame) -> List[DetectedPattern]:
        """
        Detect Power of Three (AMD) phases:
        - Accumulation: Low volatility consolidation
        - Manipulation: False breakout / stop hunt
        - Distribution: True move in intended direction
        """
        patterns = []

        if len(data) < 30:
            return patterns

        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values

        # Calculate volatility (ATR-like)
        ranges = highs - lows
        avg_range = np.mean(ranges[-20:])

        # Look for AMD pattern in last 30 candles
        for i in range(10, len(data) - 10):
            # Accumulation: Low range period (5+ candles)
            local_ranges = ranges[i - 5:i]
            is_accumulation = all(r < avg_range * 0.7 for r in local_ranges)

            if is_accumulation:
                # Look for manipulation (spike) followed by distribution
                manip_idx = i
                spike_up = highs[manip_idx] > max(highs[i - 5:i])
                spike_down = lows[manip_idx] < min(lows[i - 5:i])

                if spike_up or spike_down:
                    # Check for reversal (distribution)
                    if manip_idx + 3 < len(data):
                        if spike_up and closes[manip_idx + 3] < closes[manip_idx]:
                            patterns.append(DetectedPattern(
                                pattern_type=PatternType.MANIPULATION,
                                start_index=i - 5,
                                end_index=manip_idx + 3,
                                price_high=highs[manip_idx],
                                price_low=min(lows[i - 5:i]),
                                confidence=0.7,
                                timestamp=data.index[manip_idx] if hasattr(data.index[manip_idx], 'isoformat') else datetime.now(),
                                details={
                                    'phase': 'AMD_bearish',
                                    'manipulation_high': highs[manip_idx],
                                    'distribution_direction': 'bearish'
                                }
                            ))

                        elif spike_down and closes[manip_idx + 3] > closes[manip_idx]:
                            patterns.append(DetectedPattern(
                                pattern_type=PatternType.MANIPULATION,
                                start_index=i - 5,
                                end_index=manip_idx + 3,
                                price_high=max(highs[i - 5:i]),
                                price_low=lows[manip_idx],
                                confidence=0.7,
                                timestamp=data.index[manip_idx] if hasattr(data.index[manip_idx], 'isoformat') else datetime.now(),
                                details={
                                    'phase': 'AMD_bullish',
                                    'manipulation_low': lows[manip_idx],
                                    'distribution_direction': 'bullish'
                                }
                            ))

        return patterns[-3:]

    def get_pattern_summary(self, patterns: List[DetectedPattern]) -> Dict:
        """
        Get summary of detected patterns
        """
        if not patterns:
            return {
                'total_patterns': 0,
                'bullish_signals': 0,
                'bearish_signals': 0,
                'bias': 'neutral',
                'confidence': 0
            }

        bullish_patterns = [
            PatternType.BULLISH_ORDER_BLOCK,
            PatternType.BULLISH_FVG,
            PatternType.BREAK_OF_STRUCTURE_BULLISH,
            PatternType.CHANGE_OF_CHARACTER_BULLISH,
            PatternType.LIQUIDITY_SWEEP_LOW,
            PatternType.OPTIMAL_TRADE_ENTRY
        ]

        bearish_patterns = [
            PatternType.BEARISH_ORDER_BLOCK,
            PatternType.BEARISH_FVG,
            PatternType.BREAK_OF_STRUCTURE_BEARISH,
            PatternType.CHANGE_OF_CHARACTER_BEARISH,
            PatternType.LIQUIDITY_SWEEP_HIGH
        ]

        bullish_count = sum(1 for p in patterns if p.pattern_type in bullish_patterns)
        bearish_count = sum(1 for p in patterns if p.pattern_type in bearish_patterns)

        avg_confidence = np.mean([p.confidence for p in patterns])

        if bullish_count > bearish_count:
            bias = 'bullish'
        elif bearish_count > bullish_count:
            bias = 'bearish'
        else:
            bias = 'neutral'

        return {
            'total_patterns': len(patterns),
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'bias': bias,
            'confidence': round(avg_confidence, 2),
            'pattern_types': [p.pattern_type.value for p in patterns]
        }


# Singleton instance
pattern_recognizer = ICTPatternRecognizer()
