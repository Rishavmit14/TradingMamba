"""
Smart Money Analysis Engine - ML-POWERED

Core implementation of Smart Money (Inner Circle Trader) methodology for market analysis.
This module uses the ML's LEARNED KNOWLEDGE to identify patterns.

IMPORTANT: Patterns are ONLY detected if the ML has learned them from training videos.
If ML hasn't learned a pattern type, it will NOT be detected.

This module analyzes price data to identify:
- Market Structure (BOS, CHoCH, swing points)
- Order Blocks
- Fair Value Gaps
- Liquidity Levels
- Premium/Discount Zones
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# Import ML Pattern Engine
from ..ml.ml_pattern_engine import get_ml_engine, MLPatternEngine

logger = logging.getLogger(__name__)


class Bias(Enum):
    """Market bias direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class MarketStructure(Enum):
    """Market structure state"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    CONSOLIDATION = "consolidation"


@dataclass
class SwingPoint:
    """A swing high or low point"""
    index: int
    price: float
    type: str  # 'high' or 'low'
    timestamp: Optional[datetime] = None
    strength: int = 1  # How many candles confirm this swing


@dataclass
class OrderBlock:
    """An Smart Money Order Block"""
    start_index: int
    end_index: int
    high: float
    low: float
    type: str  # 'bullish' or 'bearish'
    mitigated: bool = False
    timestamp: Optional[datetime] = None
    strength: float = 0.0  # Based on move after OB


@dataclass
class FairValueGap:
    """A Fair Value Gap (Imbalance)"""
    index: int
    high: float
    low: float
    type: str  # 'bullish' or 'bearish'
    filled: bool = False
    fill_percentage: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class LiquidityLevel:
    """A liquidity level (stop loss cluster)"""
    price: float
    type: str  # 'buy_side' or 'sell_side'
    strength: float = 0.0  # Based on number of swing points
    timestamp: Optional[datetime] = None
    swept: bool = False


@dataclass
class StructureEvent:
    """A market structure event (BOS or CHoCH)"""
    type: str  # 'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish'
    level: float
    timestamp: Optional[datetime] = None
    description: str = ""


@dataclass
class SmartMoneyAnalysisResult:
    """Complete Smart Money analysis result"""
    swing_points: List[SwingPoint] = field(default_factory=list)
    market_structure: MarketStructure = MarketStructure.CONSOLIDATION
    structure_events: List[StructureEvent] = field(default_factory=list)
    order_blocks: List[OrderBlock] = field(default_factory=list)
    fair_value_gaps: List[FairValueGap] = field(default_factory=list)
    liquidity_levels: Dict[str, List[LiquidityLevel]] = field(default_factory=dict)
    premium_discount: Dict = field(default_factory=dict)
    bias: Bias = Bias.NEUTRAL
    bias_confidence: float = 0.0
    bias_reasoning: str = ""
    current_price: float = 0.0
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    # ML Knowledge tracking
    ml_patterns_used: List[str] = field(default_factory=list)  # Patterns ML detected
    ml_patterns_not_learned: List[str] = field(default_factory=list)  # Patterns ML can't detect yet
    ml_confidence_scores: Dict[str, float] = field(default_factory=dict)  # Confidence per pattern type


class SmartMoneyAnalyzer:
    """
    ML-Powered Smart Money methodology analysis engine

    IMPORTANT: This analyzer uses ONLY patterns that the ML has learned from training.
    If the ML hasn't been trained on a pattern type, it will NOT be detected.

    Implements the key Smart Money concepts for market analysis:
    - Swing point identification
    - Market structure analysis (BOS/CHoCH)
    - Order block detection (if ML learned)
    - Fair value gap identification (if ML learned)
    - Liquidity mapping
    - Premium/Discount zone calculation
    """

    def __init__(self, lookback_swing: int = 5, use_ml: bool = True):
        """
        Initialize the Smart Money Analyzer

        Args:
            lookback_swing: Number of candles to look back for swing detection
            use_ml: Whether to use ML knowledge (True) or fallback to basic (False)
        """
        self.lookback_swing = lookback_swing
        self.use_ml = use_ml
        self.ml_engine: Optional[MLPatternEngine] = None

        if use_ml:
            try:
                self.ml_engine = get_ml_engine()
                learned = self.ml_engine.get_learned_patterns()
                if learned:
                    logger.info(f"SmartMoneyAnalyzer initialized with ML knowledge: {learned}")
                else:
                    logger.warning("ML Engine loaded but has NO learned patterns!")
            except Exception as e:
                logger.error(f"Failed to load ML engine: {e}")
                self.ml_engine = None

    def analyze(self, data: 'pd.DataFrame') -> SmartMoneyAnalysisResult:
        """
        Run complete Smart Money analysis on OHLCV data using ML knowledge.

        IMPORTANT: Only patterns the ML has learned will be detected.
        Patterns not learned will be empty and flagged in ml_patterns_not_learned.

        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                  Index should be datetime

        Returns:
            SmartMoneyAnalysisResult with all analysis components
        """
        if pd is None:
            raise ImportError("pandas is required for Smart Money analysis")

        if len(data) < self.lookback_swing * 2:
            logger.warning("Insufficient data for analysis")
            return SmartMoneyAnalysisResult()

        # Track ML knowledge usage
        ml_patterns_used = []
        ml_patterns_not_learned = []
        ml_confidence_scores = {}

        # Step 1: Find swing points (basic analysis, always available)
        swing_points = self.find_swing_points(data)

        # Step 2: Analyze market structure (basic analysis)
        structure, events = self.analyze_market_structure(swing_points)

        # Step 3: Find order blocks - ONLY IF ML LEARNED
        order_blocks = []
        if self._can_detect('order_block'):
            order_blocks = self.find_order_blocks(data, structure)
            if order_blocks:
                ml_patterns_used.append('order_block')
                ml_confidence_scores['order_block'] = self._get_ml_confidence('order_block')
        else:
            ml_patterns_not_learned.append('order_block')
            logger.info("Order Blocks NOT detected - ML hasn't learned this pattern yet")

        # Step 4: Find fair value gaps - ONLY IF ML LEARNED
        fvgs = []
        if self._can_detect('fvg'):
            fvgs = self.find_fair_value_gaps(data)
            if fvgs:
                ml_patterns_used.append('fvg')
                ml_confidence_scores['fvg'] = self._get_ml_confidence('fvg')
        else:
            ml_patterns_not_learned.append('fvg')
            logger.info("FVGs NOT detected - ML hasn't learned this pattern yet")

        # Step 5: Map liquidity levels (basic analysis - swing-based)
        liquidity = self.find_liquidity_levels(swing_points, data)

        # Step 6: Calculate premium/discount (basic analysis)
        premium_discount = self.calculate_premium_discount(data, swing_points)

        # Step 7: Determine overall bias
        # Adjust confidence based on ML knowledge
        bias, confidence, reasoning = self.determine_bias(
            structure, premium_discount, events
        )

        # Adjust bias confidence based on ML pattern detection
        if ml_patterns_used:
            # Boost confidence if ML detected patterns
            avg_ml_confidence = sum(ml_confidence_scores.values()) / len(ml_confidence_scores)
            confidence = min(confidence + (avg_ml_confidence * 0.2), 1.0)
            reasoning += f" [ML detected: {', '.join(ml_patterns_used)}]"
        elif ml_patterns_not_learned:
            # Lower confidence if key patterns couldn't be detected
            confidence = confidence * 0.7
            reasoning += f" [ML needs training on: {', '.join(ml_patterns_not_learned)}]"

        return SmartMoneyAnalysisResult(
            swing_points=swing_points,
            market_structure=structure,
            structure_events=events,
            order_blocks=order_blocks,
            fair_value_gaps=fvgs,
            liquidity_levels=liquidity,
            premium_discount=premium_discount,
            bias=bias,
            bias_confidence=confidence,
            bias_reasoning=reasoning,
            current_price=float(data['close'].iloc[-1]),
            ml_patterns_used=ml_patterns_used,
            ml_patterns_not_learned=ml_patterns_not_learned,
            ml_confidence_scores=ml_confidence_scores,
        )

    def _can_detect(self, pattern_type: str) -> bool:
        """Check if ML can detect a pattern type"""
        if not self.use_ml or not self.ml_engine:
            # Fallback to basic detection if ML not available
            return True
        return self.ml_engine.can_detect_pattern(pattern_type)

    def _get_ml_confidence(self, pattern_type: str) -> float:
        """Get ML's confidence for a pattern type"""
        if not self.ml_engine:
            return 0.5  # Default confidence
        return self.ml_engine.get_pattern_confidence(pattern_type)

    def find_swing_points(self, data: 'pd.DataFrame') -> List[SwingPoint]:
        """
        Identify swing highs and lows

        A swing high is a high that is higher than the surrounding candles.
        A swing low is a low that is lower than the surrounding candles.
        """
        swing_points = []
        highs = data['high'].values
        lows = data['low'].values

        for i in range(self.lookback_swing, len(data) - self.lookback_swing):
            # Get surrounding values
            left_highs = highs[i - self.lookback_swing:i]
            right_highs = highs[i + 1:i + self.lookback_swing + 1]
            left_lows = lows[i - self.lookback_swing:i]
            right_lows = lows[i + 1:i + self.lookback_swing + 1]

            # Check for swing high
            if highs[i] > max(left_highs) and highs[i] > max(right_highs):
                # Calculate strength (how many candles it's higher than)
                strength = sum(1 for h in np.concatenate([left_highs, right_highs]) if highs[i] > h)

                swing_points.append(SwingPoint(
                    index=i,
                    price=float(highs[i]),
                    type='high',
                    timestamp=data.index[i] if hasattr(data.index[i], 'timestamp') else None,
                    strength=strength
                ))

            # Check for swing low
            if lows[i] < min(left_lows) and lows[i] < min(right_lows):
                strength = sum(1 for l in np.concatenate([left_lows, right_lows]) if lows[i] < l)

                swing_points.append(SwingPoint(
                    index=i,
                    price=float(lows[i]),
                    type='low',
                    timestamp=data.index[i] if hasattr(data.index[i], 'timestamp') else None,
                    strength=strength
                ))

        return sorted(swing_points, key=lambda x: x.index)

    def analyze_market_structure(
        self,
        swing_points: List[SwingPoint]
    ) -> Tuple[MarketStructure, List[StructureEvent]]:
        """
        Analyze market structure to determine trend and identify BOS/CHoCH

        BOS (Break of Structure): Continuation pattern
        CHoCH (Change of Character): Reversal pattern
        """
        events = []

        if len(swing_points) < 4:
            return MarketStructure.CONSOLIDATION, events

        # Get recent swing highs and lows
        recent_highs = [sp for sp in swing_points if sp.type == 'high'][-4:]
        recent_lows = [sp for sp in swing_points if sp.type == 'low'][-4:]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return MarketStructure.CONSOLIDATION, events

        # Determine structure based on HH/HL or LH/LL pattern
        hh = recent_highs[-1].price > recent_highs[-2].price  # Higher High
        hl = recent_lows[-1].price > recent_lows[-2].price    # Higher Low
        lh = recent_highs[-1].price < recent_highs[-2].price  # Lower High
        ll = recent_lows[-1].price < recent_lows[-2].price    # Lower Low

        # Bullish structure: HH + HL
        if hh and hl:
            structure = MarketStructure.BULLISH
            events.append(StructureEvent(
                type='bullish_structure',
                level=recent_highs[-1].price,
                timestamp=recent_highs[-1].timestamp,
                description='Higher High and Higher Low confirmed'
            ))

        # Bearish structure: LH + LL
        elif lh and ll:
            structure = MarketStructure.BEARISH
            events.append(StructureEvent(
                type='bearish_structure',
                level=recent_lows[-1].price,
                timestamp=recent_lows[-1].timestamp,
                description='Lower High and Lower Low confirmed'
            ))

        else:
            structure = MarketStructure.CONSOLIDATION

        # Check for BOS
        bos_event = self._check_bos(recent_highs, recent_lows, structure)
        if bos_event:
            events.append(bos_event)

        # Check for CHoCH
        choch_event = self._check_choch(recent_highs, recent_lows, structure)
        if choch_event:
            events.append(choch_event)

        return structure, events

    def _check_bos(
        self,
        recent_highs: List[SwingPoint],
        recent_lows: List[SwingPoint],
        structure: MarketStructure
    ) -> Optional[StructureEvent]:
        """Check for Break of Structure"""
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None

        if structure == MarketStructure.BULLISH:
            # Bullish BOS: Price breaks above previous swing high
            if recent_highs[-1].price > recent_highs[-2].price:
                return StructureEvent(
                    type='bos_bullish',
                    level=recent_highs[-2].price,
                    timestamp=recent_highs[-1].timestamp,
                    description=f'Bullish BOS at {recent_highs[-2].price:.5f}'
                )

        elif structure == MarketStructure.BEARISH:
            # Bearish BOS: Price breaks below previous swing low
            if recent_lows[-1].price < recent_lows[-2].price:
                return StructureEvent(
                    type='bos_bearish',
                    level=recent_lows[-2].price,
                    timestamp=recent_lows[-1].timestamp,
                    description=f'Bearish BOS at {recent_lows[-2].price:.5f}'
                )

        return None

    def _check_choch(
        self,
        recent_highs: List[SwingPoint],
        recent_lows: List[SwingPoint],
        structure: MarketStructure
    ) -> Optional[StructureEvent]:
        """Check for Change of Character"""
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None

        # CHoCH in bullish structure: Price breaks below previous HL
        if structure == MarketStructure.BULLISH:
            if recent_lows[-1].price < recent_lows[-2].price:
                return StructureEvent(
                    type='choch_bearish',
                    level=recent_lows[-2].price,
                    timestamp=recent_lows[-1].timestamp,
                    description='Bearish CHoCH - potential trend reversal'
                )

        # CHoCH in bearish structure: Price breaks above previous LH
        elif structure == MarketStructure.BEARISH:
            if recent_highs[-1].price > recent_highs[-2].price:
                return StructureEvent(
                    type='choch_bullish',
                    level=recent_highs[-2].price,
                    timestamp=recent_highs[-1].timestamp,
                    description='Bullish CHoCH - potential trend reversal'
                )

        return None

    def find_order_blocks(
        self,
        data: 'pd.DataFrame',
        structure: MarketStructure
    ) -> List[OrderBlock]:
        """
        Find Order Blocks using ML-learned parameters.

        The detection sensitivity is based on what ML learned from training videos.
        - Higher frequency in training = stricter detection
        - Teaching contexts inform what characteristics to look for

        Bullish OB: Last bearish candle before a bullish impulse
        Bearish OB: Last bullish candle before a bearish impulse
        """
        order_blocks = []
        current_price = data['close'].iloc[-1]

        # Get ML-learned parameters
        ml_params = {}
        if self.ml_engine:
            ml_params = self.ml_engine.get_detection_parameters('order_block')
            confidence_multiplier = ml_params.get('confidence_multiplier', 0.5)
            min_move_strength = ml_params.get('min_move_strength', 0.3)
        else:
            confidence_multiplier = 0.5
            min_move_strength = 0.3

        for i in range(2, len(data) - 1):
            # Current candle info
            is_bearish = data['close'].iloc[i] < data['open'].iloc[i]
            is_bullish = data['close'].iloc[i] > data['open'].iloc[i]

            # Next candle info
            next_close = data['close'].iloc[i + 1]
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]

            # Bullish Order Block
            # Bearish candle followed by strong bullish move (closes above the high)
            if is_bearish and next_close > current_high:
                # Calculate move strength
                move = (next_close - current_high) / current_high
                strength = min(move * 100, 1.0)

                # Apply ML-learned minimum strength threshold
                if strength < min_move_strength:
                    continue

                # Apply ML confidence to strength
                adjusted_strength = strength * confidence_multiplier

                ob = OrderBlock(
                    start_index=i,
                    end_index=i,
                    high=float(current_high),
                    low=float(current_low),
                    type='bullish',
                    mitigated=current_price < current_low,
                    timestamp=data.index[i] if hasattr(data.index, '__getitem__') else None,
                    strength=adjusted_strength
                )
                order_blocks.append(ob)

            # Bearish Order Block
            # Bullish candle followed by strong bearish move (closes below the low)
            if is_bullish and next_close < current_low:
                move = (current_low - next_close) / current_low
                strength = min(move * 100, 1.0)

                # Apply ML-learned minimum strength threshold
                if strength < min_move_strength:
                    continue

                # Apply ML confidence to strength
                adjusted_strength = strength * confidence_multiplier

                ob = OrderBlock(
                    start_index=i,
                    end_index=i,
                    high=float(current_high),
                    low=float(current_low),
                    type='bearish',
                    mitigated=current_price > current_high,
                    timestamp=data.index[i] if hasattr(data.index, '__getitem__') else None,
                    strength=adjusted_strength
                )
                order_blocks.append(ob)

        # Return only unmitigated order blocks, most recent first
        unmitigated = [ob for ob in order_blocks if not ob.mitigated]
        return sorted(unmitigated, key=lambda x: x.start_index, reverse=True)[:10]

    def find_fair_value_gaps(self, data: 'pd.DataFrame') -> List[FairValueGap]:
        """
        Find Fair Value Gaps (Imbalances) using ML-learned parameters.

        FVG was the most frequently observed pattern in ML training (31 instances),
        so detection uses learned characteristics for high confidence.

        Bullish FVG: Gap between candle 1's high and candle 3's low
        Bearish FVG: Gap between candle 1's low and candle 3's high
        """
        fvgs = []
        current_price = data['close'].iloc[-1]

        # Get ML-learned parameters
        ml_params = {}
        if self.ml_engine:
            ml_params = self.ml_engine.get_detection_parameters('fvg')
            min_gap_size_pct = ml_params.get('min_gap_size_pct', 0.0001)
            confidence_multiplier = ml_params.get('confidence_multiplier', 0.7)
        else:
            min_gap_size_pct = 0.0001
            confidence_multiplier = 0.7

        for i in range(2, len(data)):
            candle1_high = data['high'].iloc[i - 2]
            candle1_low = data['low'].iloc[i - 2]
            candle3_high = data['high'].iloc[i]
            candle3_low = data['low'].iloc[i]

            # Bullish FVG: Candle 3's low is above Candle 1's high
            if candle3_low > candle1_high:
                gap_high = float(candle3_low)
                gap_low = float(candle1_high)

                # ML-learned minimum gap size filter
                gap_size_pct = (gap_high - gap_low) / gap_low
                if gap_size_pct < min_gap_size_pct:
                    continue

                # Check if filled
                filled = current_price < gap_low
                fill_pct = 0.0
                if not filled and current_price < gap_high:
                    fill_pct = (gap_high - current_price) / (gap_high - gap_low)

                fvgs.append(FairValueGap(
                    index=i - 1,
                    high=gap_high,
                    low=gap_low,
                    type='bullish',
                    filled=filled,
                    fill_percentage=fill_pct,
                    timestamp=data.index[i - 1] if hasattr(data.index, '__getitem__') else None
                ))

            # Bearish FVG: Candle 3's high is below Candle 1's low
            if candle3_high < candle1_low:
                gap_high = float(candle1_low)
                gap_low = float(candle3_high)

                # ML-learned minimum gap size filter
                gap_size_pct = (gap_high - gap_low) / gap_high
                if gap_size_pct < min_gap_size_pct:
                    continue

                filled = current_price > gap_high
                fill_pct = 0.0
                if not filled and current_price > gap_low:
                    fill_pct = (current_price - gap_low) / (gap_high - gap_low)

                fvgs.append(FairValueGap(
                    index=i - 1,
                    high=gap_high,
                    low=gap_low,
                    type='bearish',
                    filled=filled,
                    fill_percentage=fill_pct,
                    timestamp=data.index[i - 1] if hasattr(data.index, '__getitem__') else None
                ))

        # Return unfilled FVGs, most recent first
        unfilled = [fvg for fvg in fvgs if not fvg.filled]
        return sorted(unfilled, key=lambda x: x.index, reverse=True)[:10]

    def find_liquidity_levels(
        self,
        swing_points: List[SwingPoint],
        data: 'pd.DataFrame'
    ) -> Dict[str, List[LiquidityLevel]]:
        """
        Map liquidity levels (where stop losses are likely clustered)

        Buy-side liquidity: Above swing highs
        Sell-side liquidity: Below swing lows
        """
        current_price = float(data['close'].iloc[-1])

        # Buy-side liquidity (above current price)
        buy_side = []
        for sp in swing_points:
            if sp.type == 'high' and sp.price > current_price:
                buy_side.append(LiquidityLevel(
                    price=sp.price,
                    type='buy_side',
                    strength=sp.strength / 10,  # Normalize
                    timestamp=sp.timestamp,
                    swept=False
                ))

        # Sell-side liquidity (below current price)
        sell_side = []
        for sp in swing_points:
            if sp.type == 'low' and sp.price < current_price:
                sell_side.append(LiquidityLevel(
                    price=sp.price,
                    type='sell_side',
                    strength=sp.strength / 10,
                    timestamp=sp.timestamp,
                    swept=False
                ))

        # Find equal highs/lows (stronger liquidity)
        equal_highs = self._find_equal_levels(
            [sp for sp in swing_points if sp.type == 'high']
        )
        equal_lows = self._find_equal_levels(
            [sp for sp in swing_points if sp.type == 'low']
        )

        return {
            'buy_side': sorted(buy_side, key=lambda x: x.price)[:5],
            'sell_side': sorted(sell_side, key=lambda x: x.price, reverse=True)[:5],
            'equal_highs': equal_highs,
            'equal_lows': equal_lows
        }

    def _find_equal_levels(
        self,
        swing_points: List[SwingPoint],
        tolerance: float = 0.001
    ) -> List[Dict]:
        """Find equal highs or lows within tolerance"""
        equal_levels = []

        for i, sp1 in enumerate(swing_points):
            for sp2 in swing_points[i + 1:]:
                if abs(sp1.price - sp2.price) / sp1.price < tolerance:
                    equal_levels.append({
                        'level': (sp1.price + sp2.price) / 2,
                        'count': 2,
                        'points': [sp1.index, sp2.index]
                    })

        return equal_levels

    def calculate_premium_discount(
        self,
        data: 'pd.DataFrame',
        swing_points: List[SwingPoint]
    ) -> Dict:
        """
        Calculate premium/discount zones

        Premium: Above 50% of range (look for sells)
        Discount: Below 50% of range (look for buys)
        """
        recent_highs = [sp for sp in swing_points if sp.type == 'high']
        recent_lows = [sp for sp in swing_points if sp.type == 'low']

        if not recent_highs or not recent_lows:
            return {
                'zone': 'neutral',
                'percentage': 50.0,
                'equilibrium': 0.0
            }

        # Use last 5 swing points for range
        range_high = max(sp.price for sp in recent_highs[-5:])
        range_low = min(sp.price for sp in recent_lows[-5:])

        equilibrium = (range_high + range_low) / 2
        current_price = float(data['close'].iloc[-1])

        # Calculate position in range (0-100%)
        range_size = range_high - range_low
        if range_size == 0:
            percentage = 50.0
        else:
            percentage = ((current_price - range_low) / range_size) * 100

        # Determine zone
        if percentage >= 70:
            zone = 'premium'
        elif percentage <= 30:
            zone = 'discount'
        else:
            zone = 'equilibrium'

        return {
            'zone': zone,
            'percentage': round(percentage, 2),
            'range_high': range_high,
            'range_low': range_low,
            'equilibrium': equilibrium,
            'current_price': current_price
        }

    def determine_bias(
        self,
        structure: MarketStructure,
        premium_discount: Dict,
        events: List[StructureEvent]
    ) -> Tuple[Bias, float, str]:
        """
        Determine overall market bias based on structure and position

        Returns:
            Tuple of (bias, confidence, reasoning)
        """
        zone = premium_discount.get('zone', 'neutral')

        # Bullish structure
        if structure == MarketStructure.BULLISH:
            if zone == 'discount':
                return (
                    Bias.BULLISH,
                    0.8,
                    "Bullish structure + price in discount zone - ideal long setup"
                )
            elif zone == 'equilibrium':
                return (
                    Bias.BULLISH,
                    0.6,
                    "Bullish structure + price at equilibrium - wait for discount"
                )
            else:  # premium
                return (
                    Bias.NEUTRAL,
                    0.4,
                    "Bullish structure but price in premium - avoid longs here"
                )

        # Bearish structure
        elif structure == MarketStructure.BEARISH:
            if zone == 'premium':
                return (
                    Bias.BEARISH,
                    0.8,
                    "Bearish structure + price in premium zone - ideal short setup"
                )
            elif zone == 'equilibrium':
                return (
                    Bias.BEARISH,
                    0.6,
                    "Bearish structure + price at equilibrium - wait for premium"
                )
            else:  # discount
                return (
                    Bias.NEUTRAL,
                    0.4,
                    "Bearish structure but price in discount - avoid shorts here"
                )

        # Consolidation
        return (
            Bias.NEUTRAL,
            0.3,
            "Market in consolidation - wait for structure break"
        )
