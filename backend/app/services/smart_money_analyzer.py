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
    mitigated_order_blocks: List[OrderBlock] = field(default_factory=list)
    # New ICT pattern results (from Audio-First Training)
    displacements: List[Dict] = field(default_factory=list)
    ote_zones: List[Dict] = field(default_factory=list)
    breaker_blocks: List[Dict] = field(default_factory=list)
    buy_sell_stops: Dict = field(default_factory=dict)
    kill_zone_active: bool = False
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

    def __init__(self, lookback_swing: int = 5, use_ml: bool = True, params: Optional[Dict] = None,
                 ml_engine: Optional[MLPatternEngine] = None):
        """
        Initialize the Smart Money Analyzer

        Args:
            lookback_swing: Number of candles to look back for swing detection
            use_ml: Whether to use ML knowledge (True) or fallback to basic (False)
            params: Optional parameter overrides from Tier 2 optimizer
            ml_engine: Optional pre-configured MLPatternEngine (for playlist isolation).
                       If provided, uses this engine instead of the global singleton.
        """
        self.params = params or {}
        self.lookback_swing = int(self.params.get('swing_lookback', lookback_swing))
        self.use_ml = use_ml
        self.ml_engine: Optional[MLPatternEngine] = None

        if ml_engine is not None:
            # Use injected engine (playlist-isolated)
            self.ml_engine = ml_engine
            learned = self.ml_engine.get_learned_patterns()
            logger.info(f"SmartMoneyAnalyzer initialized with injected ML engine: {learned}")
        elif use_ml:
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
        mitigated_obs = []
        if self._can_detect('order_block'):
            order_blocks, mitigated_obs = self.find_order_blocks(data, structure)
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

        # Initialize new pattern containers
        displacements = []
        ote_zones = []
        breaker_blocks = []
        buy_sell_stops = {}
        kill_zone_active = False

        # Step 5: Detect displacement - ONLY IF ML LEARNED
        if self._can_detect('displacement'):
            displacements = self.find_displacement(data)
            if displacements:
                ml_patterns_used.append('displacement')
                ml_confidence_scores['displacement'] = self._get_ml_confidence('displacement')
        else:
            ml_patterns_not_learned.append('displacement')

        # Step 6: Detect Fibonacci OTE zone - ONLY IF ML LEARNED
        if self._can_detect('optimal_trade_entry') or self._can_detect('fibonacci_ote'):
            ote_zones = self.find_ote_zone(data, swing_points)
            if ote_zones:
                if self._can_detect('optimal_trade_entry'):
                    ml_patterns_used.append('optimal_trade_entry')
                    ml_confidence_scores['optimal_trade_entry'] = self._get_ml_confidence('optimal_trade_entry')
                if self._can_detect('fibonacci_ote'):
                    ml_patterns_used.append('fibonacci_ote')
                    ml_confidence_scores['fibonacci_ote'] = self._get_ml_confidence('fibonacci_ote')
        else:
            if not self._can_detect('optimal_trade_entry'):
                ml_patterns_not_learned.append('optimal_trade_entry')
            if not self._can_detect('fibonacci_ote'):
                ml_patterns_not_learned.append('fibonacci_ote')

        # Step 7: Detect buy/sell stops - ONLY IF ML LEARNED
        if self._can_detect('buy_stops') or self._can_detect('sell_stops'):
            buy_sell_stops = self.find_buy_sell_stops(swing_points, data)
            if buy_sell_stops.get('buy_stops'):
                ml_patterns_used.append('buy_stops')
                ml_confidence_scores['buy_stops'] = self._get_ml_confidence('buy_stops')
            if buy_sell_stops.get('sell_stops'):
                ml_patterns_used.append('sell_stops')
                ml_confidence_scores['sell_stops'] = self._get_ml_confidence('sell_stops')
        else:
            if not self._can_detect('buy_stops'):
                ml_patterns_not_learned.append('buy_stops')
            if not self._can_detect('sell_stops'):
                ml_patterns_not_learned.append('sell_stops')

        # Step 8: Detect breaker blocks - ONLY IF ML LEARNED
        if self._can_detect('breaker_block'):
            breaker_blocks = self.find_breaker_blocks(data, order_blocks)
            if breaker_blocks:
                ml_patterns_used.append('breaker_block')
                ml_confidence_scores['breaker_block'] = self._get_ml_confidence('breaker_block')
        else:
            ml_patterns_not_learned.append('breaker_block')

        # Step 9: Kill zone detection - ONLY IF ML LEARNED
        if self._can_detect('kill_zone'):
            kill_zone_active = self._is_kill_zone_active()
            if kill_zone_active:
                ml_patterns_used.append('kill_zone')
                ml_confidence_scores['kill_zone'] = self._get_ml_confidence('kill_zone')
        else:
            ml_patterns_not_learned.append('kill_zone')

        # Step 10: Equal highs/lows - ONLY IF ML LEARNED
        if self._can_detect('equal_highs_lows'):
            ml_patterns_used.append('equal_highs_lows')
            ml_confidence_scores['equal_highs_lows'] = self._get_ml_confidence('equal_highs_lows')
        else:
            ml_patterns_not_learned.append('equal_highs_lows')

        # Step 10a: Inducement detection - ONLY IF ML LEARNED
        # Inducement = first valid pullback on left side of swing high/low
        if self._can_detect('inducement'):
            ml_patterns_used.append('inducement')
            ml_confidence_scores['inducement'] = self._get_ml_confidence('inducement')
        else:
            ml_patterns_not_learned.append('inducement')

        # Step 10b: Smart Money Trap detection - ONLY IF ML LEARNED
        # False breakout + quick reversal = retail trap
        if self._can_detect('smart_money_trap'):
            ml_patterns_used.append('smart_money_trap')
            ml_confidence_scores['smart_money_trap'] = self._get_ml_confidence('smart_money_trap')
        else:
            ml_patterns_not_learned.append('smart_money_trap')

        # Step 10c: Premium/Discount zone detection (ML-enhanced) - ONLY IF ML LEARNED
        if self._can_detect('premium_discount'):
            ml_patterns_used.append('premium_discount')
            ml_confidence_scores['premium_discount'] = self._get_ml_confidence('premium_discount')
        else:
            ml_patterns_not_learned.append('premium_discount')

        # Step 10d: Valid Pullback detection - ONLY IF ML LEARNED
        # Pullback with liquidity sweep confirmation
        if self._can_detect('valid_pullback'):
            ml_patterns_used.append('valid_pullback')
            ml_confidence_scores['valid_pullback'] = self._get_ml_confidence('valid_pullback')
        else:
            ml_patterns_not_learned.append('valid_pullback')

        # Step 10e: Break of Structure / CHoCH - ONLY IF ML LEARNED
        if self._can_detect('break_of_structure'):
            ml_patterns_used.append('break_of_structure')
            ml_confidence_scores['break_of_structure'] = self._get_ml_confidence('break_of_structure')
        else:
            ml_patterns_not_learned.append('break_of_structure')

        if self._can_detect('change_of_character'):
            ml_patterns_used.append('change_of_character')
            ml_confidence_scores['change_of_character'] = self._get_ml_confidence('change_of_character')
        else:
            ml_patterns_not_learned.append('change_of_character')

        # Step 10f: Liquidity Sweep detection - ONLY IF ML LEARNED
        if self._can_detect('liquidity_sweep'):
            ml_patterns_used.append('liquidity_sweep')
            ml_confidence_scores['liquidity_sweep'] = self._get_ml_confidence('liquidity_sweep')
        else:
            ml_patterns_not_learned.append('liquidity_sweep')

        # Step 11: Map liquidity levels (basic analysis - swing-based)
        liquidity = self.find_liquidity_levels(swing_points, data)

        # Step 12: Calculate premium/discount (basic analysis)
        premium_discount = self.calculate_premium_discount(data, swing_points)

        # Step 13: Determine overall bias
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
            mitigated_order_blocks=mitigated_obs,
            fair_value_gaps=fvgs,
            liquidity_levels=liquidity,
            premium_discount=premium_discount,
            bias=bias,
            bias_confidence=confidence,
            bias_reasoning=reasoning,
            current_price=float(data['close'].iloc[-1]),
            # New ICT patterns from Audio-First Training
            displacements=displacements,
            ote_zones=ote_zones,
            breaker_blocks=breaker_blocks,
            buy_sell_stops=buy_sell_stops,
            kill_zone_active=kill_zone_active,
            # ML tracking
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

        # Emit HH/HL/LH/LL labels as structure events for chart visualization
        if hh:
            events.append(StructureEvent(type='higher_high', level=recent_highs[-1].price,
                                          timestamp=recent_highs[-1].timestamp, description='Higher High'))
        if hl:
            events.append(StructureEvent(type='higher_low', level=recent_lows[-1].price,
                                          timestamp=recent_lows[-1].timestamp, description='Higher Low'))
        if lh:
            events.append(StructureEvent(type='lower_high', level=recent_highs[-1].price,
                                          timestamp=recent_highs[-1].timestamp, description='Lower High'))
        if ll:
            events.append(StructureEvent(type='lower_low', level=recent_lows[-1].price,
                                          timestamp=recent_lows[-1].timestamp, description='Lower Low'))

        # Also emit previous swing point labels for fuller structure visualization
        if len(recent_highs) >= 3 and len(recent_lows) >= 3:
            prev_hh = recent_highs[-2].price > recent_highs[-3].price
            prev_hl = recent_lows[-2].price > recent_lows[-3].price
            prev_lh = recent_highs[-2].price < recent_highs[-3].price
            prev_ll = recent_lows[-2].price < recent_lows[-3].price
            if prev_hh:
                events.append(StructureEvent(type='higher_high', level=recent_highs[-2].price,
                                              timestamp=recent_highs[-2].timestamp, description='Higher High'))
            if prev_hl:
                events.append(StructureEvent(type='higher_low', level=recent_lows[-2].price,
                                              timestamp=recent_lows[-2].timestamp, description='Higher Low'))
            if prev_lh:
                events.append(StructureEvent(type='lower_high', level=recent_highs[-2].price,
                                              timestamp=recent_highs[-2].timestamp, description='Lower High'))
            if prev_ll:
                events.append(StructureEvent(type='lower_low', level=recent_lows[-2].price,
                                              timestamp=recent_lows[-2].timestamp, description='Lower Low'))

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

        # Get ML-learned parameters (with Tier 2 override support)
        ml_params = {}
        if self.ml_engine:
            ml_params = self.ml_engine.get_detection_parameters('order_block')
            confidence_multiplier = ml_params.get('confidence_multiplier', 0.5)
            min_move_strength = ml_params.get('min_move_strength', 0.3)
        else:
            confidence_multiplier = 0.5
            min_move_strength = 0.3
        # Tier 2 optimizer overrides
        confidence_multiplier = self.params.get('ob_confidence_multiplier', confidence_multiplier)
        min_move_strength = self.params.get('ob_min_move_strength', min_move_strength)

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

        # Return unmitigated + mitigated order blocks separately
        all_sorted = sorted(order_blocks, key=lambda x: x.start_index, reverse=True)
        unmitigated = [ob for ob in all_sorted if not ob.mitigated][:10]
        mitigated = [ob for ob in all_sorted if ob.mitigated][:5]
        return unmitigated, mitigated

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

        # Get ML-learned parameters (with Tier 2 override support)
        ml_params = {}
        if self.ml_engine:
            ml_params = self.ml_engine.get_detection_parameters('fvg')
            min_gap_size_pct = ml_params.get('min_gap_size_pct', 0.0001)
            confidence_multiplier = ml_params.get('confidence_multiplier', 0.7)
        else:
            min_gap_size_pct = 0.0001
            confidence_multiplier = 0.7
        # Tier 2 optimizer overrides
        min_gap_size_pct = self.params.get('fvg_min_gap_pct', min_gap_size_pct)

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

                # Check if filled by any subsequent candle trading into the gap
                filled = False
                fill_pct = 0.0
                for j in range(i + 1, len(data)):
                    sub_low = float(data['low'].iloc[j])
                    if sub_low <= gap_low:
                        filled = True
                        fill_pct = 1.0
                        break
                    elif sub_low < gap_high:
                        fill_pct = max(fill_pct, (gap_high - sub_low) / (gap_high - gap_low))
                if fill_pct >= 0.8:
                    filled = True

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

                # Check if filled by any subsequent candle trading into the gap
                filled = False
                fill_pct = 0.0
                for j in range(i + 1, len(data)):
                    sub_high = float(data['high'].iloc[j])
                    if sub_high >= gap_high:
                        filled = True
                        fill_pct = 1.0
                        break
                    elif sub_high > gap_low:
                        fill_pct = max(fill_pct, (sub_high - gap_low) / (gap_high - gap_low))
                if fill_pct >= 0.8:
                    filled = True

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
        tolerance: float = None,
    ) -> List[Dict]:
        if tolerance is None:
            tolerance = self.params.get('equal_level_tolerance', 0.001)
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

    # =========================================================================
    # NEW PATTERN DETECTION METHODS (from Audio-First Training)
    # =========================================================================

    def find_displacement(self, data: 'pd.DataFrame') -> List[Dict]:
        """
        Find displacement candles (strong institutional moves).

        Displacement = large body candle with small wicks, indicating
        strong directional intent. Body > 70% of total range.
        """
        displacements = []
        if len(data) < 3:
            return displacements

        for i in range(max(0, len(data) - 20), len(data)):
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]
            open_p = data['open'].iloc[i]
            close = data['close'].iloc[i]

            total_range = high - low
            if total_range == 0:
                continue

            body = abs(close - open_p)
            body_ratio = body / total_range

            # Displacement: body > threshold of range AND significant size
            disp_body_threshold = self.params.get('displacement_body_ratio', 0.70)
            disp_range_mult = self.params.get('displacement_range_mult', 1.2)
            if body_ratio >= disp_body_threshold:
                # Check if it's a significant candle (above average range)
                avg_range = (data['high'] - data['low']).tail(20).mean()
                if total_range > avg_range * disp_range_mult:
                    direction = 'bullish' if close > open_p else 'bearish'
                    displacements.append({
                        'index': i,
                        'type': direction,
                        'high': float(high),
                        'low': float(low),
                        'body_ratio': float(body_ratio),
                        'range_vs_avg': float(total_range / avg_range),
                        'timestamp': data.index[i] if hasattr(data.index, '__getitem__') else None,
                    })

        return displacements[-5:]  # Return last 5

    def find_ote_zone(
        self,
        data: 'pd.DataFrame',
        swing_points: List[SwingPoint]
    ) -> List[Dict]:
        """
        Find Optimal Trade Entry zones (62-79% Fibonacci retracement).

        ICT teaches: The OTE is the sweet spot between the 62% and 79%
        retracement levels of a significant swing.
        """
        ote_zones = []

        recent_highs = [sp for sp in swing_points if sp.type == 'high'][-3:]
        recent_lows = [sp for sp in swing_points if sp.type == 'low'][-3:]

        if not recent_highs or not recent_lows:
            return ote_zones

        # Find the most recent significant swing
        for i in range(len(recent_highs)):
            for j in range(len(recent_lows)):
                sh = recent_highs[-(i+1)]
                sl = recent_lows[-(j+1)]

                swing_range = abs(sh.price - sl.price)
                if swing_range == 0:
                    continue

                # Calculate OTE zone (Fibonacci retracement levels)
                fib_low = self.params.get('ote_fib_low', 0.62)
                fib_high = self.params.get('ote_fib_high', 0.79)
                if sh.index > sl.index:
                    # Upswing: retracement goes down
                    ote_high = sh.price - (swing_range * fib_low)
                    ote_low = sh.price - (swing_range * fib_high)
                    direction = 'bullish'  # Buy in the OTE of an upswing
                else:
                    # Downswing: retracement goes up
                    ote_low = sl.price + (swing_range * fib_low)
                    ote_high = sl.price + (swing_range * fib_high)
                    direction = 'bearish'  # Sell in the OTE of a downswing

                current_price = float(data['close'].iloc[-1])

                ote_zones.append({
                    'type': direction,
                    'ote_high': float(ote_high),
                    'ote_low': float(ote_low),
                    'swing_high': float(sh.price),
                    'swing_low': float(sl.price),
                    'fib_62': float(sh.price - swing_range * fib_low) if sh.index > sl.index else float(sl.price + swing_range * fib_low),
                    'fib_79': float(sh.price - swing_range * fib_high) if sh.index > sl.index else float(sl.price + swing_range * fib_high),
                    'price_in_ote': ote_low <= current_price <= ote_high,
                })

                if ote_zones:
                    return ote_zones[:2]  # Return top 2

        return ote_zones

    def find_buy_sell_stops(
        self,
        swing_points: List[SwingPoint],
        data: 'pd.DataFrame'
    ) -> Dict[str, List[Dict]]:
        """
        Find buy stops (above equal/clustered highs) and sell stops
        (below equal/clustered lows).

        ICT teaches: Smart money targets these liquidity pools.
        """
        current_price = float(data['close'].iloc[-1])
        tolerance = 0.001  # 0.1% tolerance for "equal" levels

        buy_stops = []
        sell_stops = []

        # Find equal highs (buy stops above)
        highs = [sp for sp in swing_points if sp.type == 'high']
        for i, sp1 in enumerate(highs):
            for sp2 in highs[i + 1:]:
                if abs(sp1.price - sp2.price) / sp1.price < tolerance:
                    level = (sp1.price + sp2.price) / 2
                    if level > current_price:
                        buy_stops.append({
                            'level': float(level),
                            'type': 'equal_highs',
                            'count': 2,
                            'distance_pct': float((level - current_price) / current_price * 100),
                        })

        # Find equal lows (sell stops below)
        lows = [sp for sp in swing_points if sp.type == 'low']
        for i, sp1 in enumerate(lows):
            for sp2 in lows[i + 1:]:
                if abs(sp1.price - sp2.price) / sp1.price < tolerance:
                    level = (sp1.price + sp2.price) / 2
                    if level < current_price:
                        sell_stops.append({
                            'level': float(level),
                            'type': 'equal_lows',
                            'count': 2,
                            'distance_pct': float((current_price - level) / current_price * 100),
                        })

        return {
            'buy_stops': sorted(buy_stops, key=lambda x: x['level'])[:5],
            'sell_stops': sorted(sell_stops, key=lambda x: x['level'], reverse=True)[:5],
        }

    def find_breaker_blocks(
        self,
        data: 'pd.DataFrame',
        order_blocks: List[OrderBlock]
    ) -> List[Dict]:
        """
        Find breaker blocks (mitigated order blocks that become support/resistance).

        ICT teaches: When an order block fails (gets mitigated), it becomes
        a breaker that acts as the opposite (support becomes resistance, etc.)
        """
        breakers = []
        current_price = float(data['close'].iloc[-1])

        # Check all candles for failed OB patterns
        for i in range(3, len(data) - 2):
            open_p = data['open'].iloc[i]
            close = data['close'].iloc[i]
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]

            is_bearish = close < open_p
            is_bullish = close > open_p

            # Look for bullish candle that gets broken below (becomes bearish breaker)
            if is_bullish:
                # Check if subsequent candles broke below this candle's low
                for j in range(i + 1, min(i + 10, len(data))):
                    if data['close'].iloc[j] < low:
                        # This is a failed bullish OB → bearish breaker
                        # Check if price has come back to test it
                        if current_price <= high and current_price >= low:
                            breakers.append({
                                'index': i,
                                'type': 'bearish_breaker',
                                'high': float(high),
                                'low': float(low),
                                'being_tested': True,
                            })
                        break

            # Look for bearish candle that gets broken above (becomes bullish breaker)
            if is_bearish:
                for j in range(i + 1, min(i + 10, len(data))):
                    if data['close'].iloc[j] > high:
                        if current_price >= low and current_price <= high:
                            breakers.append({
                                'index': i,
                                'type': 'bullish_breaker',
                                'high': float(high),
                                'low': float(low),
                                'being_tested': True,
                            })
                        break

        return breakers[-5:]  # Return last 5

    def _is_kill_zone_active(self) -> bool:
        """
        Check if current time is in an ICT Kill Zone.

        London Kill Zone: 2:00-5:00 AM EST (7:00-10:00 UTC)
        NY Kill Zone: 7:00-10:00 AM EST (12:00-15:00 UTC)
        Asian Kill Zone: 8:00 PM - 12:00 AM EST (1:00-5:00 UTC)
        """
        now = datetime.utcnow()
        hour = now.hour

        kill_zones = [
            (1, 5),    # Asian
            (7, 10),   # London
            (12, 15),  # New York
        ]

        for start, end in kill_zones:
            if start <= hour < end:
                return True

        return False
