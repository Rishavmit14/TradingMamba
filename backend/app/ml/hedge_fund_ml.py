"""
Hedge Fund Level ML Enhancements

This module adds the capabilities that separate retail traders from professionals:
1. Pattern Grading System (A+ to F)
2. Historical Validation (backtest patterns)
3. Multi-Timeframe Confluence checking
4. Statistical Edge Tracking (win rate, R:R, expectancy)

All features are 100% FREE using:
- yfinance for historical data
- Local compute for calculations
- No paid APIs required

=============================================================================
THE GENIUS STUDENT PHILOSOPHY
=============================================================================

Average Student (Current State):
- "I see an FVG here" - stores it, done

Genius Student (Target State):
- "I see an FVG here"
- "It's in discount zone - good location"
- "ICT called this type 'beautiful' in training"
- "Historical data shows 78% fill rate"
- "H4 has Order Block confluence"
- "Grade: A - High probability setup"
- "Entry at 50% of FVG, SL below, TP at previous high"

This is the difference between a retail trader and a hedge fund.
=============================================================================
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed. Install with: pip install yfinance")

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas/numpy not installed. Install with: pip install pandas numpy")


class PatternGrade(Enum):
    """Pattern quality grades - ICT methodology"""
    A_PLUS = "A+"  # Perfect setup - institutional, kill zone, HTF confluence
    A = "A"        # Excellent - clear pattern, good location, high probability
    B = "B"        # Good - valid pattern but minor issues
    C = "C"        # Average - pattern exists but weak characteristics
    D = "D"        # Poor - questionable validity, avoid trading
    F = "F"        # Invalid - not a real pattern, misidentified


@dataclass
class GradingCriteria:
    """Criteria for grading a pattern"""
    size_score: float = 0.0          # Size/Magnitude (0-1)
    location_score: float = 0.0      # Premium/Discount zone (0-1)
    structure_score: float = 0.0     # Alignment with market structure (0-1)
    confluence_score: float = 0.0    # Confluence with other patterns (0-1)
    freshness_score: float = 0.0     # Is this a fresh/untested level? (0-1)
    timeframe_score: float = 0.0     # Higher TF = higher score (0-1)
    historical_score: float = 0.0    # Historical success rate (0-1)

    def total_score(self) -> float:
        """Calculate weighted total score"""
        weights = {
            'size': 0.10,
            'location': 0.20,        # Location is key in ICT
            'structure': 0.20,       # Market structure alignment
            'confluence': 0.15,      # Multiple factors
            'freshness': 0.15,       # Fresh levels work better
            'timeframe': 0.10,       # HTF preferred
            'historical': 0.10,      # Backtest results
        }

        return (
            self.size_score * weights['size'] +
            self.location_score * weights['location'] +
            self.structure_score * weights['structure'] +
            self.confluence_score * weights['confluence'] +
            self.freshness_score * weights['freshness'] +
            self.timeframe_score * weights['timeframe'] +
            self.historical_score * weights['historical']
        )

    def to_grade(self) -> PatternGrade:
        """Convert score to letter grade"""
        score = self.total_score()
        if score >= 0.90:
            return PatternGrade.A_PLUS
        elif score >= 0.80:
            return PatternGrade.A
        elif score >= 0.65:
            return PatternGrade.B
        elif score >= 0.50:
            return PatternGrade.C
        elif score >= 0.35:
            return PatternGrade.D
        else:
            return PatternGrade.F


@dataclass
class GradedPattern:
    """A pattern with its quality grade"""
    pattern_type: str
    grade: PatternGrade
    criteria: GradingCriteria
    total_score: float
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    trade_recommendation: str = ""
    invalidation: str = ""
    raw_data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'pattern_type': self.pattern_type,
            'grade': self.grade.value,
            'total_score': round(self.total_score, 3),
            'criteria': asdict(self.criteria),
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'trade_recommendation': self.trade_recommendation,
            'invalidation': self.invalidation,
        }


@dataclass
class PatternValidation:
    """Historical validation result for a pattern"""
    pattern_type: str
    total_tested: int
    fill_rate: float          # % of patterns that filled/respected
    win_rate: float           # % of trades that hit TP before SL
    avg_rr_achieved: float    # Average R:R achieved
    avg_time_to_fill: float   # Hours until pattern played out
    best_entry_type: str      # limit/market/confirmation
    best_timeframe: str       # Which TF works best
    notes: List[str] = field(default_factory=list)


@dataclass
class ConfluenceResult:
    """Multi-timeframe confluence analysis result"""
    pattern_type: str
    primary_timeframe: str
    confluence_score: float  # 0-1
    aligned_timeframes: List[str]
    conflicting_timeframes: List[str]
    confluence_factors: List[str]
    recommendation: str  # STRONG / MODERATE / WEAK / AVOID


@dataclass
class EdgeStatistics:
    """Statistical edge tracking for a pattern type"""
    pattern_type: str
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    pending: int = 0
    win_rate: float = 0.0
    avg_rr: float = 0.0
    expectancy: float = 0.0  # (win_rate * avg_win) - (loss_rate * avg_loss)
    profit_factor: float = 0.0  # gross profit / gross loss
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    best_day: str = ""
    best_session: str = ""
    last_updated: Optional[datetime] = None


class PatternGrader:
    """
    Grade patterns from A+ to F based on multiple criteria.

    This is what separates professionals from amateurs:
    - Not all FVGs are equal
    - Location matters (premium/discount)
    - Confluence matters (multiple factors)
    - Freshness matters (first test vs multiple tests)
    """

    def __init__(self, knowledge_base: Optional[Dict] = None):
        self.knowledge_base = knowledge_base or {}

    def grade_pattern(
        self,
        pattern_type: str,
        pattern_data: Dict,
        market_context: Dict,
        historical_stats: Optional[Dict] = None
    ) -> GradedPattern:
        """
        Grade a single pattern based on ICT methodology.

        Args:
            pattern_type: Type of pattern (fvg, order_block, etc.)
            pattern_data: Raw pattern data (price levels, etc.)
            market_context: Current market context (bias, structure, etc.)
            historical_stats: Historical performance of this pattern type
        """
        criteria = GradingCriteria()
        strengths = []
        weaknesses = []

        # 1. SIZE/MAGNITUDE SCORING
        criteria.size_score = self._score_size(pattern_type, pattern_data)
        if criteria.size_score >= 0.7:
            strengths.append("Large, significant pattern")
        elif criteria.size_score < 0.4:
            weaknesses.append("Small pattern - may not hold")

        # 2. LOCATION SCORING (Premium/Discount Zone)
        criteria.location_score = self._score_location(
            pattern_type, pattern_data, market_context
        )
        if criteria.location_score >= 0.8:
            strengths.append("Optimal location (discount for buys, premium for sells)")
        elif criteria.location_score < 0.4:
            weaknesses.append("Poor location - not in optimal zone")

        # 3. STRUCTURE ALIGNMENT
        criteria.structure_score = self._score_structure(
            pattern_type, pattern_data, market_context
        )
        if criteria.structure_score >= 0.7:
            strengths.append("Aligned with market structure")
        elif criteria.structure_score < 0.4:
            weaknesses.append("Against current market structure")

        # 4. CONFLUENCE SCORING
        criteria.confluence_score = self._score_confluence(
            pattern_type, pattern_data, market_context
        )
        if criteria.confluence_score >= 0.7:
            strengths.append("Multiple confluence factors")
        elif criteria.confluence_score < 0.3:
            weaknesses.append("No confluence - pattern alone")

        # 5. FRESHNESS SCORING
        criteria.freshness_score = self._score_freshness(pattern_data)
        if criteria.freshness_score >= 0.8:
            strengths.append("Fresh, untested level")
        elif criteria.freshness_score < 0.4:
            weaknesses.append("Already tested multiple times")

        # 6. TIMEFRAME SCORING
        criteria.timeframe_score = self._score_timeframe(pattern_data)
        if criteria.timeframe_score >= 0.7:
            strengths.append("Higher timeframe pattern")
        elif criteria.timeframe_score < 0.3:
            weaknesses.append("Low timeframe - more noise")

        # 7. HISTORICAL SCORING
        if historical_stats:
            criteria.historical_score = self._score_historical(
                pattern_type, historical_stats
            )
            if criteria.historical_score >= 0.7:
                strengths.append(f"Strong historical performance ({historical_stats.get('win_rate', 0):.0%} win rate)")
            elif criteria.historical_score < 0.4:
                weaknesses.append("Weak historical performance")
        else:
            criteria.historical_score = 0.5  # Neutral if no data

        # Calculate final grade
        total_score = criteria.total_score()
        grade = criteria.to_grade()

        # Generate trade recommendation
        recommendation = self._generate_recommendation(grade, strengths, weaknesses)

        # Generate invalidation criteria
        invalidation = self._generate_invalidation(pattern_type, pattern_data)

        return GradedPattern(
            pattern_type=pattern_type,
            grade=grade,
            criteria=criteria,
            total_score=total_score,
            strengths=strengths,
            weaknesses=weaknesses,
            trade_recommendation=recommendation,
            invalidation=invalidation,
            raw_data=pattern_data
        )

    def _score_size(self, pattern_type: str, data: Dict) -> float:
        """Score pattern size/magnitude"""
        # Get size from pattern data
        size_pct = data.get('size_pct', data.get('gap_size_pct', 0))

        if pattern_type in ['fvg', 'fair_value_gap']:
            # FVG: Larger gaps are more significant
            if size_pct >= 0.5:  # 0.5%+ gap
                return 0.9
            elif size_pct >= 0.3:
                return 0.7
            elif size_pct >= 0.15:
                return 0.5
            else:
                return 0.3

        elif pattern_type in ['order_block', 'ob']:
            # OB: Check candle body size and move after
            move_strength = data.get('move_strength', 0.5)
            return min(move_strength * 1.2, 1.0)

        else:
            # Default sizing
            return data.get('significance', 0.5)

    def _score_location(
        self,
        pattern_type: str,
        data: Dict,
        context: Dict
    ) -> float:
        """Score location relative to premium/discount zones"""
        zone = context.get('current_zone', 'equilibrium')
        bias = context.get('bias', 'neutral')

        # For bullish patterns, discount zone is ideal
        # For bearish patterns, premium zone is ideal
        pattern_bias = data.get('characteristic', data.get('bias', 'neutral'))

        if pattern_bias == 'bullish':
            if zone == 'discount':
                return 0.95  # Perfect - buying in discount
            elif zone == 'equilibrium':
                return 0.6   # OK - fair value
            else:
                return 0.3   # Poor - buying in premium

        elif pattern_bias == 'bearish':
            if zone == 'premium':
                return 0.95  # Perfect - selling in premium
            elif zone == 'equilibrium':
                return 0.6   # OK - fair value
            else:
                return 0.3   # Poor - selling in discount

        return 0.5  # Neutral

    def _score_structure(
        self,
        pattern_type: str,
        data: Dict,
        context: Dict
    ) -> float:
        """Score alignment with market structure"""
        market_bias = context.get('bias', 'neutral')
        pattern_bias = data.get('characteristic', data.get('bias', 'neutral'))

        # Check if pattern aligns with market structure
        if market_bias == pattern_bias:
            return 0.9  # Aligned
        elif market_bias == 'neutral':
            return 0.6  # Neutral market
        else:
            return 0.3  # Counter-trend

    def _score_confluence(
        self,
        pattern_type: str,
        data: Dict,
        context: Dict
    ) -> float:
        """Score confluence with other factors"""
        confluence_count = 0

        # Check for other patterns nearby
        other_patterns = context.get('nearby_patterns', [])
        if other_patterns:
            confluence_count += min(len(other_patterns), 3)

        # Check for key levels
        key_levels = context.get('key_levels', [])
        if key_levels:
            confluence_count += 1

        # Check for session timing (kill zones)
        in_kill_zone = context.get('in_kill_zone', False)
        if in_kill_zone:
            confluence_count += 2  # Kill zones are important

        # Check for HTF alignment
        htf_aligned = context.get('htf_aligned', False)
        if htf_aligned:
            confluence_count += 2

        # Score based on confluence factors
        if confluence_count >= 5:
            return 0.95
        elif confluence_count >= 3:
            return 0.75
        elif confluence_count >= 1:
            return 0.5
        else:
            return 0.2

    def _score_freshness(self, data: Dict) -> float:
        """Score how fresh/untested the level is"""
        touch_count = data.get('touch_count', data.get('test_count', 0))

        if touch_count == 0:
            return 1.0  # Completely fresh
        elif touch_count == 1:
            return 0.7  # First retest
        elif touch_count == 2:
            return 0.5  # Second retest
        else:
            return 0.3  # Multiple retests - weakening

    def _score_timeframe(self, data: Dict) -> float:
        """Score based on pattern timeframe"""
        tf = data.get('timeframe', 'M15')

        tf_scores = {
            'M1': 0.2,
            'M5': 0.3,
            'M15': 0.5,
            'M30': 0.6,
            'H1': 0.7,
            'H4': 0.85,
            'D1': 0.95,
            'W1': 1.0,
        }

        return tf_scores.get(tf, 0.5)

    def _score_historical(
        self,
        pattern_type: str,
        stats: Dict
    ) -> float:
        """Score based on historical performance"""
        win_rate = stats.get('win_rate', 0.5)
        fill_rate = stats.get('fill_rate', 0.5)

        # Combine win rate and fill rate
        combined = (win_rate * 0.6) + (fill_rate * 0.4)
        return min(combined, 1.0)

    def _generate_recommendation(
        self,
        grade: PatternGrade,
        strengths: List[str],
        weaknesses: List[str]
    ) -> str:
        """Generate trade recommendation based on grade"""
        if grade == PatternGrade.A_PLUS:
            return "STRONG BUY/SELL - Institutional quality setup. Full position size."
        elif grade == PatternGrade.A:
            return "TAKE TRADE - High probability setup. Standard position size."
        elif grade == PatternGrade.B:
            return "CONSIDER - Valid setup with minor issues. Reduced position size."
        elif grade == PatternGrade.C:
            return "WAIT - Pattern exists but conditions not ideal. Wait for better setup."
        elif grade == PatternGrade.D:
            return "AVOID - Poor quality pattern. High risk of failure."
        else:
            return "DO NOT TRADE - Invalid pattern identification."

    def _generate_invalidation(
        self,
        pattern_type: str,
        data: Dict
    ) -> str:
        """Generate invalidation criteria"""
        if pattern_type in ['fvg', 'fair_value_gap']:
            return "Invalidated if price closes through the FVG without respect"
        elif pattern_type in ['order_block', 'ob']:
            return "Invalidated if price closes below (bullish OB) or above (bearish OB) the block"
        elif pattern_type in ['breaker_block', 'breaker']:
            return "Invalidated if price fails to hold at the breaker level"
        else:
            return "Invalidated if key level is broken with conviction"

    def grade_all_patterns(
        self,
        patterns: List[Dict],
        market_context: Dict,
        historical_stats: Optional[Dict] = None
    ) -> List[GradedPattern]:
        """Grade multiple patterns and sort by quality"""
        graded = []

        for pattern in patterns:
            pattern_type = pattern.get('type', pattern.get('pattern_type', 'unknown'))
            graded_pattern = self.grade_pattern(
                pattern_type=pattern_type,
                pattern_data=pattern,
                market_context=market_context,
                historical_stats=historical_stats.get(pattern_type) if historical_stats else None
            )
            graded.append(graded_pattern)

        # Sort by total score (best first)
        graded.sort(key=lambda x: x.total_score, reverse=True)

        return graded


class HistoricalValidator:
    """
    Validate patterns against historical data.

    This answers the question: "Does this pattern actually work?"
    - Fetch historical data after pattern formed
    - Check if price respected the pattern
    - Calculate success rates
    """

    def __init__(self, data_dir: str = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent.parent.parent / "data"

        self.validation_cache = self.data_dir / "pattern_validations.json"
        self._load_cache()

    def _load_cache(self):
        """Load cached validation results"""
        self.validations = {}
        if self.validation_cache.exists():
            try:
                with open(self.validation_cache) as f:
                    self.validations = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load validation cache: {e}")

    def _save_cache(self):
        """Save validation results to cache"""
        try:
            self.validation_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(self.validation_cache, 'w') as f:
                json.dump(self.validations, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save validation cache: {e}")

    def validate_pattern(
        self,
        pattern_type: str,
        symbol: str,
        pattern_time: datetime,
        pattern_levels: Dict,  # high, low, entry, etc.
        lookforward_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Validate a single pattern against historical data.

        Args:
            pattern_type: Type of pattern (fvg, order_block, etc.)
            symbol: Trading symbol (EURUSD, BTCUSD, etc.)
            pattern_time: When the pattern formed
            pattern_levels: Price levels of the pattern
            lookforward_hours: How far forward to check

        Returns:
            Validation result with fill status, time to fill, etc.
        """
        if not HAS_YFINANCE or not HAS_PANDAS:
            logger.warning("yfinance/pandas required for historical validation")
            return {'validated': False, 'reason': 'Missing dependencies'}

        try:
            # Convert symbol to yfinance format
            yf_symbol = self._convert_symbol(symbol)

            # Fetch historical data
            end_time = pattern_time + timedelta(hours=lookforward_hours)
            data = yf.download(
                yf_symbol,
                start=pattern_time,
                end=end_time,
                interval='1h',
                progress=False
            )

            if data.empty:
                return {'validated': False, 'reason': 'No historical data available'}

            # Check if pattern was respected
            result = self._check_pattern_outcome(
                pattern_type=pattern_type,
                pattern_levels=pattern_levels,
                price_data=data
            )

            return result

        except Exception as e:
            logger.error(f"Historical validation failed: {e}")
            return {'validated': False, 'reason': str(e)}

    def _convert_symbol(self, symbol: str) -> str:
        """Convert trading symbol to yfinance format"""
        # Common conversions
        conversions = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD',
            'SPX': '^GSPC',
            'NQ': 'NQ=F',
            'ES': 'ES=F',
        }

        return conversions.get(symbol.upper(), symbol)

    def _check_pattern_outcome(
        self,
        pattern_type: str,
        pattern_levels: Dict,
        price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check how a pattern played out"""
        result = {
            'validated': True,
            'filled': False,
            'respected': False,
            'time_to_fill_hours': None,
            'max_adverse_excursion': 0,
            'max_favorable_excursion': 0,
        }

        if pattern_type in ['fvg', 'fair_value_gap']:
            # FVG: Check if price returned to fill the gap
            fvg_high = pattern_levels.get('high', 0)
            fvg_low = pattern_levels.get('low', 0)

            for i, (idx, row) in enumerate(price_data.iterrows()):
                low_price = row['Low']
                high_price = row['High']

                # Check if price entered the FVG
                if low_price <= fvg_high and high_price >= fvg_low:
                    result['filled'] = True
                    result['time_to_fill_hours'] = i + 1

                    # Check if it respected (bounced from FVG)
                    if i < len(price_data) - 1:
                        next_close = price_data.iloc[i + 1]['Close']
                        # If bullish FVG and price bounced up, respected
                        if pattern_levels.get('bias') == 'bullish' and next_close > fvg_high:
                            result['respected'] = True
                        elif pattern_levels.get('bias') == 'bearish' and next_close < fvg_low:
                            result['respected'] = True
                    break

        elif pattern_type in ['order_block', 'ob']:
            # OB: Check if price returned and respected the block
            ob_high = pattern_levels.get('high', 0)
            ob_low = pattern_levels.get('low', 0)

            for i, (idx, row) in enumerate(price_data.iterrows()):
                low_price = row['Low']
                high_price = row['High']

                # Check if price touched the OB
                if low_price <= ob_high and high_price >= ob_low:
                    result['filled'] = True
                    result['time_to_fill_hours'] = i + 1

                    # Check if it held (didn't break through)
                    if pattern_levels.get('bias') == 'bullish':
                        # Bullish OB should hold as support
                        if low_price >= ob_low:
                            result['respected'] = True
                    else:
                        # Bearish OB should hold as resistance
                        if high_price <= ob_high:
                            result['respected'] = True
                    break

        return result

    def get_pattern_statistics(
        self,
        pattern_type: str,
        symbol: str = None
    ) -> PatternValidation:
        """Get aggregated statistics for a pattern type"""
        key = f"{pattern_type}_{symbol}" if symbol else pattern_type

        if key not in self.validations:
            return PatternValidation(
                pattern_type=pattern_type,
                total_tested=0,
                fill_rate=0.0,
                win_rate=0.0,
                avg_rr_achieved=0.0,
                avg_time_to_fill=0.0,
                best_entry_type="unknown",
                best_timeframe="unknown"
            )

        stats = self.validations[key]

        return PatternValidation(
            pattern_type=pattern_type,
            total_tested=stats.get('total', 0),
            fill_rate=stats.get('fill_rate', 0),
            win_rate=stats.get('win_rate', 0),
            avg_rr_achieved=stats.get('avg_rr', 0),
            avg_time_to_fill=stats.get('avg_time', 0),
            best_entry_type=stats.get('best_entry', 'limit'),
            best_timeframe=stats.get('best_tf', 'H1'),
            notes=stats.get('notes', [])
        )


class MultiTimeframeAnalyzer:
    """
    Analyze patterns across multiple timeframes.

    ICT teaches: Higher timeframe patterns are stronger.
    Professional traders check confluence across M15, H1, H4, D1.
    """

    TIMEFRAMES = ['M5', 'M15', 'H1', 'H4', 'D1']
    TF_WEIGHTS = {
        'M5': 0.1,
        'M15': 0.2,
        'H1': 0.25,
        'H4': 0.25,
        'D1': 0.2,
    }

    def __init__(self):
        pass

    def analyze_confluence(
        self,
        primary_pattern: Dict,
        primary_tf: str,
        all_tf_patterns: Dict[str, List[Dict]]  # TF -> patterns
    ) -> ConfluenceResult:
        """
        Analyze multi-timeframe confluence for a pattern.

        Args:
            primary_pattern: The main pattern being analyzed
            primary_tf: Timeframe of the primary pattern
            all_tf_patterns: Patterns found on all timeframes
        """
        aligned = []
        conflicting = []
        factors = []

        primary_bias = primary_pattern.get('characteristic',
                                          primary_pattern.get('bias', 'neutral'))
        primary_type = primary_pattern.get('type', primary_pattern.get('pattern_type', ''))

        # Check each timeframe
        for tf in self.TIMEFRAMES:
            if tf == primary_tf:
                continue

            tf_patterns = all_tf_patterns.get(tf, [])

            # Check for similar patterns on this TF
            for pattern in tf_patterns:
                pattern_bias = pattern.get('characteristic',
                                          pattern.get('bias', 'neutral'))
                pattern_type = pattern.get('type', pattern.get('pattern_type', ''))

                # Check alignment
                if pattern_bias == primary_bias:
                    aligned.append(tf)

                    # Same pattern type is stronger confluence
                    if pattern_type == primary_type:
                        factors.append(f"{primary_type} confirmed on {tf}")
                    else:
                        factors.append(f"{pattern_type} supports bias on {tf}")

                elif pattern_bias != 'neutral' and pattern_bias != primary_bias:
                    conflicting.append(tf)
                    factors.append(f"WARNING: {pattern_type} conflicts on {tf}")

        # Calculate confluence score
        aligned_weight = sum(self.TF_WEIGHTS.get(tf, 0.1) for tf in aligned)
        conflicting_weight = sum(self.TF_WEIGHTS.get(tf, 0.1) for tf in conflicting)

        # Score: More aligned = higher, conflicts reduce score
        confluence_score = aligned_weight - (conflicting_weight * 0.5)
        confluence_score = max(0, min(1, confluence_score + 0.3))  # Base score of 0.3

        # Generate recommendation
        if confluence_score >= 0.75 and len(conflicting) == 0:
            recommendation = "STRONG"
        elif confluence_score >= 0.5:
            recommendation = "MODERATE"
        elif confluence_score >= 0.3 and len(conflicting) <= 1:
            recommendation = "WEAK"
        else:
            recommendation = "AVOID"

        return ConfluenceResult(
            pattern_type=primary_type,
            primary_timeframe=primary_tf,
            confluence_score=confluence_score,
            aligned_timeframes=list(set(aligned)),
            conflicting_timeframes=list(set(conflicting)),
            confluence_factors=factors,
            recommendation=recommendation
        )


class EdgeTracker:
    """
    Track statistical edge of patterns over time.

    Professional traders track:
    - Win rate per pattern type
    - Average R:R achieved
    - Expectancy (expected $ per trade)
    - Profit factor (gross profit / gross loss)
    """

    def __init__(self, data_dir: str = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent.parent.parent / "data"

        self.stats_file = self.data_dir / "edge_statistics.json"
        self._load_stats()

    def _load_stats(self):
        """Load saved statistics"""
        self.stats: Dict[str, EdgeStatistics] = {}

        if self.stats_file.exists():
            try:
                with open(self.stats_file) as f:
                    data = json.load(f)

                for pattern_type, stats in data.items():
                    self.stats[pattern_type] = EdgeStatistics(
                        pattern_type=pattern_type,
                        **stats
                    )
            except Exception as e:
                logger.warning(f"Failed to load edge statistics: {e}")

    def _save_stats(self):
        """Save statistics to file"""
        try:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for pattern_type, stats in self.stats.items():
                data[pattern_type] = {
                    'total_signals': stats.total_signals,
                    'wins': stats.wins,
                    'losses': stats.losses,
                    'breakeven': stats.breakeven,
                    'pending': stats.pending,
                    'win_rate': stats.win_rate,
                    'avg_rr': stats.avg_rr,
                    'expectancy': stats.expectancy,
                    'profit_factor': stats.profit_factor,
                    'max_consecutive_wins': stats.max_consecutive_wins,
                    'max_consecutive_losses': stats.max_consecutive_losses,
                    'best_day': stats.best_day,
                    'best_session': stats.best_session,
                    'last_updated': stats.last_updated.isoformat() if stats.last_updated else None,
                }

            with open(self.stats_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save edge statistics: {e}")

    def record_trade(
        self,
        pattern_type: str,
        outcome: str,  # 'win', 'loss', 'breakeven'
        rr_achieved: float = 0.0,
        session: str = "",
        day_of_week: str = ""
    ):
        """Record a trade outcome"""
        if pattern_type not in self.stats:
            self.stats[pattern_type] = EdgeStatistics(pattern_type=pattern_type)

        stats = self.stats[pattern_type]
        stats.total_signals += 1

        if outcome == 'win':
            stats.wins += 1
        elif outcome == 'loss':
            stats.losses += 1
        else:
            stats.breakeven += 1

        # Update calculated stats
        total_closed = stats.wins + stats.losses + stats.breakeven
        if total_closed > 0:
            stats.win_rate = stats.wins / total_closed

        # Update average R:R (simplified - would need full history for accurate calc)
        if rr_achieved != 0:
            stats.avg_rr = (stats.avg_rr * (total_closed - 1) + rr_achieved) / total_closed

        # Calculate expectancy: (win_rate * avg_win) - (loss_rate * avg_loss)
        # Simplified: assuming avg_win = avg_rr, avg_loss = 1
        if total_closed > 0:
            loss_rate = stats.losses / total_closed
            stats.expectancy = (stats.win_rate * stats.avg_rr) - (loss_rate * 1.0)

            # Profit factor
            if stats.losses > 0:
                stats.profit_factor = (stats.wins * stats.avg_rr) / stats.losses
            else:
                stats.profit_factor = float('inf') if stats.wins > 0 else 0

        stats.last_updated = datetime.now()
        self._save_stats()

    def get_edge_summary(self, pattern_type: str = None) -> Dict[str, Any]:
        """Get edge statistics summary"""
        if pattern_type:
            if pattern_type not in self.stats:
                return {'pattern_type': pattern_type, 'no_data': True}
            stats = self.stats[pattern_type]
            return {
                'pattern_type': pattern_type,
                'total_signals': stats.total_signals,
                'win_rate': f"{stats.win_rate:.1%}",
                'avg_rr': f"{stats.avg_rr:.2f}R",
                'expectancy': f"{stats.expectancy:.2f}R per trade",
                'profit_factor': f"{stats.profit_factor:.2f}" if stats.profit_factor != float('inf') else "Infinity",
                'has_edge': stats.expectancy > 0,
            }
        else:
            # Return summary for all patterns
            summary = {}
            for pt, stats in self.stats.items():
                summary[pt] = {
                    'win_rate': f"{stats.win_rate:.1%}",
                    'expectancy': f"{stats.expectancy:.2f}R",
                    'has_edge': stats.expectancy > 0,
                }
            return summary

    def get_best_patterns(self, min_signals: int = 10) -> List[str]:
        """Get patterns with positive expectancy (edge)"""
        best = []
        for pattern_type, stats in self.stats.items():
            if stats.total_signals >= min_signals and stats.expectancy > 0:
                best.append({
                    'pattern_type': pattern_type,
                    'expectancy': stats.expectancy,
                    'win_rate': stats.win_rate,
                    'total_signals': stats.total_signals,
                })

        # Sort by expectancy
        best.sort(key=lambda x: x['expectancy'], reverse=True)
        return best


# Global instances
_pattern_grader: Optional[PatternGrader] = None
_historical_validator: Optional[HistoricalValidator] = None
_mtf_analyzer: Optional[MultiTimeframeAnalyzer] = None
_edge_tracker: Optional[EdgeTracker] = None


def get_pattern_grader() -> PatternGrader:
    """Get global PatternGrader instance"""
    global _pattern_grader
    if _pattern_grader is None:
        _pattern_grader = PatternGrader()
    return _pattern_grader


def get_historical_validator() -> HistoricalValidator:
    """Get global HistoricalValidator instance"""
    global _historical_validator
    if _historical_validator is None:
        _historical_validator = HistoricalValidator()
    return _historical_validator


def get_mtf_analyzer() -> MultiTimeframeAnalyzer:
    """Get global MultiTimeframeAnalyzer instance"""
    global _mtf_analyzer
    if _mtf_analyzer is None:
        _mtf_analyzer = MultiTimeframeAnalyzer()
    return _mtf_analyzer


def get_edge_tracker() -> EdgeTracker:
    """Get global EdgeTracker instance"""
    global _edge_tracker
    if _edge_tracker is None:
        _edge_tracker = EdgeTracker()
    return _edge_tracker
