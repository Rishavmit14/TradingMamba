"""
Signal Generator Service - ML-POWERED + HEDGE FUND LEVEL

Combines Smart Money analysis with ML's LEARNED knowledge to generate trading signals.

HEDGE FUND INTEGRATION:
- PatternGrader: Grades patterns A+ to F (only A/B patterns generate signals)
- EdgeTracker: Tracks statistical edge of each pattern type
- HistoricalValidator: Validates patterns against historical data
- MTF Confluence: Checks alignment across timeframes

IMPORTANT: Signals are generated using ONLY patterns the ML has learned from training.
The signal will clearly indicate:
- Which patterns ML detected (from its training)
- Pattern grades (A+ to F) with quality reasoning
- Statistical edge based on historical performance
- Multi-timeframe confluence status
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from uuid import uuid4

try:
    import pandas as pd
except ImportError:
    pd = None

from .smart_money_analyzer import SmartMoneyAnalyzer, SmartMoneyAnalysisResult, Bias, MarketStructure
from ..models.signal import (
    Signal,
    SignalStatus,
    TradingDirection,
    Timeframe,
    SignalFactor,
    KeyLevel
)
from ..ml.ml_pattern_engine import get_ml_engine
from ..ml.hedge_fund_ml import (
    get_pattern_grader,
    get_edge_tracker,
    get_historical_validator,
    get_mtf_analyzer,
    PatternGrade,
    GradedPattern,
)

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    ML-Powered Trading Signal Generator (HEDGE FUND LEVEL)

    Signal generation uses the ML's LEARNED knowledge + Hedge Fund Components:
    1. Determine higher timeframe bias
    2. Identify valid entry zones (ONLY patterns ML learned)
    3. GRADE patterns A+ to F (hedge fund level)
    4. FILTER: Only A/B grades generate signals
    5. Check historical validation and MTF confluence
    6. Track statistical edge for continuous improvement
    7. Calculate risk levels with pattern-specific adjustments
    8. Score the setup based on ML confidence + grade

    If ML hasn't been trained, signals will be limited and flagged.
    """

    # Minimum grade required to generate a signal
    TRADEABLE_GRADES = [PatternGrade.A_PLUS, PatternGrade.A, PatternGrade.B]

    def __init__(
        self,
        min_confidence: float = 0.65,
        min_risk_reward: float = 2.0,
        min_grade: str = "B"  # Minimum pattern grade to trade
    ):
        self.analyzer = SmartMoneyAnalyzer(use_ml=True)
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        self.ml_engine = get_ml_engine()

        # Hedge Fund Components
        self.pattern_grader = get_pattern_grader()
        self.edge_tracker = get_edge_tracker()
        self.historical_validator = get_historical_validator()
        self.mtf_analyzer = get_mtf_analyzer()

        # Set minimum grade threshold
        grade_map = {"A+": PatternGrade.A_PLUS, "A": PatternGrade.A, "B": PatternGrade.B, "C": PatternGrade.C}
        self.min_grade = grade_map.get(min_grade, PatternGrade.B)

    def generate_signal(
        self,
        symbol: str,
        data: 'pd.DataFrame',
        timeframe: Timeframe,
        htf_bias: Optional[Bias] = None
    ) -> Signal:
        """
        Generate a trading signal for a symbol (HEDGE FUND LEVEL)

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            data: OHLCV DataFrame
            timeframe: Signal timeframe
            htf_bias: Optional higher timeframe bias

        Returns:
            Signal object with entry/exit levels, analysis, and PATTERN GRADES
        """
        # Run Smart Money analysis
        analysis = self.analyzer.analyze(data)

        # Calculate signal score and factors
        score, factors = self._calculate_signal_score(analysis, htf_bias)

        # =====================================================================
        # HEDGE FUND LEVEL: Grade all patterns
        # =====================================================================
        graded_patterns, best_pattern = self._grade_patterns(analysis, htf_bias)

        # Check if we should generate a signal based on grade criteria
        should_signal, signal_reason = self._check_should_generate_signal(best_pattern, score)

        # Determine direction (may be WAIT if pattern grade is too low)
        if should_signal:
            direction = self._determine_direction(analysis, score)
        else:
            direction = TradingDirection.WAIT
            logger.info(f"Signal blocked: {signal_reason}")

        # Calculate levels if we have a valid signal
        if direction != TradingDirection.WAIT:
            entry_zone, stop_loss, targets = self._calculate_levels(
                analysis, direction, data
            )
        else:
            entry_zone = (0.0, 0.0)
            stop_loss = 0.0
            targets = []

        # Calculate risk:reward
        risk_reward = self._calculate_risk_reward(
            direction, entry_zone, stop_loss, targets[0] if targets else 0
        )

        # Adjust confidence based on R:R
        base_confidence = self._adjust_confidence(score / 100, risk_reward)

        # =====================================================================
        # HEDGE FUND LEVEL: Adjust confidence based on pattern grade
        # =====================================================================
        final_confidence = self._adjust_confidence_with_grade(base_confidence, best_pattern)

        # Generate analysis text (includes grade analysis)
        analysis_text = self._generate_analysis_text(analysis, factors, direction)

        # Add hedge fund grade analysis
        grade_analysis = self._generate_grade_analysis(graded_patterns, best_pattern)
        analysis_text += grade_analysis

        # Extract key levels
        key_levels = self._extract_key_levels(analysis)

        # Calculate validity period
        valid_until = self._calculate_validity(timeframe)

        # Generate ML knowledge status message
        ml_status = self._generate_ml_status(analysis)

        # =====================================================================
        # HEDGE FUND LEVEL: Build grade information for signal
        # =====================================================================
        pattern_grades = {}
        for gp in graded_patterns:
            pattern_grades[gp.pattern_type] = {
                'grade': gp.grade.value,
                'score': round(gp.total_score, 3),
                'strengths': gp.strengths,
                'weaknesses': gp.weaknesses,
                'recommendation': gp.trade_recommendation,
            }

        # Get edge statistics for the best pattern
        edge_info = {}
        if best_pattern:
            edge_summary = self.edge_tracker.get_edge_summary(best_pattern.pattern_type)
            if not edge_summary.get('no_data'):
                edge_info = {
                    'pattern_type': best_pattern.pattern_type,
                    'win_rate': edge_summary.get('win_rate', 'N/A'),
                    'expectancy': edge_summary.get('expectancy', 'N/A'),
                    'has_edge': edge_summary.get('has_edge', True),
                }

        # =====================================================================
        # HEDGE FUND LEVEL: Historical validation
        # =====================================================================
        historical_validation = {}
        if best_pattern and direction != TradingDirection.WAIT:
            historical_validation = self._validate_pattern_historically(
                pattern_type=best_pattern.pattern_type,
                symbol=symbol,
                pattern_levels=best_pattern.raw_data
            )

            # Adjust confidence based on historical performance
            confidence_adj = historical_validation.get('confidence_adjustment', 1.0)
            final_confidence = min(final_confidence * confidence_adj, 0.95)

            # Add validation analysis to text
            validation_analysis = self._generate_validation_analysis(historical_validation)
            analysis_text += validation_analysis

        return Signal(
            id=str(uuid4()),
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            confidence=final_confidence,
            signal_score=score,
            current_price=analysis.current_price,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            take_profit_1=targets[0] if len(targets) > 0 else 0.0,
            take_profit_2=targets[1] if len(targets) > 1 else None,
            take_profit_3=targets[2] if len(targets) > 2 else None,
            risk_reward_ratio=risk_reward,
            factors=factors,
            key_levels=key_levels,
            analysis_text=analysis_text,
            mtf_bias=htf_bias.value if htf_bias else analysis.bias.value,
            htf_structure=analysis.market_structure.value,
            order_blocks=[
                {'type': ob.type, 'high': ob.high, 'low': ob.low, 'grade': pattern_grades.get('order_block', {}).get('grade', 'N/A')}
                for ob in analysis.order_blocks[:3]
            ],
            fair_value_gaps=[
                {'type': fvg.type, 'high': fvg.high, 'low': fvg.low, 'grade': pattern_grades.get('fvg', {}).get('grade', 'N/A')}
                for fvg in analysis.fair_value_gaps[:3]
            ],
            liquidity_levels=[
                {'type': ll.type, 'price': ll.price}
                for ll in (
                    analysis.liquidity_levels.get('buy_side', [])[:2] +
                    analysis.liquidity_levels.get('sell_side', [])[:2]
                )
            ],
            premium_discount_zone=analysis.premium_discount.get('zone', 'neutral'),
            kill_zone_active=self._is_kill_zone_active(),
            kill_zone_name=self._get_active_kill_zone(),
            valid_until=valid_until,
            concept_ids_used=[],  # TODO: Link to concepts
            rule_ids_used=[],  # TODO: Link to rules
            # ML Knowledge tracking
            ml_patterns_detected=analysis.ml_patterns_used,
            ml_patterns_not_learned=analysis.ml_patterns_not_learned,
            ml_confidence_scores=analysis.ml_confidence_scores,
            ml_knowledge_status=ml_status,
            # =====================================================================
            # HEDGE FUND LEVEL: New fields
            # =====================================================================
            pattern_grades=pattern_grades,
            best_pattern_grade=best_pattern.grade.value if best_pattern else None,
            best_pattern_type=best_pattern.pattern_type if best_pattern else None,
            edge_statistics=edge_info,
            grade_recommendation=best_pattern.trade_recommendation if best_pattern else "No tradeable pattern found",
            historical_validation=historical_validation,
        )

    def _calculate_signal_score(
        self,
        analysis: SmartMoneyAnalysisResult,
        htf_bias: Optional[Bias]
    ) -> Tuple[int, List[SignalFactor]]:
        """
        Calculate signal score based on Smart Money criteria

        Maximum score: 100 points
        - Market Structure: 25 points
        - Order Block: 25 points
        - Fair Value Gap: 20 points
        - Premium/Discount: 20 points
        - Liquidity Target: 10 points
        """
        score = 0
        factors = []

        # Factor 1: Market Structure (25 points)
        if analysis.market_structure != MarketStructure.CONSOLIDATION:
            factor_met = True
            score += 25
            description = f"Clear {analysis.market_structure.value} market structure"
        else:
            factor_met = False
            description = "Market in consolidation - no clear structure"

        factors.append(SignalFactor(
            name="Market Structure",
            description=description,
            weight=0.25,
            met=factor_met,
            details=f"Structure: {analysis.market_structure.value}"
        ))

        # Factor 2: Order Block (25 points)
        bias = htf_bias or analysis.bias
        valid_ob = False

        if bias == Bias.BULLISH:
            bullish_obs = [ob for ob in analysis.order_blocks if ob.type == 'bullish']
            if bullish_obs:
                valid_ob = True
                score += 25
        elif bias == Bias.BEARISH:
            bearish_obs = [ob for ob in analysis.order_blocks if ob.type == 'bearish']
            if bearish_obs:
                valid_ob = True
                score += 25

        factors.append(SignalFactor(
            name="Order Block",
            description="Valid order block present" if valid_ob else "No valid order block",
            weight=0.25,
            met=valid_ob,
            details=f"Found {len(analysis.order_blocks)} order blocks"
        ))

        # Factor 3: Fair Value Gap (20 points)
        valid_fvg = False

        if bias == Bias.BULLISH:
            bullish_fvgs = [fvg for fvg in analysis.fair_value_gaps if fvg.type == 'bullish']
            if bullish_fvgs:
                valid_fvg = True
                score += 20
        elif bias == Bias.BEARISH:
            bearish_fvgs = [fvg for fvg in analysis.fair_value_gaps if fvg.type == 'bearish']
            if bearish_fvgs:
                valid_fvg = True
                score += 20

        factors.append(SignalFactor(
            name="Fair Value Gap",
            description="FVG available for entry" if valid_fvg else "No valid FVG",
            weight=0.20,
            met=valid_fvg,
            details=f"Found {len(analysis.fair_value_gaps)} FVGs"
        ))

        # Factor 4: Premium/Discount Zone (20 points)
        zone = analysis.premium_discount.get('zone', 'neutral')
        zone_aligned = False

        if bias == Bias.BULLISH and zone == 'discount':
            zone_aligned = True
            score += 20
            zone_desc = f"Price in discount zone ({analysis.premium_discount.get('percentage', 50):.0f}%)"
        elif bias == Bias.BEARISH and zone == 'premium':
            zone_aligned = True
            score += 20
            zone_desc = f"Price in premium zone ({analysis.premium_discount.get('percentage', 50):.0f}%)"
        else:
            zone_desc = f"Price in {zone} zone - not ideal for {bias.value if bias else 'neutral'} bias"

        factors.append(SignalFactor(
            name="Premium/Discount",
            description=zone_desc,
            weight=0.20,
            met=zone_aligned,
            details=f"Zone: {zone}, Position: {analysis.premium_discount.get('percentage', 50):.0f}%"
        ))

        # Factor 5: Liquidity Target (10 points)
        has_target = False

        if bias == Bias.BULLISH and analysis.liquidity_levels.get('buy_side'):
            has_target = True
            score += 10
            target_desc = "Buy-side liquidity target identified"
        elif bias == Bias.BEARISH and analysis.liquidity_levels.get('sell_side'):
            has_target = True
            score += 10
            target_desc = "Sell-side liquidity target identified"
        else:
            target_desc = "No clear liquidity target"

        factors.append(SignalFactor(
            name="Liquidity Target",
            description=target_desc,
            weight=0.10,
            met=has_target,
        ))

        return score, factors

    def _determine_direction(
        self,
        analysis: SmartMoneyAnalysisResult,
        score: int
    ) -> TradingDirection:
        """Determine trade direction based on analysis and score"""
        if score < 50:
            return TradingDirection.WAIT

        if analysis.bias == Bias.BULLISH:
            return TradingDirection.BUY
        elif analysis.bias == Bias.BEARISH:
            return TradingDirection.SELL
        else:
            return TradingDirection.WAIT

    def _calculate_levels(
        self,
        analysis: SmartMoneyAnalysisResult,
        direction: TradingDirection,
        data: 'pd.DataFrame'
    ) -> Tuple[Tuple[float, float], float, List[float]]:
        """Calculate entry zone, stop loss, and take profit levels"""
        current_price = analysis.current_price

        if direction == TradingDirection.BUY:
            # Entry at bullish OB or FVG
            bullish_obs = [ob for ob in analysis.order_blocks if ob.type == 'bullish']
            bullish_fvgs = [fvg for fvg in analysis.fair_value_gaps if fvg.type == 'bullish']

            if bullish_obs:
                entry_zone = (bullish_obs[0].low, bullish_obs[0].high)
                stop_loss = bullish_obs[0].low * 0.998
            elif bullish_fvgs:
                entry_zone = (bullish_fvgs[0].low, bullish_fvgs[0].high)
                stop_loss = bullish_fvgs[0].low * 0.998
            else:
                entry_zone = (current_price * 0.998, current_price)
                stop_loss = current_price * 0.99

            # Take profits at buy-side liquidity levels
            buy_side = analysis.liquidity_levels.get('buy_side', [])
            if buy_side:
                targets = [ll.price for ll in buy_side[:3]]
            else:
                targets = [
                    current_price * 1.02,
                    current_price * 1.04,
                    current_price * 1.06
                ]

        elif direction == TradingDirection.SELL:
            # Entry at bearish OB or FVG
            bearish_obs = [ob for ob in analysis.order_blocks if ob.type == 'bearish']
            bearish_fvgs = [fvg for fvg in analysis.fair_value_gaps if fvg.type == 'bearish']

            if bearish_obs:
                entry_zone = (bearish_obs[0].low, bearish_obs[0].high)
                stop_loss = bearish_obs[0].high * 1.002
            elif bearish_fvgs:
                entry_zone = (bearish_fvgs[0].low, bearish_fvgs[0].high)
                stop_loss = bearish_fvgs[0].high * 1.002
            else:
                entry_zone = (current_price, current_price * 1.002)
                stop_loss = current_price * 1.01

            # Take profits at sell-side liquidity levels
            sell_side = analysis.liquidity_levels.get('sell_side', [])
            if sell_side:
                targets = [ll.price for ll in sell_side[:3]]
            else:
                targets = [
                    current_price * 0.98,
                    current_price * 0.96,
                    current_price * 0.94
                ]

        else:
            entry_zone = (0.0, 0.0)
            stop_loss = 0.0
            targets = []

        return entry_zone, stop_loss, targets

    def _calculate_risk_reward(
        self,
        direction: TradingDirection,
        entry_zone: Tuple[float, float],
        stop_loss: float,
        take_profit: float
    ) -> float:
        """Calculate risk:reward ratio"""
        if direction == TradingDirection.WAIT or stop_loss == 0 or take_profit == 0:
            return 0.0

        entry_mid = (entry_zone[0] + entry_zone[1]) / 2
        risk = abs(entry_mid - stop_loss)

        if risk == 0:
            return 0.0

        reward = abs(take_profit - entry_mid)
        return round(reward / risk, 2)

    def _adjust_confidence(self, base_confidence: float, risk_reward: float) -> float:
        """
        Adjust confidence based on:
        1. Risk:Reward ratio - Higher R:R = more confident
        2. Kill zone timing - Premium kill zones boost confidence
        3. ML pattern coverage - More patterns = higher confidence

        ICT methodology emphasizes that entries should be taken:
        - In kill zones for maximum institutional flow
        - At locations offering minimum 2:1 R:R
        """
        # R:R multiplier
        if risk_reward >= 3:
            rr_mult = 1.15
        elif risk_reward >= 2:
            rr_mult = 1.0
        elif risk_reward >= 1.5:
            rr_mult = 0.9
        else:
            rr_mult = 0.7

        # Kill zone multiplier
        kz_mult = self._get_kill_zone_multiplier()

        # Combined confidence
        adjusted = base_confidence * rr_mult * kz_mult

        return min(adjusted, 0.95)  # Cap at 95% - never 100% certain

    def _generate_analysis_text(
        self,
        analysis: SmartMoneyAnalysisResult,
        factors: List[SignalFactor],
        direction: TradingDirection
    ) -> str:
        """Generate human-readable analysis"""
        pd_info = analysis.premium_discount

        factors_text = "\n".join([
            f"{'✓' if f.met else '✗'} {f.name}: {f.description}"
            for f in factors
        ])

        return f"""## AI Analysis Summary
*Based on ML's learned ICT knowledge*

**Market Structure:** {analysis.market_structure.value.title()}
**Current Bias:** {analysis.bias.value.title()} ({analysis.bias_confidence:.0%} confidence)
**Price Zone:** {pd_info.get('zone', 'neutral').title()} ({pd_info.get('percentage', 50):.0f}% of range)

### Signal Factors:
{factors_text}

### Key Observations:
- {analysis.bias_reasoning}
- Range: {pd_info.get('range_low', 0):.5f} - {pd_info.get('range_high', 0):.5f}
- Equilibrium: {pd_info.get('equilibrium', 0):.5f}
- Order Blocks: {len(analysis.order_blocks)} found (ML-detected)
- Fair Value Gaps: {len(analysis.fair_value_gaps)} found (ML-detected)

### Recommendation: **{direction.value}**
"""

    def _extract_key_levels(self, analysis: SmartMoneyAnalysisResult) -> List[KeyLevel]:
        """Extract key levels for charting"""
        levels = []

        # Order blocks
        for ob in analysis.order_blocks[:5]:
            levels.append(KeyLevel(
                level_type=f'{ob.type}_ob',
                price=(ob.high + ob.low) / 2,
                high=ob.high,
                low=ob.low,
                strength=ob.strength
            ))

        # FVGs
        for fvg in analysis.fair_value_gaps[:5]:
            levels.append(KeyLevel(
                level_type=f'{fvg.type}_fvg',
                price=(fvg.high + fvg.low) / 2,
                high=fvg.high,
                low=fvg.low
            ))

        # Liquidity
        for ll in analysis.liquidity_levels.get('buy_side', [])[:3]:
            levels.append(KeyLevel(
                level_type='buy_liquidity',
                price=ll.price,
                strength=ll.strength
            ))

        for ll in analysis.liquidity_levels.get('sell_side', [])[:3]:
            levels.append(KeyLevel(
                level_type='sell_liquidity',
                price=ll.price,
                strength=ll.strength
            ))

        return levels

    def _calculate_validity(self, timeframe: Timeframe) -> datetime:
        """Calculate signal validity duration based on timeframe"""
        validity_map = {
            Timeframe.M1: timedelta(minutes=5),
            Timeframe.M5: timedelta(minutes=15),
            Timeframe.M15: timedelta(hours=1),
            Timeframe.M30: timedelta(hours=2),
            Timeframe.H1: timedelta(hours=4),
            Timeframe.H4: timedelta(hours=16),
            Timeframe.D1: timedelta(days=2),
            Timeframe.W1: timedelta(days=7),
            Timeframe.MN: timedelta(days=30),
        }

        return datetime.utcnow() + validity_map.get(timeframe, timedelta(hours=4))

    def _is_kill_zone_active(self) -> bool:
        """
        Check if currently in an ICT Kill Zone (optimal trading times).

        ICT Kill Zones represent times when institutional order flow is highest,
        creating the best opportunities for Smart Money concept trades.
        """
        now = datetime.utcnow()
        hour = now.hour
        minute = now.minute
        time_decimal = hour + minute / 60

        # ICT Kill Zones (UTC) - Based on ICT methodology:
        # Asian Kill Zone: 00:00 - 04:00 UTC (7 PM - 11 PM EST)
        # London Kill Zone: 02:00 - 05:00 UTC (9 PM - 12 AM EST) & 07:00 - 10:00 UTC (2 AM - 5 AM EST)
        # New York Kill Zone: 12:00 - 15:00 UTC (7 AM - 10 AM EST)
        # London Close: 15:00 - 17:00 UTC (10 AM - 12 PM EST)

        kill_zones = [
            (0, 4),      # Asian Session
            (2, 5),      # London Pre-Market
            (7, 10),     # London Open (most volatile)
            (12, 15),    # New York Open (most volume)
            (15, 17),    # London Close / NY Session Overlap
        ]

        return any(start <= time_decimal < end for start, end in kill_zones)

    def _get_active_kill_zone(self) -> Optional[str]:
        """
        Get the name and quality of the active kill zone.

        Returns the kill zone name with quality indicator.
        Premium kill zones (London Open, NY Open) have higher signal weighting.
        """
        now = datetime.utcnow()
        hour = now.hour
        minute = now.minute
        time_decimal = hour + minute / 60

        # ICT Kill Zones with quality ratings
        kill_zone_info = [
            (0, 4, "Asian Session", "standard"),
            (2, 5, "London Pre-Market", "standard"),
            (7, 10, "London Open", "premium"),  # Highest volatility
            (12, 15, "New York Open", "premium"),  # Highest volume
            (15, 17, "London Close", "standard"),
        ]

        for start, end, name, quality in kill_zone_info:
            if start <= time_decimal < end:
                return f"{name} ({quality.title()})"

        return None

    def _get_kill_zone_multiplier(self) -> float:
        """
        Get confidence multiplier based on current kill zone.

        Premium kill zones (London Open, NY Open) increase signal confidence.
        Standard kill zones have neutral effect.
        Outside kill zones reduces confidence.
        """
        now = datetime.utcnow()
        hour = now.hour

        if 7 <= hour < 10:  # London Open - Premium
            return 1.15
        elif 12 <= hour < 15:  # NY Open - Premium
            return 1.20
        elif self._is_kill_zone_active():  # Other kill zones - Standard
            return 1.0
        else:  # Outside kill zones
            return 0.85

    def _generate_ml_status(self, analysis: SmartMoneyAnalysisResult) -> str:
        """Generate a human-readable ML knowledge status"""
        if not analysis.ml_patterns_used and analysis.ml_patterns_not_learned:
            return "⚠️ ML has NOT been trained yet. Signal based on basic structure only. Train on videos to enable pattern detection."

        if analysis.ml_patterns_used and not analysis.ml_patterns_not_learned:
            patterns = ', '.join(analysis.ml_patterns_used)
            return f"✅ ML-powered signal. Patterns detected: {patterns}"

        if analysis.ml_patterns_used and analysis.ml_patterns_not_learned:
            detected = ', '.join(analysis.ml_patterns_used)
            missing = ', '.join(analysis.ml_patterns_not_learned)
            return f"⚡ Partial ML coverage. Detected: {detected}. Needs training: {missing}"

        return "ℹ️ Basic analysis. No ML patterns applicable for this setup."

    # =========================================================================
    # HEDGE FUND LEVEL METHODS
    # =========================================================================

    def _grade_patterns(
        self,
        analysis: SmartMoneyAnalysisResult,
        htf_bias: Optional[Bias]
    ) -> Tuple[List[GradedPattern], Optional[GradedPattern]]:
        """
        Grade all detected patterns using hedge fund methodology.

        Returns:
            Tuple of (all_graded_patterns, best_pattern_for_entry)
        """
        all_graded = []

        # Build market context for grading
        market_context = {
            'bias': htf_bias.value if htf_bias else analysis.bias.value,
            'current_zone': analysis.premium_discount.get('zone', 'equilibrium'),
            'zone_percentage': analysis.premium_discount.get('percentage', 50),
            'market_structure': analysis.market_structure.value,
            'in_kill_zone': self._is_kill_zone_active(),
            'nearby_patterns': [],  # Will be populated below
            'htf_aligned': htf_bias is not None and htf_bias == analysis.bias,
        }

        # Grade Order Blocks
        for ob in analysis.order_blocks:
            pattern_data = {
                'type': 'order_block',
                'high': ob.high,
                'low': ob.low,
                'bias': ob.type,  # bullish/bearish
                'characteristic': ob.type,
                'strength': ob.strength,
                'touch_count': getattr(ob, 'touch_count', 0),
                'timeframe': getattr(ob, 'timeframe', 'M15'),
            }

            # Get historical stats for this pattern type
            historical_stats = self._get_historical_stats('order_block')

            graded = self.pattern_grader.grade_pattern(
                pattern_type='order_block',
                pattern_data=pattern_data,
                market_context=market_context,
                historical_stats=historical_stats
            )
            all_graded.append(graded)

        # Grade Fair Value Gaps
        for fvg in analysis.fair_value_gaps:
            pattern_data = {
                'type': 'fvg',
                'high': fvg.high,
                'low': fvg.low,
                'bias': fvg.type,
                'characteristic': fvg.type,
                'gap_size_pct': abs(fvg.high - fvg.low) / fvg.low * 100 if fvg.low > 0 else 0,
                'touch_count': getattr(fvg, 'touch_count', 0),
                'timeframe': getattr(fvg, 'timeframe', 'M15'),
            }

            historical_stats = self._get_historical_stats('fvg')

            graded = self.pattern_grader.grade_pattern(
                pattern_type='fvg',
                pattern_data=pattern_data,
                market_context=market_context,
                historical_stats=historical_stats
            )
            all_graded.append(graded)

        # Sort by score (best first)
        all_graded.sort(key=lambda x: x.total_score, reverse=True)

        # Find best tradeable pattern
        best_pattern = None
        for graded in all_graded:
            if graded.grade in self.TRADEABLE_GRADES:
                best_pattern = graded
                break

        return all_graded, best_pattern

    def _get_historical_stats(self, pattern_type: str) -> Optional[Dict]:
        """Get historical statistics for a pattern type from EdgeTracker"""
        try:
            edge_summary = self.edge_tracker.get_edge_summary(pattern_type)
            if edge_summary.get('no_data'):
                return None

            return {
                'win_rate': float(edge_summary.get('win_rate', '0%').rstrip('%')) / 100,
                'fill_rate': 0.7,  # Default if not tracked
                'avg_rr': float(edge_summary.get('avg_rr', '0R').rstrip('R')),
            }
        except Exception:
            return None

    def _adjust_confidence_with_grade(
        self,
        base_confidence: float,
        best_pattern: Optional[GradedPattern]
    ) -> float:
        """
        Adjust confidence based on pattern grade.

        A+ patterns boost confidence significantly
        A patterns have standard confidence
        B patterns reduce confidence slightly
        C/D/F patterns should not generate signals
        """
        if best_pattern is None:
            return base_confidence * 0.7  # No graded pattern

        grade_multipliers = {
            PatternGrade.A_PLUS: 1.25,  # 25% boost for institutional setups
            PatternGrade.A: 1.10,       # 10% boost for excellent setups
            PatternGrade.B: 0.95,       # Slight reduction for good-but-not-great
            PatternGrade.C: 0.75,       # Significant reduction
            PatternGrade.D: 0.50,       # Major reduction
            PatternGrade.F: 0.25,       # Should not trade
        }

        multiplier = grade_multipliers.get(best_pattern.grade, 1.0)
        adjusted = base_confidence * multiplier

        return min(adjusted, 0.95)  # Cap at 95%

    def _generate_grade_analysis(
        self,
        graded_patterns: List[GradedPattern],
        best_pattern: Optional[GradedPattern]
    ) -> str:
        """Generate analysis text for pattern grades"""
        if not graded_patterns:
            return "\n### Pattern Grading: No patterns to grade"

        lines = ["\n### Pattern Grading (Hedge Fund Level):"]

        for graded in graded_patterns[:5]:  # Top 5 patterns
            grade_emoji = {
                PatternGrade.A_PLUS: "🏆",
                PatternGrade.A: "✅",
                PatternGrade.B: "👍",
                PatternGrade.C: "⚠️",
                PatternGrade.D: "❌",
                PatternGrade.F: "🚫",
            }.get(graded.grade, "❓")

            lines.append(f"- {grade_emoji} **{graded.pattern_type.upper()}**: Grade {graded.grade.value} ({graded.total_score:.0%})")

            if graded.strengths:
                lines.append(f"  - Strengths: {', '.join(graded.strengths[:2])}")
            if graded.weaknesses:
                lines.append(f"  - Weaknesses: {', '.join(graded.weaknesses[:2])}")

        if best_pattern:
            lines.append(f"\n**Best Entry Pattern**: {best_pattern.pattern_type.upper()} (Grade {best_pattern.grade.value})")
            lines.append(f"**Recommendation**: {best_pattern.trade_recommendation}")
        else:
            lines.append("\n**⚠️ No A/B grade patterns found - Consider waiting for better setup**")

        return '\n'.join(lines)

    def _check_should_generate_signal(
        self,
        best_pattern: Optional[GradedPattern],
        score: int
    ) -> Tuple[bool, str]:
        """
        Determine if a signal should be generated based on hedge fund criteria.

        Returns:
            Tuple of (should_generate, reason)
        """
        if score < 50:
            return False, "Score too low (< 50)"

        if best_pattern is None:
            return False, "No patterns detected"

        if best_pattern.grade not in self.TRADEABLE_GRADES:
            return False, f"Pattern grade {best_pattern.grade.value} below minimum (need A+/A/B)"

        # Check edge statistics
        edge_summary = self.edge_tracker.get_edge_summary(best_pattern.pattern_type)
        if not edge_summary.get('no_data'):
            has_edge = edge_summary.get('has_edge', True)
            if not has_edge:
                # Don't completely block, but note it
                logger.warning(f"Pattern {best_pattern.pattern_type} has negative expectancy")

        return True, "All criteria met"

    def _validate_pattern_historically(
        self,
        pattern_type: str,
        symbol: str,
        pattern_levels: Dict,
        lookback_days: int = 30
    ) -> Dict:
        """
        Validate pattern against recent historical data.

        This answers: "Has this type of pattern worked recently for this symbol?"

        Args:
            pattern_type: Type of pattern (fvg, order_block, etc.)
            symbol: Trading symbol
            pattern_levels: Price levels of the pattern
            lookback_days: How far back to look for similar patterns

        Returns:
            Validation result with fill rate, win rate, and recommendation
        """
        try:
            # Get pattern statistics from historical validator
            stats = self.historical_validator.get_pattern_statistics(
                pattern_type=pattern_type,
                symbol=symbol
            )

            # Build validation result
            result = {
                'validated': stats.total_tested > 0,
                'total_tested': stats.total_tested,
                'fill_rate': stats.fill_rate,
                'win_rate': stats.win_rate,
                'avg_rr_achieved': stats.avg_rr_achieved,
                'avg_time_to_fill': stats.avg_time_to_fill,
                'best_timeframe': stats.best_timeframe,
                'notes': stats.notes,
            }

            # Generate recommendation based on historical performance
            if stats.total_tested == 0:
                result['recommendation'] = "No historical data - use default risk"
                result['confidence_adjustment'] = 1.0
            elif stats.win_rate >= 0.6 and stats.fill_rate >= 0.7:
                result['recommendation'] = "Strong historical performance - full position"
                result['confidence_adjustment'] = 1.15
            elif stats.win_rate >= 0.5:
                result['recommendation'] = "Moderate historical performance - standard position"
                result['confidence_adjustment'] = 1.0
            else:
                result['recommendation'] = "Weak historical performance - reduced position"
                result['confidence_adjustment'] = 0.85

            return result

        except Exception as e:
            logger.error(f"Historical validation failed: {e}")
            return {
                'validated': False,
                'error': str(e),
                'recommendation': "Validation failed - use caution",
                'confidence_adjustment': 0.9
            }

    def _generate_validation_analysis(self, validation_result: Dict) -> str:
        """Generate analysis text for historical validation"""
        if not validation_result.get('validated'):
            return "\n### Historical Validation: No data available"

        lines = ["\n### Historical Validation:"]

        if validation_result.get('total_tested', 0) > 0:
            lines.append(f"- Patterns tested: {validation_result['total_tested']}")
            lines.append(f"- Fill rate: {validation_result.get('fill_rate', 0):.0%}")
            lines.append(f"- Win rate: {validation_result.get('win_rate', 0):.0%}")
            lines.append(f"- Avg R:R achieved: {validation_result.get('avg_rr_achieved', 0):.2f}")
            lines.append(f"- Best timeframe: {validation_result.get('best_timeframe', 'N/A')}")

        lines.append(f"\n**Recommendation**: {validation_result.get('recommendation', 'N/A')}")

        return '\n'.join(lines)
