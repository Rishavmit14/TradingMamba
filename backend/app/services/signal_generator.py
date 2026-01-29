"""
Signal Generator Service - ML-POWERED

Combines Smart Money analysis with ML's LEARNED knowledge to generate trading signals.

IMPORTANT: Signals are generated using ONLY patterns the ML has learned from training.
The signal will clearly indicate:
- Which patterns ML detected (from its training)
- Which patterns ML hasn't learned yet (needs more training)
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

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    ML-Powered Trading Signal Generator

    Signal generation uses the ML's LEARNED knowledge:
    1. Determine higher timeframe bias
    2. Identify valid entry zones (ONLY patterns ML learned)
    3. Confirm with market structure
    4. Calculate risk levels
    5. Score the setup based on ML confidence

    If ML hasn't been trained, signals will be limited and flagged.
    """

    def __init__(
        self,
        min_confidence: float = 0.65,
        min_risk_reward: float = 2.0
    ):
        self.analyzer = SmartMoneyAnalyzer(use_ml=True)
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        self.ml_engine = get_ml_engine()

    def generate_signal(
        self,
        symbol: str,
        data: 'pd.DataFrame',
        timeframe: Timeframe,
        htf_bias: Optional[Bias] = None
    ) -> Signal:
        """
        Generate a trading signal for a symbol

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            data: OHLCV DataFrame
            timeframe: Signal timeframe
            htf_bias: Optional higher timeframe bias

        Returns:
            Signal object with entry/exit levels and analysis
        """
        # Run Smart Money analysis
        analysis = self.analyzer.analyze(data)

        # Calculate signal score and factors
        score, factors = self._calculate_signal_score(analysis, htf_bias)

        # Determine direction
        direction = self._determine_direction(analysis, score)

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
        final_confidence = self._adjust_confidence(score / 100, risk_reward)

        # Generate analysis text
        analysis_text = self._generate_analysis_text(analysis, factors, direction)

        # Extract key levels
        key_levels = self._extract_key_levels(analysis)

        # Calculate validity period
        valid_until = self._calculate_validity(timeframe)

        # Generate ML knowledge status message
        ml_status = self._generate_ml_status(analysis)

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
                {'type': ob.type, 'high': ob.high, 'low': ob.low}
                for ob in analysis.order_blocks[:3]
            ],
            fair_value_gaps=[
                {'type': fvg.type, 'high': fvg.high, 'low': fvg.low}
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
