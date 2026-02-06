"""
Pattern Validator Module - ICT Rule Validation

This module validates detected patterns against ICT (Inner Circle Trader) rules
to ensure they meet the required criteria for high-probability setups.

Validation Rules:
- Displacement: Strong impulsive move required for Order Blocks
- Structure Break: BOS/CHoCH must occur for certain patterns
- Prerequisites: Some patterns require base patterns to exist first
- Time of Day: Kill zones increase pattern validity
- Premium/Discount: Location matters for directional bias

The goal is to reduce false positives and improve pattern quality scoring.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, time

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Pattern validation status"""
    VALID = "valid"           # All rules satisfied
    PARTIAL = "partial"       # Some rules satisfied, reduced confidence
    PENDING = "pending"       # Waiting for prerequisite patterns
    INVALID = "invalid"       # Failed critical rules


@dataclass
class ValidationRule:
    """A validation rule definition"""
    name: str
    description: str
    is_critical: bool = False  # Critical rules must pass for pattern to be valid
    weight: float = 1.0        # Weight for confidence calculation


@dataclass
class RuleResult:
    """Result of a single rule check"""
    rule_name: str
    passed: bool
    message: str
    confidence_impact: float = 0.0  # How much this affects confidence (-1 to +1)


@dataclass
class PatternValidationResult:
    """Complete validation result for a pattern"""
    pattern_type: str
    status: ValidationStatus
    original_confidence: float
    adjusted_confidence: float
    rules_checked: List[RuleResult] = field(default_factory=list)
    rules_passed: int = 0
    rules_failed: int = 0
    critical_failures: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'pattern_type': self.pattern_type,
            'status': self.status.value,
            'original_confidence': round(self.original_confidence, 3),
            'adjusted_confidence': round(self.adjusted_confidence, 3),
            'rules_checked': [
                {
                    'rule': r.rule_name,
                    'passed': r.passed,
                    'message': r.message,
                    'impact': round(r.confidence_impact, 3)
                }
                for r in self.rules_checked
            ],
            'rules_passed': self.rules_passed,
            'rules_failed': self.rules_failed,
            'critical_failures': self.critical_failures,
            'recommendations': self.recommendations
        }


# ICT Pattern Rules Configuration
# Each pattern type has specific rules that must be validated
ICT_PATTERN_RULES = {
    # Order Blocks
    'order_block': {
        'requires_displacement': True,
        'requires_structure_break': False,  # Recommended but not required
        'displacement_min_atr_multiple': 1.5,
        'max_candles_since_formation': 50,
        'must_be_unmitigated': True,
        'preferred_location': 'premium_discount',  # Better if in discount for bullish
    },
    'valid_order_block': {
        'requires_displacement': True,
        'requires_structure_break': True,  # Structure break confirms validity
        'displacement_min_atr_multiple': 2.0,  # Higher requirement
        'max_candles_since_formation': 30,
        'must_be_unmitigated': True,
        'preferred_location': 'premium_discount',
    },
    'bullish_order_block': {
        'requires_displacement': True,
        'displacement_direction': 'bullish',
        'displacement_min_atr_multiple': 1.5,
        'must_be_unmitigated': True,
        'preferred_zone': 'discount',
    },
    'bearish_order_block': {
        'requires_displacement': True,
        'displacement_direction': 'bearish',
        'displacement_min_atr_multiple': 1.5,
        'must_be_unmitigated': True,
        'preferred_zone': 'premium',
    },

    # Order Block Mitigation
    'order_block_mitigation': {
        'prerequisite_patterns': ['order_block', 'bullish_order_block', 'bearish_order_block'],
        'requires_price_reentry': True,
        'mitigation_must_complete': True,
    },

    # Breaker Blocks
    'breaker_block': {
        'requires_structure_break': True,  # BOS/CHoCH must occur
        'requires_ob_failure': True,  # Order block must have failed
        'requires_displacement': True,
        'displacement_min_atr_multiple': 1.5,
    },

    # Fair Value Gaps
    'fvg': {
        'min_gap_size_atr': 0.3,  # Minimum gap size as ATR multiple
        'requires_displacement': True,  # FVG should form from impulsive move
        'max_fill_percentage': 50,  # Less than 50% filled is better
    },
    'bullish_fvg': {
        'min_gap_size_atr': 0.3,
        'requires_displacement': True,
        'displacement_direction': 'bullish',
        'preferred_zone': 'discount',
    },
    'bearish_fvg': {
        'min_gap_size_atr': 0.3,
        'requires_displacement': True,
        'displacement_direction': 'bearish',
        'preferred_zone': 'premium',
    },

    # Market Structure
    'bos_bullish': {
        'requires_swing_break': True,
        'break_must_be_clean': True,  # Close above, not just wick
        'requires_displacement': False,  # BOS can happen without displacement
    },
    'bos_bearish': {
        'requires_swing_break': True,
        'break_must_be_clean': True,
        'requires_displacement': False,
    },
    'choch_bullish': {
        'requires_swing_break': True,
        'break_must_be_clean': True,
        'must_break_trend': True,  # Must break from bearish to bullish
        'preferred_with_displacement': True,
    },
    'choch_bearish': {
        'requires_swing_break': True,
        'break_must_be_clean': True,
        'must_break_trend': True,
        'preferred_with_displacement': True,
    },

    # Liquidity
    'liquidity_sweep': {
        'requires_quick_reversal': True,  # Price must reverse quickly
        'max_candles_beyond_level': 3,  # Can't stay beyond level long
        'requires_displacement_after': True,  # Displacement after sweep
    },
    'liquidity_sweep_high': {
        'requires_quick_reversal': True,
        'reversal_direction': 'bearish',
        'max_candles_beyond_level': 3,
    },
    'liquidity_sweep_low': {
        'requires_quick_reversal': True,
        'reversal_direction': 'bullish',
        'max_candles_beyond_level': 3,
    },

    # Entry Patterns
    'optimal_trade_entry': {
        'prerequisite_patterns': ['fvg', 'order_block'],  # OTE requires these
        'requires_retracement': True,  # Must retrace to OTE zone
        'retracement_fib_levels': [0.62, 0.705, 0.79],
        'requires_structure_confirmation': True,
    },

    # Displacement/Expansion
    'displacement': {
        'min_atr_multiple': 2.0,
        'max_candles': 3,  # Displacement happens in 1-3 candles
        'requires_volume_increase': False,  # Not required for forex
    },

    # Smart Money Trap / Inducement
    'smart_money_trap': {
        'requires_liquidity_sweep': True,
        'requires_reversal': True,
        'trap_confirmation_candles': 3,
    },
    'inducement': {
        'requires_liquidity_target': True,
        'must_precede_real_move': True,
    },
}


class PatternValidator:
    """
    Validates detected patterns against ICT rules.

    This validator checks each pattern against specific ICT methodology rules
    to determine if the pattern is valid, partially valid, or invalid.

    Usage:
        validator = PatternValidator(atr_period=14)
        result = validator.validate_pattern(
            pattern_type='order_block',
            pattern_data={...},
            price_data=df,
            detected_patterns=[...]
        )
    """

    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        self._atr_cache: Dict[str, float] = {}

    def validate_pattern(
        self,
        pattern_type: str,
        pattern_data: Dict,
        price_data: Any,  # pd.DataFrame
        detected_patterns: List[Dict] = None,
        current_market_structure: str = None
    ) -> PatternValidationResult:
        """
        Validate a single pattern against ICT rules.

        Args:
            pattern_type: Type of pattern (e.g., 'order_block', 'fvg')
            pattern_data: Pattern details (price levels, indices, etc.)
            price_data: OHLCV DataFrame
            detected_patterns: Other detected patterns (for prerequisite checks)
            current_market_structure: Current market structure ('bullish', 'bearish')

        Returns:
            PatternValidationResult with status and confidence adjustment
        """
        detected_patterns = detected_patterns or []

        # Normalize pattern type
        pattern_type_normalized = pattern_type.lower().replace(' ', '_')

        # Get rules for this pattern type
        rules = ICT_PATTERN_RULES.get(pattern_type_normalized, {})

        # If no specific rules, check generic rules
        if not rules:
            # Try to find matching rules by substring
            for key, value in ICT_PATTERN_RULES.items():
                if key in pattern_type_normalized or pattern_type_normalized in key:
                    rules = value
                    break

        # If still no rules, return valid with original confidence
        if not rules:
            original_confidence = pattern_data.get('confidence', 0.7)
            return PatternValidationResult(
                pattern_type=pattern_type,
                status=ValidationStatus.VALID,
                original_confidence=original_confidence,
                adjusted_confidence=original_confidence,
                recommendations=['No specific validation rules for this pattern type']
            )

        # Calculate ATR for displacement checks
        atr = self._calculate_atr(price_data)

        # Original confidence
        original_confidence = pattern_data.get('confidence', 0.7)

        # Run validation checks
        rule_results = []
        confidence_multiplier = 1.0
        critical_failures = []
        recommendations = []

        # Check displacement requirement
        if rules.get('requires_displacement'):
            result = self._check_displacement(
                pattern_data, price_data, atr,
                rules.get('displacement_min_atr_multiple', 1.5),
                rules.get('displacement_direction')
            )
            rule_results.append(result)
            if not result.passed:
                confidence_multiplier *= 0.6
                if rules.get('requires_displacement') == 'critical':
                    critical_failures.append('Missing displacement')
                recommendations.append('Look for displacement (strong impulsive move) to confirm this pattern')

        # Check structure break requirement
        if rules.get('requires_structure_break'):
            result = self._check_structure_break(
                pattern_data, detected_patterns, current_market_structure
            )
            rule_results.append(result)
            if not result.passed:
                confidence_multiplier *= 0.5
                critical_failures.append('No structure break confirmation')
                recommendations.append('Wait for BOS/CHoCH to confirm structure shift')

        # Check prerequisite patterns
        if rules.get('prerequisite_patterns'):
            result = self._check_prerequisites(
                rules['prerequisite_patterns'], detected_patterns
            )
            rule_results.append(result)
            if not result.passed:
                confidence_multiplier *= 0.4
                recommendations.append(f"Required patterns not found: {rules['prerequisite_patterns']}")

        # Check if unmitigated (for order blocks)
        if rules.get('must_be_unmitigated'):
            result = self._check_unmitigated(pattern_data, price_data)
            rule_results.append(result)
            if not result.passed:
                confidence_multiplier *= 0.3
                critical_failures.append('Pattern has been mitigated')
                recommendations.append('This pattern has already been tested - look for fresh patterns')

        # Check location (premium/discount)
        if rules.get('preferred_zone'):
            result = self._check_zone_location(
                pattern_data, price_data, rules['preferred_zone']
            )
            rule_results.append(result)
            if not result.passed:
                confidence_multiplier *= 0.8  # Minor penalty
                recommendations.append(f"Pattern not in optimal {rules['preferred_zone']} zone")

        # Check quick reversal (for liquidity sweeps)
        if rules.get('requires_quick_reversal'):
            result = self._check_quick_reversal(
                pattern_data, price_data,
                rules.get('max_candles_beyond_level', 3)
            )
            rule_results.append(result)
            if not result.passed:
                confidence_multiplier *= 0.5
                recommendations.append('Reversal too slow - may not be a valid sweep')

        # Calculate final confidence
        adjusted_confidence = original_confidence * confidence_multiplier
        adjusted_confidence = max(0.1, min(0.99, adjusted_confidence))  # Clamp to 0.1-0.99

        # Determine validation status
        rules_passed = sum(1 for r in rule_results if r.passed)
        rules_failed = len(rule_results) - rules_passed

        if critical_failures:
            status = ValidationStatus.INVALID
        elif rules_failed == 0:
            status = ValidationStatus.VALID
        elif rules_passed > rules_failed:
            status = ValidationStatus.PARTIAL
        else:
            status = ValidationStatus.PARTIAL if adjusted_confidence > 0.4 else ValidationStatus.INVALID

        return PatternValidationResult(
            pattern_type=pattern_type,
            status=status,
            original_confidence=original_confidence,
            adjusted_confidence=adjusted_confidence,
            rules_checked=rule_results,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            critical_failures=critical_failures,
            recommendations=recommendations
        )

    def validate_all_patterns(
        self,
        patterns: List[Dict],
        price_data: Any,
        current_market_structure: str = None
    ) -> Dict[str, PatternValidationResult]:
        """
        Validate all detected patterns.

        Returns:
            Dict mapping pattern ID to validation result
        """
        results = {}

        for i, pattern in enumerate(patterns):
            pattern_type = pattern.get('pattern_type', pattern.get('type', 'unknown'))
            pattern_id = f"{pattern_type}_{i}"

            result = self.validate_pattern(
                pattern_type=pattern_type,
                pattern_data=pattern,
                price_data=price_data,
                detected_patterns=patterns,
                current_market_structure=current_market_structure
            )

            results[pattern_id] = result

        return results

    def _calculate_atr(self, price_data: Any) -> float:
        """Calculate Average True Range"""
        if not HAS_PANDAS or price_data is None or len(price_data) < self.atr_period:
            return 0.0

        cache_key = f"atr_{len(price_data)}"
        if cache_key in self._atr_cache:
            return self._atr_cache[cache_key]

        high = price_data['high'].values
        low = price_data['low'].values
        close = price_data['close'].values

        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr[0] = tr1[0]  # First value

        # Calculate ATR
        atr = np.mean(tr[-self.atr_period:])

        self._atr_cache[cache_key] = atr
        return atr

    def _check_displacement(
        self,
        pattern_data: Dict,
        price_data: Any,
        atr: float,
        min_multiple: float,
        direction: str = None
    ) -> RuleResult:
        """Check if there's displacement (strong impulsive move) near the pattern"""
        if not HAS_PANDAS or atr == 0:
            return RuleResult(
                rule_name='displacement',
                passed=True,
                message='Cannot verify displacement - assuming valid',
                confidence_impact=0
            )

        pattern_index = pattern_data.get('end_index', pattern_data.get('index', -1))
        if pattern_index < 0:
            pattern_index = len(price_data) - 10

        # Look at candles after pattern formation
        lookforward = min(10, len(price_data) - pattern_index - 1)
        if lookforward <= 0:
            return RuleResult(
                rule_name='displacement',
                passed=False,
                message='Not enough data to check displacement',
                confidence_impact=-0.2
            )

        # Check for large candle bodies
        for i in range(lookforward):
            idx = pattern_index + i + 1
            if idx >= len(price_data):
                break

            candle_body = abs(price_data['close'].iloc[idx] - price_data['open'].iloc[idx])

            if candle_body >= atr * min_multiple:
                # Check direction if specified
                if direction:
                    is_bullish = price_data['close'].iloc[idx] > price_data['open'].iloc[idx]
                    if direction == 'bullish' and not is_bullish:
                        continue
                    if direction == 'bearish' and is_bullish:
                        continue

                return RuleResult(
                    rule_name='displacement',
                    passed=True,
                    message=f'Displacement found: {candle_body/atr:.1f}x ATR',
                    confidence_impact=0.1
                )

        return RuleResult(
            rule_name='displacement',
            passed=False,
            message=f'No displacement found (need {min_multiple}x ATR move)',
            confidence_impact=-0.2
        )

    def _check_structure_break(
        self,
        pattern_data: Dict,
        detected_patterns: List[Dict],
        market_structure: str = None
    ) -> RuleResult:
        """Check if there's a structure break (BOS/CHoCH) confirming the pattern"""
        # Look for BOS/CHoCH in detected patterns
        structure_patterns = [
            p for p in detected_patterns
            if any(x in str(p.get('pattern_type', '')).lower()
                   for x in ['bos', 'choch', 'break_of_structure', 'change_of_character'])
        ]

        if structure_patterns:
            return RuleResult(
                rule_name='structure_break',
                passed=True,
                message=f'Structure break confirmed: {len(structure_patterns)} found',
                confidence_impact=0.15
            )

        return RuleResult(
            rule_name='structure_break',
            passed=False,
            message='No structure break (BOS/CHoCH) confirmation',
            confidence_impact=-0.2
        )

    def _check_prerequisites(
        self,
        required_patterns: List[str],
        detected_patterns: List[Dict]
    ) -> RuleResult:
        """Check if prerequisite patterns exist"""
        detected_types = [
            str(p.get('pattern_type', p.get('type', ''))).lower()
            for p in detected_patterns
        ]

        found = []
        missing = []

        for required in required_patterns:
            required_lower = required.lower()
            if any(required_lower in dt or dt in required_lower for dt in detected_types):
                found.append(required)
            else:
                missing.append(required)

        if found:
            return RuleResult(
                rule_name='prerequisites',
                passed=True,
                message=f'Prerequisites found: {found}',
                confidence_impact=0.1
            )

        return RuleResult(
            rule_name='prerequisites',
            passed=False,
            message=f'Missing prerequisites: {missing}',
            confidence_impact=-0.3
        )

    def _check_unmitigated(
        self,
        pattern_data: Dict,
        price_data: Any
    ) -> RuleResult:
        """Check if pattern (like OB) has been mitigated"""
        # Check if pattern has mitigated flag
        if pattern_data.get('mitigated', False):
            return RuleResult(
                rule_name='unmitigated',
                passed=False,
                message='Pattern has been mitigated',
                confidence_impact=-0.4
            )

        # Check fill percentage for FVGs
        fill_pct = pattern_data.get('fill_percentage', 0)
        if fill_pct > 80:
            return RuleResult(
                rule_name='unmitigated',
                passed=False,
                message=f'Pattern {fill_pct:.0f}% filled',
                confidence_impact=-0.3
            )

        return RuleResult(
            rule_name='unmitigated',
            passed=True,
            message='Pattern is fresh/unmitigated',
            confidence_impact=0.1
        )

    def _check_zone_location(
        self,
        pattern_data: Dict,
        price_data: Any,
        preferred_zone: str
    ) -> RuleResult:
        """Check if pattern is in the preferred premium/discount zone"""
        if not HAS_PANDAS or price_data is None or len(price_data) < 20:
            return RuleResult(
                rule_name='zone_location',
                passed=True,
                message='Cannot verify zone - assuming valid',
                confidence_impact=0
            )

        # Calculate recent range
        recent_high = price_data['high'].iloc[-20:].max()
        recent_low = price_data['low'].iloc[-20:].min()
        range_mid = (recent_high + recent_low) / 2

        # Get pattern price level
        pattern_price = pattern_data.get('price_low', pattern_data.get('low',
                         pattern_data.get('price', range_mid)))

        is_in_discount = pattern_price < range_mid
        is_in_premium = pattern_price > range_mid

        if preferred_zone == 'discount' and is_in_discount:
            return RuleResult(
                rule_name='zone_location',
                passed=True,
                message='Pattern in optimal discount zone',
                confidence_impact=0.1
            )
        elif preferred_zone == 'premium' and is_in_premium:
            return RuleResult(
                rule_name='zone_location',
                passed=True,
                message='Pattern in optimal premium zone',
                confidence_impact=0.1
            )

        return RuleResult(
            rule_name='zone_location',
            passed=False,
            message=f'Pattern not in preferred {preferred_zone} zone',
            confidence_impact=-0.1
        )

    def _check_quick_reversal(
        self,
        pattern_data: Dict,
        price_data: Any,
        max_candles: int
    ) -> RuleResult:
        """Check if price reversed quickly after sweep (for liquidity patterns)"""
        pattern_index = pattern_data.get('index', pattern_data.get('end_index', -1))

        if pattern_index < 0 or not HAS_PANDAS:
            return RuleResult(
                rule_name='quick_reversal',
                passed=True,
                message='Cannot verify reversal timing',
                confidence_impact=0
            )

        # Check candles after the pattern
        lookforward = min(max_candles + 2, len(price_data) - pattern_index - 1)
        if lookforward < 2:
            return RuleResult(
                rule_name='quick_reversal',
                passed=True,
                message='Not enough data to verify reversal',
                confidence_impact=0
            )

        # Check for reversal within max_candles
        pattern_type = str(pattern_data.get('pattern_type', '')).lower()

        if 'high' in pattern_type or 'sell' in pattern_type:
            # Should reverse down
            high_idx = pattern_index
            for i in range(1, lookforward):
                if price_data['close'].iloc[pattern_index + i] < price_data['open'].iloc[pattern_index + i]:
                    if i <= max_candles:
                        return RuleResult(
                            rule_name='quick_reversal',
                            passed=True,
                            message=f'Reversal in {i} candles',
                            confidence_impact=0.1
                        )
        else:
            # Should reverse up
            for i in range(1, lookforward):
                if price_data['close'].iloc[pattern_index + i] > price_data['open'].iloc[pattern_index + i]:
                    if i <= max_candles:
                        return RuleResult(
                            rule_name='quick_reversal',
                            passed=True,
                            message=f'Reversal in {i} candles',
                            confidence_impact=0.1
                        )

        return RuleResult(
            rule_name='quick_reversal',
            passed=False,
            message=f'No quick reversal within {max_candles} candles',
            confidence_impact=-0.2
        )


# Utility function to get a global validator instance
_validator_instance: Optional[PatternValidator] = None

def get_pattern_validator() -> PatternValidator:
    """Get or create the global pattern validator instance"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = PatternValidator()
    return _validator_instance
