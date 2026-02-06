"""
Pattern Conflict Resolver Module - Confluence & Conflict Detection

This module handles situations where multiple patterns overlap or conflict:

1. CONFLUENCE DETECTION:
   - Identifies when multiple patterns align at similar price levels
   - Boosts confidence when confluence exists (OB + FVG = stronger setup)
   - Labels confluence zones for the UI

2. CONFLICT DETECTION:
   - Identifies when bullish and bearish patterns conflict
   - Flags zones where signals are unclear
   - Prevents trading in conflicting zones

3. CONFLICT RESOLUTION:
   - Uses HTF (higher timeframe) bias to resolve conflicts
   - Applies pattern priority hierarchy
   - Returns "WAIT" signal when conflicts are unresolvable

ICT teaches that confluence increases probability while conflicts suggest waiting.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of pattern conflicts"""
    DIRECTIONAL = "directional"           # Bullish vs bearish at same level
    TIMEFRAME = "timeframe"               # LTF disagrees with HTF
    STRUCTURE = "structure"               # Pattern conflicts with market structure
    CONTRADICTORY = "contradictory"       # Mutually exclusive patterns


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts"""
    DEFER_TO_HTF = "defer_to_htf"         # Higher timeframe wins
    DEFER_TO_STRUCTURE = "defer_to_structure"  # Market structure wins
    HIGHER_PRIORITY = "higher_priority"   # Higher priority pattern wins
    WAIT = "wait"                         # Don't trade - wait for clarity
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Higher confidence wins


# Pattern Priority Hierarchy (higher number = higher priority)
# Based on ICT methodology importance
PATTERN_PRIORITY = {
    # Market Structure (highest priority)
    'break_of_structure': 100,
    'bos_bullish': 100,
    'bos_bearish': 100,
    'change_of_character': 95,
    'choch_bullish': 95,
    'choch_bearish': 95,
    'market_structure_shift': 95,

    # Validated Order Blocks
    'valid_order_block': 90,
    'institutional_order_flow': 88,

    # Standard Order Blocks
    'order_block': 80,
    'bullish_order_block': 80,
    'bearish_order_block': 80,

    # Breaker Blocks (failed OBs become breakers)
    'breaker_block': 75,

    # Fair Value Gaps
    'fvg': 70,
    'bullish_fvg': 70,
    'bearish_fvg': 70,
    'fair_value_gap': 70,

    # Displacement
    'displacement': 65,
    'expansion': 65,

    # Entry Patterns
    'optimal_trade_entry': 60,
    'ote': 60,
    'silver_bullet': 58,

    # Liquidity
    'liquidity_sweep': 55,
    'liquidity_sweep_high': 55,
    'liquidity_sweep_low': 55,
    'buyside_liquidity': 50,
    'sellside_liquidity': 50,
    'equal_highs': 50,
    'equal_lows': 50,

    # Swing Points
    'swing_high': 40,
    'swing_low': 40,
    'higher_high': 40,
    'higher_low': 40,
    'lower_high': 40,
    'lower_low': 40,

    # Premium/Discount
    'premium_zone': 30,
    'discount_zone': 30,

    # Other
    'smart_money_trap': 45,
    'inducement': 45,
}

# Confluence Pairs - patterns that strengthen each other
CONFLUENCE_PAIRS = {
    # Order Block + FVG = very strong
    ('order_block', 'fvg'): {
        'name': 'OB+FVG Confluence',
        'confidence_boost': 0.15,
        'description': 'Order Block with Fair Value Gap - institutional entry zone'
    },
    ('bullish_order_block', 'bullish_fvg'): {
        'name': 'Bullish OB+FVG',
        'confidence_boost': 0.15,
        'description': 'Bullish OB with bullish FVG - high probability long entry'
    },
    ('bearish_order_block', 'bearish_fvg'): {
        'name': 'Bearish OB+FVG',
        'confidence_boost': 0.15,
        'description': 'Bearish OB with bearish FVG - high probability short entry'
    },

    # Order Block + Displacement = institutional
    ('order_block', 'displacement'): {
        'name': 'Institutional OB',
        'confidence_boost': 0.12,
        'description': 'Displacement confirms institutional activity at OB'
    },

    # FVG + Displacement = impulsive
    ('fvg', 'displacement'): {
        'name': 'Impulsive FVG',
        'confidence_boost': 0.10,
        'description': 'FVG formed from impulsive move - likely to be respected'
    },

    # Breaker + FVG
    ('breaker_block', 'fvg'): {
        'name': 'Breaker+FVG',
        'confidence_boost': 0.12,
        'description': 'Breaker block with FVG - failed OB becoming support/resistance'
    },

    # Structure + OB
    ('bos_bullish', 'bullish_order_block'): {
        'name': 'BOS with OB',
        'confidence_boost': 0.15,
        'description': 'Structure break with order block - trend continuation'
    },
    ('bos_bearish', 'bearish_order_block'): {
        'name': 'BOS with OB',
        'confidence_boost': 0.15,
        'description': 'Structure break with order block - trend continuation'
    },

    # CHoCH + OB = trend reversal setup
    ('choch_bullish', 'bullish_order_block'): {
        'name': 'CHoCH Reversal Setup',
        'confidence_boost': 0.18,
        'description': 'Character change with OB - high probability reversal'
    },
    ('choch_bearish', 'bearish_order_block'): {
        'name': 'CHoCH Reversal Setup',
        'confidence_boost': 0.18,
        'description': 'Character change with OB - high probability reversal'
    },

    # Liquidity Sweep + Reversal patterns
    ('liquidity_sweep_high', 'bearish_order_block'): {
        'name': 'Sweep & Dump',
        'confidence_boost': 0.14,
        'description': 'Buy side liquidity taken, bearish OB for entry'
    },
    ('liquidity_sweep_low', 'bullish_order_block'): {
        'name': 'Sweep & Pump',
        'confidence_boost': 0.14,
        'description': 'Sell side liquidity taken, bullish OB for entry'
    },

    # OTE + other patterns
    ('optimal_trade_entry', 'fvg'): {
        'name': 'OTE in FVG',
        'confidence_boost': 0.12,
        'description': 'Optimal trade entry within fair value gap'
    },
    ('optimal_trade_entry', 'order_block'): {
        'name': 'OTE at OB',
        'confidence_boost': 0.12,
        'description': 'Optimal trade entry at order block'
    },
}

# Conflicting Pattern Pairs - patterns that contradict each other
CONFLICTING_PAIRS = {
    # Bullish vs Bearish at same level
    ('bullish_order_block', 'bearish_order_block'): {
        'conflict_type': ConflictType.DIRECTIONAL,
        'severity': 'high',
        'resolution': ResolutionStrategy.DEFER_TO_STRUCTURE,
        'description': 'Bullish and bearish OBs at same level'
    },
    ('bullish_fvg', 'bearish_fvg'): {
        'conflict_type': ConflictType.DIRECTIONAL,
        'severity': 'medium',
        'resolution': ResolutionStrategy.DEFER_TO_HTF,
        'description': 'Bullish and bearish FVGs overlapping'
    },
    ('bos_bullish', 'bos_bearish'): {
        'conflict_type': ConflictType.CONTRADICTORY,
        'severity': 'high',
        'resolution': ResolutionStrategy.WAIT,
        'description': 'Contradictory structure breaks'
    },
    ('choch_bullish', 'choch_bearish'): {
        'conflict_type': ConflictType.CONTRADICTORY,
        'severity': 'high',
        'resolution': ResolutionStrategy.WAIT,
        'description': 'Contradictory character changes'
    },

    # Structure contradictions
    ('bos_bullish', 'bearish_order_block'): {
        'conflict_type': ConflictType.STRUCTURE,
        'severity': 'medium',
        'resolution': ResolutionStrategy.DEFER_TO_STRUCTURE,
        'description': 'Bullish structure but bearish OB'
    },
    ('bos_bearish', 'bullish_order_block'): {
        'conflict_type': ConflictType.STRUCTURE,
        'severity': 'medium',
        'resolution': ResolutionStrategy.DEFER_TO_STRUCTURE,
        'description': 'Bearish structure but bullish OB'
    },
}


@dataclass
class PatternOverlap:
    """Represents two overlapping patterns"""
    pattern1_type: str
    pattern1_index: int
    pattern2_type: str
    pattern2_index: int
    price_overlap_percentage: float  # How much they overlap in price
    is_confluence: bool
    is_conflict: bool
    confluence_info: Optional[Dict] = None
    conflict_info: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            'pattern1': self.pattern1_type,
            'pattern2': self.pattern2_type,
            'overlap_pct': round(self.price_overlap_percentage, 2),
            'is_confluence': self.is_confluence,
            'is_conflict': self.is_conflict,
            'confluence': self.confluence_info,
            'conflict': self.conflict_info
        }


@dataclass
class PatternConflict:
    """Represents a conflict between patterns"""
    patterns_involved: List[str]
    conflict_type: ConflictType
    severity: str  # 'low', 'medium', 'high'
    price_level: float
    description: str
    resolution: Optional[str] = None
    resolved: bool = False

    def to_dict(self) -> Dict:
        return {
            'patterns': self.patterns_involved,
            'type': self.conflict_type.value,
            'severity': self.severity,
            'price_level': self.price_level,
            'description': self.description,
            'resolution': self.resolution,
            'resolved': self.resolved
        }


@dataclass
class ConflictResolutionResult:
    """Result of conflict resolution"""
    original_conflicts: int
    resolved_conflicts: int
    unresolved_conflicts: int
    resolutions: List[Dict] = field(default_factory=list)
    recommendation: str = ""
    should_wait: bool = False
    confidence_adjustment: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'original_conflicts': self.original_conflicts,
            'resolved': self.resolved_conflicts,
            'unresolved': self.unresolved_conflicts,
            'resolutions': self.resolutions,
            'recommendation': self.recommendation,
            'should_wait': self.should_wait,
            'confidence_adjustment': round(self.confidence_adjustment, 3)
        }


class PatternConflictResolver:
    """
    Detects and resolves pattern conflicts and identifies confluences.

    Usage:
        resolver = PatternConflictResolver()

        # Find overlaps
        overlaps = resolver.find_pattern_overlaps(patterns)

        # Detect conflicts
        conflicts = resolver.detect_conflicts(patterns)

        # Resolve conflicts
        resolution = resolver.resolve_conflicts(
            conflicts,
            htf_bias='bullish',
            market_structure='bullish'
        )
    """

    def __init__(self, price_overlap_threshold: float = 0.3):
        """
        Args:
            price_overlap_threshold: Minimum price overlap (0-1) to consider patterns related
        """
        self.price_overlap_threshold = price_overlap_threshold

    def find_pattern_overlaps(
        self,
        patterns: List[Dict]
    ) -> Tuple[List[PatternOverlap], List[PatternOverlap]]:
        """
        Find all overlapping patterns and categorize as confluence or conflict.

        Returns:
            Tuple of (confluences, conflicts)
        """
        confluences = []
        conflicts = []

        for i, p1 in enumerate(patterns):
            for j, p2 in enumerate(patterns):
                if i >= j:  # Avoid duplicate comparisons
                    continue

                overlap_pct = self._calculate_price_overlap(p1, p2)

                if overlap_pct < self.price_overlap_threshold:
                    continue  # Not overlapping enough

                p1_type = self._normalize_pattern_type(p1.get('pattern_type', p1.get('type', '')))
                p2_type = self._normalize_pattern_type(p2.get('pattern_type', p2.get('type', '')))

                # Check for confluence
                confluence_info = self._check_confluence(p1_type, p2_type)

                # Check for conflict
                conflict_info = self._check_conflict(p1_type, p2_type)

                if confluence_info or conflict_info:
                    overlap = PatternOverlap(
                        pattern1_type=p1_type,
                        pattern1_index=i,
                        pattern2_type=p2_type,
                        pattern2_index=j,
                        price_overlap_percentage=overlap_pct,
                        is_confluence=confluence_info is not None,
                        is_conflict=conflict_info is not None,
                        confluence_info=confluence_info,
                        conflict_info=conflict_info
                    )

                    if confluence_info:
                        confluences.append(overlap)
                    if conflict_info:
                        conflicts.append(overlap)

        return confluences, conflicts

    def detect_conflicts(self, patterns: List[Dict]) -> List[PatternConflict]:
        """
        Detect all conflicts among patterns.

        Returns:
            List of PatternConflict objects
        """
        _, conflict_overlaps = self.find_pattern_overlaps(patterns)

        conflicts = []
        for overlap in conflict_overlaps:
            if overlap.conflict_info:
                # Get price level from patterns
                p1_idx = overlap.pattern1_index
                p2_idx = overlap.pattern2_index
                price_level = self._get_pattern_price(patterns[p1_idx])

                conflict = PatternConflict(
                    patterns_involved=[overlap.pattern1_type, overlap.pattern2_type],
                    conflict_type=overlap.conflict_info.get('conflict_type', ConflictType.DIRECTIONAL),
                    severity=overlap.conflict_info.get('severity', 'medium'),
                    price_level=price_level,
                    description=overlap.conflict_info.get('description', 'Pattern conflict detected')
                )
                conflicts.append(conflict)

        return conflicts

    def detect_confluences(self, patterns: List[Dict]) -> List[Dict]:
        """
        Detect all confluences among patterns.

        Returns:
            List of confluence dictionaries with boost info
        """
        confluence_overlaps, _ = self.find_pattern_overlaps(patterns)

        confluences = []
        for overlap in confluence_overlaps:
            if overlap.confluence_info:
                confluences.append({
                    'patterns': [overlap.pattern1_type, overlap.pattern2_type],
                    'name': overlap.confluence_info.get('name', 'Confluence'),
                    'confidence_boost': overlap.confluence_info.get('confidence_boost', 0.1),
                    'description': overlap.confluence_info.get('description', ''),
                    'price_overlap': overlap.price_overlap_percentage
                })

        return confluences

    def resolve_conflicts(
        self,
        conflicts: List[PatternConflict],
        htf_bias: str = None,
        market_structure: str = None,
        pattern_confidences: Dict[str, float] = None
    ) -> ConflictResolutionResult:
        """
        Attempt to resolve pattern conflicts.

        Args:
            conflicts: List of detected conflicts
            htf_bias: Higher timeframe bias ('bullish', 'bearish', None)
            market_structure: Current market structure ('bullish', 'bearish', 'consolidation')
            pattern_confidences: Dict mapping pattern types to their confidence scores

        Returns:
            ConflictResolutionResult
        """
        pattern_confidences = pattern_confidences or {}
        resolutions = []
        resolved_count = 0
        unresolved_count = 0
        total_confidence_adjustment = 0.0

        for conflict in conflicts:
            resolution = self._resolve_single_conflict(
                conflict, htf_bias, market_structure, pattern_confidences
            )

            if resolution['resolved']:
                resolved_count += 1
                conflict.resolved = True
                conflict.resolution = resolution['winning_pattern']
            else:
                unresolved_count += 1
                total_confidence_adjustment -= 0.1  # Penalty for unresolved conflicts

            resolutions.append(resolution)

        # Determine overall recommendation
        should_wait = unresolved_count > 0 and any(
            c.severity == 'high' for c in conflicts if not c.resolved
        )

        if should_wait:
            recommendation = "WAIT - High severity conflicts detected. Wait for clarity before trading."
        elif unresolved_count > 0:
            recommendation = f"CAUTION - {unresolved_count} minor conflicts. Consider reduced position size."
        elif resolved_count > 0:
            recommendation = f"PROCEED - {resolved_count} conflicts resolved using market context."
        else:
            recommendation = "CLEAR - No significant conflicts detected."

        return ConflictResolutionResult(
            original_conflicts=len(conflicts),
            resolved_conflicts=resolved_count,
            unresolved_conflicts=unresolved_count,
            resolutions=resolutions,
            recommendation=recommendation,
            should_wait=should_wait,
            confidence_adjustment=total_confidence_adjustment
        )

    def calculate_confluence_boost(self, patterns: List[Dict]) -> float:
        """
        Calculate total confidence boost from confluences.

        Returns:
            Total confidence boost (can be added to base confidence)
        """
        confluences = self.detect_confluences(patterns)
        total_boost = sum(c.get('confidence_boost', 0) for c in confluences)
        # Cap at 0.25 to prevent over-confidence
        return min(0.25, total_boost)

    def _normalize_pattern_type(self, pattern_type: str) -> str:
        """Normalize pattern type string for comparison"""
        return str(pattern_type).lower().replace(' ', '_').replace('-', '_')

    def _calculate_price_overlap(self, p1: Dict, p2: Dict) -> float:
        """Calculate percentage of price overlap between two patterns"""
        p1_high = p1.get('price_high', p1.get('high', p1.get('price', 0)))
        p1_low = p1.get('price_low', p1.get('low', p1_high))
        p2_high = p2.get('price_high', p2.get('high', p2.get('price', 0)))
        p2_low = p2.get('price_low', p2.get('low', p2_high))

        # Ensure high > low
        if p1_high < p1_low:
            p1_high, p1_low = p1_low, p1_high
        if p2_high < p2_low:
            p2_high, p2_low = p2_low, p2_high

        # Calculate overlap
        overlap_high = min(p1_high, p2_high)
        overlap_low = max(p1_low, p2_low)

        if overlap_high <= overlap_low:
            return 0.0  # No overlap

        overlap_range = overlap_high - overlap_low
        p1_range = max(p1_high - p1_low, 0.0001)
        p2_range = max(p2_high - p2_low, 0.0001)

        # Return average overlap percentage
        return (overlap_range / p1_range + overlap_range / p2_range) / 2

    def _check_confluence(self, p1_type: str, p2_type: str) -> Optional[Dict]:
        """Check if two pattern types form a confluence"""
        # Try both orderings
        key1 = (p1_type, p2_type)
        key2 = (p2_type, p1_type)

        if key1 in CONFLUENCE_PAIRS:
            return CONFLUENCE_PAIRS[key1]
        if key2 in CONFLUENCE_PAIRS:
            return CONFLUENCE_PAIRS[key2]

        # Try partial matching
        for (t1, t2), info in CONFLUENCE_PAIRS.items():
            if (t1 in p1_type or p1_type in t1) and (t2 in p2_type or p2_type in t2):
                return info
            if (t1 in p2_type or p2_type in t1) and (t2 in p1_type or p1_type in t2):
                return info

        return None

    def _check_conflict(self, p1_type: str, p2_type: str) -> Optional[Dict]:
        """Check if two pattern types conflict"""
        # Try both orderings
        key1 = (p1_type, p2_type)
        key2 = (p2_type, p1_type)

        if key1 in CONFLICTING_PAIRS:
            return CONFLICTING_PAIRS[key1]
        if key2 in CONFLICTING_PAIRS:
            return CONFLICTING_PAIRS[key2]

        # Check for implicit directional conflict
        p1_bullish = 'bullish' in p1_type or 'bull' in p1_type
        p1_bearish = 'bearish' in p1_type or 'bear' in p1_type
        p2_bullish = 'bullish' in p2_type or 'bull' in p2_type
        p2_bearish = 'bearish' in p2_type or 'bear' in p2_type

        if (p1_bullish and p2_bearish) or (p1_bearish and p2_bullish):
            # Same pattern type with opposite directions
            p1_base = p1_type.replace('bullish', '').replace('bearish', '').replace('bull', '').replace('bear', '').strip('_')
            p2_base = p2_type.replace('bullish', '').replace('bearish', '').replace('bull', '').replace('bear', '').strip('_')

            if p1_base == p2_base or p1_base in p2_base or p2_base in p1_base:
                return {
                    'conflict_type': ConflictType.DIRECTIONAL,
                    'severity': 'medium',
                    'resolution': ResolutionStrategy.DEFER_TO_STRUCTURE,
                    'description': f'Opposing {p1_base} patterns detected'
                }

        return None

    def _get_pattern_price(self, pattern: Dict) -> float:
        """Get representative price level from pattern"""
        high = pattern.get('price_high', pattern.get('high', 0))
        low = pattern.get('price_low', pattern.get('low', high))
        return (high + low) / 2 if high and low else high or low or 0

    def _resolve_single_conflict(
        self,
        conflict: PatternConflict,
        htf_bias: str,
        market_structure: str,
        pattern_confidences: Dict[str, float]
    ) -> Dict:
        """Resolve a single conflict"""
        resolution = {
            'conflict': conflict.to_dict(),
            'resolved': False,
            'strategy_used': None,
            'winning_pattern': None,
            'reason': None
        }

        conflict_info = None
        for (t1, t2), info in CONFLICTING_PAIRS.items():
            if t1 in conflict.patterns_involved[0] and t2 in conflict.patterns_involved[1]:
                conflict_info = info
                break
            if t2 in conflict.patterns_involved[0] and t1 in conflict.patterns_involved[1]:
                conflict_info = info
                break

        strategy = conflict_info.get('resolution', ResolutionStrategy.DEFER_TO_STRUCTURE) if conflict_info else ResolutionStrategy.DEFER_TO_STRUCTURE

        # Try to resolve based on strategy
        if strategy == ResolutionStrategy.DEFER_TO_HTF:
            if htf_bias in ['bullish', 'bearish']:
                winning = self._get_pattern_by_bias(conflict.patterns_involved, htf_bias)
                if winning:
                    resolution['resolved'] = True
                    resolution['strategy_used'] = 'htf_bias'
                    resolution['winning_pattern'] = winning
                    resolution['reason'] = f'HTF bias is {htf_bias}'

        elif strategy == ResolutionStrategy.DEFER_TO_STRUCTURE:
            if market_structure in ['bullish', 'bearish']:
                winning = self._get_pattern_by_bias(conflict.patterns_involved, market_structure)
                if winning:
                    resolution['resolved'] = True
                    resolution['strategy_used'] = 'market_structure'
                    resolution['winning_pattern'] = winning
                    resolution['reason'] = f'Market structure is {market_structure}'

        elif strategy == ResolutionStrategy.HIGHER_PRIORITY:
            p1_priority = self._get_pattern_priority(conflict.patterns_involved[0])
            p2_priority = self._get_pattern_priority(conflict.patterns_involved[1])

            if p1_priority != p2_priority:
                winning = conflict.patterns_involved[0] if p1_priority > p2_priority else conflict.patterns_involved[1]
                resolution['resolved'] = True
                resolution['strategy_used'] = 'priority'
                resolution['winning_pattern'] = winning
                resolution['reason'] = f'Higher priority pattern'

        elif strategy == ResolutionStrategy.CONFIDENCE_WEIGHTED:
            c1 = pattern_confidences.get(conflict.patterns_involved[0], 0.5)
            c2 = pattern_confidences.get(conflict.patterns_involved[1], 0.5)

            if abs(c1 - c2) > 0.15:  # Significant confidence difference
                winning = conflict.patterns_involved[0] if c1 > c2 else conflict.patterns_involved[1]
                resolution['resolved'] = True
                resolution['strategy_used'] = 'confidence'
                resolution['winning_pattern'] = winning
                resolution['reason'] = f'Higher confidence ({max(c1, c2):.0%})'

        elif strategy == ResolutionStrategy.WAIT:
            resolution['resolved'] = False
            resolution['strategy_used'] = 'wait'
            resolution['reason'] = 'Conflict requires waiting for clarity'

        # Fallback: try confidence-based resolution
        if not resolution['resolved'] and pattern_confidences:
            c1 = pattern_confidences.get(conflict.patterns_involved[0], 0.5)
            c2 = pattern_confidences.get(conflict.patterns_involved[1], 0.5)

            if abs(c1 - c2) > 0.2:
                winning = conflict.patterns_involved[0] if c1 > c2 else conflict.patterns_involved[1]
                resolution['resolved'] = True
                resolution['strategy_used'] = 'confidence_fallback'
                resolution['winning_pattern'] = winning
                resolution['reason'] = f'Significantly higher confidence'

        return resolution

    def _get_pattern_by_bias(self, patterns: List[str], bias: str) -> Optional[str]:
        """Get the pattern that matches the given bias"""
        for p in patterns:
            if bias == 'bullish' and ('bullish' in p or 'bull' in p or 'long' in p):
                return p
            if bias == 'bearish' and ('bearish' in p or 'bear' in p or 'short' in p):
                return p
        return None

    def _get_pattern_priority(self, pattern_type: str) -> int:
        """Get priority for a pattern type"""
        normalized = self._normalize_pattern_type(pattern_type)

        # Direct match
        if normalized in PATTERN_PRIORITY:
            return PATTERN_PRIORITY[normalized]

        # Partial match
        for key, priority in PATTERN_PRIORITY.items():
            if key in normalized or normalized in key:
                return priority

        return 50  # Default priority


# Utility function to get a global resolver instance
_resolver_instance: Optional[PatternConflictResolver] = None

def get_conflict_resolver() -> PatternConflictResolver:
    """Get or create the global conflict resolver instance"""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = PatternConflictResolver()
    return _resolver_instance
