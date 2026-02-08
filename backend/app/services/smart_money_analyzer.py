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

# Import Pattern Validation and Conflict Resolution
try:
    from ..ml.pattern_validator import PatternValidator, get_pattern_validator, PatternValidationResult
    from ..ml.pattern_conflict_resolver import PatternConflictResolver, get_conflict_resolver
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False
    PatternValidator = None
    PatternConflictResolver = None

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
    """A swing high or low point with ICT validation (3 SMC Rules)"""
    index: int
    price: float
    type: str  # 'high' or 'low'
    timestamp: Optional[datetime] = None
    strength: int = 1  # How many candles confirm this swing
    # ICT validation fields (populated by validate_swing_points)
    is_validated: bool = False      # Passed all 3 SMC rules?
    is_strong: bool = False         # Strong swing per Video 7?
    has_idm: bool = False           # Was IDM swept before this swing?
    idm_sweep_type: str = 'none'    # 'body' | 'wick' | 'none'
    is_impulse: bool = False        # No pullback = impulse (Video 4 Rule 3)
    validity: str = 'raw'           # 'validated' | 'weak' | 'impulse' | 'raw'
    reasoning: str = ''             # Per-swing explanation
    label: str = ''                 # Chart label (e.g., "HH ✓" or "HL ?")
    associated_idm_price: float = 0.0   # Price of the IDM that validated this swing
    associated_idm_index: int = -1      # Index of the IDM candle
    associated_idm_time: Optional[datetime] = None  # Timestamp of the IDM candle


@dataclass
class OrderBlock:
    """An ICT Order Block with V14 validation (last opposite candle before BOS)"""
    start_index: int
    end_index: int
    high: float
    low: float
    type: str  # 'bullish' or 'bearish'
    mitigated: bool = False
    timestamp: Optional[datetime] = None
    strength: float = 0.0  # Based on move after OB
    # ICT V14 enhancement fields
    bos_validity: str = 'none'        # 'confirmed' | 'unconfirmed' | 'none' (inherited from BOS)
    bos_type: str = ''                # 'higher_high' | 'lower_low' | ''
    is_last_opposite: bool = False    # True = verified last opposite candle before BOS
    pd_zone: str = 'unknown'          # 'premium' | 'discount' | 'unknown'
    zone_aligned: bool = False        # Bullish OB in discount / Bearish OB in premium
    fvg_confluence: bool = False      # OB overlaps with an FVG (highest probability)
    lifecycle: str = 'fresh'          # 'fresh' | 'partial' | 'mitigated' | 'invalid'
    mitigation_pct: float = 0.0       # How much of OB zone has been filled (0-1)
    reasoning: str = ''               # Per-OB explanation


@dataclass
class FairValueGap:
    """A Fair Value Gap (Imbalance) with ICT Video 13 validation"""
    index: int
    high: float
    low: float
    type: str  # 'bullish' or 'bearish'
    filled: bool = False
    fill_percentage: float = 0.0
    timestamp: Optional[datetime] = None
    # ICT enhancement fields (Video 13 + V12 PD zone alignment)
    pd_zone: str = 'unknown'            # 'premium' | 'discount' | 'unknown'
    zone_aligned: bool = False           # True = FVG in correct zone for its type
    lifecycle: str = 'fresh'             # 'fresh' | 'partial' | 'filled' | 'spent'
    reasoning: str = ''                  # Per-FVG explanation


@dataclass
class LiquidityLevel:
    """A liquidity level (stop loss cluster) with ICT Video 2/6 validation"""
    price: float
    type: str  # 'buy_side' or 'sell_side'
    strength: float = 0.0  # Based on swing validation
    timestamp: Optional[datetime] = None
    swept: bool = False
    # ICT enhancement fields (Video 2, 6)
    sweep_type: str = 'none'             # 'wick' | 'body' | 'none' (V6 3 SMC Rules)
    swing_validity: str = 'raw'          # Inherited from swing's ICT validation
    is_equal_level: bool = False         # Equal highs/lows (stronger pool)
    reasoning: str = ''                  # Per-level explanation


@dataclass
class StructureEvent:
    """A market structure event (BOS or CHoCH) with ICT validation"""
    type: str  # 'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish'
    level: float
    timestamp: Optional[datetime] = None
    description: str = ""
    # ICT validation fields
    validity: str = 'unknown'       # 'confirmed' | 'unconfirmed' | 'fake' | 'unknown'
    reasoning: str = ''             # Per-event explanation
    label: str = ''                 # Chart label (e.g., "BOS ↑ ✓")
    associated_idm_price: float = 0.0   # Price of IDM that validated this swing
    associated_idm_index: int = -1      # Index of the IDM candle
    associated_idm_time: Optional[datetime] = None  # Timestamp of the IDM candle
    end_time: Optional[datetime] = None  # Timestamp where this ray should end (next same-group event)
    # V6 3-way classification (BOS enhancement)
    classification: str = 'unknown'  # 'swing_hl' | 'bos' | 'liquidity_sweep' | 'fake_bos' | 'unknown'


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
    inducements: List[Dict] = field(default_factory=list)
    eng_liq_zones: List[Dict] = field(default_factory=list)  # Engineered Liquidity zones (V8/V9)
    kill_zone_active: bool = False
    # ML Knowledge tracking
    ml_patterns_used: List[str] = field(default_factory=list)  # Patterns ML detected
    ml_patterns_not_learned: List[str] = field(default_factory=list)  # Patterns ML can't detect yet
    ml_confidence_scores: Dict[str, float] = field(default_factory=dict)  # Confidence per pattern type

    # Pattern Validation Results (ICT rule checking)
    pattern_validations: Dict = field(default_factory=dict)  # Pattern ID -> ValidationResult
    validation_summary: str = ""  # Human-readable summary of validation

    # Confluence & Conflict Detection
    pattern_confluences: List[Dict] = field(default_factory=list)  # Detected confluences
    pattern_conflicts: List[Dict] = field(default_factory=list)  # Detected conflicts
    conflict_resolutions: List[Dict] = field(default_factory=list)  # How conflicts were resolved
    has_unresolved_conflicts: bool = False  # True if should wait to trade
    confluence_confidence_boost: float = 0.0  # Bonus confidence from confluences


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

        # Step 1: Find raw swing points (basic analysis, always available)
        swing_points = self.find_swing_points(data)

        # Step 1a: Detect inducements EARLY (needed for swing validation)
        # IDM detection feeds into swing validation → market structure
        inducements = []
        if self._can_detect('inducement'):
            inducements = self.find_inducement(data, swing_points)

        # Step 1b: Validate swing points using IDM results (ICT 3 SMC Rules)
        swing_points = self.validate_swing_points(data, swing_points, inducements)

        # Step 2: Analyze market structure using validated swings
        structure, events = self.analyze_market_structure(swing_points, inducements)

        # Step 3: Find order blocks - ONLY IF ML LEARNED
        order_blocks = []
        mitigated_obs = []
        if self._can_detect('order_block'):
            order_blocks, mitigated_obs = self.find_order_blocks(data, structure, swing_points)
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

        # Initialize new pattern containers (inducements already detected in Step 1a)
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

        # Step 10a: Inducement ML tracking (detection already done in Step 1a)
        if self._can_detect('inducement'):
            if inducements:
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

        # Step 11: Map liquidity levels (ICT V2/V6 — sweep detection)
        liquidity = self.find_liquidity_levels(swing_points, data)

        # Step 11b: Detect Engineered Liquidity zones (ICT V8/V9)
        eng_liq_zones = self.find_engineered_liquidity(swing_points, liquidity)

        # Step 11c: Enhance CHoCH events with Rule 2 (ENG_LIQ) + Confirmation (V10)
        all_highs = [sp for sp in swing_points if sp.type == 'high']
        all_lows = [sp for sp in swing_points if sp.type == 'low']
        for event in events:
            if event.type not in ('choch_bullish', 'choch_bearish'):
                continue
            choch_level = event.level

            # V9 Rule 2: Check if CHoCH break is near an ENG_LIQ zone (trap)
            rule2_pass = True
            for zone in eng_liq_zones:
                zone_level = zone.get('level', 0)
                if zone_level > 0 and abs(zone_level - choch_level) / max(zone_level, 1e-8) < 0.02:
                    rule2_pass = False
                    break

            # V10 Confirmation: Check if opposite swing structure formed after CHoCH
            confirmation_state = 'unconfirmed'
            if event.type == 'choch_bearish':
                # Need LL + LH after the CHoCH point
                choch_lows_after = [sp for sp in all_lows if sp.price <= choch_level]
                choch_highs_after = [sp for sp in all_highs if len(choch_lows_after) > 0]
                has_ll = len(choch_lows_after) >= 2 and choch_lows_after[-1].price < choch_lows_after[-2].price
                has_lh = len(all_highs) >= 2 and all_highs[-1].price < all_highs[-2].price
                if has_ll and has_lh:
                    confirmation_state = 'confirmed'
            elif event.type == 'choch_bullish':
                # Need HH + HL after the CHoCH point
                choch_highs_after = [sp for sp in all_highs if sp.price >= choch_level]
                has_hh = len(choch_highs_after) >= 2 and choch_highs_after[-1].price > choch_highs_after[-2].price
                has_hl = len(all_lows) >= 2 and all_lows[-1].price > all_lows[-2].price
                if has_hh and has_hl:
                    confirmation_state = 'confirmed'

            # Update reasoning with Rule 2 result and confirmation
            r2_result = '\u2705' if rule2_pass else '\u274c (near ENG LIQ zone)'
            updated_reasoning = event.reasoning.replace(
                'R2(ENG LIQ): pending',
                f'R2(ENG LIQ): {r2_result}'
            )
            conf_tag = f' | Confirmation(V10): {confirmation_state.upper()}'
            updated_reasoning += conf_tag

            # Re-evaluate validity with Rule 2
            if not rule2_pass and event.validity != 'fake':
                event.validity = 'fake'
                event.label = event.label.replace('\u2713', 'FAKE').replace('?', 'FAKE')
                event.description = event.description.replace('Rules 1+3 pass', 'FAKE — ENG LIQ trap (V9 Rule 2)')
            elif confirmation_state == 'confirmed' and event.validity == 'confirmed':
                event.description = event.description.replace('awaiting R2 post-check', 'CONFIRMED (V10)')
            event.reasoning = updated_reasoning

        # Step 12: Calculate premium/discount (ICT Video 12 — validated swing range)
        premium_discount = self.calculate_premium_discount(data, swing_points, structure)

        # Step 12b: Enrich FVGs with PD zone alignment (ICT V13 + V12)
        if fvgs and premium_discount.get('equilibrium'):
            eq = premium_discount['equilibrium']
            for fvg in fvgs:
                fvg_mid = (fvg.high + fvg.low) / 2
                fvg.pd_zone = 'premium' if fvg_mid >= eq else 'discount'
                # ICT V13: Bullish FVG in discount = highest probability
                # Bearish FVG in premium = highest probability
                fvg.zone_aligned = (
                    (fvg.type == 'bullish' and fvg.pd_zone == 'discount') or
                    (fvg.type == 'bearish' and fvg.pd_zone == 'premium')
                )
                # Lifecycle classification (V13)
                if fvg.fill_percentage >= 0.8:
                    fvg.lifecycle = 'spent'
                elif fvg.fill_percentage > 0:
                    fvg.lifecycle = 'partial'
                else:
                    fvg.lifecycle = 'fresh'
                # Per-FVG reasoning
                zone_tag = "ALIGNED" if fvg.zone_aligned else "wrong zone"
                lifecycle_tag = fvg.lifecycle.upper()
                fvg.reasoning = (
                    f"{fvg.type.title()} FVG ({lifecycle_tag}) in {fvg.pd_zone} "
                    f"[{zone_tag}] | Fill: {fvg.fill_percentage:.0%}"
                )

        # Step 12c: Enrich OBs with PD zone alignment + FVG confluence (ICT V14 + V12 + V13)
        if (order_blocks or mitigated_obs) and premium_discount.get('equilibrium'):
            eq = premium_discount['equilibrium']
            all_obs = list(order_blocks) + list(mitigated_obs)
            for ob in all_obs:
                ob_mid = (ob.high + ob.low) / 2
                ob.pd_zone = 'premium' if ob_mid >= eq else 'discount'
                # V12: Bullish OB in discount = highest probability; Bearish OB in premium
                ob.zone_aligned = (
                    (ob.type == 'bullish' and ob.pd_zone == 'discount') or
                    (ob.type == 'bearish' and ob.pd_zone == 'premium')
                )
                # V14+V13: Check FVG confluence (OB zone overlaps any FVG)
                for fvg in fvgs:
                    if fvg.filled:
                        continue
                    # Zones overlap if OB high >= FVG low AND OB low <= FVG high
                    if ob.high >= fvg.low and ob.low <= fvg.high:
                        ob.fvg_confluence = True
                        break
                # Update reasoning with PD + FVG info
                zone_tag = 'ALIGNED' if ob.zone_aligned else 'wrong zone'
                fvg_tag = 'OB+FVG \u2713' if ob.fvg_confluence else 'no FVG'
                ob.reasoning += f" | PD: {ob.pd_zone} [{zone_tag}] | {fvg_tag}"

        # Step 12a: Validate patterns against ICT rules and detect conflicts
        pattern_validations = {}
        pattern_confluences = []
        pattern_conflicts = []
        conflict_resolutions = []
        has_unresolved_conflicts = False
        confluence_confidence_boost = 0.0
        validation_summary = ""

        if HAS_VALIDATION and ml_patterns_used:
            try:
                # Collect all detected patterns into a unified format
                all_detected_patterns = self._collect_detected_patterns(
                    order_blocks, fvgs, displacements, ote_zones,
                    breaker_blocks, events, swing_points, inducements
                )

                # Get market structure string for validation
                market_structure_str = structure.value if structure else 'neutral'

                # Run pattern validation
                validator = get_pattern_validator()
                validations = validator.validate_all_patterns(
                    all_detected_patterns, data, market_structure_str
                )
                pattern_validations = {k: v.to_dict() for k, v in validations.items()}

                # Generate validation summary
                valid_count = sum(1 for v in validations.values() if v.status.value == 'valid')
                partial_count = sum(1 for v in validations.values() if v.status.value == 'partial')
                invalid_count = sum(1 for v in validations.values() if v.status.value == 'invalid')
                validation_summary = f"Validated {len(validations)} patterns: {valid_count} valid, {partial_count} partial, {invalid_count} invalid"

                # Update confidence scores based on validation
                for pattern_id, validation in validations.items():
                    pattern_type = validation.pattern_type.split('_')[0]  # Get base type
                    if pattern_type in ml_confidence_scores:
                        # Adjust confidence based on validation
                        original = ml_confidence_scores[pattern_type]
                        adjusted = validation.adjusted_confidence
                        ml_confidence_scores[pattern_type] = (original + adjusted) / 2

                # Run conflict detection and resolution
                resolver = get_conflict_resolver()
                detected_confluences = resolver.detect_confluences(all_detected_patterns)
                detected_conflicts = resolver.detect_conflicts(all_detected_patterns)

                pattern_confluences = detected_confluences
                pattern_conflicts = [c.to_dict() for c in detected_conflicts]

                # Calculate confluence boost
                confluence_confidence_boost = resolver.calculate_confluence_boost(all_detected_patterns)

                # Resolve conflicts if any
                if detected_conflicts:
                    # Determine HTF bias from structure
                    htf_bias = structure.value if structure != MarketStructure.CONSOLIDATION else None

                    resolution_result = resolver.resolve_conflicts(
                        detected_conflicts,
                        htf_bias=htf_bias,
                        market_structure=market_structure_str,
                        pattern_confidences=ml_confidence_scores
                    )

                    conflict_resolutions = resolution_result.resolutions
                    has_unresolved_conflicts = resolution_result.should_wait

                    if resolution_result.recommendation:
                        validation_summary += f" | {resolution_result.recommendation}"

                logger.info(f"Pattern validation complete: {validation_summary}")

            except Exception as e:
                logger.warning(f"Pattern validation failed (non-critical): {e}")

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

            # Apply confluence confidence boost
            if confluence_confidence_boost > 0:
                confidence = min(confidence + confluence_confidence_boost, 0.95)
                reasoning += f" [Confluence boost: +{confluence_confidence_boost:.0%}]"

            # Reduce confidence if unresolved conflicts exist
            if has_unresolved_conflicts:
                confidence = confidence * 0.6
                reasoning += " [WARNING: Unresolved conflicts - consider waiting]"

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
            inducements=inducements,
            eng_liq_zones=eng_liq_zones,
            kill_zone_active=kill_zone_active,
            # ML tracking
            ml_patterns_used=ml_patterns_used,
            ml_patterns_not_learned=ml_patterns_not_learned,
            ml_confidence_scores=ml_confidence_scores,
            # Pattern validation and conflict resolution
            pattern_validations=pattern_validations,
            validation_summary=validation_summary,
            pattern_confluences=pattern_confluences,
            pattern_conflicts=pattern_conflicts,
            conflict_resolutions=conflict_resolutions,
            has_unresolved_conflicts=has_unresolved_conflicts,
            confluence_confidence_boost=confluence_confidence_boost,
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

    def _collect_detected_patterns(
        self,
        order_blocks: List[OrderBlock],
        fvgs: List[FairValueGap],
        displacements: List[Dict],
        ote_zones: List[Dict],
        breaker_blocks: List[Dict],
        events: List[StructureEvent],
        swing_points: List[SwingPoint],
        inducements: List[Dict] = None
    ) -> List[Dict]:
        """
        Collect all detected patterns into a unified format for validation.

        Returns:
            List of pattern dictionaries with consistent format:
            {
                'pattern_type': str,
                'index': int,
                'price_high': float,
                'price_low': float,
                'confidence': float,
                ...original fields
            }
        """
        patterns = []

        # Convert Order Blocks
        for i, ob in enumerate(order_blocks):
            patterns.append({
                'pattern_type': f"{ob.type}_order_block",
                'index': ob.start_index,
                'end_index': ob.end_index,
                'price_high': ob.high,
                'price_low': ob.low,
                'confidence': ob.strength if ob.strength else 0.7,
                'mitigated': ob.mitigated,
            })

        # Convert FVGs
        for i, fvg in enumerate(fvgs):
            patterns.append({
                'pattern_type': f"{fvg.type}_fvg",
                'index': fvg.index,
                'price_high': fvg.high,
                'price_low': fvg.low,
                'confidence': 0.7 - (fvg.fill_percentage / 100 * 0.3),  # Lower conf if filled
                'filled': fvg.filled,
                'fill_percentage': fvg.fill_percentage,
            })

        # Convert Displacements
        for i, disp in enumerate(displacements):
            patterns.append({
                'pattern_type': 'displacement',
                'index': disp.get('index', 0),
                'end_index': disp.get('end_index', disp.get('index', 0)),
                'price_high': disp.get('high', disp.get('price_high', 0)),
                'price_low': disp.get('low', disp.get('price_low', 0)),
                'confidence': disp.get('confidence', 0.75),
                **{k: v for k, v in disp.items() if k not in ['index', 'end_index', 'high', 'low', 'confidence']}
            })

        # Convert OTE Zones
        for i, ote in enumerate(ote_zones):
            patterns.append({
                'pattern_type': 'optimal_trade_entry',
                'index': ote.get('index', 0),
                'price_high': ote.get('high', ote.get('price_high', 0)),
                'price_low': ote.get('low', ote.get('price_low', 0)),
                'confidence': ote.get('confidence', 0.7),
                **{k: v for k, v in ote.items() if k not in ['index', 'high', 'low', 'confidence']}
            })

        # Convert Breaker Blocks
        for i, bb in enumerate(breaker_blocks):
            patterns.append({
                'pattern_type': 'breaker_block',
                'index': bb.get('index', 0),
                'price_high': bb.get('high', bb.get('price_high', 0)),
                'price_low': bb.get('low', bb.get('price_low', 0)),
                'confidence': bb.get('confidence', 0.7),
                **{k: v for k, v in bb.items() if k not in ['index', 'high', 'low', 'confidence']}
            })

        # Convert Structure Events (BOS/CHoCH)
        for i, event in enumerate(events):
            patterns.append({
                'pattern_type': event.type,
                'index': 0,  # Structure events don't have a specific index
                'price_high': event.level,
                'price_low': event.level,
                'confidence': 0.8,  # Structure breaks are generally reliable
            })

        # Convert Swing Points (for validation reference)
        for i, sp in enumerate(swing_points):
            patterns.append({
                'pattern_type': f"swing_{sp.type}",
                'index': sp.index,
                'price_high': sp.price if sp.type == 'high' else sp.price,
                'price_low': sp.price if sp.type == 'low' else sp.price,
                'confidence': min(0.5 + sp.strength * 0.05, 0.9),  # Higher strength = higher confidence
            })

        # Convert Inducements
        if inducements:
            for i, idm in enumerate(inducements):
                patterns.append({
                    'pattern_type': f"{idm['type']}_inducement",
                    'index': idm.get('index', 0),
                    'price_high': idm.get('high', 0),
                    'price_low': idm.get('low', 0),
                    'confidence': 0.75,
                    'taken_out': idm.get('taken_out', False),
                })

        return patterns

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

    def validate_swing_points(
        self,
        data: 'pd.DataFrame',
        swing_points: List[SwingPoint],
        inducements: List[Dict]
    ) -> List[SwingPoint]:
        """
        Apply ICT 3 SMC Rules (Video 6) to classify each swing as validated/weak/impulse.

        3 SMC Rules for Swing H/L:
        - Rule 1: IDM required (must have inducement before swing)
        - Rule 2: ONLY body close below/above IDM (wick-only = weak)
        - Rule 3: ONLY body close above/below previous same-type swing

        Strong swing (Video 7): All 3 rules met.
        Weak swing: Any rule failed.
        Impulse: IDM is impulse-type (Video 4 Rule 3).

        Returns swing_points with validation fields populated.
        """
        if not swing_points:
            return swing_points

        closes = data['close'].values
        opens = data['open'].values

        for i, sp in enumerate(swing_points):
            # Find IDMs associated with this swing (parent_swing_index matches)
            sp_idms = [idm for idm in inducements
                       if idm.get('parent_swing_index') == sp.index]

            # Find previous same-type swing for Rule 3
            prev_same = None
            for j in range(i - 1, -1, -1):
                if swing_points[j].type == sp.type:
                    prev_same = swing_points[j]
                    break

            # ---- Rule 1: IDM exists? ----
            has_idm = len(sp_idms) > 0

            # ---- Rule 2: IDM body-close sweep? Track primary IDM ----
            best_sweep = 'none'
            is_impulse = False
            primary_idm_price = 0.0
            primary_idm_index = -1
            primary_idm_time = None
            if sp_idms:
                # Pick the best IDM (body sweep > wick > none, then largest depth)
                best_idm = None
                for idm in sp_idms:
                    if idm.get('is_impulse'):
                        is_impulse = True
                    st = idm.get('sweep_type', 'none')
                    if st == 'body':
                        best_sweep = 'body'
                        if not best_idm or best_idm.get('sweep_type') != 'body':
                            best_idm = idm
                    elif st == 'wick' and best_sweep != 'body':
                        best_sweep = 'wick'
                        if not best_idm:
                            best_idm = idm
                    elif not best_idm:
                        best_idm = idm
                if best_idm:
                    primary_idm_price = best_idm.get('price', 0.0)
                    primary_idm_index = best_idm.get('index', -1)
                    primary_idm_time = best_idm.get('timestamp', None)
            has_body_sweep = best_sweep == 'body'

            # ---- Rule 3: Body close beyond previous same-type swing? ----
            body_closed_beyond = False
            if prev_same and sp.type == 'high':
                # For swing high: check if any candle body-closed above prev swing high
                search_end = min(sp.index + 1, len(closes))
                for k in range(prev_same.index + 1, search_end):
                    body_high = max(closes[k], opens[k])
                    if body_high > prev_same.price:
                        body_closed_beyond = True
                        break
            elif prev_same and sp.type == 'low':
                # For swing low: check if any candle body-closed below prev swing low
                search_end = min(sp.index + 1, len(closes))
                for k in range(prev_same.index + 1, search_end):
                    body_low = min(closes[k], opens[k])
                    if body_low < prev_same.price:
                        body_closed_beyond = True
                        break
            elif not prev_same:
                body_closed_beyond = True  # First swing of type = auto-valid

            # ---- Classify ----
            all_3_met = has_idm and has_body_sweep and body_closed_beyond
            is_strong = all_3_met and not is_impulse

            if is_impulse:
                validity = 'impulse'
            elif all_3_met:
                validity = 'validated'
            elif has_idm:
                validity = 'weak'
            else:
                validity = 'raw'

            # ---- Build per-swing reasoning ----
            price_str = f'{sp.price:.2f}' if sp.price > 100 else f'{sp.price:.5f}'
            swing_type_str = 'Swing High' if sp.type == 'high' else 'Swing Low'
            reasoning_lines = [f'{swing_type_str} @ {price_str}']

            r1_status = '\u2705' if has_idm else '\u274c'
            r2_status = '\u2705' if has_body_sweep else ('\u26a0\ufe0f' if best_sweep == 'wick' else '\u274c')
            r3_status = '\u2705' if body_closed_beyond else '\u274c'

            if has_idm and primary_idm_price > 0:
                idm_price_str = f'{primary_idm_price:.2f}' if primary_idm_price > 100 else f'{primary_idm_price:.5f}'
                reasoning_lines.append(f'Rule 1 (IDM exists): {r1_status} IDM {idm_price_str} exists')
            else:
                reasoning_lines.append(f'Rule 1 (IDM exists): {r1_status} No IDM found')
            if has_idm:
                reasoning_lines.append(f'Rule 2 (Body close sweep): {r2_status} sweep={best_sweep}')
            else:
                reasoning_lines.append(f'Rule 2 (Body close sweep): {r2_status} N/A (no IDM)')
            reasoning_lines.append(f'Rule 3 (Body beyond prev swing): {r3_status} {"Yes" if body_closed_beyond else "No"}')

            if is_impulse:
                reasoning_lines.append('IMPULSE swing - no pullback, low/high IS inducement (Video 4 Rule 3)')

            validity_str = validity.upper()
            reasoning_lines.append(f'Classification: {validity_str} | Strong: {"Yes" if is_strong else "No"}')

            reasoning_str = ' | '.join(reasoning_lines)

            # ---- Build chart label ----
            # Label will be set by analyze_market_structure based on HH/HL/LH/LL context
            validity_tag = {'validated': '\u2713', 'weak': '?', 'impulse': '\u26a1', 'raw': ''}.get(validity, '')
            label = f'{swing_type_str[:2]} {validity_tag}'.strip()

            # Update swing point fields
            sp.is_validated = all_3_met
            sp.is_strong = is_strong
            sp.has_idm = has_idm
            sp.idm_sweep_type = best_sweep
            sp.is_impulse = is_impulse
            sp.validity = validity
            sp.reasoning = reasoning_str
            sp.label = label
            sp.associated_idm_price = primary_idm_price
            sp.associated_idm_index = primary_idm_index
            sp.associated_idm_time = primary_idm_time

        return swing_points

    def analyze_market_structure(
        self,
        swing_points: List[SwingPoint],
        inducements: Optional[List[Dict]] = None
    ) -> Tuple[MarketStructure, List[StructureEvent]]:
        """
        Analyze market structure to determine trend and identify BOS/CHoCH.

        Uses ICT 3 SMC Rules (Video 6) when swing points have been validated:
        - HH/HL/LH/LL events carry validation status from validate_swing_points()
        - BOS requires IDM taken + body close beyond level (Video 6)
        - CHoCH checks strong/weak swing + 4 fake CHoCH rules (Video 7-10)
        """
        events = []
        inducements = inducements or []

        if len(swing_points) < 4:
            return MarketStructure.CONSOLIDATION, events

        # Get ALL swing highs and lows (not just recent)
        all_highs = [sp for sp in swing_points if sp.type == 'high']
        all_lows = [sp for sp in swing_points if sp.type == 'low']

        if len(all_highs) < 2 or len(all_lows) < 2:
            return MarketStructure.CONSOLIDATION, events

        # Recent swings for trend determination
        recent_highs = all_highs[-4:]
        recent_lows = all_lows[-4:]

        # Determine structure based on HH/HL or LH/LL pattern (recent only)
        hh = recent_highs[-1].price > recent_highs[-2].price
        hl = recent_lows[-1].price > recent_lows[-2].price
        lh = recent_highs[-1].price < recent_highs[-2].price
        ll = recent_lows[-1].price < recent_lows[-2].price

        # Helper to build validated structure event from a swing point
        # V6 3-way classification: swing_hl (strictest) | bos (middle) | liquidity_sweep | fake_bos
        def _classify_v6(sp):
            """Classify per V6 3 SMC Rules table: Swing H/L vs BOS vs Liq Sweep."""
            has_idm = sp.has_idm
            sweep_type = sp.idm_sweep_type  # 'body' | 'wick' | 'none'
            body_closed = sp.is_validated   # Rule 3: body closed beyond prev swing

            if not has_idm and not body_closed:
                return 'unknown'
            if not has_idm and body_closed:
                return 'fake_bos'       # No IDM → invalid per Rule 1
            if has_idm and not body_closed:
                return 'liquidity_sweep'  # IDM taken but no body close = wick sweep of level
            # has_idm AND body_closed → BOS or Swing H/L (Rule 2 differentiates)
            if sweep_type == 'body':
                return 'swing_hl'       # Strictest: body close below IDM
            return 'bos'                # Wick OR body for IDM = BOS (middle)

        def _ms_event(event_type, sp, description):
            """Create a StructureEvent with ICT validation + V6 3-way classification."""
            validity = sp.validity if sp.validity != 'raw' else 'unknown'
            classification = _classify_v6(sp)
            tag = {'validated': '\u2713', 'weak': '?', 'impulse': '\u26a1', 'raw': '', 'unknown': ''}.get(validity, '')
            # Add classification tag for BOS/Swing H/L distinction
            cls_tag = {'swing_hl': ' [SHL]', 'bos': ' [BOS]', 'liquidity_sweep': ' [LS]',
                       'fake_bos': ' [FAKE]'}.get(classification, '')
            base_label = {'higher_high': 'HH', 'higher_low': 'HL',
                          'lower_high': 'LH', 'lower_low': 'LL'}.get(event_type, event_type)
            label = f'{base_label} {tag}'.strip()
            return StructureEvent(
                type=event_type, level=sp.price,
                timestamp=sp.timestamp,
                description=f'{description} ({validity}{cls_tag})',
                validity=validity, reasoning=sp.reasoning, label=label,
                associated_idm_price=sp.associated_idm_price,
                associated_idm_index=sp.associated_idm_index,
                associated_idm_time=sp.associated_idm_time,
                classification=classification
            )

        # Emit HH/HL/LH/LL labels for ALL consecutive swing pairs (full chart history)
        for i in range(1, len(all_highs)):
            if all_highs[i].price > all_highs[i - 1].price:
                events.append(_ms_event('higher_high', all_highs[i], 'Higher High'))
            elif all_highs[i].price < all_highs[i - 1].price:
                events.append(_ms_event('lower_high', all_highs[i], 'Lower High'))

        for i in range(1, len(all_lows)):
            if all_lows[i].price > all_lows[i - 1].price:
                events.append(_ms_event('higher_low', all_lows[i], 'Higher Low'))
            elif all_lows[i].price < all_lows[i - 1].price:
                events.append(_ms_event('lower_low', all_lows[i], 'Lower Low'))

        # Compute end_time for each MS event: ray ends at the next same-group event
        # Groups: highs (HH/LH) and lows (HL/LL)
        high_events = [e for e in events if e.type in ('higher_high', 'lower_high')]
        low_events = [e for e in events if e.type in ('higher_low', 'lower_low')]
        # Sort by timestamp within each group
        high_events.sort(key=lambda e: e.timestamp if e.timestamp else datetime.min)
        low_events.sort(key=lambda e: e.timestamp if e.timestamp else datetime.min)
        for group in (high_events, low_events):
            for i in range(len(group)):
                if i + 1 < len(group):
                    group[i].end_time = group[i + 1].timestamp
                # Last event in group: end_time stays None (ray extends to right edge)

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

        # Check for BOS with ICT rules
        bos_event = self._check_bos(recent_highs, recent_lows, structure, inducements)
        if bos_event:
            events.append(bos_event)

        # Check for CHoCH with fake detection
        choch_event = self._check_choch(recent_highs, recent_lows, structure, inducements)
        if choch_event:
            events.append(choch_event)

        return structure, events

    def _check_bos(
        self,
        recent_highs: List[SwingPoint],
        recent_lows: List[SwingPoint],
        structure: MarketStructure,
        inducements: Optional[List[Dict]] = None
    ) -> Optional[StructureEvent]:
        """
        Check for Break of Structure with ICT V5/V6 3 SMC Rules + 3-way classification.

        V5 BOS 2-Rule System:
        - Rule 1: IDM must be taken (wick OR body — both valid for BOS)
        - Rule 2: Wick OR body for IDM sweep (BOS is middle ground per V6)
        - Rule 3: Body close beyond the level required (wick only = Liquidity Sweep)

        V6 3-Way Classification (BOS vs Liq Sweep vs Swing H/L):
        - Swing H/L: IDM body-close only (strictest Rule 2)
        - BOS: IDM wick or body (middle Rule 2)
        - Liquidity Sweep: No body close beyond level (Rule 3 wick only)
        """
        inducements = inducements or []
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None

        def _build_bos_event(direction, breaking_swing, broken_level):
            """Build a BOS StructureEvent with V6 3-way classification."""
            price_str = f'{broken_level:.2f}' if broken_level > 100 else f'{broken_level:.5f}'
            arrow = '\u2191' if direction == 'bullish' else '\u2193'

            has_idm = breaking_swing.has_idm
            sweep_type = breaking_swing.idm_sweep_type  # 'body' | 'wick' | 'none'
            is_body_close = breaking_swing.is_validated

            # V6 3-Way Classification
            if not has_idm and not is_body_close:
                classification = 'unknown'
            elif not has_idm and is_body_close:
                classification = 'fake_bos'
            elif has_idm and not is_body_close:
                classification = 'liquidity_sweep'
            elif sweep_type == 'body':
                classification = 'swing_hl'
            else:
                classification = 'bos'

            cls_label = {
                'swing_hl': 'Swing H/L',
                'bos': 'BOS',
                'liquidity_sweep': 'Liq Sweep',
                'fake_bos': 'Fake BOS',
                'unknown': 'Unclassified'
            }.get(classification, 'Unknown')

            # Build detailed reasoning per V5/V6
            reasoning_parts = [f'{direction.title()} BOS at {price_str}']
            if has_idm:
                reasoning_parts.append(f'Rule 1: IDM taken \u2705 ({sweep_type} sweep)')
            else:
                reasoning_parts.append('Rule 1: No IDM taken \u274c')
            # Rule 2 (V6 differentiator): how was IDM taken?
            if has_idm:
                if sweep_type == 'body':
                    reasoning_parts.append('Rule 2: IDM body close \u2705 (qualifies as Swing H/L)')
                else:
                    reasoning_parts.append('Rule 2: IDM wick sweep \u2705 (BOS level, not Swing H/L)')
            else:
                reasoning_parts.append('Rule 2: N/A (no IDM)')
            if is_body_close:
                reasoning_parts.append('Rule 3: Body close beyond level \u2705')
            else:
                reasoning_parts.append('Rule 3: No body close \u274c (wick only = Liq Sweep)')
            reasoning_parts.append(f'V6 Classification: {cls_label}')

            # Validity
            validity = 'confirmed' if has_idm and is_body_close else 'unconfirmed'
            tag = '\u2713' if validity == 'confirmed' else '?'

            return StructureEvent(
                type=f'bos_{direction}',
                level=broken_level,
                timestamp=breaking_swing.timestamp,
                description=f'{direction.title()} BOS at {price_str} ({validity} — {cls_label})',
                validity=validity,
                reasoning=' | '.join(reasoning_parts),
                label=f'BOS {arrow} {tag}',
                classification=classification
            )

        if structure == MarketStructure.BULLISH:
            if recent_highs[-1].price > recent_highs[-2].price:
                return _build_bos_event('bullish', recent_highs[-1], recent_highs[-2].price)

        elif structure == MarketStructure.BEARISH:
            if recent_lows[-1].price < recent_lows[-2].price:
                return _build_bos_event('bearish', recent_lows[-1], recent_lows[-2].price)

        return None

    def _check_choch(
        self,
        recent_highs: List[SwingPoint],
        recent_lows: List[SwingPoint],
        structure: MarketStructure,
        inducements: Optional[List[Dict]] = None
    ) -> Optional[StructureEvent]:
        """
        Check for Change of Character with V9 4-rule fake CHoCH detection.

        Video 7-10 Complete CHoCH System:
        - CHoCH = breaking previous swing in OPPOSITE direction (reversal)
        - 4 Fake CHoCH Rules (V9): ALL must pass for valid CHoCH
          Rule 1: Broken swing must be STRONG (not weak/sweeping candle)
          Rule 2: Must NOT create ENG LIQ zone (checked in post-processing)
          Rule 3: IDM must be taken before break
          Rule 4: Must NOT be higher TF inducement (single-TF = N/A)
        - Confirmation (V10): LL+LH or HH+HL must form after CHoCH
        """
        inducements = inducements or []
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None

        def _build_choch_event(direction, broken_swing, breaking_swing, broken_level):
            """Build a CHoCH StructureEvent with V9 4-rule validation + V6 classification."""
            price_str = f'{broken_level:.2f}' if broken_level > 100 else f'{broken_level:.5f}'
            arrow = '\u2191' if direction == 'bullish' else '\u2193'

            reasoning_parts = [f'{direction.title()} CHoCH at {price_str}']

            # V9 Rule 1: Was broken swing STRONG?
            is_broken_strong = broken_swing.is_strong
            if is_broken_strong:
                reasoning_parts.append('R1(Strong Swing): \u2705')
            else:
                reasoning_parts.append(f'R1(Strong Swing): \u274c ({broken_swing.validity})')

            # V9 Rule 2: ENG LIQ creation (checked in post-processing step)
            rule2_pass = True  # Placeholder — enhanced in analyze() after ENG_LIQ detection
            reasoning_parts.append('R2(ENG LIQ): pending')

            # V9 Rule 3: Was IDM taken before break?
            breaker_has_idm = breaking_swing.has_idm
            if breaker_has_idm:
                reasoning_parts.append(f'R3(IDM taken): \u2705 ({breaking_swing.idm_sweep_type})')
            else:
                reasoning_parts.append('R3(IDM taken): \u274c')

            # V9 Rule 4: Multi-TF check (single-TF analysis = N/A)
            reasoning_parts.append('R4(MTF IDM): N/A (single TF)')

            # V6 3-way classification for the breaking swing
            classification = 'unknown'
            if not breaker_has_idm and not breaking_swing.is_validated:
                classification = 'unknown'
            elif not breaker_has_idm:
                classification = 'fake_bos'
            elif not breaking_swing.is_validated:
                classification = 'liquidity_sweep'
            elif breaking_swing.idm_sweep_type == 'body':
                classification = 'swing_hl'
            else:
                classification = 'bos'

            # Rules that can be checked now (1 + 3)
            rules_checked = is_broken_strong and breaker_has_idm

            if rules_checked:
                validity = 'confirmed'
                tag = '\u2713'
                desc_suffix = 'Rules 1+3 pass — awaiting R2 post-check'
            elif not is_broken_strong:
                validity = 'fake'
                tag = 'FAKE'
                desc_suffix = 'FAKE — weak swing broken (V9 Rule 1)'
            elif not breaker_has_idm:
                validity = 'fake'
                tag = 'FAKE'
                desc_suffix = 'FAKE — no IDM before break (V9 Rule 3)'
            else:
                validity = 'unconfirmed'
                tag = '?'
                desc_suffix = 'unconfirmed'

            reasoning_parts.append(f'Verdict: {validity.upper()} CHoCH')

            return StructureEvent(
                type=f'choch_{direction}',
                level=broken_level,
                timestamp=breaking_swing.timestamp,
                description=f'{direction.title()} CHoCH — {desc_suffix}',
                validity=validity,
                reasoning=' | '.join(reasoning_parts),
                label=f'CHoCH {arrow} {tag}',
                classification=classification
            )

        # CHoCH in bullish structure: Price breaks below previous HL (reversal)
        if structure == MarketStructure.BULLISH:
            if recent_lows[-1].price < recent_lows[-2].price:
                return _build_choch_event(
                    'bearish', recent_lows[-2], recent_lows[-1], recent_lows[-2].price
                )

        # CHoCH in bearish structure: Price breaks above previous LH (reversal)
        elif structure == MarketStructure.BEARISH:
            if recent_highs[-1].price > recent_highs[-2].price:
                return _build_choch_event(
                    'bullish', recent_highs[-2], recent_highs[-1], recent_highs[-2].price
                )

        return None

    def find_order_blocks(
        self,
        data: 'pd.DataFrame',
        structure: MarketStructure,
        swing_points: Optional[List[SwingPoint]] = None
    ) -> Tuple[List[OrderBlock], List[OrderBlock]]:
        """
        Find Order Blocks using ICT V14 rules.

        ICT Definition: OB = last opposite candle before a confirmed BOS.
        - Bullish OB = last bearish candle before bullish BOS (HH event)
        - Bearish OB = last bullish candle before bearish BOS (LL event)

        Validation chain (V5 → V14):
        - BOS must be valid (V5 Rule 1: IDM taken + Rule 2: body close beyond swing)
        - OB must be the LAST opposite candle before BOS
        - Mitigation lifecycle: fresh → partial → mitigated → invalid
        """
        order_blocks = []
        closes = data['close'].values
        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values

        # Get ML-learned parameters (with Tier 2 override support)
        ml_params = {}
        if self.ml_engine:
            ml_params = self.ml_engine.get_detection_parameters('order_block')
            confidence_multiplier = ml_params.get('confidence_multiplier', 0.5)
        else:
            confidence_multiplier = 0.5
        confidence_multiplier = self.params.get('ob_confidence_multiplier', confidence_multiplier)

        # Helper: format price for reasoning
        def _pfmt(p):
            return f'{p:.2f}' if p > 100 else f'{p:.5f}'

        # Helper: calculate mitigation lifecycle for an OB
        def _calc_mitigation(ob_high, ob_low, ob_type, from_index):
            ob_range = ob_high - ob_low
            if ob_range <= 0:
                return 0.0, 'fresh'
            max_penetration = 0.0
            for k in range(from_index, len(closes)):
                if ob_type == 'bullish':
                    # Bullish OB = support zone; mitigation = price dips into zone from above
                    if lows[k] <= ob_high:
                        penetration = (ob_high - lows[k]) / ob_range
                        max_penetration = max(max_penetration, min(penetration, 1.0))
                else:
                    # Bearish OB = resistance zone; mitigation = price rises into zone from below
                    if highs[k] >= ob_low:
                        penetration = (highs[k] - ob_low) / ob_range
                        max_penetration = max(max_penetration, min(penetration, 1.0))
            lifecycle = 'fresh'
            if max_penetration >= 1.0:
                lifecycle = 'invalid'
            elif max_penetration >= 0.5:
                lifecycle = 'mitigated'
            elif max_penetration > 0:
                lifecycle = 'partial'
            return max_penetration, lifecycle

        # Track used OB indices to prevent duplicates
        used_ob_indices = set()

        # ICT V14 approach: Find OBs from structure breaks (HH/LL events via swing_points)
        if swing_points and len(swing_points) >= 4:
            all_highs = [sp for sp in swing_points if sp.type == 'high']
            all_lows = [sp for sp in swing_points if sp.type == 'low']

            # Bullish OBs from HH events (bullish BOS)
            for i in range(1, len(all_highs)):
                if all_highs[i].price > all_highs[i - 1].price:  # HH = bullish BOS
                    bos_swing = all_highs[i]
                    bos_index = bos_swing.index

                    # BOS validity from swing validation (V5 rules)
                    bos_valid = 'confirmed' if bos_swing.is_validated else (
                        'unconfirmed' if bos_swing.has_idm else 'none'
                    )

                    # Find LAST bearish candle before BOS swing (search backward)
                    ob_candle_idx = None
                    for k in range(bos_index - 1, max(bos_index - 25, 0), -1):
                        if k < len(closes) and closes[k] < opens[k]:  # Bearish candle
                            ob_candle_idx = k
                            break

                    if ob_candle_idx is not None and ob_candle_idx not in used_ob_indices:
                        used_ob_indices.add(ob_candle_idx)
                        ob_high = float(highs[ob_candle_idx])
                        ob_low = float(lows[ob_candle_idx])

                        # Strength: how far did price move after OB
                        move = (bos_swing.price - ob_high) / ob_high if ob_high > 0 else 0
                        strength = min(abs(move) * 100, 1.0) * confidence_multiplier

                        # Mitigation lifecycle
                        mit_pct, lifecycle = _calc_mitigation(ob_high, ob_low, 'bullish', bos_index + 1)
                        mitigated = lifecycle in ('mitigated', 'invalid')

                        # Build per-OB reasoning (V14)
                        reasoning_parts = [f"Bullish OB {_pfmt(ob_low)}-{_pfmt(ob_high)}"]
                        reasoning_parts.append(f"BOS: HH at {_pfmt(bos_swing.price)} ({bos_valid})")
                        reasoning_parts.append('Last bearish candle before BOS \u2713')
                        reasoning_parts.append(f"Lifecycle: {lifecycle.upper()} (fill: {mit_pct:.0%})")

                        ob = OrderBlock(
                            start_index=ob_candle_idx,
                            end_index=ob_candle_idx,
                            high=ob_high,
                            low=ob_low,
                            type='bullish',
                            mitigated=mitigated,
                            timestamp=data.index[ob_candle_idx] if hasattr(data.index, '__getitem__') else None,
                            strength=strength,
                            bos_validity=bos_valid,
                            bos_type='higher_high',
                            is_last_opposite=True,
                            lifecycle=lifecycle,
                            mitigation_pct=mit_pct,
                            reasoning=' | '.join(reasoning_parts),
                        )
                        order_blocks.append(ob)

            # Bearish OBs from LL events (bearish BOS)
            for i in range(1, len(all_lows)):
                if all_lows[i].price < all_lows[i - 1].price:  # LL = bearish BOS
                    bos_swing = all_lows[i]
                    bos_index = bos_swing.index

                    bos_valid = 'confirmed' if bos_swing.is_validated else (
                        'unconfirmed' if bos_swing.has_idm else 'none'
                    )

                    # Find LAST bullish candle before BOS swing
                    ob_candle_idx = None
                    for k in range(bos_index - 1, max(bos_index - 25, 0), -1):
                        if k < len(closes) and closes[k] > opens[k]:  # Bullish candle
                            ob_candle_idx = k
                            break

                    if ob_candle_idx is not None and ob_candle_idx not in used_ob_indices:
                        used_ob_indices.add(ob_candle_idx)
                        ob_high = float(highs[ob_candle_idx])
                        ob_low = float(lows[ob_candle_idx])

                        move = (ob_low - bos_swing.price) / ob_low if ob_low > 0 else 0
                        strength = min(abs(move) * 100, 1.0) * confidence_multiplier

                        mit_pct, lifecycle = _calc_mitigation(ob_high, ob_low, 'bearish', bos_index + 1)
                        mitigated = lifecycle in ('mitigated', 'invalid')

                        reasoning_parts = [f"Bearish OB {_pfmt(ob_low)}-{_pfmt(ob_high)}"]
                        reasoning_parts.append(f"BOS: LL at {_pfmt(bos_swing.price)} ({bos_valid})")
                        reasoning_parts.append('Last bullish candle before BOS \u2713')
                        reasoning_parts.append(f"Lifecycle: {lifecycle.upper()} (fill: {mit_pct:.0%})")

                        ob = OrderBlock(
                            start_index=ob_candle_idx,
                            end_index=ob_candle_idx,
                            high=ob_high,
                            low=ob_low,
                            type='bearish',
                            mitigated=mitigated,
                            timestamp=data.index[ob_candle_idx] if hasattr(data.index, '__getitem__') else None,
                            strength=strength,
                            bos_validity=bos_valid,
                            bos_type='lower_low',
                            is_last_opposite=True,
                            lifecycle=lifecycle,
                            mitigation_pct=mit_pct,
                            reasoning=' | '.join(reasoning_parts),
                        )
                        order_blocks.append(ob)

        # Fallback: generic detection if no structure-based OBs found
        if not order_blocks:
            current_price = data['close'].iloc[-1]
            min_move_strength = 0.3
            if self.ml_engine:
                min_move_strength = ml_params.get('min_move_strength', 0.3)
            min_move_strength = self.params.get('ob_min_move_strength', min_move_strength)

            for i in range(2, len(data) - 1):
                is_bearish = closes[i] < opens[i]
                is_bullish = closes[i] > opens[i]
                next_close = closes[i + 1]
                c_high = float(highs[i])
                c_low = float(lows[i])

                if is_bearish and next_close > c_high:
                    move = (next_close - c_high) / c_high
                    strength = min(move * 100, 1.0)
                    if strength < min_move_strength:
                        continue
                    mit_pct, lifecycle = _calc_mitigation(c_high, c_low, 'bullish', i + 2)
                    mitigated = lifecycle in ('mitigated', 'invalid')
                    ob = OrderBlock(
                        start_index=i, end_index=i,
                        high=c_high, low=c_low, type='bullish',
                        mitigated=mitigated,
                        timestamp=data.index[i] if hasattr(data.index, '__getitem__') else None,
                        strength=strength * confidence_multiplier,
                        lifecycle=lifecycle, mitigation_pct=mit_pct,
                        reasoning=f"Bullish OB {_pfmt(c_low)}-{_pfmt(c_high)} | Generic detection (no BOS link)",
                    )
                    order_blocks.append(ob)

                if is_bullish and next_close < c_low:
                    move = (c_low - next_close) / c_low
                    strength = min(move * 100, 1.0)
                    if strength < min_move_strength:
                        continue
                    mit_pct, lifecycle = _calc_mitigation(c_high, c_low, 'bearish', i + 2)
                    mitigated = lifecycle in ('mitigated', 'invalid')
                    ob = OrderBlock(
                        start_index=i, end_index=i,
                        high=c_high, low=c_low, type='bearish',
                        mitigated=mitigated,
                        timestamp=data.index[i] if hasattr(data.index, '__getitem__') else None,
                        strength=strength * confidence_multiplier,
                        lifecycle=lifecycle, mitigation_pct=mit_pct,
                        reasoning=f"Bearish OB {_pfmt(c_low)}-{_pfmt(c_high)} | Generic detection (no BOS link)",
                    )
                    order_blocks.append(ob)

        # Sort by recency, separate unmitigated vs mitigated
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
        Map liquidity levels with ICT Video 2/6 validation.

        ICT Rules:
        - Buy-side liquidity: Above swing highs (buy stops, short SLs)
        - Sell-side liquidity: Below swing lows (sell stops, long SLs)
        - Validated swings = stronger liquidity pools
        - Sweep detection: wick vs body per Video 6 3 SMC Rules
        - Equal highs/lows = strongest pools (V2)
        """
        current_price = float(data['close'].iloc[-1])
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        opens = data['open'].values

        # Find equal highs/lows first (for is_equal_level marking)
        equal_highs = self._find_equal_levels(
            [sp for sp in swing_points if sp.type == 'high']
        )
        equal_lows = self._find_equal_levels(
            [sp for sp in swing_points if sp.type == 'low']
        )
        eq_high_levels = {round(eh['level'], 5) for eh in equal_highs}
        eq_low_levels = {round(el['level'], 5) for el in equal_lows}

        # Buy-side liquidity (above current price)
        buy_side = []
        for sp in swing_points:
            if sp.type == 'high' and sp.price > current_price:
                # Sweep detection: did any candle after this swing wick above it?
                swept = False
                sweep_type = 'none'
                for j in range(sp.index + 1, len(data)):
                    if highs[j] > sp.price:
                        body_high = max(closes[j], opens[j])
                        if body_high > sp.price:
                            swept = True
                            sweep_type = 'body'
                        else:
                            swept = True
                            sweep_type = 'wick'
                        break

                # Strength: validated swings = stronger pools
                base_strength = sp.strength / 10
                if sp.validity in ('validated', 'weak'):
                    base_strength += 0.2
                is_eq = any(abs(sp.price - lvl) / sp.price < 0.001 for lvl in eq_high_levels)
                if is_eq:
                    base_strength += 0.3

                reasoning = f"BSL ${sp.price:,.2f}" if sp.price > 100 else f"BSL {sp.price:.5f}"
                if is_eq:
                    reasoning += " (EQH)"
                if swept:
                    reasoning += f" SWEPT ({sweep_type})"
                else:
                    reasoning += " UNSWEPT"

                buy_side.append(LiquidityLevel(
                    price=sp.price,
                    type='buy_side',
                    strength=min(base_strength, 1.0),
                    timestamp=sp.timestamp,
                    swept=swept,
                    sweep_type=sweep_type,
                    swing_validity=sp.validity,
                    is_equal_level=is_eq,
                    reasoning=reasoning,
                ))

        # Sell-side liquidity (below current price)
        sell_side = []
        for sp in swing_points:
            if sp.type == 'low' and sp.price < current_price:
                # Sweep detection: did any candle after this swing wick below it?
                swept = False
                sweep_type = 'none'
                for j in range(sp.index + 1, len(data)):
                    if lows[j] < sp.price:
                        body_low = min(closes[j], opens[j])
                        if body_low < sp.price:
                            swept = True
                            sweep_type = 'body'
                        else:
                            swept = True
                            sweep_type = 'wick'
                        break

                # Strength: validated swings = stronger pools
                base_strength = sp.strength / 10
                if sp.validity in ('validated', 'weak'):
                    base_strength += 0.2
                is_eq = any(abs(sp.price - lvl) / sp.price < 0.001 for lvl in eq_low_levels)
                if is_eq:
                    base_strength += 0.3

                reasoning = f"SSL ${sp.price:,.2f}" if sp.price > 100 else f"SSL {sp.price:.5f}"
                if is_eq:
                    reasoning += " (EQL)"
                if swept:
                    reasoning += f" SWEPT ({sweep_type})"
                else:
                    reasoning += " UNSWEPT"

                sell_side.append(LiquidityLevel(
                    price=sp.price,
                    type='sell_side',
                    strength=min(base_strength, 1.0),
                    timestamp=sp.timestamp,
                    swept=swept,
                    sweep_type=sweep_type,
                    swing_validity=sp.validity,
                    is_equal_level=is_eq,
                    reasoning=reasoning,
                ))

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

    def find_engineered_liquidity(
        self,
        swing_points: List[SwingPoint],
        liquidity: Dict[str, List[LiquidityLevel]],
    ) -> List[Dict]:
        """
        Detect Engineered Liquidity zones (ICT Video 8/9).

        ENG LIQ = areas where retail S/R traders are trapped by Smart Money.
        Detection heuristics:
        1. Equal highs/lows clusters (strongest ENG LIQ — obvious retail S/R)
        2. Consolidation zones (3+ swings within tight range)

        Used by: Fake CHoCH Rule 2 (ENG LIQ creation = indicator of fake CHoCH)
        """
        eng_liq_zones = []

        # Type 1: Equal highs/lows as ENG LIQ (strongest signal)
        for eq in liquidity.get('equal_highs', []):
            eng_liq_zones.append({
                'type': 'equal_highs',
                'level': eq['level'],
                'count': eq.get('count', 2),
                'indices': eq.get('points', []),
                'strength': 0.8 + (eq.get('count', 2) - 2) * 0.1,
                'reasoning': f"ENG LIQ (EQH): Equal highs at {eq['level']:.5f} — retail resistance trap",
            })
        for eq in liquidity.get('equal_lows', []):
            eng_liq_zones.append({
                'type': 'equal_lows',
                'level': eq['level'],
                'count': eq.get('count', 2),
                'indices': eq.get('points', []),
                'strength': 0.8 + (eq.get('count', 2) - 2) * 0.1,
                'reasoning': f"ENG LIQ (EQL): Equal lows at {eq['level']:.5f} — retail support trap",
            })

        # Type 2: Consolidation zones (3+ swings within 2% range)
        if len(swing_points) >= 6:
            # Look at the most recent 20 swings for consolidation
            recent = swing_points[-20:]
            tolerance = 0.02  # 2% range = consolidation

            # Group swings by price proximity
            for i, sp1 in enumerate(recent):
                cluster = [sp1]
                for sp2 in recent[i + 1:]:
                    if abs(sp1.price - sp2.price) / sp1.price < tolerance:
                        cluster.append(sp2)
                if len(cluster) >= 3:
                    avg_price = sum(s.price for s in cluster) / len(cluster)
                    # Avoid duplicates: only add if not already covered by equal levels
                    already_covered = any(
                        abs(ez['level'] - avg_price) / avg_price < tolerance
                        for ez in eng_liq_zones
                    )
                    if not already_covered:
                        eng_liq_zones.append({
                            'type': 'consolidation',
                            'level': avg_price,
                            'count': len(cluster),
                            'indices': [s.index for s in cluster],
                            'strength': 0.5 + len(cluster) * 0.1,
                            'reasoning': f"ENG LIQ (Consolidation): {len(cluster)} swings near {avg_price:.5f} — range trap",
                        })

        return eng_liq_zones

    def calculate_premium_discount(
        self,
        data: 'pd.DataFrame',
        swing_points: List[SwingPoint],
        structure: 'MarketStructure' = None
    ) -> Dict:
        """
        Calculate premium/discount zones using ICT methodology (Video 12).

        ICT Rules:
        - Range = most recent validated swing high to validated swing low
        - Equilibrium = exact 50% midpoint (fair value line)
        - Premium = above 50% (expensive), Discount = below 50% (cheap)
        - Sub-zones: deep_discount (0-25%), discount (25-50%),
                     premium (50-75%), deep_premium (75-100%)
        - SM buys in discount, sells in premium
        - OTE = pullback to discount (bullish) or premium (bearish)
        """
        # Prefer validated/strong swings for range definition (ICT: use validated structure)
        validated_highs = [sp for sp in swing_points if sp.type == 'high'
                          and sp.validity in ('validated', 'weak')]
        validated_lows = [sp for sp in swing_points if sp.type == 'low'
                         and sp.validity in ('validated', 'weak')]

        # Fallback to all swings if no validated ones exist
        all_highs = validated_highs if validated_highs else [
            sp for sp in swing_points if sp.type == 'high']
        all_lows = validated_lows if validated_lows else [
            sp for sp in swing_points if sp.type == 'low']

        if not all_highs or not all_lows:
            return {
                'zone': 'neutral',
                'zone_simple': 'neutral',
                'sub_zone': 'neutral',
                'percentage': 50.0,
                'equilibrium': 0.0,
                'reasoning': 'Insufficient swing points for PD zone calculation'
            }

        # Use the ACTIVE swing range: most recent validated swing high + low
        # This defines the current trading range per ICT Video 12
        range_swing_high = all_highs[-1]
        range_swing_low = all_lows[-1]
        range_high = range_swing_high.price
        range_low = range_swing_low.price

        # Ensure range_high > range_low (swap if needed for proper zone calc)
        if range_high < range_low:
            range_high, range_low = range_low, range_high
            range_swing_high, range_swing_low = range_swing_low, range_swing_high

        equilibrium = (range_high + range_low) / 2
        current_price = float(data['close'].iloc[-1])

        # Calculate position in range (0-100%)
        range_size = range_high - range_low
        if range_size == 0:
            percentage = 50.0
        else:
            percentage = ((current_price - range_low) / range_size) * 100

        # ICT Zone determination: 50% is THE dividing line (Video 12)
        # Above 50% = premium, below 50% = discount
        if percentage >= 50:
            zone_simple = 'premium'
        else:
            zone_simple = 'discount'

        # Sub-zone classification (Video 12: finer precision)
        if percentage >= 75:
            sub_zone = 'deep_premium'
            zone = 'deep_premium'
        elif percentage >= 50:
            sub_zone = 'premium'
            zone = 'premium'
        elif percentage >= 25:
            sub_zone = 'discount'
            zone = 'discount'
        else:
            sub_zone = 'deep_discount'
            zone = 'deep_discount'

        # Build per-zone reasoning (ICT Video 12 rules)
        reasoning_parts = []
        price_fmt = f"${current_price:,.2f}" if current_price > 100 else f"{current_price:.5f}"
        eq_fmt = f"${equilibrium:,.2f}" if equilibrium > 100 else f"{equilibrium:.5f}"
        hi_fmt = f"${range_high:,.2f}" if range_high > 100 else f"{range_high:.5f}"
        lo_fmt = f"${range_low:,.2f}" if range_low > 100 else f"{range_low:.5f}"

        reasoning_parts.append(f"Range: {lo_fmt} → {hi_fmt} | EQ: {eq_fmt} | Price: {price_fmt} ({percentage:.1f}%)")

        # Swing quality note
        if validated_highs and validated_lows:
            reasoning_parts.append(f"Range defined by validated swings (ICT-confirmed)")
        else:
            reasoning_parts.append(f"Range from raw swings (no validated swings available)")

        # Zone-specific trading implications per ICT
        struct_val = structure.value if structure else 'consolidation'
        if struct_val == 'bullish':
            if zone_simple == 'discount':
                reasoning_parts.append("Bullish + Discount = IDEAL long zone (SM buys here)")
            else:
                reasoning_parts.append("Bullish + Premium = Avoid new longs (SM takes profit here)")
        elif struct_val == 'bearish':
            if zone_simple == 'premium':
                reasoning_parts.append("Bearish + Premium = IDEAL short zone (SM sells here)")
            else:
                reasoning_parts.append("Bearish + Discount = Avoid new shorts (SM covers here)")
        else:
            reasoning_parts.append("Consolidation = Wait for structure break before trading PD zones")

        reasoning = " | ".join(reasoning_parts)

        # Timestamps for range swing points (for frontend display)
        range_high_time = range_swing_high.timestamp
        range_low_time = range_swing_low.timestamp

        return {
            'zone': zone,
            'zone_simple': zone_simple,
            'sub_zone': sub_zone,
            'percentage': round(percentage, 2),
            'range_high': range_high,
            'range_low': range_low,
            'equilibrium': equilibrium,
            'current_price': current_price,
            'reasoning': reasoning,
            'range_high_validity': range_swing_high.validity,
            'range_low_validity': range_swing_low.validity,
            'range_high_time': range_high_time.isoformat() if range_high_time else None,
            'range_low_time': range_low_time.isoformat() if range_low_time else None,
        }

    def determine_bias(
        self,
        structure: MarketStructure,
        premium_discount: Dict,
        events: List[StructureEvent]
    ) -> Tuple[Bias, float, str]:
        """
        Determine overall market bias based on structure and PD zone position.

        ICT Video 12: SM buys in discount (bullish), sells in premium (bearish).
        Uses zone_simple (binary 50/50 split) for core bias determination,
        and sub_zone for confidence grading.

        Returns:
            Tuple of (bias, confidence, reasoning)
        """
        zone_simple = premium_discount.get('zone_simple', premium_discount.get('zone', 'neutral'))
        sub_zone = premium_discount.get('sub_zone', zone_simple)

        # Bullish structure
        if structure == MarketStructure.BULLISH:
            if sub_zone == 'deep_discount':
                return (
                    Bias.BULLISH,
                    0.85,
                    "Bullish structure + deep discount zone - strongest long setup (V12 OTE)"
                )
            elif sub_zone == 'discount':
                return (
                    Bias.BULLISH,
                    0.75,
                    "Bullish structure + discount zone - ideal long setup (SM buys here)"
                )
            elif sub_zone == 'premium':
                return (
                    Bias.NEUTRAL,
                    0.4,
                    "Bullish structure but price in premium - avoid new longs (SM takes profit)"
                )
            else:  # deep_premium
                return (
                    Bias.NEUTRAL,
                    0.3,
                    "Bullish structure but deep premium - high risk for longs (overextended)"
                )

        # Bearish structure
        elif structure == MarketStructure.BEARISH:
            if sub_zone == 'deep_premium':
                return (
                    Bias.BEARISH,
                    0.85,
                    "Bearish structure + deep premium zone - strongest short setup (V12 OTE)"
                )
            elif sub_zone == 'premium':
                return (
                    Bias.BEARISH,
                    0.75,
                    "Bearish structure + premium zone - ideal short setup (SM sells here)"
                )
            elif sub_zone == 'discount':
                return (
                    Bias.NEUTRAL,
                    0.4,
                    "Bearish structure but price in discount - avoid new shorts (SM covers)"
                )
            else:  # deep_discount
                return (
                    Bias.NEUTRAL,
                    0.3,
                    "Bearish structure but deep discount - high risk for shorts (oversold)"
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

    def _check_liquidity_sweep(
        self, data, idm_price, idm_index, parent_index, direction
    ):
        """
        ICT Video 3 Rule: Liquidity sweep is THE ONLY confirmation for valid IDM.

        Reference candle rule (Video 3):
        - Bullish: find highest-high candle before pullback -> its LOW is sweep target
        - Bearish: find lowest-low candle before pullback -> its HIGH is sweep target

        The pullback must break through the reference candle's level to be valid.
        Both wick-based and body-based sweeps are valid (Video 3).

        Returns dict: {'type': 'wick'|'body'|'none', 'index': int|None, 'reasoning': list}
        """
        opens = data['open'].values
        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        reasoning = []

        if idm_index <= 0:
            reasoning.append('IDM at edge of data - cannot determine reference candle')
            return {'type': 'none', 'index': None, 'reasoning': reasoning}

        # Find reference candle: peak of impulse before pullback (up to 10 candles back)
        search_back = max(0, idm_index - 10)

        if direction == 'bullish':
            # Reference = candle with highest HIGH before pullback (Video 3)
            ref_idx = idm_index - 1
            for k in range(idm_index - 2, search_back - 1, -1):
                if highs[k] > highs[ref_idx]:
                    ref_idx = k
            reference_level = lows[ref_idx]  # LOW of highest candle = sweep target

            # Check if pullback candle swept below reference level
            # Only check the pullback candle itself (it's the local minimum by definition)
            if lows[idm_index] < reference_level:
                body_low = min(closes[idm_index], opens[idm_index])
                if body_low < reference_level:
                    reasoning.append(f'BODY sweep: pullback closed below ref candle low')
                    reasoning.append(f'Ref: peak candle {ref_idx} (H={highs[ref_idx]:.5f}, L={reference_level:.5f})')
                    reasoning.append('Strongest confirmation - body closed below (Video 3)')
                    return {'type': 'body', 'index': idm_index, 'reasoning': reasoning}
                else:
                    reasoning.append(f'WICK sweep: pullback wicked below ref candle low')
                    reasoning.append(f'Ref: peak candle {ref_idx} (H={highs[ref_idx]:.5f}, L={reference_level:.5f})')
                    reasoning.append('Valid sweep - wick pierced, body held (Video 3)')
                    return {'type': 'wick', 'index': idm_index, 'reasoning': reasoning}

            reasoning.append(f'Pullback did NOT sweep below ref candle low ({reference_level:.5f})')
            reasoning.append(f'Ref: peak candle {ref_idx} (H={highs[ref_idx]:.5f}, L={reference_level:.5f})')
            reasoning.append('Unconfirmed - no liquidity sweep of reference level (Video 3)')
            return {'type': 'none', 'index': None, 'reasoning': reasoning}

        else:  # bearish
            # Reference = candle with lowest LOW before pullback (Video 3)
            ref_idx = idm_index - 1
            for k in range(idm_index - 2, search_back - 1, -1):
                if lows[k] < lows[ref_idx]:
                    ref_idx = k
            reference_level = highs[ref_idx]  # HIGH of lowest candle = sweep target

            # Check if pullback candle swept above reference level
            # Only check the pullback candle itself (it's the local maximum by definition)
            if highs[idm_index] > reference_level:
                body_high = max(closes[idm_index], opens[idm_index])
                if body_high > reference_level:
                    reasoning.append(f'BODY sweep: pullback closed above ref candle high')
                    reasoning.append(f'Ref: trough candle {ref_idx} (L={lows[ref_idx]:.5f}, H={reference_level:.5f})')
                    reasoning.append('Strongest confirmation - body closed above (Video 3)')
                    return {'type': 'body', 'index': idm_index, 'reasoning': reasoning}
                else:
                    reasoning.append(f'WICK sweep: pullback wicked above ref candle high')
                    reasoning.append(f'Ref: trough candle {ref_idx} (L={lows[ref_idx]:.5f}, H={reference_level:.5f})')
                    reasoning.append('Valid sweep - wick pierced, body held (Video 3)')
                    return {'type': 'wick', 'index': idm_index, 'reasoning': reasoning}

            reasoning.append(f'Pullback did NOT sweep above ref candle high ({reference_level:.5f})')
            reasoning.append(f'Ref: trough candle {ref_idx} (L={lows[ref_idx]:.5f}, H={reference_level:.5f})')
            reasoning.append('Unconfirmed - no liquidity sweep of reference level (Video 3)')
            return {'type': 'none', 'index': None, 'reasoning': reasoning}

    def _build_inducement(
        self, idm_type, idm_price, idm_index, parent_swing,
        data, is_impulse, sweep_type, sweep_index=None, reasoning_parts=None,
        depth=0.0
    ):
        """Build an inducement dict with validation status and per-IDM reasoning."""
        zone_width = idm_price * 0.003  # Tighter 0.3% zone for pin-point accuracy

        # Determine validity based on ICT rules
        if is_impulse:
            validity = 'impulse_trap'
            confidence = 0.6
        elif sweep_type == 'body':
            validity = 'valid'
            confidence = 0.85
        elif sweep_type == 'wick':
            validity = 'valid'
            confidence = 0.7
        else:
            validity = 'unconfirmed'
            confidence = 0.4

        # Build human-readable reasoning
        lines = reasoning_parts or []
        direction_str = 'Bullish' if idm_type == 'bullish' else 'Bearish'
        price_str = f'{idm_price:.2f}' if idm_price > 100 else f'{idm_price:.5f}'
        parent_str = f'{parent_swing.price:.2f}' if parent_swing.price > 100 else f'{parent_swing.price:.5f}'

        header = f'{direction_str} IDM @ {price_str}'
        if is_impulse:
            header += f' (impulse before swing {"high" if idm_type == "bullish" else "low"} @ {parent_str})'
        else:
            header += f' (pullback before swing {"high" if idm_type == "bullish" else "low"} @ {parent_str})'

        if depth > 0:
            lines.insert(0, f'Pullback depth: {depth:.1%}')
        lines.insert(0, header)
        lines.append(f'Validity: {validity.upper()} | Sweep: {sweep_type} | Confidence: {confidence:.0%}')

        # Build label for chart
        validity_tag = {'valid': '\u2713', 'unconfirmed': '?', 'impulse_trap': '\u26a1TRAP', 'shifted': '\u2197'}.get(validity, '')
        label = f'{direction_str[:4]} IDM {validity_tag}'

        return {
            'type': idm_type,
            'high': idm_price + zone_width,
            'low': idm_price - zone_width,
            'price': idm_price,
            'index': idm_index,
            'parent_swing_index': parent_swing.index,
            'parent_swing_price': float(parent_swing.price),
            'taken_out': sweep_type != 'none',  # Backward compat
            'timestamp': data.index[idm_index] if hasattr(data.index, '__getitem__') else None,
            # New ICT-rule fields
            'validity': validity,
            'sweep_type': sweep_type,
            'sweep_index': sweep_index,
            'is_impulse': is_impulse,
            'confidence': confidence,
            'depth': depth,
            'reasoning': ' | '.join(lines),
            'label': label,
        }

    def find_inducement(
        self,
        data: 'pd.DataFrame',
        swing_points: List[SwingPoint]
    ) -> List[Dict]:
        """
        Find inducement zones using full ICT rules from 16-video training.

        ICT Rules implemented:
        1. IDM = first pullback on left side of swing high/low (Video 1-2)
        2. LIQUIDITY SWEEP is THE ONLY confirmation (Video 3)
        3. Both wick and body sweeps are valid (Video 3)
        4. No sweep = unconfirmed IDM (Video 3)
        5. Impulse swing (no pullback) = low/high IS inducement (Video 4 Rule 3)
        6. Multiple IDMs: prioritize LARGEST, then MOST RECENT (Video 8)
        7. Inducement shifts when new high formed without sweep (Video 4 Rule 1)
        """
        inducements = []
        if len(swing_points) < 3:
            return inducements

        highs = data['high'].values
        lows = data['low'].values

        # ---- BULLISH INDUCEMENT (pullback LOW before swing HIGH) ----
        for i, sh in enumerate(swing_points):
            if sh.type != 'high':
                continue

            # Find previous swing low as search boundary
            prev_swing_low_idx = 0
            prev_swing_low_price = 0.0
            for sp in swing_points:
                if sp.type == 'low' and sp.index < sh.index:
                    prev_swing_low_idx = sp.index
                    prev_swing_low_price = sp.price

            if prev_swing_low_idx == 0 and sh.index < 5:
                continue

            search_start = max(prev_swing_low_idx + 1, sh.index - 20)

            # Collect ALL pullback candidates (Video 8: need all for size comparison)
            pullback_candidates = []
            for j in range(sh.index - 1, search_start - 1, -1):
                if j <= 0 or j >= len(data) - 1:
                    continue
                if lows[j] < lows[j - 1] and lows[j] < lows[j + 1]:
                    pullback_depth = (sh.price - lows[j]) / sh.price
                    if pullback_depth > 0.003:  # 0.3% minimum (more sensitive)
                        pullback_candidates.append({
                            'index': j,
                            'price': float(lows[j]),
                            'depth': pullback_depth,
                        })

            # RULE 7 (Video 4 Rule 3): Impulse swing = no pullback, low IS inducement
            if not pullback_candidates and prev_swing_low_idx > 0:
                impulse_depth = (sh.price - prev_swing_low_price) / sh.price
                if impulse_depth > 0.005:
                    inducements.append(self._build_inducement(
                        idm_type='bullish',
                        idm_price=prev_swing_low_price,
                        idm_index=prev_swing_low_idx,
                        parent_swing=sh,
                        data=data,
                        is_impulse=True,
                        sweep_type='none',
                        depth=impulse_depth,
                        reasoning_parts=[
                            'IMPULSE SWING - no pullback detected (Video 4 Rule 3)',
                            'The swing low itself IS the inducement (BIGGEST retail trap)',
                            'Retail sees this as valid HL/support but Smart Money knows it is IDM',
                        ],
                    ))
                continue

            if not pullback_candidates:
                continue

            # RULE 8 (Video 8): Sort by depth DESC, then index DESC (largest first, recent tiebreak)
            pullback_candidates.sort(key=lambda x: (-x['depth'], -x['index']))
            primary = pullback_candidates[0]

            # RULE 2-4 (Video 3): Liquidity sweep validation
            sweep = self._check_liquidity_sweep(
                data, primary['price'], primary['index'], sh.index, 'bullish'
            )

            inducements.append(self._build_inducement(
                idm_type='bullish',
                idm_price=primary['price'],
                idm_index=primary['index'],
                parent_swing=sh,
                data=data,
                is_impulse=False,
                sweep_type=sweep['type'],
                sweep_index=sweep.get('index'),
                depth=primary['depth'],
                reasoning_parts=sweep['reasoning'],
            ))

            # RULE 6 (Video 4 Rule 2): Wick sweep + multiple pullbacks = secondary IDM
            if sweep['type'] == 'wick' and len(pullback_candidates) > 1:
                secondary = pullback_candidates[1]
                sweep2 = self._check_liquidity_sweep(
                    data, secondary['price'], secondary['index'], sh.index, 'bullish'
                )
                inducements.append(self._build_inducement(
                    idm_type='bullish',
                    idm_price=secondary['price'],
                    idm_index=secondary['index'],
                    parent_swing=sh,
                    data=data,
                    is_impulse=False,
                    sweep_type=sweep2['type'],
                    sweep_index=sweep2.get('index'),
                    depth=secondary['depth'],
                    reasoning_parts=[
                        'SECONDARY IDM (Video 4 Rule 2: wick sweep created two levels)',
                    ] + sweep2['reasoning'],
                ))

        # ---- BEARISH INDUCEMENT (pullback HIGH before swing LOW) ----
        for i, sl in enumerate(swing_points):
            if sl.type != 'low':
                continue

            prev_swing_high_idx = 0
            prev_swing_high_price = 0.0
            for sp in swing_points:
                if sp.type == 'high' and sp.index < sl.index:
                    prev_swing_high_idx = sp.index
                    prev_swing_high_price = sp.price

            if prev_swing_high_idx == 0 and sl.index < 5:
                continue

            search_start = max(prev_swing_high_idx + 1, sl.index - 20)

            pullback_candidates = []
            for j in range(sl.index - 1, search_start - 1, -1):
                if j <= 0 or j >= len(data) - 1:
                    continue
                if highs[j] > highs[j - 1] and highs[j] > highs[j + 1]:
                    pullback_depth = (highs[j] - sl.price) / sl.price
                    if pullback_depth > 0.003:
                        pullback_candidates.append({
                            'index': j,
                            'price': float(highs[j]),
                            'depth': pullback_depth,
                        })

            # Impulse detection (bearish)
            if not pullback_candidates and prev_swing_high_idx > 0:
                impulse_depth = (prev_swing_high_price - sl.price) / sl.price
                if impulse_depth > 0.005:
                    inducements.append(self._build_inducement(
                        idm_type='bearish',
                        idm_price=prev_swing_high_price,
                        idm_index=prev_swing_high_idx,
                        parent_swing=sl,
                        data=data,
                        is_impulse=True,
                        sweep_type='none',
                        depth=impulse_depth,
                        reasoning_parts=[
                            'IMPULSE SWING - no pullback detected (Video 4 Rule 3)',
                            'The swing high itself IS the inducement (BIGGEST retail trap)',
                            'Retail sees this as valid LH/resistance but Smart Money knows it is IDM',
                        ],
                    ))
                continue

            if not pullback_candidates:
                continue

            pullback_candidates.sort(key=lambda x: (-x['depth'], -x['index']))
            primary = pullback_candidates[0]

            sweep = self._check_liquidity_sweep(
                data, primary['price'], primary['index'], sl.index, 'bearish'
            )

            inducements.append(self._build_inducement(
                idm_type='bearish',
                idm_price=primary['price'],
                idm_index=primary['index'],
                parent_swing=sl,
                data=data,
                is_impulse=False,
                sweep_type=sweep['type'],
                sweep_index=sweep.get('index'),
                depth=primary['depth'],
                reasoning_parts=sweep['reasoning'],
            ))

            if sweep['type'] == 'wick' and len(pullback_candidates) > 1:
                secondary = pullback_candidates[1]
                sweep2 = self._check_liquidity_sweep(
                    data, secondary['price'], secondary['index'], sl.index, 'bearish'
                )
                inducements.append(self._build_inducement(
                    idm_type='bearish',
                    idm_price=secondary['price'],
                    idm_index=secondary['index'],
                    parent_swing=sl,
                    data=data,
                    is_impulse=False,
                    sweep_type=sweep2['type'],
                    sweep_index=sweep2.get('index'),
                    depth=secondary['depth'],
                    reasoning_parts=[
                        'SECONDARY IDM (Video 4 Rule 2: wick sweep created two levels)',
                    ] + sweep2['reasoning'],
                ))

        # RULE 5 (Video 4 Rule 1): Mark shifted IDMs
        # Only mark as shifted if a NEW higher high (bullish) or lower low (bearish)
        # formed beyond the parent swing without sweeping this IDM
        for idm in inducements:
            if idm['validity'] == 'unconfirmed':
                parent_idx = idm['parent_swing_index']
                parent_price = idm['parent_swing_price']
                expected_type = 'high' if idm['type'] == 'bullish' else 'low'
                for sp in swing_points:
                    if sp.index <= parent_idx or sp.type != expected_type:
                        continue
                    # Only shift if new swing exceeds parent (HH for bullish, LL for bearish)
                    if (idm['type'] == 'bullish' and sp.price > parent_price) or \
                       (idm['type'] == 'bearish' and sp.price < parent_price):
                        idm['validity'] = 'shifted'
                        idm['confidence'] = 0.3
                        idm['reasoning'] += ' | SHIFTED: New swing formed without sweeping this IDM (Video 4 Rule 1)'
                        label_dir = idm['label'][:4]
                        idm['label'] = f'{label_dir} IDM \u2197'
                        break

        # Sort by priority: valid > impulse > unconfirmed > shifted, then by recency
        validity_order = {'valid': 0, 'impulse_trap': 1, 'unconfirmed': 2, 'shifted': 3}
        inducements.sort(key=lambda x: (validity_order.get(x['validity'], 9), -x['index']))
        return inducements[:20]

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
