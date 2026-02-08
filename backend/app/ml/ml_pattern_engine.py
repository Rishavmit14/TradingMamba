"""
ML Pattern Engine - Uses ONLY learned knowledge from trained videos

This engine reads the ML's visual knowledge and uses it to:
1. Detect patterns based on the LOGIC ML learned from ICT videos
2. Use learned CHARACTERISTICS to identify patterns universally
3. Track which pattern TYPES ML understands (not specific instances)

=============================================================================
CRITICAL PHILOSOPHY - READ THIS BEFORE MODIFYING:
=============================================================================

ML learns HOW TO DRAW/DETECT patterns, not specific pattern instances.

Example: If ML studies 1000 FVGs across 100 videos:
- It learns WHAT an FVG is (gap between candle 1 high and candle 3 low)
- It learns HOW to identify FVGs on any chart
- It learns the CHARACTERISTICS (bullish/bearish, significance levels)
- It does NOT memorize those 1000 specific FVGs

This means:
- Frequency = how much ML studied (improves understanding, NOT a limit)
- After training on 100 videos with 5000 FVGs, ML can detect ANY FVG anywhere
- The same applies to Order Blocks, Breaker Blocks, Liquidity, etc.

SCALING CONSIDERATIONS:
- As training grows (100s of videos, 1000s of patterns), ML gets BETTER
- More examples = deeper understanding of edge cases and variations
- Teaching contexts accumulate = richer explanations of WHY patterns matter
- NEVER use frequency to LIMIT detection - that defeats the purpose

If ML hasn't learned a pattern TYPE at all, it won't detect it.
But once learned, ML applies that knowledge universally to ANY chart.
=============================================================================
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LearnedPattern:
    """
    A pattern type learned from video training.

    IMPORTANT: This represents ML's understanding of a PATTERN TYPE,
    not specific instances. ML learns the LOGIC of how to identify
    this pattern, then applies it universally.

    - frequency: How many examples ML studied (more = better understanding)
    - characteristics: The traits ML learned to look for (bullish/bearish, etc.)
    - teaching_contexts: WHY this pattern matters (ICT's explanations)
    """
    pattern_type: str
    frequency: int  # How many examples ML studied (NOT a detection limit!)
    confidence: float  # ML's understanding level of this pattern type
    characteristics: List[str] = field(default_factory=list)
    example_locations: List[str] = field(default_factory=list)  # Historical examples for reference
    teaching_contexts: List[str] = field(default_factory=list)
    visual_example_path: Optional[str] = None

    # Learned pattern logic
    learned_traits: Dict[str, Any] = field(default_factory=dict)  # bullish/bearish, significance levels, etc.


@dataclass
class MLKnowledgeBase:
    """The ML's complete knowledge from training"""
    patterns_learned: Dict[str, LearnedPattern] = field(default_factory=dict)
    total_videos_trained: int = 0
    total_frames_analyzed: int = 0
    total_chart_frames: int = 0
    training_sources: List[str] = field(default_factory=list)
    last_trained: Optional[datetime] = None

    def has_learned(self, pattern_type: str) -> bool:
        """Check if ML has learned a specific pattern type"""
        # Normalize pattern names (e.g., "Order Blocks" and "Order Block" are the same)
        normalized = self._normalize_pattern_type(pattern_type)
        return normalized in self.patterns_learned

    def get_confidence(self, pattern_type: str) -> float:
        """Get ML's confidence for detecting a pattern type"""
        normalized = self._normalize_pattern_type(pattern_type)
        if normalized in self.patterns_learned:
            return self.patterns_learned[normalized].confidence
        return 0.0  # No confidence if not learned

    def _normalize_pattern_type(self, pattern_type: str) -> str:
        """Normalize pattern type names"""
        # Convert to lowercase and handle plurals
        normalized = pattern_type.lower().strip()

        # Map variations to standard names
        mappings = {
            # Order Blocks
            'order blocks': 'order_block',
            'order block': 'order_block',
            'ob': 'order_block',
            'valid order block': 'valid_order_block',
            'valid_order_block': 'valid_order_block',
            'valid order block rules': 'valid_order_block',
            'valid_order_block_rules': 'valid_order_block',
            'valid ob': 'valid_order_block',
            'order block mitigation': 'order_block_mitigation',
            'order_block_mitigation': 'order_block_mitigation',
            'order_block_mitigation_and_trade_entry': 'order_block_mitigation',
            # Fair Value Gap
            'fvg': 'fvg',
            'fair value gap': 'fvg',
            'fair value gaps': 'fvg',
            'fair_value_gap': 'fvg',
            'valid fvg': 'fvg',
            'valid_fvg_confirmation_rules': 'fvg',
            # Breaker Block
            'breaker block': 'breaker_block',
            'breaker blocks': 'breaker_block',
            'breaker': 'breaker_block',
            # Support/Resistance
            'support/resistance': 'support_resistance',
            'support': 'support_resistance',
            'resistance': 'support_resistance',
            # Liquidity
            'liquidity': 'liquidity',
            'buy side liquidity': 'liquidity',
            'sell side liquidity': 'liquidity',
            # Liquidity Sweep
            'liquidity sweep': 'liquidity_sweep',
            'liquidity_sweep': 'liquidity_sweep',
            'sweep': 'liquidity_sweep',
            'liquidity grab': 'liquidity_sweep',
            # Market Structure
            'market structure': 'market_structure',
            'market_structure': 'market_structure',
            # Break of Structure
            'bos': 'break_of_structure',
            'break of structure': 'break_of_structure',
            'break_of_structure': 'break_of_structure',
            'structure break': 'break_of_structure',
            'bos validation': 'bos_validation',
            'bos_validation': 'bos_validation',
            'bos_validation_rules': 'bos_validation',
            'single_candle_bos': 'bos_validation',
            # Change of Character
            'choch': 'change_of_character',
            'change of character': 'change_of_character',
            'change_of_character': 'change_of_character',
            'change_of_character_choch': 'change_of_character',
            'choch confirmation': 'choch_confirmation',
            'choch_confirmation': 'choch_confirmation',
            'choch_confirmation_overview': 'choch_confirmation',
            # Fake CHoCH
            'fake choch': 'fake_choch',
            'fake_choch': 'fake_choch',
            'fake change of character': 'fake_choch',
            # Optimal Trade Entry / Fibonacci
            'optimal trade entry': 'optimal_trade_entry',
            'ote': 'optimal_trade_entry',
            'optimal_trade_entry': 'optimal_trade_entry',
            'fibonacci': 'fibonacci_ote',
            'fib': 'fibonacci_ote',
            'fibonacci retracement': 'fibonacci_ote',
            'fibonacci_ote': 'fibonacci_ote',
            # Swing Points
            'swing high': 'swing_high_low',
            'swing low': 'swing_high_low',
            'swing high/low': 'swing_high_low',
            'swing_high_low': 'swing_high_low',
            'weak swing point': 'weak_swing_point',
            'weak_swing_point': 'weak_swing_point',
            'weak_swing_point_rule': 'weak_swing_point',
            # Equal Highs/Lows
            'equal highs': 'equal_highs_lows',
            'equal lows': 'equal_highs_lows',
            'equal_highs_lows': 'equal_highs_lows',
            # Displacement
            'displacement': 'displacement',
            # Kill Zones
            'kill zone': 'kill_zone',
            'kill zones': 'kill_zone',
            'kill_zone': 'kill_zone',
            # Institutional
            'institutional': 'institutional',
            # Buy/Sell Stops
            'sell stops': 'sell_stops',
            'sell_stops': 'sell_stops',
            'buy stops': 'buy_stops',
            'buy_stops': 'buy_stops',
            # Higher High / Higher Low / Lower Low / Lower High
            'higher high': 'higher_high',
            'higher_high': 'higher_high',
            'higher low': 'higher_low',
            'higher_low': 'higher_low',
            'lower low': 'lower_low',
            'lower_low': 'lower_low',
            'lower high': 'lower_high',
            'lower_high': 'lower_high',
            # Smart Money
            'smart money': 'smart_money',
            'smart_money': 'smart_money',
            # Inducement
            'inducement': 'inducement',
            'idm': 'inducement',
            'valid inducement': 'inducement',
            'valid_inducement': 'inducement',
            'inducement shift': 'inducement_shift',
            'inducement_shift': 'inducement_shift',
            # Pullback
            'pullback': 'valid_pullback',
            'valid pullback': 'valid_pullback',
            'valid_pullback': 'valid_pullback',
            # Smart Money Traps
            'smart money trap': 'smart_money_trap',
            'smart_money_trap': 'smart_money_trap',
            'smart_money_traps': 'smart_money_trap',
            'retail trap': 'smart_money_trap',
            'retail_trap': 'smart_money_trap',
            # Premium / Discount
            'premium zone': 'premium_discount',
            'premium_zone': 'premium_discount',
            'discount zone': 'premium_discount',
            'discount_zone': 'premium_discount',
            'premium': 'premium_discount',
            'discount': 'premium_discount',
            'premium_and_discount_zones': 'premium_discount',
            # Equilibrium
            'equilibrium': 'equilibrium',
            'equilibrium_level': 'equilibrium',
            # Price Delivery Cycle
            'price delivery cycle': 'price_delivery_cycle',
            'price_delivery_cycle': 'price_delivery_cycle',
            'price cycle': 'price_delivery_cycle',
            # Expansion / Retracement
            'expansion': 'expansion',
            'retracement': 'retracement',
            # Judas Swing
            'judas swing': 'judas_swing',
            'judas_swing': 'judas_swing',
            'judas': 'judas_swing',
            'judas_swing_and_liquidity_sweep': 'judas_swing',
            # Forex Sessions
            'forex sessions': 'forex_sessions',
            'forex_sessions': 'forex_sessions',
            'forex_trading_sessions': 'forex_sessions',
            # Millions Dollar Setup
            'millions dollar setup': 'millions_dollar_setup',
            'millions_dollar_setup': 'millions_dollar_setup',
            'millions_dollar_setup_overview': 'millions_dollar_setup',
            # Engineered Liquidity
            'engineered liquidity': 'engineered_liquidity',
            'engineered_liquidity': 'engineered_liquidity',
            # Candlestick Analysis
            'candlestick analysis': 'candlestick_analysis',
            'candlestick_analysis': 'candlestick_analysis',
            # Manipulation / Accumulation / Distribution
            'manipulation': 'manipulation',
            'accumulation': 'accumulation',
            'distribution': 'distribution',
            # Risk Management
            'risk_reward_management': 'risk_management',
            'risk management': 'risk_management',
            # Swing Failure Pattern
            'swing_failure_pattern': 'swing_failure_pattern',
            'sfp': 'swing_failure_pattern',
            # Three SMC Rules
            'three_smc_rules': 'three_smc_rules',
            'three_smc_rules_framework': 'three_smc_rules',
            # High Probability Inducement
            'high_probability_inducement': 'high_probability_inducement',
            'high probability inducement': 'high_probability_inducement',
            # Power of Three
            'power of three': 'power_of_three',
            'power_of_three': 'power_of_three',
            'po3': 'power_of_three',
        }

        return mappings.get(normalized, normalized.replace(' ', '_'))


class MLPatternEngine:
    """
    Engine that uses ONLY ML's learned knowledge for pattern detection.

    KEY PHILOSOPHY:
    1. ML learns the LOGIC of patterns from ICT videos
    2. Once learned, ML can detect that pattern type UNIVERSALLY on any chart
    3. Frequency counts show how much ML studied, NOT detection limits
    4. Characteristics (bullish/bearish, significance) guide detection quality

    Example: If ML saw 31 FVG examples, it learned WHAT an FVG is.
    Now ML can detect ANY FVG on any chart - not just those 31 specific ones.

    Key Principles:
    1. If ML hasn't learned a pattern TYPE at all, it won't detect it
    2. Confidence is based on quality of understanding (teaching contexts + examples)
    3. Detection uses learned CHARACTERISTICS, not frequency limits
    """

    def __init__(self, data_dir: str = None, video_ids: Optional[List[str]] = None):
        """
        Initialize the ML Pattern Engine.

        Args:
            data_dir: Path to data directory. If None, auto-resolved from project root.
            video_ids: Optional list of video IDs to filter knowledge loading.
                       If None, loads ALL videos (backward compatible).
                       If provided, only loads knowledge from those specific videos.
        """
        # Resolve the data directory relative to the project root
        if data_dir is None:
            # Go up from backend/app/ml/ to backend/ then to project root
            project_root = Path(__file__).parent.parent.parent.parent
            self.data_dir = project_root / "data"
        else:
            self.data_dir = Path(data_dir)

        self.vision_dir = self.data_dir / "video_vision"
        self.knowledge_base: Optional[MLKnowledgeBase] = None
        self._video_ids = set(video_ids) if video_ids else None  # None = load all
        self._load_knowledge()

    def _extract_video_id_from_file(self, filepath: Path) -> str:
        """Extract video_id from a training file path.

        Examples:
            E9F_aT9f038_vision.json -> E9F_aT9f038
            -k1eBYoajIo_knowledge_base.json -> -k1eBYoajIo
        """
        stem = filepath.stem
        for suffix in ['_vision', '_knowledge_base', '_teaching_units',
                       '_vision_analysis', '_vision_progress',
                       '_selected_frames', '_summary', '_knowledge_summary']:
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        return stem

    def _should_load_file(self, filepath: Path) -> bool:
        """Check if a file should be loaded based on video_ids filter."""
        if self._video_ids is None:
            return True  # No filter, load everything
        vid = self._extract_video_id_from_file(filepath)
        return vid in self._video_ids

    def _load_knowledge(self):
        """Load all ML knowledge from trained video analyses"""
        self.knowledge_base = MLKnowledgeBase()

        if not self.vision_dir.exists():
            logger.warning(f"Vision directory not found: {self.vision_dir}")
            logger.warning("ML has NO knowledge - no patterns will be detected!")
            return

        # Load all vision analysis files (filtered by playlist if specified)
        vision_files = [f for f in self.vision_dir.glob("*_vision.json")
                        if self._should_load_file(f)]

        if not vision_files:
            if self._video_ids is not None:
                logger.info(f"No vision files match playlist filter ({len(self._video_ids)} video IDs)")
            else:
                logger.warning("No vision training files found - ML has NO knowledge!")
            # Don't return yet - audio-first training may still have knowledge

        all_patterns = {}
        total_frames = 0
        total_charts = 0
        sources = []
        latest_trained = None

        for vision_file in vision_files:
            try:
                with open(vision_file, 'r') as f:
                    data = json.load(f)

                video_id = data.get('video_id', vision_file.stem)
                title = data.get('title', video_id)
                sources.append(f"{title} ({video_id})")

                total_frames += data.get('total_frames_analyzed', 0)
                total_charts += data.get('chart_frames', 0)

                # Parse analyzed timestamp
                if 'analyzed_at' in data:
                    analyzed = datetime.fromisoformat(data['analyzed_at'])
                    if latest_trained is None or analyzed > latest_trained:
                        latest_trained = analyzed

                # Extract pattern frequencies
                pattern_freq = data.get('pattern_frequency', {})
                for pattern_type, count in pattern_freq.items():
                    normalized = self.knowledge_base._normalize_pattern_type(pattern_type)
                    if normalized not in all_patterns:
                        all_patterns[normalized] = {
                            'frequency': 0,
                            'characteristics': [],
                            'locations': [],
                            'teaching_contexts': [],
                            'visual_path': None
                        }
                    all_patterns[normalized]['frequency'] += count

                # Extract characteristics and teaching contexts
                for pattern in data.get('all_patterns', []):
                    pattern_type = pattern.get('type', '')
                    normalized = self.knowledge_base._normalize_pattern_type(pattern_type)

                    if normalized in all_patterns:
                        if pattern.get('characteristic'):
                            all_patterns[normalized]['characteristics'].append(
                                pattern['characteristic']
                            )
                        if pattern.get('location'):
                            all_patterns[normalized]['locations'].append(
                                pattern['location']
                            )

                # Extract teaching contexts from key moments
                for moment in data.get('key_moments', []):
                    for pattern in moment.get('patterns', []):
                        pattern_type = pattern.get('type', '')
                        normalized = self.knowledge_base._normalize_pattern_type(pattern_type)

                        if normalized in all_patterns:
                            if moment.get('teaching_point'):
                                all_patterns[normalized]['teaching_contexts'].append(
                                    moment['teaching_point']
                                )

                # Extract visual example paths
                for concept in data.get('visual_concepts', []):
                    concept_name = concept.get('concept', '')
                    normalized = self.knowledge_base._normalize_pattern_type(concept_name)

                    if normalized in all_patterns and concept.get('frame_path'):
                        all_patterns[normalized]['visual_path'] = concept['frame_path']

            except Exception as e:
                logger.error(f"Error loading vision file {vision_file}: {e}")
                continue

        # Calculate UNDERSTANDING LEVEL based on training depth
        # NOTE: This is about HOW WELL ML understands the pattern logic,
        # NOT a limit on detection. More examples = better understanding.
        total_pattern_observations = sum(p['frequency'] for p in all_patterns.values())

        for pattern_type, data in all_patterns.items():
            # Confidence = ML's understanding level of this pattern TYPE
            # This affects detection QUALITY, not quantity
            # More examples studied = higher confidence in edge cases
            #
            # Example: If ML saw 1000 FVGs across 100 videos:
            # - High confidence that it understands FVG variations
            # - Can accurately detect ANY FVG on any chart
            # - Better at distinguishing valid vs invalid FVGs

            if total_pattern_observations > 0:
                # More examples = deeper understanding (but diminishing returns)
                base_confidence = min(data['frequency'] / total_pattern_observations * 2, 0.7)
            else:
                base_confidence = 0.1

            # Teaching contexts from ICT videos boost understanding
            # Each context adds nuance to when/why patterns matter
            teaching_boost = min(len(data['teaching_contexts']) * 0.05, 0.25)

            confidence = min(base_confidence + teaching_boost, 0.95)

            # Deduplicate lists - handle both string and dict items
            def unique_items(items, max_items=10):
                seen = set()
                result = []
                for item in items:
                    # Convert dicts to strings for deduplication
                    key = str(item) if isinstance(item, dict) else item
                    if key not in seen:
                        seen.add(key)
                        result.append(item if isinstance(item, str) else str(item))
                        if len(result) >= max_items:
                            break
                return result

            self.knowledge_base.patterns_learned[pattern_type] = LearnedPattern(
                pattern_type=pattern_type,
                frequency=data['frequency'],
                confidence=confidence,
                characteristics=unique_items(data['characteristics'], 10),
                example_locations=unique_items(data['locations'], 5),
                teaching_contexts=unique_items(data['teaching_contexts'], 5),
                visual_example_path=data['visual_path']
            )

        # NOTE: knowledge_base.json from transcript training is NOT loaded as patterns_learned.
        # It contains generic topic categories (e.g., "institutional", "price_action", "entry_models")
        # from keyword matching on transcripts — these are NOT actual chart patterns that the ML
        # learned to detect from video analysis. Only audio-first training (Claude Code expert
        # analysis) and vision training produce real, actionable patterns.
        # The knowledge_base.json data is still used by the SmartMoneyKnowledgeBase for concept
        # classification and text-based queries, but should not appear in the Pattern Filter.

        # =========================================================================
        # AUDIO-FIRST TRAINING KNOWLEDGE (from 20 ICT video trainings)
        # These contain rich LLM-generated summaries for each concept
        # =========================================================================
        audio_first_dir = self.data_dir / 'audio_first_training'
        audio_first_videos = 0
        if audio_first_dir.exists():
            kb_files = [f for f in audio_first_dir.glob("*_knowledge_base.json")
                        if self._should_load_file(f)]
            logger.info(f"Loading audio-first training from {len(kb_files)} knowledge bases"
                        f"{' (playlist-filtered)' if self._video_ids is not None else ''}...")

            for kb_file in kb_files:
                try:
                    with open(kb_file, 'r') as f:
                        kb_data = json.load(f)

                    video_id = kb_data.get('video_id', kb_file.stem.replace('_knowledge_base', ''))
                    audio_first_videos += 1

                    # Track as training source
                    source_label = f"Audio-First Training: {video_id}"
                    if source_label not in sources:
                        sources.append(source_label)

                    # Parse timestamp
                    if 'generated_at' in kb_data:
                        try:
                            generated = datetime.fromisoformat(kb_data['generated_at'])
                            if latest_trained is None or generated > latest_trained:
                                latest_trained = generated
                        except (ValueError, TypeError):
                            pass

                    # Extract concepts from this video's knowledge base
                    concepts = kb_data.get('concepts', {})
                    stats = kb_data.get('processing_stats', {})

                    total_frames += stats.get('vision_analyses', 0)

                    for concept_name, concept_data in concepts.items():
                        normalized = self.knowledge_base._normalize_pattern_type(concept_name)

                        llm_summary = concept_data.get('llm_summary', '')
                        concept_stats = concept_data.get('statistics', {})
                        teaching_types = concept_data.get('teaching_types', {})

                        # Calculate frequency from teaching units
                        teaching_units = concept_stats.get('teaching_units', 1)
                        frames_analyzed = concept_stats.get('frames_analyzed', 0)

                        # Detect if this is expert-trained data (Claude Code)
                        generation_method = kb_data.get('generation_method', '')
                        is_expert = generation_method == "Claude Code expert analysis"

                        if normalized in self.knowledge_base.patterns_learned:
                            # MERGE with existing pattern - audio-first supplements
                            existing = self.knowledge_base.patterns_learned[normalized]
                            existing.frequency += teaching_units

                            # Store LLM summary in learned_traits
                            # Prefer Claude Code expert summaries over MLX-VLM output
                            existing_summary = existing.learned_traits.get('llm_summary', '')
                            existing_is_expert = existing.learned_traits.get('expert_trained', False)
                            if is_expert and not existing_is_expert:
                                # Expert data always wins over non-expert
                                existing.learned_traits['llm_summary'] = llm_summary
                                existing.learned_traits['expert_trained'] = True
                            elif len(llm_summary) > len(existing_summary) and not existing_is_expert:
                                existing.learned_traits['llm_summary'] = llm_summary

                            # Add teaching context from summary (first 120 chars)
                            if llm_summary and llm_summary[:120] not in existing.teaching_contexts:
                                existing.teaching_contexts.append(llm_summary[:120])
                                existing.teaching_contexts = existing.teaching_contexts[:8]

                            # Accumulate stats
                            existing.learned_traits['total_teaching_seconds'] = (
                                existing.learned_traits.get('total_teaching_seconds', 0) +
                                concept_stats.get('teaching_duration_seconds', 0)
                            )
                            existing.learned_traits['total_words'] = (
                                existing.learned_traits.get('total_words', 0) +
                                concept_stats.get('word_count', 0)
                            )
                            existing.learned_traits['total_frames'] = (
                                existing.learned_traits.get('total_frames', 0) + frames_analyzed
                            )
                            existing.learned_traits['deictic_references'] = (
                                existing.learned_traits.get('deictic_references', 0) +
                                concept_stats.get('deictic_references', 0)
                            )
                            existing.learned_traits['video_ids'] = list(set(
                                existing.learned_traits.get('video_ids', []) + [video_id]
                            ))

                            # Boost confidence from audio-first training
                            existing.confidence = min(existing.confidence + 0.05, 0.95)

                            # Add characteristic
                            char = f"Audio-first trained ({teaching_units} teaching units, {frames_analyzed} frames)"
                            if char not in existing.characteristics:
                                existing.characteristics.append(char)

                        else:
                            # NEW pattern from audio-first training
                            # Graduated confidence formula based on training depth
                            confidence = 0.50  # Base for audio-first
                            if teaching_units >= 3:
                                confidence += 0.15
                            if frames_analyzed >= 5:
                                confidence += 0.10
                            if concept_stats.get('word_count', 0) >= 200:
                                confidence += 0.05
                            if concept_stats.get('deictic_references', 0) >= 5:
                                confidence += 0.05
                            # Bonus for expert-trained data (Claude Code)
                            if is_expert:
                                confidence += 0.10
                            confidence = min(confidence, 0.95)

                            # Build teaching contexts from LLM summary
                            teaching_contexts = []
                            if llm_summary:
                                # Take first sentence as primary teaching context
                                first_sentence = llm_summary.split('.')[0] + '.'
                                if len(first_sentence) > 10:
                                    teaching_contexts.append(first_sentence[:150])

                            # Build characteristics
                            characteristics = [
                                f"Audio-first trained ({teaching_units} teaching units, {frames_analyzed} frames)",
                                f"Learned from ICT video {video_id}",
                            ]
                            if is_expert:
                                characteristics.append("Expert-trained via Claude Code")

                            # Build learned traits
                            learned_traits = {
                                'llm_summary': llm_summary,
                                'total_teaching_seconds': concept_stats.get('teaching_duration_seconds', 0),
                                'total_words': concept_stats.get('word_count', 0),
                                'total_frames': frames_analyzed,
                                'deictic_references': concept_stats.get('deictic_references', 0),
                                'teaching_types': teaching_types,
                                'video_ids': [video_id],
                                'audio_first': True,
                                'expert_trained': is_expert,
                            }

                            self.knowledge_base.patterns_learned[normalized] = LearnedPattern(
                                pattern_type=normalized,
                                frequency=teaching_units,
                                confidence=confidence,
                                characteristics=characteristics,
                                example_locations=[],
                                teaching_contexts=teaching_contexts,
                                visual_example_path=None,
                                learned_traits=learned_traits,
                            )

                            all_patterns[normalized] = {'frequency': teaching_units}
                            logger.info(f"  + Audio-first concept: {normalized} (confidence={confidence:.2f}, "
                                       f"units={teaching_units}, frames={frames_analyzed})")

                except Exception as e:
                    logger.error(f"Error loading audio-first KB {kb_file}: {e}")
                    continue

            logger.info(f"Audio-first training: loaded {audio_first_videos} videos")

        self.knowledge_base.total_videos_trained = len(vision_files) + audio_first_videos
        self.knowledge_base.total_frames_analyzed = total_frames
        self.knowledge_base.total_chart_frames = total_charts
        self.knowledge_base.training_sources = sources
        self.knowledge_base.last_trained = latest_trained

        logger.info(f"ML Knowledge loaded: {len(self.knowledge_base.patterns_learned)} pattern types "
                    f"from {len(vision_files)} vision files + knowledge base + {audio_first_videos} audio-first trainings")
        for pt, p in self.knowledge_base.patterns_learned.items():
            logger.info(f"  - {pt}: frequency={p.frequency}, confidence={p.confidence:.2f}")

    def get_learned_patterns(self) -> List[str]:
        """Get list of pattern types the ML has learned"""
        if not self.knowledge_base:
            return []
        return list(self.knowledge_base.patterns_learned.keys())

    def get_unlearned_patterns(self) -> List[str]:
        """Get list of common ICT pattern types the ML has NOT learned yet"""
        common_patterns = [
            'fvg', 'order_block', 'breaker_block', 'market_structure',
            'liquidity', 'optimal_trade_entry', 'fibonacci_ote',
            'displacement', 'kill_zone', 'institutional', 'smart_money',
            'buy_stops', 'sell_stops', 'equal_highs_lows', 'swing_high_low',
            'higher_high', 'support_resistance',
        ]

        learned = self.get_learned_patterns()
        return [p for p in common_patterns if p not in learned]

    def get_pattern_confidence(self, pattern_type: str) -> float:
        """Get ML's confidence for a specific pattern type"""
        if not self.knowledge_base:
            return 0.0
        return self.knowledge_base.get_confidence(pattern_type)

    def can_detect_pattern(self, pattern_type: str) -> bool:
        """Check if ML can detect a pattern type (has learned it)"""
        if not self.knowledge_base:
            return False
        return self.knowledge_base.has_learned(pattern_type)

    def get_detection_parameters(self, pattern_type: str) -> Dict[str, Any]:
        """
        Get ML-learned parameters for pattern detection.

        IMPORTANT: This does NOT use frequency to limit detection!
        Frequency just shows how much ML studied the pattern.
        Parameters come from LEARNED CHARACTERISTICS, not counts.

        Once ML understands a pattern type, it can detect it universally.
        """
        if not self.knowledge_base or not self.can_detect_pattern(pattern_type):
            return {}

        normalized = self.knowledge_base._normalize_pattern_type(pattern_type)
        pattern = self.knowledge_base.patterns_learned.get(normalized)

        if not pattern:
            return {}

        # Base parameters - ML has learned this pattern type
        params = {
            'enabled': True,
            'confidence_multiplier': pattern.confidence,
            'min_confidence_threshold': 0.3,
            # ML has studied this pattern and can now detect it universally
            'pattern_understood': True,
            'examples_studied': pattern.frequency,  # For reference only, not detection limit
        }

        # Extract learned traits from characteristics
        learned_traits = self._extract_learned_traits(pattern)
        params['learned_traits'] = learned_traits

        # Pattern-specific parameters based on ICT methodology (universal rules)
        # These are standard ICT rules that ML learned from videos
        if normalized == 'fvg':
            # ICT FVG: 3-candle pattern with gap between candle 1 and 3
            params['lookback_candles'] = 3
            params['min_gap_size_pct'] = 0.0001  # Detect any valid FVG
            # Add learned characteristics for significance scoring
            params['bullish_traits'] = learned_traits.get('bullish_count', 0)
            params['bearish_traits'] = learned_traits.get('bearish_count', 0)

        elif normalized == 'order_block':
            # ICT Order Block: Zone where institutional orders were placed
            params['min_move_strength'] = 0.3  # Universal threshold
            params['require_impulse'] = True   # ICT methodology
            params['institutional_zone'] = learned_traits.get('institutional', False)

        elif normalized == 'breaker_block':
            # ICT Breaker: Failed order block that becomes support/resistance
            params['require_prior_structure'] = True  # Must have broken structure
            params['high_significance'] = learned_traits.get('high_significance', 0)

        elif normalized == 'support_resistance':
            # Standard S/R levels
            params['tolerance_pct'] = 0.001

        return params

    def _extract_learned_traits(self, pattern: LearnedPattern) -> Dict[str, Any]:
        """
        Extract pattern traits from ML's learned characteristics.

        This analyzes what ML learned about the pattern:
        - Is it typically bullish or bearish?
        - What significance levels did ICT emphasize?
        - Any special characteristics mentioned?
        """
        traits = {
            'bullish_count': 0,
            'bearish_count': 0,
            'high_significance': 0,
            'institutional': False,
            'key_characteristics': [],
        }

        for char in pattern.characteristics:
            char_lower = char.lower()

            # Count bullish/bearish mentions
            if 'bullish' in char_lower:
                traits['bullish_count'] += 1
            if 'bearish' in char_lower:
                traits['bearish_count'] += 1

            # Check significance
            if 'high' in char_lower and 'significance' in char_lower:
                traits['high_significance'] += 1
            if 'significant' in char_lower:
                traits['high_significance'] += 1

            # Check institutional nature
            if 'institutional' in char_lower or 'large order' in char_lower:
                traits['institutional'] = True

            # Store key characteristics
            if char and len(char) > 5:
                traits['key_characteristics'].append(char[:100])

        return traits

    def _load_concept_definitions(self) -> Dict[str, Any]:
        """Load concept definitions from knowledge_base.json (actual ICT definitions from transcripts)"""
        try:
            kb_path = self.data_dir / "ml_models" / "knowledge_base.json"
            if kb_path.exists():
                with open(kb_path) as f:
                    data = json.load(f)
                    return data.get('concept_definitions', {})
        except Exception as e:
            logger.warning(f"Could not load concept definitions: {e}")
        return {}

    def _extract_core_definition(self, examples: list, concept: str) -> str:
        """
        Extract the core definition from transcript examples.

        ICT typically says "what is a [concept]? it is..." so we look for that pattern.
        We search through ALL examples to find the actual definition.
        """
        # For FVG - look for the canonical definition
        if concept in ['fvg', 'fair_value_gap']:
            for ex in examples:
                if 'range in price delivery' in ex.lower():
                    start = ex.lower().find('it is a range')
                    if start == -1:
                        start = ex.lower().find('range in price delivery')
                        if start > 10:
                            start -= 10  # Include "it is a" if present
                    if start != -1:
                        end = min(start + 150, len(ex))
                        definition = ex[start:end].strip()
                        if not definition.endswith('.'):
                            definition = definition.rsplit(' ', 1)[0] + '...'
                        return f"ICT defines FVG as: {definition}"

        # For Order Blocks - look for the canonical definition
        if concept == 'order_block':
            for ex in examples:
                ex_lower = ex.lower()
                # Look for the actual ICT definition pattern
                if 'down candle' in ex_lower and ('before' in ex_lower or 'bullish order block' in ex_lower):
                    return f"ICT defines Order Block as: {ex[:150]}..."
                if 'that is a bullish order block' in ex_lower or 'that is a bearish order block' in ex_lower:
                    return f"ICT defines Order Block as: {ex[:150]}..."

            # Fallback: provide the standard ICT definition
            return "ICT defines Order Block as: The last down candle before price moves higher (bullish OB) or last up candle before price moves lower (bearish OB) - indicates institutional order placement."

        # For Breaker Blocks
        if concept in ['breaker_block', 'breaker']:
            for ex in examples:
                if 'breaker' in ex.lower() and ('failed' in ex.lower() or 'broken' in ex.lower()):
                    return f"ICT defines Breaker: {ex[:150]}..."
            return "ICT defines Breaker: A failed Order Block that becomes support/resistance after price breaks through it."

        # For Liquidity
        if concept == 'liquidity':
            for ex in examples:
                if 'buy stop' in ex.lower() or 'sell stop' in ex.lower() or 'liquidity pool' in ex.lower():
                    return f"ICT on Liquidity: {ex[:150]}..."

        # Default: use first non-day-specific example
        for ex in examples:
            if not self._is_day_specific_context(ex):
                clean = ex[:120].strip()
                if not clean.endswith('.'):
                    clean = clean.rsplit(' ', 1)[0] + '...'
                return clean

        # Absolute fallback
        clean = examples[0][:120].strip() if examples else "Pattern understood from ICT methodology"
        if not clean.endswith('.'):
            clean = clean.rsplit(' ', 1)[0] + '...'
        return clean

    def _is_day_specific_context(self, context: str) -> bool:
        """
        Check if a teaching context is day-specific (not a general rule).

        Day-specific contexts like "Thursday's high" or "Wednesday's range" are
        specific to that video's market conditions, not general ICT methodology.
        """
        day_markers = [
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            "monday's", "tuesday's", "wednesday's", "thursday's", "friday's",
            'today', 'yesterday', 'this week', 'last week',
            'this morning', 'this afternoon',
            # Specific prices that are day-specific
            '106.6', '1.06', 'pipettes',
        ]
        context_lower = context.lower()
        return any(marker in context_lower for marker in day_markers)

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Get a summary of what the AI knows.

        Focus on PATTERN TYPES learned and understanding level,
        not frequency counts as detection limits.
        """
        if not self.knowledge_base:
            return {
                'status': 'no_training',
                'message': 'AI has not been trained yet. No patterns can be detected.',
                'patterns_learned': [],
                'patterns_not_learned': self.get_unlearned_patterns(),
                'total_videos': 0,
                'total_frames': 0,
            }

        patterns_summary = []
        for pt, p in self.knowledge_base.patterns_learned.items():
            traits = self._extract_learned_traits(p)

            # Determine understanding level
            if p.frequency >= 10 and len(p.teaching_contexts) > 0:
                understanding = 'expert'  # Many examples + teaching context
            elif p.frequency >= 5:
                understanding = 'proficient'  # Good number of examples
            elif len(p.teaching_contexts) > 0:
                understanding = 'intermediate'  # Has teaching context
            else:
                understanding = 'basic'  # Just recognizes pattern

            patterns_summary.append({
                'type': pt,
                'understanding_level': understanding,
                'confidence': round(p.confidence, 2),
                'has_teaching': len(p.teaching_contexts) > 0,
                'has_visual': p.visual_example_path is not None,
                'examples_studied': p.frequency,  # Reference info, not a limit
                'can_detect_universally': True,  # Once learned, can detect anywhere
                'learned_traits': {
                    'bullish_bias': traits['bullish_count'] > traits['bearish_count'],
                    'bearish_bias': traits['bearish_count'] > traits['bullish_count'],
                    'high_significance': traits['high_significance'] > 0,
                    'institutional': traits['institutional'],
                },
            })

        return {
            'status': 'trained',
            'message': f'AI trained on {self.knowledge_base.total_videos_trained} ICT video(s). '
                      f'Can detect {len(patterns_summary)} pattern types universally.',
            'patterns_learned': patterns_summary,
            'patterns_not_learned': self.get_unlearned_patterns(),
            'total_videos': self.knowledge_base.total_videos_trained,
            'total_frames': self.knowledge_base.total_frames_analyzed,
            'chart_frames': self.knowledge_base.total_chart_frames,
            'training_sources': self.knowledge_base.training_sources,
            'last_trained': self.knowledge_base.last_trained.isoformat()
                          if self.knowledge_base.last_trained else None,
        }

    def get_pattern_reasoning(self, pattern_type: str) -> Dict[str, Any]:
        """
        Get AI's reasoning for a pattern type based on what it learned from videos.

        Focus on the LOGIC learned, not just counts:
        - What characteristics define this pattern?
        - How did ICT explain it?
        - What significance does it have?

        The AI can now detect this pattern universally on any chart.
        """
        if not self.knowledge_base:
            return {
                'can_explain': False,
                'reason': 'AI has not been trained yet.',
            }

        normalized = self.knowledge_base._normalize_pattern_type(pattern_type)
        if normalized not in self.knowledge_base.patterns_learned:
            return {
                'can_explain': False,
                'reason': f'AI has not learned {pattern_type} from any videos yet.',
            }

        pattern = self.knowledge_base.patterns_learned[normalized]
        traits = self._extract_learned_traits(pattern)

        return {
            'can_explain': True,
            'pattern_type': normalized,
            'confidence': pattern.confidence,
            # Explain what AI learned, not just how many times
            'understanding_level': 'well_understood' if pattern.frequency >= 5 else 'basic_understanding',
            'examples_studied': pattern.frequency,  # Reference info only
            # Focus on learned LOGIC
            'learned_traits': traits,
            'characteristics': pattern.characteristics,
            'teaching_contexts': pattern.teaching_contexts,
            'visual_example': pattern.visual_example_path,
            # Explain detection capability
            'detection_note': f"AI can detect any {pattern_type} pattern on any chart based on learned characteristics",
        }

    def generate_ml_reasoning(self, detected_patterns: List[str], bias: str, zone: str,
                              detected_pattern_details: List[Dict] = None) -> str:
        """
        Generate a human-readable explanation of WHY ML made its analysis decision.

        This is based ONLY on what ML learned from videos, not generic SMC concepts.
        Focus on the LOGIC and CHARACTERISTICS learned, not frequency counts.

        IMPORTANT: Uses concept_definitions (actual ICT definitions from transcripts)
        NOT teaching_contexts from vision files (which may contain day-specific commentary).
        """
        if not self.knowledge_base or not self.knowledge_base.patterns_learned:
            return "⚠️ AI has no trained knowledge. Analysis is based on basic price structure only."

        reasoning_parts = []

        # Load concept definitions from knowledge base for proper definitions
        concept_definitions = self._load_concept_definitions()

        # Explain based on patterns detected
        for pattern in detected_patterns:
            normalized = self.knowledge_base._normalize_pattern_type(pattern)
            if normalized in self.knowledge_base.patterns_learned:
                learned = self.knowledge_base.patterns_learned[normalized]

                # Extract learned traits for this pattern
                traits = self._extract_learned_traits(learned)

                # PRIORITY 0: Use LLM summary from audio-first training (richest source)
                llm_summary = learned.learned_traits.get('llm_summary', '')
                if llm_summary and len(llm_summary) > 20:
                    # Use first 400 chars of the LLM-generated summary for richer reasoning
                    summary_text = llm_summary[:400].strip()
                    if not summary_text.endswith('.'):
                        # Cut at last complete sentence
                        last_period = summary_text.rfind('.')
                        if last_period > 50:
                            summary_text = summary_text[:last_period + 1]
                        else:
                            summary_text = summary_text + '...'
                    reasoning_parts.append(
                        f"🧠 **{pattern.upper()}**: {summary_text}"
                    )
                    # Skip other priorities since we have the best source
                    # Still add characteristic insights below
                    if traits['bullish_count'] > traits['bearish_count']:
                        reasoning_parts.append(f"   ↳ Typically bullish pattern in current context")
                    elif traits['bearish_count'] > traits['bullish_count']:
                        reasoning_parts.append(f"   ↳ Typically bearish pattern in current context")
                    if traits['high_significance'] > 0:
                        reasoning_parts.append(f"   ↳ High significance level as taught by ICT")
                    if traits['institutional']:
                        reasoning_parts.append(f"   ↳ Indicates institutional activity")
                    continue

                # PRIORITY 1: Use concept_definitions (actual ICT definitions from transcripts)
                # This is the REAL definition, not day-specific teaching commentary
                # Check both normalized form (fvg) and full form (fair_value_gap)
                concept_key = None
                if normalized in concept_definitions and concept_definitions[normalized].get('examples'):
                    concept_key = normalized
                elif normalized == 'fvg' and 'fair_value_gap' in concept_definitions:
                    concept_key = 'fair_value_gap'
                elif normalized == 'order_block' and 'order_block' in concept_definitions:
                    concept_key = 'order_block'
                elif normalized == 'breaker_block' and 'breaker' in concept_definitions:
                    concept_key = 'breaker'

                if concept_key and concept_definitions[concept_key].get('examples'):
                    # Get ALL examples and find the best definition
                    examples = concept_definitions[concept_key]['examples']
                    # Extract the core definition from the examples
                    clean_def = self._extract_core_definition(examples, concept_key)
                    reasoning_parts.append(
                        f"📚 **{pattern.upper()}**: {clean_def}"
                    )
                # PRIORITY 2: Fall back to teaching_contexts only if no definition
                elif learned.teaching_contexts:
                    # Filter out day-specific contexts (Thursday, Monday, etc.)
                    valid_contexts = [c for c in learned.teaching_contexts
                                     if not self._is_day_specific_context(c)]
                    if valid_contexts:
                        context = valid_contexts[0]
                        reasoning_parts.append(
                            f"📚 **{pattern.upper()}**: {context[:120]}..."
                        )
                    else:
                        reasoning_parts.append(
                            f"📊 **{pattern.upper()}**: AI understands this pattern from ICT methodology. "
                            f"(confidence: {learned.confidence:.0%})"
                        )
                else:
                    # Describe based on learned characteristics
                    reasoning_parts.append(
                        f"📊 **{pattern.upper()}**: AI understands this pattern from ICT methodology. "
                        f"(confidence: {learned.confidence:.0%})"
                    )

                # Add characteristic-based insight
                if traits['bullish_count'] > traits['bearish_count']:
                    reasoning_parts.append(f"   ↳ Typically bullish pattern in current context")
                elif traits['bearish_count'] > traits['bullish_count']:
                    reasoning_parts.append(f"   ↳ Typically bearish pattern in current context")

                if traits['high_significance'] > 0:
                    reasoning_parts.append(f"   ↳ High significance level as taught by ICT")

                if traits['institutional']:
                    reasoning_parts.append(f"   ↳ Indicates institutional activity")

        # Per-IDM reasoning (pin-point detail for each plotted inducement)
        if detected_pattern_details:
            idm_details = [p for p in detected_pattern_details
                           if 'inducement' in p.get('pattern_type', '')]
            if idm_details:
                reasoning_parts.append("")
                reasoning_parts.append("🔍 **INDUCEMENT DETAILS** (per-IDM analysis):")
                for idm in idm_details:
                    validity = idm.get('validity', 'unknown')
                    emoji = {
                        'valid': '✅', 'unconfirmed': '⚠️',
                        'impulse_trap': '🪤', 'shifted': '↗️'
                    }.get(validity, '❓')
                    idm_reasoning = idm.get('reasoning', 'No reasoning available')
                    reasoning_parts.append(f"   {emoji} {idm_reasoning}")

        # Explain unlearned patterns
        unlearned = self.get_unlearned_patterns()
        if unlearned:
            reasoning_parts.append(
                f"\n⚠️ AI hasn't learned: {', '.join(unlearned)}. "
                f"Train more videos to expand AI's pattern knowledge."
            )

        # Add bias explanation
        if detected_patterns:
            reasoning_parts.append(
                f"\n🎯 **Bias**: {bias.upper()} based on AI-detected patterns in {zone} zone."
            )

        if not reasoning_parts:
            return "ℹ️ No AI patterns detected in current data. Signal based on basic structure analysis."

        return "\n".join(reasoning_parts)

    def get_entry_exit_reasoning(self, direction: str, patterns_found: Dict[str, Any]) -> Dict[str, str]:
        """
        Get AI-based reasoning for entry, exit, and stop loss placement.

        Based ONLY on what AI learned from ICT videos.
        Focus on the LOGIC taught, not frequency counts.
        """
        if not self.knowledge_base or not self.knowledge_base.patterns_learned:
            return {
                'entry_reason': "Basic structure-based entry (AI not trained)",
                'stop_reason': "Default risk management (AI not trained)",
                'target_reason': "Standard R:R ratio (AI not trained)",
            }

        entry_reason = ""
        stop_reason = ""
        target_reason = ""

        # Entry reasoning based on FVG
        if 'fvg' in self.knowledge_base.patterns_learned:
            fvg = self.knowledge_base.patterns_learned['fvg']
            traits = self._extract_learned_traits(fvg)

            if fvg.teaching_contexts:
                # Use the actual teaching from ICT
                entry_reason = f"Entry at FVG - ICT methodology: \"{fvg.teaching_contexts[0][:80]}...\""
            else:
                # Explain based on learned traits
                bias = "bullish" if traits['bullish_count'] > traits['bearish_count'] else "bearish"
                entry_reason = f"Entry at FVG (AI learned {bias} Fair Value Gap identification)"

        # Entry/Stop at Order Block
        if 'order_block' in self.knowledge_base.patterns_learned:
            ob = self.knowledge_base.patterns_learned['order_block']
            traits = self._extract_learned_traits(ob)

            if ob.teaching_contexts:
                stop_reason = f"Stop beyond Order Block - ICT taught: \"{ob.teaching_contexts[0][:80]}...\""
            else:
                institutional = "institutional zone" if traits['institutional'] else "significant price zone"
                stop_reason = f"Stop beyond Order Block ({institutional} as per ICT)"

            if not entry_reason:
                entry_reason = f"Entry at Order Block (AI understands this as institutional zone)"

        # Entry at Optimal Trade Entry (62-79% Fibonacci retracement)
        if 'optimal_trade_entry' in self.knowledge_base.patterns_learned:
            ote = self.knowledge_base.patterns_learned['optimal_trade_entry']
            llm_summary = ote.learned_traits.get('llm_summary', '')
            if llm_summary:
                entry_reason = f"Entry at OTE zone (62-79% Fib retracement) - ICT: \"{llm_summary[:80]}...\""
            elif not entry_reason:
                entry_reason = "Entry at Optimal Trade Entry (62-79% Fibonacci retracement as taught by ICT)"

        # Displacement confirms institutional intent
        if 'displacement' in self.knowledge_base.patterns_learned:
            disp = self.knowledge_base.patterns_learned['displacement']
            llm_summary = disp.learned_traits.get('llm_summary', '')
            if llm_summary:
                if entry_reason:
                    entry_reason += f" | Displacement confirms institutional intent"
                else:
                    entry_reason = f"Entry confirmed by displacement (institutional move)"

        # Kill Zone timing confirmation
        if 'kill_zone' in self.knowledge_base.patterns_learned:
            kz = self.knowledge_base.patterns_learned['kill_zone']
            if entry_reason:
                entry_reason += " | Kill Zone timing active"
            else:
                entry_reason = "Entry during Kill Zone (high-probability ICT session time)"

        # Buy/Sell Stops as liquidity targets
        if 'buy_stops' in self.knowledge_base.patterns_learned or 'sell_stops' in self.knowledge_base.patterns_learned:
            if not stop_reason:
                stop_reason = "Stop beyond buy/sell stop liquidity level (ICT methodology)"

        # Default if patterns not learned
        if not entry_reason:
            entry_reason = "Entry based on price structure (FVG/Order Block not yet learned)"
        if not stop_reason:
            stop_reason = "Stop based on swing structure (Order Block not yet learned)"

        # Target reasoning based on liquidity/structure
        if 'liquidity' in self.knowledge_base.patterns_learned:
            liq = self.knowledge_base.patterns_learned['liquidity']
            if liq.teaching_contexts:
                target_reason = f"Target at liquidity - {liq.teaching_contexts[0][:60]}..."
            else:
                target_reason = "Target at next liquidity level (swing high/low) as per ICT"
        else:
            target_reason = "Target at next liquidity level (swing high/low)"

        return {
            'entry_reason': entry_reason,
            'stop_reason': stop_reason,
            'target_reason': target_reason,
        }

    def reload_knowledge(self):
        """Reload knowledge after new training"""
        logger.info("Reloading ML knowledge base...")
        self._load_knowledge()


# Global instance for the application
_ml_engine: Optional[MLPatternEngine] = None


def get_ml_engine() -> MLPatternEngine:
    """Get or create the global ML Pattern Engine instance"""
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = MLPatternEngine()
    return _ml_engine


def reload_ml_knowledge():
    """Reload ML knowledge (call after training)"""
    global _ml_engine
    if _ml_engine:
        _ml_engine.reload_knowledge()
    else:
        _ml_engine = MLPatternEngine()


# =============================================================================
# HEDGE FUND ML INTEGRATION
# =============================================================================
# These features bridge the gap between retail and institutional trading:
# - Pattern Grading (A+ to F)
# - Historical Validation
# - Multi-Timeframe Confluence
# - Statistical Edge Tracking
# =============================================================================

try:
    from .hedge_fund_ml import (
        PatternGrader,
        HistoricalValidator,
        MultiTimeframeAnalyzer,
        EdgeTracker,
        PatternGrade,
        GradedPattern,
        get_pattern_grader,
        get_historical_validator,
        get_mtf_analyzer,
        get_edge_tracker,
    )
    HEDGE_FUND_ML_AVAILABLE = True
    logger.info("Hedge Fund ML features loaded successfully")
except ImportError as e:
    HEDGE_FUND_ML_AVAILABLE = False
    logger.warning(f"Hedge Fund ML features not available: {e}")


def grade_pattern(
    pattern_type: str,
    pattern_data: Dict,
    market_context: Dict,
    historical_stats: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Grade a pattern from A+ to F (Hedge Fund Level Quality Assessment).

    This is what separates professionals from amateurs:
    - Not all patterns are equal
    - Location, confluence, freshness all matter
    - Only trade A/B grade patterns
    """
    if not HEDGE_FUND_ML_AVAILABLE:
        logger.warning("Hedge Fund ML not available for pattern grading")
        return None

    grader = get_pattern_grader()
    graded = grader.grade_pattern(
        pattern_type=pattern_type,
        pattern_data=pattern_data,
        market_context=market_context,
        historical_stats=historical_stats
    )

    return graded.to_dict()


def analyze_confluence(
    primary_pattern: Dict,
    primary_tf: str,
    all_tf_patterns: Dict[str, List[Dict]]
) -> Optional[Dict]:
    """
    Analyze multi-timeframe confluence for a pattern.

    ICT teaches: Higher timeframe patterns are stronger.
    Professional traders check M15, H1, H4, D1 for confluence.
    """
    if not HEDGE_FUND_ML_AVAILABLE:
        logger.warning("Hedge Fund ML not available for confluence analysis")
        return None

    mtf = get_mtf_analyzer()
    result = mtf.analyze_confluence(
        primary_pattern=primary_pattern,
        primary_tf=primary_tf,
        all_tf_patterns=all_tf_patterns
    )

    return {
        'pattern_type': result.pattern_type,
        'primary_timeframe': result.primary_timeframe,
        'confluence_score': result.confluence_score,
        'aligned_timeframes': result.aligned_timeframes,
        'conflicting_timeframes': result.conflicting_timeframes,
        'confluence_factors': result.confluence_factors,
        'recommendation': result.recommendation,
    }


def record_trade_outcome(
    pattern_type: str,
    outcome: str,
    rr_achieved: float = 0.0,
    session: str = "",
    day_of_week: str = ""
):
    """
    Record a trade outcome for statistical edge tracking.

    Track over time to build a statistical edge profile:
    - Which patterns have positive expectancy?
    - What's the real win rate?
    - What R:R do you actually achieve?
    """
    if not HEDGE_FUND_ML_AVAILABLE:
        logger.warning("Hedge Fund ML not available for edge tracking")
        return

    tracker = get_edge_tracker()
    tracker.record_trade(
        pattern_type=pattern_type,
        outcome=outcome,
        rr_achieved=rr_achieved,
        session=session,
        day_of_week=day_of_week
    )


def get_edge_statistics(pattern_type: str = None) -> Optional[Dict]:
    """
    Get statistical edge for pattern(s).

    Returns win rate, expectancy, profit factor.
    Use this to know which patterns to focus on.
    """
    if not HEDGE_FUND_ML_AVAILABLE:
        return None

    tracker = get_edge_tracker()
    return tracker.get_edge_summary(pattern_type)


def get_best_performing_patterns(min_signals: int = 10) -> List[Dict]:
    """
    Get patterns with proven positive expectancy.

    Only trade patterns with statistical edge!
    """
    if not HEDGE_FUND_ML_AVAILABLE:
        return []

    tracker = get_edge_tracker()
    return tracker.get_best_patterns(min_signals)
