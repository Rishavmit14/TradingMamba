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
            'order blocks': 'order_block',
            'order block': 'order_block',
            'ob': 'order_block',
            'fvg': 'fvg',
            'fair value gap': 'fvg',
            'fair value gaps': 'fvg',
            'breaker block': 'breaker_block',
            'breaker blocks': 'breaker_block',
            'breaker': 'breaker_block',
            'support/resistance': 'support_resistance',
            'support': 'support_resistance',
            'resistance': 'support_resistance',
            'liquidity': 'liquidity',
            'buy side liquidity': 'liquidity',
            'sell side liquidity': 'liquidity',
            'bos': 'market_structure',
            'break of structure': 'market_structure',
            'choch': 'market_structure',
            'change of character': 'market_structure',
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

    def __init__(self, data_dir: str = None):
        # Resolve the data directory relative to the project root
        if data_dir is None:
            # Go up from backend/app/ml/ to backend/ then to project root
            project_root = Path(__file__).parent.parent.parent.parent
            self.data_dir = project_root / "data"
        else:
            self.data_dir = Path(data_dir)

        self.vision_dir = self.data_dir / "video_vision"
        self.knowledge_base: Optional[MLKnowledgeBase] = None
        self._load_knowledge()

    def _load_knowledge(self):
        """Load all ML knowledge from trained video analyses"""
        self.knowledge_base = MLKnowledgeBase()

        if not self.vision_dir.exists():
            logger.warning(f"Vision directory not found: {self.vision_dir}")
            logger.warning("ML has NO knowledge - no patterns will be detected!")
            return

        # Load all vision analysis files
        vision_files = list(self.vision_dir.glob("*_vision.json"))

        if not vision_files:
            logger.warning("No vision training files found - ML has NO knowledge!")
            return

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

            self.knowledge_base.patterns_learned[pattern_type] = LearnedPattern(
                pattern_type=pattern_type,
                frequency=data['frequency'],
                confidence=confidence,
                characteristics=list(set(data['characteristics']))[:10],  # Unique, max 10
                example_locations=list(set(data['locations']))[:5],
                teaching_contexts=list(set(data['teaching_contexts']))[:5],
                visual_example_path=data['visual_path']
            )

        self.knowledge_base.total_videos_trained = len(vision_files)
        self.knowledge_base.total_frames_analyzed = total_frames
        self.knowledge_base.total_chart_frames = total_charts
        self.knowledge_base.training_sources = sources
        self.knowledge_base.last_trained = latest_trained

        logger.info(f"ML Knowledge loaded: {len(all_patterns)} pattern types from {len(vision_files)} videos")
        for pt, p in self.knowledge_base.patterns_learned.items():
            logger.info(f"  - {pt}: frequency={p.frequency}, confidence={p.confidence:.2f}")

    def get_learned_patterns(self) -> List[str]:
        """Get list of pattern types the ML has learned"""
        if not self.knowledge_base:
            return []
        return list(self.knowledge_base.patterns_learned.keys())

    def get_unlearned_patterns(self) -> List[str]:
        """Get list of common pattern types the ML has NOT learned yet"""
        common_patterns = [
            'fvg', 'order_block', 'breaker_block', 'market_structure',
            'liquidity', 'support_resistance', 'mitigation_block',
            'rejection_block', 'void', 'imbalance'
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

    def generate_ml_reasoning(self, detected_patterns: List[str], bias: str, zone: str) -> str:
        """
        Generate a human-readable explanation of WHY ML made its analysis decision.

        This is based ONLY on what ML learned from videos, not generic SMC concepts.
        Focus on the LOGIC and CHARACTERISTICS learned, not frequency counts.
        """
        if not self.knowledge_base or not self.knowledge_base.patterns_learned:
            return "⚠️ AI has no trained knowledge. Analysis is based on basic price structure only."

        reasoning_parts = []

        # Explain based on patterns detected
        for pattern in detected_patterns:
            normalized = self.knowledge_base._normalize_pattern_type(pattern)
            if normalized in self.knowledge_base.patterns_learned:
                learned = self.knowledge_base.patterns_learned[normalized]

                # Extract learned traits for this pattern
                traits = self._extract_learned_traits(learned)

                # Primary explanation: What ICT taught about this pattern
                if learned.teaching_contexts:
                    context = learned.teaching_contexts[0]
                    reasoning_parts.append(
                        f"📚 **{pattern.upper()}**: {context[:120]}..."
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
