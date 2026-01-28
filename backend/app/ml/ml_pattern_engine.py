"""
ML Pattern Engine - Uses ONLY learned knowledge from trained videos

This engine reads the ML's visual knowledge and uses it to:
1. Detect patterns based on what the ML learned
2. Calculate confidence based on how often ML saw the pattern
3. Track which patterns ML knows vs doesn't know

IMPORTANT: This replaces hardcoded pattern detection.
If ML hasn't learned a pattern type, it will NOT detect it.
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
    """A pattern learned from video training"""
    pattern_type: str
    frequency: int  # How many times ML saw this pattern
    confidence: float  # Calculated from frequency
    characteristics: List[str] = field(default_factory=list)
    example_locations: List[str] = field(default_factory=list)
    teaching_contexts: List[str] = field(default_factory=list)
    visual_example_path: Optional[str] = None


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

    Key Principles:
    1. If ML hasn't learned a pattern type, it WILL NOT detect it
    2. Confidence is based on how often ML saw the pattern in training
    3. Detection parameters are derived from ML's observations
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

        # Calculate confidence based on frequency
        total_pattern_observations = sum(p['frequency'] for p in all_patterns.values())

        for pattern_type, data in all_patterns.items():
            # Confidence formula:
            # - Base: frequency relative to total observations
            # - Boost: having teaching contexts increases confidence
            # - Cap at 0.95 (never 100% certain)

            if total_pattern_observations > 0:
                base_confidence = min(data['frequency'] / total_pattern_observations * 2, 0.7)
            else:
                base_confidence = 0.1

            # Teaching boost (up to +0.25)
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

        Returns parameters based on what ML learned, NOT hardcoded values.
        """
        if not self.knowledge_base or not self.can_detect_pattern(pattern_type):
            return {}

        normalized = self.knowledge_base._normalize_pattern_type(pattern_type)
        pattern = self.knowledge_base.patterns_learned.get(normalized)

        if not pattern:
            return {}

        # Base parameters from ML knowledge
        params = {
            'enabled': True,
            'confidence_multiplier': pattern.confidence,
            'min_confidence_threshold': 0.3,  # Only detect if ML is at least 30% confident
        }

        # Pattern-specific parameters derived from ML observations
        if normalized == 'fvg':
            # FVG was seen 31 times - high frequency means strict detection
            params['min_gap_size_pct'] = 0.0001 if pattern.frequency > 20 else 0.0005
            params['lookback_candles'] = 3  # Standard FVG is 3-candle pattern

        elif normalized == 'order_block':
            # Order blocks seen less frequently - more selective
            params['min_move_strength'] = 0.5 if pattern.frequency > 5 else 0.3
            params['require_impulse'] = pattern.frequency > 3

        elif normalized == 'breaker_block':
            # Only 1 breaker seen - very selective
            params['high_selectivity'] = True
            params['min_prior_structure'] = True

        elif normalized == 'support_resistance':
            # Basic support/resistance
            params['tolerance_pct'] = 0.001

        return params

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get a summary of what the ML knows"""
        if not self.knowledge_base:
            return {
                'status': 'no_training',
                'message': 'ML has not been trained yet. No patterns can be detected.',
                'patterns_learned': [],
                'patterns_not_learned': self.get_unlearned_patterns(),
                'total_videos': 0,
                'total_frames': 0,
            }

        patterns_summary = []
        for pt, p in self.knowledge_base.patterns_learned.items():
            patterns_summary.append({
                'type': pt,
                'frequency': p.frequency,
                'confidence': round(p.confidence, 2),
                'has_teaching': len(p.teaching_contexts) > 0,
                'has_visual': p.visual_example_path is not None,
            })

        return {
            'status': 'trained',
            'message': f'ML trained on {self.knowledge_base.total_videos_trained} videos, '
                      f'analyzed {self.knowledge_base.total_frames_analyzed} frames',
            'patterns_learned': patterns_summary,
            'patterns_not_learned': self.get_unlearned_patterns(),
            'total_videos': self.knowledge_base.total_videos_trained,
            'total_frames': self.knowledge_base.total_frames_analyzed,
            'chart_frames': self.knowledge_base.total_chart_frames,
            'training_sources': self.knowledge_base.training_sources,
            'last_trained': self.knowledge_base.last_trained.isoformat()
                          if self.knowledge_base.last_trained else None,
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
