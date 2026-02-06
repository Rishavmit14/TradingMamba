"""
Video Knowledge Index - Bridge between Video Training JSON and ML Features

Reads all knowledge_base.json and teaching_units.json files produced by the
audio-first training pipeline and builds a structured index that can be
converted into numerical features for ML models.

This is the critical bridge that makes "train on videos" actually feed into
model.fit() - it converts structured LLM-analyzed knowledge into numbers.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard ICT concept names (normalized)
CONCEPT_NAMES = [
    'order_block', 'fvg', 'displacement', 'optimal_trade_entry',
    'fibonacci', 'liquidity', 'market_structure', 'breaker',
    'kill_zone', 'equal_highs', 'equal_lows', 'buy_stops',
    'sell_stops', 'swing_high', 'swing_low', 'institutional',
    'smart_money', 'higher_high', 'higher_low', 'lower_high', 'lower_low',
]

# Map raw concept names from JSON to normalized names
CONCEPT_ALIASES = {
    'optimal trade entry': 'optimal_trade_entry',
    'order block': 'order_block',
    'order blocks': 'order_block',
    'fair value gap': 'fvg',
    'fair_value_gap': 'fvg',
    'market structure': 'market_structure',
    'kill zone': 'kill_zone',
    'equal highs': 'equal_highs',
    'equal lows': 'equal_lows',
    'buy stops': 'buy_stops',
    'sell stops': 'sell_stops',
    'swing high': 'swing_high',
    'swing low': 'swing_low',
    'higher high': 'higher_high',
    'higher low': 'higher_low',
    'lower high': 'lower_high',
    'lower low': 'lower_low',
    'smart money': 'smart_money',
    'breaker block': 'breaker',
    'breaker_block': 'breaker',
}

# =============================================================================
# ICT PATTERN RULES — Extracted from 16 Forex Minions training videos via Claude Code
# =============================================================================
# These rules define what each pattern requires, implies, and how to validate it.
# Source: Claude Code Max plan analysis of training data (FREE, no API cost)
# =============================================================================

ICT_PATTERN_RULES = {
    # -------------------------------------------------------------------------
    # CORE PATTERNS
    # -------------------------------------------------------------------------
    'order_block': {
        'requires_displacement': True,
        'requires_structure_break': True,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,  # has bullish/bearish variant
        'validation_rules': [
            'Must have displacement (strong move) away from the block',
            'Candle body must close beyond the order block for confirmation',
            'Look for return to the order block for entry',
        ],
    },
    'valid_order_block': {
        'requires_displacement': True,
        'requires_structure_break': True,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,
        'validation_rules': [
            'Order block must lead to break of structure',
            'Must have liquidity sweep before the order block forms',
            'Entry on return to order block after displacement',
            'Stop loss below/above the order block',
        ],
    },
    'fvg': {
        'requires_displacement': True,
        'requires_structure_break': False,
        'significance_when_institutional': False,
        'fibonacci_related': False,
        'directional': True,
        'validation_rules': [
            'Gap between candle 1 high and candle 3 low (bullish)',
            'Gap between candle 1 low and candle 3 high (bearish)',
            'Valid when price returns to fill the gap',
        ],
    },
    'displacement': {
        'requires_displacement': False,  # IS displacement
        'requires_structure_break': False,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,
    },
    'optimal_trade_entry': {
        'requires_displacement': True,
        'requires_structure_break': True,
        'significance_when_institutional': True,
        'fibonacci_related': True,  # 62-79% retracement
        'directional': True,
        'validation_rules': [
            'Entry at 62-79% Fibonacci retracement of impulse move',
            'Must have prior displacement/impulse',
            'Confluence with order block or FVG increases probability',
        ],
    },
    'liquidity': {
        'requires_displacement': False,
        'requires_structure_break': False,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': False,
        'validation_rules': [
            'Equal highs/lows create liquidity pools',
            'Stop losses cluster above highs and below lows',
            'Smart money targets these pools before reversing',
        ],
    },
    'breaker': {
        'requires_displacement': True,
        'requires_structure_break': True,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,
    },
    'kill_zone': {
        'requires_displacement': False,
        'requires_structure_break': False,
        'significance_when_institutional': False,
        'fibonacci_related': False,
        'directional': False,
        'validation_rules': [
            'London: 2am-5am EST',
            'New York: 7am-10am EST',
            'Highest probability setups occur in these windows',
        ],
    },

    # -------------------------------------------------------------------------
    # INDUCEMENT PATTERNS (from Forex Minions)
    # -------------------------------------------------------------------------
    'inducement': {
        'requires_displacement': False,
        'requires_structure_break': False,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,
        'validation_rules': [
            'First valid pullback on the left side of a swing high/low',
            'Acts as a trap before order block/supply/demand zone',
            '70% of forex trading works around inducement',
            'Must be swept (liquidity taken) before valid move continues',
        ],
    },
    'inducement_shift': {
        'requires_displacement': True,
        'requires_structure_break': False,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,
        'validation_rules': [
            'Rule 1: No sweep of inducement = shift to impulse low/high',
            'Rule 2: Sweep but no candle close beyond = shift to impulse low/high',
            'Rule 3: Impulse swing creates new inducement at its extreme',
        ],
    },
    'valid_pullback': {
        'requires_displacement': False,
        'requires_structure_break': False,
        'significance_when_institutional': False,
        'fibonacci_related': False,
        'directional': True,
        'validation_rules': [
            'Pullback must sweep the highest/lowest candle (liquidity sweep)',
            'Candle body must close back inside the range',
            'Creates valid inducement for continuation',
        ],
    },
    'smart_money_trap': {
        'requires_displacement': False,
        'requires_structure_break': False,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,
        'validation_rules': [
            'Retail traders enter early expecting breakout',
            'Smart money sweeps their stops before reversing',
            'Look for failed breakouts with quick reversals',
        ],
    },

    # -------------------------------------------------------------------------
    # STRUCTURE PATTERNS (from Forex Minions)
    # -------------------------------------------------------------------------
    'break_of_structure': {
        'requires_displacement': True,
        'requires_structure_break': True,  # IS the structure break
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,
        'validation_rules': [
            'Candle body must CLOSE beyond the swing point',
            'Wick-only break is NOT valid BOS',
            'Single candle BOS = strongest confirmation',
            'Multi-candle BOS = valid but weaker',
        ],
    },
    'change_of_character': {
        'requires_displacement': True,
        'requires_structure_break': True,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,
        'validation_rules': [
            'First break of structure in opposite direction',
            'Signals potential trend reversal',
            'Requires confirmation (not just indication)',
            'Fake CHoCH occurs when inducement not swept',
        ],
    },
    'fake_choch': {
        'requires_displacement': False,
        'requires_structure_break': True,
        'significance_when_institutional': False,
        'fibonacci_related': False,
        'directional': True,
        'validation_rules': [
            'CHoCH that fails because inducement was not swept',
            'Looks like reversal but continues original trend',
            'Always check if inducement was taken before CHoCH',
        ],
    },

    # -------------------------------------------------------------------------
    # PREMIUM/DISCOUNT PATTERNS (from Forex Minions)
    # -------------------------------------------------------------------------
    'premium_discount': {
        'requires_displacement': False,
        'requires_structure_break': False,
        'significance_when_institutional': True,
        'fibonacci_related': True,
        'directional': False,
        'validation_rules': [
            'Premium zone: above 50% (equilibrium) of range - sell zone',
            'Discount zone: below 50% of range - buy zone',
            'Equilibrium is the 50% level of the range',
            'Look for shorts in premium, longs in discount',
        ],
    },
    'equilibrium': {
        'requires_displacement': False,
        'requires_structure_break': False,
        'significance_when_institutional': False,
        'fibonacci_related': True,
        'directional': False,
        'validation_rules': [
            'The 50% level of any price range',
            'Divides premium from discount zones',
            'Price often reacts at equilibrium',
        ],
    },

    # -------------------------------------------------------------------------
    # PRICE DELIVERY PATTERNS (from Forex Minions)
    # -------------------------------------------------------------------------
    'price_delivery_cycle': {
        'requires_displacement': True,
        'requires_structure_break': False,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': False,
        'validation_rules': [
            'Cycle: Expansion → Retracement → Reversal/Continuation',
            'Expansion = strong directional move (displacement)',
            'Retracement = pullback to key levels',
            'Cycle repeats at all timeframes',
        ],
    },
    'expansion': {
        'requires_displacement': True,  # IS expansion
        'requires_structure_break': False,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,
    },
    'retracement': {
        'requires_displacement': False,
        'requires_structure_break': False,
        'significance_when_institutional': False,
        'fibonacci_related': True,
        'directional': True,
    },
    'judas_swing': {
        'requires_displacement': True,
        'requires_structure_break': False,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,
        'validation_rules': [
            'Fake move at session open to trap retail traders',
            'Sweeps liquidity in one direction',
            'Real move occurs in opposite direction',
            'Common in London and NY sessions',
        ],
    },

    # -------------------------------------------------------------------------
    # LIQUIDITY PATTERNS (from Forex Minions)
    # -------------------------------------------------------------------------
    'liquidity_sweep': {
        'requires_displacement': False,
        'requires_structure_break': False,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': True,
        'validation_rules': [
            'Price briefly breaks a level to take stops',
            'Candle wick extends beyond the level',
            'Body closes back inside the range',
            'Confirms valid pullback and inducement',
        ],
    },
    'equal_highs_lows': {
        'requires_displacement': False,
        'requires_structure_break': False,
        'significance_when_institutional': True,
        'fibonacci_related': False,
        'directional': False,
        'validation_rules': [
            'Multiple highs/lows at same level = liquidity pool',
            'Buy stops above equal highs',
            'Sell stops below equal lows',
            'Smart money targets these before reversing',
        ],
    },
}


class ConceptProfile:
    """Profile of a single ICT concept learned from videos."""

    def __init__(self, name: str):
        self.name = name
        self.teaching_depth = 0          # total teaching units
        self.total_words = 0             # words spoken about this concept
        self.total_frames = 0            # chart frames analyzed
        self.total_deictic_refs = 0      # visual pointer references
        self.video_count = 0             # how many videos taught this
        self.video_ids = set()
        self.llm_summaries: List[str] = []
        self.teaching_types: Counter = Counter()
        self.co_occurring_concepts: Counter = Counter()
        self.characteristics: List[str] = []
        self.teaching_contexts: List[str] = []

    @property
    def deictic_density(self) -> float:
        """Deictic references per teaching unit."""
        if self.teaching_depth == 0:
            return 0.0
        return self.total_deictic_refs / self.teaching_depth

    @property
    def best_llm_summary(self) -> str:
        """Return the longest (most detailed) LLM summary."""
        if not self.llm_summaries:
            return ""
        return max(self.llm_summaries, key=len)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'teaching_depth': self.teaching_depth,
            'total_words': self.total_words,
            'total_frames': self.total_frames,
            'total_deictic_refs': self.total_deictic_refs,
            'deictic_density': round(self.deictic_density, 3),
            'video_count': self.video_count,
            'teaching_types': dict(self.teaching_types),
            'co_occurring_concepts': dict(self.co_occurring_concepts.most_common(10)),
            'characteristics': self.characteristics[:10],
            'has_llm_summary': len(self.llm_summaries) > 0,
        }


class VideoKnowledgeIndex:
    """
    Index of all video-learned knowledge, optimized for ML feature extraction.

    Reads knowledge_base.json and teaching_units.json files from the
    audio-first training pipeline and builds:
    1. Concept profiles (teaching depth, word count, frame count, etc.)
    2. Co-occurrence matrix (which concepts appear together)
    3. Pattern rules (ICT methodology rules per concept)
    4. Normalized scores for ML feature extraction
    """

    def __init__(self, data_dir: str = None, video_ids: Optional[set] = None):
        """
        Initialize the Video Knowledge Index.

        Args:
            data_dir: Path to data directory. If None, auto-resolved from project root.
            video_ids: Optional set of video IDs to filter loading.
                       If None, loads ALL videos (backward compatible).
                       If provided, only loads knowledge from those specific videos.
        """
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent.parent / "data")
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / "audio_first_training"
        self._video_ids_filter = video_ids  # None = load all

        # Built indexes
        self.concept_profiles: Dict[str, ConceptProfile] = {}
        self.co_occurrence_matrix: Optional[np.ndarray] = None
        self.concept_index: Dict[str, int] = {}  # concept_name -> matrix index
        self.pattern_rules: Dict[str, Dict] = ICT_PATTERN_RULES.copy()

        # Normalization constants (computed from data)
        self._max_teaching_depth = 1
        self._max_words = 1
        self._max_frames = 1
        self._max_deictic = 1

        # Load state
        self._loaded = False
        self._n_videos = 0
        self._n_knowledge_bases = 0

        # Auto-load
        self._load()

    def _normalize_concept(self, raw_name: str) -> str:
        """Normalize a concept name to standard form."""
        name = raw_name.lower().strip()
        if name in CONCEPT_ALIASES:
            return CONCEPT_ALIASES[name]
        # Try underscore version
        underscored = name.replace(' ', '_')
        if underscored in CONCEPT_ALIASES:
            return CONCEPT_ALIASES[underscored]
        # Return underscored if it's a known concept
        if underscored in CONCEPT_NAMES:
            return underscored
        return underscored

    def _extract_video_id_from_file(self, filepath: Path) -> str:
        """Extract video_id from a training file path."""
        stem = filepath.stem
        for suffix in ['_knowledge_base', '_teaching_units', '_vision_analysis',
                       '_vision_progress', '_selected_frames', '_summary',
                       '_knowledge_summary']:
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        return stem

    def _should_load_file(self, filepath: Path) -> bool:
        """Check if a file should be loaded based on video_ids filter."""
        if self._video_ids_filter is None:
            return True
        vid = self._extract_video_id_from_file(filepath)
        return vid in self._video_ids_filter

    def _load(self):
        """Load all knowledge base and teaching unit files."""
        if not self.training_dir.exists():
            logger.info("No audio_first_training directory found - video knowledge empty")
            return

        # Load knowledge base files (filtered by playlist if specified)
        kb_files = sorted(f for f in self.training_dir.glob("*_knowledge_base.json")
                          if self._should_load_file(f))
        tu_files = sorted(f for f in self.training_dir.glob("*_teaching_units.json")
                          if self._should_load_file(f))

        if not kb_files:
            logger.info("No knowledge base files found - video knowledge empty")
            return

        filter_msg = f" (playlist-filtered from {len(self._video_ids_filter)} videos)" if self._video_ids_filter else ""
        logger.info(f"Loading video knowledge from {len(kb_files)} knowledge bases, {len(tu_files)} teaching unit files{filter_msg}")

        # Phase 1: Load knowledge bases (concept profiles)
        for kb_path in kb_files:
            try:
                with open(kb_path) as f:
                    kb_data = json.load(f)
                self._process_knowledge_base(kb_data)
            except Exception as e:
                logger.warning(f"Failed to load {kb_path.name}: {e}")

        # Phase 2: Load teaching units (co-occurrence data)
        all_teaching_units = []
        for tu_path in tu_files:
            try:
                with open(tu_path) as f:
                    units = json.load(f)
                all_teaching_units.extend(units)
            except Exception as e:
                logger.warning(f"Failed to load {tu_path.name}: {e}")

        # Phase 3: Build co-occurrence matrix from teaching units
        self._build_co_occurrence(all_teaching_units)

        # Phase 4: Compute normalization constants
        self._compute_normalization()

        self._n_knowledge_bases = len(kb_files)
        self._loaded = True

        logger.info(
            f"Video knowledge loaded: {len(self.concept_profiles)} concepts, "
            f"{self._n_videos} videos, co-occurrence matrix {self.co_occurrence_matrix.shape if self.co_occurrence_matrix is not None else 'None'}"
        )

    def _process_knowledge_base(self, kb_data: Dict):
        """Process a single knowledge_base.json file."""
        video_id = kb_data.get('video_id', 'unknown')
        stats = kb_data.get('processing_stats', {})
        concepts = kb_data.get('concepts', {})

        if not concepts:
            return

        self._n_videos += 1

        for raw_name, concept_data in concepts.items():
            name = self._normalize_concept(raw_name)

            if name not in self.concept_profiles:
                self.concept_profiles[name] = ConceptProfile(name)

            profile = self.concept_profiles[name]
            concept_stats = concept_data.get('statistics', {})

            # Accumulate statistics
            profile.teaching_depth += concept_stats.get('teaching_units', 1)
            profile.total_words += concept_stats.get('word_count', 0)
            profile.total_frames += concept_stats.get('frames_analyzed', 0)
            profile.total_deictic_refs += concept_stats.get('deictic_references', 0)

            # Track video sources
            profile.video_ids.add(video_id)
            profile.video_count = len(profile.video_ids)

            # Store LLM summary
            llm_summary = concept_data.get('llm_summary', '')
            if llm_summary:
                profile.llm_summaries.append(llm_summary)

            # Accumulate teaching types
            for tt, count in concept_data.get('teaching_types', {}).items():
                profile.teaching_types[tt] += count

    def _build_co_occurrence(self, teaching_units: List[Dict]):
        """Build co-occurrence matrix from teaching units."""
        if not teaching_units or not self.concept_profiles:
            return

        # Assign indices to concepts
        concepts = sorted(self.concept_profiles.keys())
        self.concept_index = {c: i for i, c in enumerate(concepts)}
        n = len(concepts)

        if n == 0:
            return

        # Count co-occurrences
        raw_matrix = np.zeros((n, n), dtype=np.float64)
        concept_counts = np.zeros(n, dtype=np.float64)

        for unit in teaching_units:
            # Get normalized concepts in this teaching unit
            raw_concepts = unit.get('detected_concepts', [])
            unit_concepts = set()
            for rc in raw_concepts:
                normalized = self._normalize_concept(rc)
                if normalized in self.concept_index:
                    unit_concepts.add(normalized)

            # Update co-occurrence for all pairs
            unit_list = sorted(unit_concepts)
            for c in unit_list:
                idx = self.concept_index[c]
                concept_counts[idx] += 1

                # Also update co-occurring concepts in profiles
                for other_c in unit_list:
                    if other_c != c:
                        self.concept_profiles[c].co_occurring_concepts[other_c] += 1

            for i_idx in range(len(unit_list)):
                for j_idx in range(i_idx + 1, len(unit_list)):
                    ci = self.concept_index[unit_list[i_idx]]
                    cj = self.concept_index[unit_list[j_idx]]
                    raw_matrix[ci, cj] += 1
                    raw_matrix[cj, ci] += 1

        # Normalize: co_occurrence[i][j] = count(i,j) / count(i)
        # This gives P(j appears | i appears)
        self.co_occurrence_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            if concept_counts[i] > 0:
                self.co_occurrence_matrix[i, :] = raw_matrix[i, :] / concept_counts[i]

        logger.info(f"Co-occurrence matrix built: {n}x{n}, {int(np.sum(raw_matrix))} total co-occurrences")

    def _compute_normalization(self):
        """Compute normalization constants for feature extraction."""
        if not self.concept_profiles:
            return

        self._max_teaching_depth = max(p.teaching_depth for p in self.concept_profiles.values()) or 1
        self._max_words = max(p.total_words for p in self.concept_profiles.values()) or 1
        self._max_frames = max(p.total_frames for p in self.concept_profiles.values()) or 1
        self._max_deictic = max(p.total_deictic_refs for p in self.concept_profiles.values()) or 1

    # ==================== Feature Extraction Methods ====================

    def is_loaded(self) -> bool:
        """Whether any video knowledge exists."""
        return self._loaded and len(self.concept_profiles) > 0

    def get_concept_profile(self, concept_name: str) -> Optional[ConceptProfile]:
        """Get full profile for a concept."""
        normalized = self._normalize_concept(concept_name)
        return self.concept_profiles.get(normalized)

    def get_teaching_depth_score(self, concept_name: str) -> float:
        """Get normalized teaching depth (0-1) for a concept."""
        profile = self.get_concept_profile(concept_name)
        if not profile:
            return 0.0
        return min(profile.teaching_depth / self._max_teaching_depth, 1.0)

    def get_word_density_score(self, concept_name: str) -> float:
        """Get normalized word density (0-1) for a concept."""
        profile = self.get_concept_profile(concept_name)
        if not profile:
            return 0.0
        return min(profile.total_words / self._max_words, 1.0)

    def get_frame_density_score(self, concept_name: str) -> float:
        """Get normalized frame density (0-1) for a concept."""
        profile = self.get_concept_profile(concept_name)
        if not profile:
            return 0.0
        return min(profile.total_frames / self._max_frames, 1.0)

    def get_co_occurrence(self, concept_a: str, concept_b: str) -> float:
        """Get co-occurrence score between two concepts (0-1)."""
        if self.co_occurrence_matrix is None:
            return 0.0

        norm_a = self._normalize_concept(concept_a)
        norm_b = self._normalize_concept(concept_b)

        idx_a = self.concept_index.get(norm_a)
        idx_b = self.concept_index.get(norm_b)

        if idx_a is None or idx_b is None:
            return 0.0

        return float(self.co_occurrence_matrix[idx_a, idx_b])

    def get_context_similarity(self, detected_patterns: List[str]) -> float:
        """
        Score how well a set of detected patterns matches video-learned co-occurrence.

        If the video learned that OB + FVG appear together 72% of the time,
        and we detect both OB and FVG, the score is high.
        Returns average pairwise co-occurrence score.
        """
        if not detected_patterns or self.co_occurrence_matrix is None:
            return 0.0

        normalized = [self._normalize_concept(p) for p in detected_patterns]
        normalized = [n for n in normalized if n in self.concept_index]

        if len(normalized) < 2:
            return 0.0

        # Average pairwise co-occurrence
        total = 0.0
        count = 0
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                total += self.get_co_occurrence(normalized[i], normalized[j])
                count += 1

        return total / count if count > 0 else 0.0

    def get_strongest_pair(self, detected_patterns: List[str]) -> float:
        """Get the strongest co-occurrence pair among detected patterns."""
        if not detected_patterns or self.co_occurrence_matrix is None:
            return 0.0

        normalized = [self._normalize_concept(p) for p in detected_patterns]
        normalized = [n for n in normalized if n in self.concept_index]

        if len(normalized) < 2:
            return 0.0

        max_score = 0.0
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                score = self.get_co_occurrence(normalized[i], normalized[j])
                max_score = max(max_score, score)

        return max_score

    def has_unexpected_combination(self, detected_patterns: List[str]) -> bool:
        """Check if any pattern pair was never seen together in videos."""
        if not detected_patterns or self.co_occurrence_matrix is None:
            return False

        normalized = [self._normalize_concept(p) for p in detected_patterns]
        normalized = [n for n in normalized if n in self.concept_index]

        if len(normalized) < 2:
            return False

        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                if self.get_co_occurrence(normalized[i], normalized[j]) == 0.0:
                    return True
        return False

    def get_synergy_score(self, detected_patterns: List[str]) -> float:
        """
        Weighted synergy score considering teaching depth of each pattern.
        Higher if well-studied patterns appear together in learned combinations.
        """
        if not detected_patterns or self.co_occurrence_matrix is None:
            return 0.0

        normalized = [self._normalize_concept(p) for p in detected_patterns]
        normalized = [n for n in normalized if n in self.concept_index]

        if len(normalized) < 2:
            # Single pattern - return its teaching depth as base synergy
            if len(normalized) == 1:
                return self.get_teaching_depth_score(normalized[0]) * 0.5
            return 0.0

        total = 0.0
        count = 0
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                co_occ = self.get_co_occurrence(normalized[i], normalized[j])
                depth_i = self.get_teaching_depth_score(normalized[i])
                depth_j = self.get_teaching_depth_score(normalized[j])
                # Weight co-occurrence by teaching depth of both concepts
                total += co_occ * (depth_i + depth_j) / 2.0
                count += 1

        return total / count if count > 0 else 0.0

    def get_rule_alignment(self, concept_name: str, has_displacement: bool = False,
                           has_structure_break: bool = False,
                           has_fib_alignment: bool = False,
                           is_institutional: bool = False) -> float:
        """
        Score how well current market conditions align with ICT rules for this concept.
        Returns 0-1 where 1 means all applicable rules are satisfied.
        """
        normalized = self._normalize_concept(concept_name)
        rules = self.pattern_rules.get(normalized, {})

        if not rules:
            return 0.5  # Unknown concept - neutral

        score = 0.0
        total_rules = 0

        if rules.get('requires_displacement'):
            total_rules += 1
            if has_displacement:
                score += 1.0

        if rules.get('requires_structure_break'):
            total_rules += 1
            if has_structure_break:
                score += 1.0

        if rules.get('fibonacci_related'):
            total_rules += 1
            if has_fib_alignment:
                score += 1.0

        if rules.get('significance_when_institutional'):
            total_rules += 1
            if is_institutional:
                score += 1.0

        return score / total_rules if total_rules > 0 else 0.5

    def get_directional_bias(self, detected_patterns: List[str],
                              analysis: Dict = None) -> Tuple[float, float, float]:
        """
        Compute video-learned directional bias from detected patterns.

        Returns (bullish_score, bearish_score, confidence) all in [0, 1].
        """
        if not detected_patterns:
            return 0.0, 0.0, 0.0

        bullish = 0.0
        bearish = 0.0
        total_weight = 0.0

        for pattern in detected_patterns:
            profile = self.get_concept_profile(pattern)
            if not profile:
                continue

            depth = self.get_teaching_depth_score(pattern)
            weight = max(depth, 0.1)  # minimum weight

            # Check pattern characteristics for directional bias
            chars_text = ' '.join(profile.characteristics).lower()
            has_bullish = 'bullish' in chars_text
            has_bearish = 'bearish' in chars_text

            if has_bullish and not has_bearish:
                bullish += weight
            elif has_bearish and not has_bullish:
                bearish += weight
            # Both or neither: no directional signal from this pattern alone

            total_weight += weight

        if total_weight == 0:
            return 0.0, 0.0, 0.0

        # Use analysis bias if available
        if analysis:
            raw_bias = analysis.get('bias', '') if isinstance(analysis, dict) else getattr(analysis, 'bias', '')
            bias = str(raw_bias).lower() if raw_bias else ''
            if 'bullish' in bias:
                bullish += 0.3
                total_weight += 0.3
            elif 'bearish' in bias:
                bearish += 0.3
                total_weight += 0.3

        bullish_norm = min(bullish / total_weight, 1.0) if total_weight > 0 else 0.0
        bearish_norm = min(bearish / total_weight, 1.0) if total_weight > 0 else 0.0
        confidence = min(total_weight / 2.0, 1.0)  # Higher with more patterns

        return bullish_norm, bearish_norm, confidence

    # ==================== Status & Info ====================

    def get_status(self) -> Dict:
        """Get full status of video knowledge index."""
        if not self._loaded:
            return {
                'loaded': False,
                'concepts': 0,
                'videos': 0,
                'message': 'No video training data found',
            }

        concepts_list = []
        for name, profile in sorted(self.concept_profiles.items(),
                                     key=lambda x: x[1].teaching_depth, reverse=True):
            concepts_list.append(profile.to_dict())

        # Top co-occurrences
        top_pairs = []
        if self.co_occurrence_matrix is not None:
            concepts = sorted(self.concept_index.keys(), key=lambda c: self.concept_index[c])
            n = len(concepts)
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    score = float(self.co_occurrence_matrix[i, j])
                    if score > 0:
                        pairs.append((concepts[i], concepts[j], score))
            pairs.sort(key=lambda x: x[2], reverse=True)
            top_pairs = [
                {'concept_a': a, 'concept_b': b, 'score': round(s, 3)}
                for a, b, s in pairs[:10]
            ]

        return {
            'loaded': True,
            'concepts': len(self.concept_profiles),
            'videos': self._n_videos,
            'knowledge_bases': self._n_knowledge_bases,
            'concepts_list': concepts_list,
            'top_co_occurrences': top_pairs,
            'co_occurrence_matrix_size': self.co_occurrence_matrix.shape[0] if self.co_occurrence_matrix is not None else 0,
            'normalization': {
                'max_teaching_depth': self._max_teaching_depth,
                'max_words': self._max_words,
                'max_frames': self._max_frames,
            },
        }


# Singleton
_vk_instance = None


def get_video_knowledge() -> VideoKnowledgeIndex:
    """Get the singleton VideoKnowledgeIndex instance."""
    global _vk_instance
    if _vk_instance is None:
        _vk_instance = VideoKnowledgeIndex()
    return _vk_instance
