"""
Audio-First Rapid Training Pipeline - Complete End-to-End ML Training

This module provides a complete end-to-end pipeline for training ML from YouTube videos.
Just provide a URL and everything happens automatically.

USAGE:
    # Single video - end-to-end
    from backend.app.ml.audio_first_learning import train_from_url
    result = train_from_url('https://youtube.com/watch?v=VIDEO_ID')

    # Playlist - batch training
    from backend.app.ml.audio_first_learning import train_playlist
    results = train_playlist('https://youtube.com/playlist?list=PLxxx')

    # Or use the trainer directly for more control
    from backend.app.ml.audio_first_learning import AudioFirstTrainer
    trainer = AudioFirstTrainer()
    result = trainer.train_from_url('VIDEO_URL')

PIPELINE PHASES:
    Phase 0: Prerequisites (video_preprocessor.py)
        - Download audio from YouTube (pytubefix)
        - Extract frames at regular intervals (ffmpeg)
        - Transcribe audio (faster-whisper)

    Phase 1-3: Training
        - Parse transcript into segments
        - Detect teaching units and ICT concepts
        - Smart frame selection based on teaching context

    Phase 4: Vision Analysis
        - Analyze selected frames with context-enriched prompts
        - MLX-VLM for fast inference on Apple Silicon

    Phase 5: Knowledge Synthesis
        - Generate LLM summaries per ICT concept
        - Output: knowledge_base.json + knowledge_summary.md

KEY INNOVATION:
    Audio-First approach where audio is PRIMARY, frames are SECONDARY.
    Instead of analyzing every frame and adding audio context, we analyze
    audio first, detect teaching segments, then select the minimal set of
    frames needed to visually verify the concepts.

PERFORMANCE (for ~18 min video):
    - Phase 0: ~2-5 min (depends on download speed)
    - Phase 1-3: ~1 second
    - Phase 4: ~35 min (vision analysis)
    - Phase 5: ~3 min (knowledge synthesis)
    - Total: ~40-45 minutes
"""

import os
import re
import json
import logging
import bisect
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TranscriptWord:
    """A single word with timestamp"""
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class TranscriptSegment:
    """A segment of transcript with timing"""
    text: str
    start_time: float
    end_time: float
    words: List[TranscriptWord] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class TeachingUnit:
    """
    A logical unit of teaching - one concept or explanation.

    This is the core abstraction for Audio-First learning.
    Instead of treating each frame independently, we group
    related transcript segments into teaching units.
    """
    id: str
    start_time: float
    end_time: float
    text: str
    segments: List[TranscriptSegment]

    # Detected properties
    detected_concepts: List[str] = field(default_factory=list)
    teaching_type: str = "explanation"  # definition, example, annotation, summary
    deictic_references: List[Tuple[float, str]] = field(default_factory=list)
    confidence: float = 0.0

    # Selected frames for this unit
    selected_frames: List[str] = field(default_factory=list)
    frame_timestamps: List[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class FrameInfo:
    """Information about a video frame"""
    path: str
    timestamp: float

    # Visual analysis results (filled later)
    visual_hash: Optional[str] = None
    is_duplicate: bool = False
    visual_change_score: float = 0.0


@dataclass
class AudioFirstResult:
    """Result of Audio-First training pipeline"""
    video_id: str
    teaching_units: List[TeachingUnit]
    selected_frames: List[FrameInfo]
    total_frames_available: int
    frames_selected: int
    audio_coverage_percent: float
    processing_time_seconds: float

    # Statistics
    total_concepts_detected: int = 0
    total_deictic_references: int = 0


# =============================================================================
# ICT Concept Detection
# =============================================================================

# ICT-specific trading concepts to detect
ICT_CONCEPTS = {
    # Core Concepts
    "fair value gap": ["fair value gap", "fvg", "fair value gaps", "fvgs"],
    "order block": ["order block", "order blocks", "ob", "bullish order block", "bearish order block"],
    "liquidity": ["liquidity", "liquidity pool", "liquidity void", "buy side liquidity", "sell side liquidity"],
    "displacement": ["displacement", "displaced"],
    "imbalance": ["imbalance", "imbalances"],
    "breaker": ["breaker", "breaker block"],
    "mitigation": ["mitigation", "mitigate", "mitigated"],

    # Price Action
    "buy stops": ["buy stops", "buy stop"],
    "sell stops": ["sell stops", "sell stop"],
    "stop hunt": ["stop hunt", "stop run", "running stops"],
    "turtle soup": ["turtle soup"],
    "equal highs": ["equal highs", "equal high"],
    "equal lows": ["equal lows", "equal low"],

    # Market Structure
    "market structure": ["market structure", "structure"],
    "swing high": ["swing high", "swing highs"],
    "swing low": ["swing low", "swing lows"],
    "higher high": ["higher high", "higher highs"],
    "lower low": ["lower low", "lower lows"],

    # Time & Sessions
    "kill zone": ["kill zone", "kill zones"],
    "power of three": ["power of three", "power three"],
    "accumulation": ["accumulation"],
    "manipulation": ["manipulation"],
    "distribution": ["distribution"],

    # Entries
    "optimal trade entry": ["optimal trade entry", "ote"],
    "fibonacci": ["fibonacci", "fib"],

    # Institutional
    "smart money": ["smart money"],
    "institutional": ["institutional", "institutions"],
    "market maker": ["market maker", "market makers"],
}

# Deictic words that indicate visual reference
DEICTIC_WORDS = [
    "this", "here", "look", "see", "notice", "watch",
    "right here", "look at this", "look here", "see this",
    "as you can see", "notice how", "look at", "see how",
    "i'm drawing", "let me mark", "let me show", "pointing to",
    "highlighted", "shaded", "blue area", "this area",
    "this candle", "that candle", "this range", "that range",
]

# Teaching type indicators
TEACHING_PATTERNS = {
    "definition": [
        r"what is a[n]?\s+\w+",
        r"(\w+)\s+is\s+defined\s+as",
        r"let me explain",
        r"(\w+)\s+means",
    ],
    "example": [
        r"for example",
        r"let's take a look",
        r"here's an example",
        r"as you can see here",
        r"notice how",
    ],
    "annotation": [
        r"i'm drawing",
        r"let me mark",
        r"i'm highlighting",
        r"this area here",
        r"blue shaded",
    ],
    "summary": [
        r"so to summarize",
        r"in summary",
        r"the key point",
        r"remember that",
        r"the important thing",
    ],
}


def detect_concepts(text: str) -> List[str]:
    """Detect ICT concepts mentioned in text"""
    text_lower = text.lower()
    detected = []

    for concept, keywords in ICT_CONCEPTS.items():
        for keyword in keywords:
            if keyword in text_lower:
                if concept not in detected:
                    detected.append(concept)
                break

    return detected


def detect_deictic_references(text: str, start_time: float) -> List[Tuple[float, str]]:
    """
    Detect deictic references (words pointing to visual elements).
    Returns list of (approximate_timestamp, deictic_phrase)
    """
    references = []
    text_lower = text.lower()

    # Simple word-position based estimation
    words = text.split()
    words_per_second = len(words) / max(1, len(text) / 100)  # Rough estimate

    for deictic in DEICTIC_WORDS:
        if deictic in text_lower:
            # Find position in text
            pos = text_lower.find(deictic)
            word_pos = len(text_lower[:pos].split())

            # Estimate timestamp
            estimated_time = start_time + (word_pos / max(1, words_per_second))
            references.append((estimated_time, deictic))

    return references


def detect_teaching_type(text: str) -> str:
    """Detect the type of teaching in this segment"""
    text_lower = text.lower()

    for teaching_type, patterns in TEACHING_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return teaching_type

    return "explanation"


# =============================================================================
# Teaching Segment Detection
# =============================================================================

class TeachingSegmentDetector:
    """
    Detects logical teaching units from transcript.

    Groups related transcript segments based on:
    1. Topic continuity (same concept being discussed)
    2. Natural pauses (gaps > threshold)
    3. Transition phrases
    """

    def __init__(
        self,
        min_unit_duration: float = 5.0,
        max_unit_duration: float = 120.0,
        pause_threshold: float = 2.0,
        min_words_per_unit: int = 20,
    ):
        self.min_unit_duration = min_unit_duration
        self.max_unit_duration = max_unit_duration
        self.pause_threshold = pause_threshold
        self.min_words_per_unit = min_words_per_unit

        self.transition_phrases = [
            "okay", "now", "so", "alright", "let's",
            "moving on", "next", "another", "also",
        ]

    def detect_units(self, segments: List[TranscriptSegment]) -> List[TeachingUnit]:
        """
        Group transcript segments into teaching units.

        Returns list of TeachingUnit objects.
        """
        if not segments:
            return []

        units = []
        current_segments = []
        current_concepts = set()

        for i, segment in enumerate(segments):
            # Check if this is a boundary
            is_boundary = self._is_unit_boundary(
                segment,
                current_segments,
                current_concepts,
                segments[i-1] if i > 0 else None
            )

            if is_boundary and current_segments:
                # Create unit from accumulated segments
                unit = self._create_teaching_unit(current_segments, len(units))
                if unit.word_count >= self.min_words_per_unit:
                    units.append(unit)
                current_segments = []
                current_concepts = set()

            current_segments.append(segment)
            current_concepts.update(detect_concepts(segment.text))

        # Don't forget last unit
        if current_segments:
            unit = self._create_teaching_unit(current_segments, len(units))
            if unit.word_count >= self.min_words_per_unit:
                units.append(unit)

        logger.info(f"Detected {len(units)} teaching units from {len(segments)} segments")
        return units

    def _is_unit_boundary(
        self,
        current: TranscriptSegment,
        accumulated: List[TranscriptSegment],
        current_concepts: set,
        previous: Optional[TranscriptSegment]
    ) -> bool:
        """Determine if current segment starts a new teaching unit"""

        if not accumulated:
            return False

        # Check duration limit
        accumulated_duration = current.start_time - accumulated[0].start_time
        if accumulated_duration >= self.max_unit_duration:
            return True

        # Check for pause
        if previous:
            pause = current.start_time - previous.end_time
            if pause >= self.pause_threshold:
                return True

        # Check for transition phrase at start
        text_lower = current.text.lower().strip()
        for phrase in self.transition_phrases:
            if text_lower.startswith(phrase):
                # Only boundary if we have enough content
                if accumulated_duration >= self.min_unit_duration:
                    return True

        # Check for topic shift (new concepts not in current set)
        new_concepts = set(detect_concepts(current.text))
        if new_concepts and not new_concepts.intersection(current_concepts):
            if accumulated_duration >= self.min_unit_duration:
                return True

        return False

    def _create_teaching_unit(
        self,
        segments: List[TranscriptSegment],
        index: int
    ) -> TeachingUnit:
        """Create a TeachingUnit from accumulated segments"""

        # Combine text
        full_text = " ".join(seg.text for seg in segments)

        # Detect properties
        concepts = detect_concepts(full_text)
        teaching_type = detect_teaching_type(full_text)
        deictic_refs = detect_deictic_references(
            full_text,
            segments[0].start_time
        )

        # Calculate confidence based on concept density
        word_count = len(full_text.split())
        concept_density = len(concepts) / max(1, word_count / 50)
        confidence = min(1.0, concept_density)

        return TeachingUnit(
            id=f"unit_{index:03d}",
            start_time=segments[0].start_time,
            end_time=segments[-1].end_time,
            text=full_text,
            segments=segments,
            detected_concepts=concepts,
            teaching_type=teaching_type,
            deictic_references=deictic_refs,
            confidence=confidence,
        )


# =============================================================================
# Smart Frame Selection
# =============================================================================

class SmartFrameSelector:
    """
    Selects optimal frames for each teaching unit.

    Selection Strategy:
    1. Always capture frame at unit start (context)
    2. Capture frames at deictic references ("look here")
    3. Capture frames at visual changes
    4. For long static periods, single representative frame
    """

    def __init__(
        self,
        frames_dir: str,
        max_frames_per_unit: int = 5,
        min_frame_interval: float = 2.0,
        visual_change_threshold: float = 0.15,
    ):
        self.frames_dir = Path(frames_dir)
        self.max_frames_per_unit = max_frames_per_unit
        self.min_frame_interval = min_frame_interval
        self.visual_change_threshold = visual_change_threshold

        # Load frame pool
        self.frame_pool: Dict[float, FrameInfo] = {}
        self.sorted_timestamps: List[float] = []
        self._load_frame_pool()

    def _load_frame_pool(self):
        """Load all available frames into indexed pool"""
        if not self.frames_dir.exists():
            logger.warning(f"Frames directory not found: {self.frames_dir}")
            return

        for frame_file in self.frames_dir.glob("frame_*.jpg"):
            # Extract timestamp from filename
            match = re.search(r'frame_(\d+\.\d+)', frame_file.name)
            if match:
                timestamp = float(match.group(1))
                self.frame_pool[timestamp] = FrameInfo(
                    path=str(frame_file),
                    timestamp=timestamp
                )

        self.sorted_timestamps = sorted(self.frame_pool.keys())
        logger.info(f"Loaded {len(self.frame_pool)} frames into pool")

    def get_nearest_frame(self, timestamp: float) -> Optional[FrameInfo]:
        """Get the frame nearest to the given timestamp (O(log n))"""
        if not self.sorted_timestamps:
            return None

        # Binary search for nearest timestamp
        idx = bisect.bisect_left(self.sorted_timestamps, timestamp)

        if idx == 0:
            nearest = self.sorted_timestamps[0]
        elif idx == len(self.sorted_timestamps):
            nearest = self.sorted_timestamps[-1]
        else:
            before = self.sorted_timestamps[idx - 1]
            after = self.sorted_timestamps[idx]
            nearest = before if (timestamp - before) < (after - timestamp) else after

        return self.frame_pool.get(nearest)

    def get_frames_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[FrameInfo]:
        """Get all frames within a time range"""
        frames = []

        start_idx = bisect.bisect_left(self.sorted_timestamps, start_time)
        end_idx = bisect.bisect_right(self.sorted_timestamps, end_time)

        for idx in range(start_idx, end_idx):
            ts = self.sorted_timestamps[idx]
            frames.append(self.frame_pool[ts])

        return frames

    def select_frames_for_unit(self, unit: TeachingUnit) -> List[FrameInfo]:
        """
        Select optimal frames for a teaching unit.

        Returns list of selected FrameInfo objects.
        """
        selected: List[FrameInfo] = []
        selected_timestamps: set = set()

        def add_frame(frame: Optional[FrameInfo], reason: str = ""):
            if frame and frame.timestamp not in selected_timestamps:
                # Check minimum interval
                for ts in selected_timestamps:
                    if abs(frame.timestamp - ts) < self.min_frame_interval:
                        return
                selected.append(frame)
                selected_timestamps.add(frame.timestamp)
                logger.debug(f"Selected frame at {frame.timestamp:.2f}s: {reason}")

        # 1. Always capture frame at unit start
        start_frame = self.get_nearest_frame(unit.start_time)
        add_frame(start_frame, "unit_start")

        # 2. Capture frames at deictic references
        for timestamp, deictic in unit.deictic_references:
            deictic_frame = self.get_nearest_frame(timestamp)
            add_frame(deictic_frame, f"deictic: {deictic}")

        # 3. For definition/example types, capture at end too
        if unit.teaching_type in ["definition", "example"]:
            end_frame = self.get_nearest_frame(unit.end_time - 1.0)
            add_frame(end_frame, "unit_end")

        # 4. If unit is long and we have few frames, add middle frame
        if unit.duration > 30 and len(selected) < 2:
            mid_time = (unit.start_time + unit.end_time) / 2
            mid_frame = self.get_nearest_frame(mid_time)
            add_frame(mid_frame, "long_unit_midpoint")

        # 5. Detect visual changes if we have room for more frames
        if len(selected) < self.max_frames_per_unit:
            unit_frames = self.get_frames_in_range(unit.start_time, unit.end_time)
            change_frames = self._detect_visual_changes(unit_frames)
            for frame in change_frames:
                if len(selected) >= self.max_frames_per_unit:
                    break
                add_frame(frame, "visual_change")

        # Sort by timestamp
        selected.sort(key=lambda f: f.timestamp)

        # Update unit with selected frames
        unit.selected_frames = [f.path for f in selected]
        unit.frame_timestamps = [f.timestamp for f in selected]

        return selected

    def _detect_visual_changes(
        self,
        frames: List[FrameInfo]
    ) -> List[FrameInfo]:
        """
        Detect frames where significant visual changes occur.

        Uses simple histogram comparison for speed.
        More sophisticated methods can be added later.
        """
        if len(frames) < 2:
            return []

        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.warning("OpenCV not available for visual change detection")
            return []

        change_frames = []
        prev_hist = None

        for frame in frames:
            try:
                img = cv2.imread(frame.path)
                if img is None:
                    continue

                # Calculate histogram
                hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                                   [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                if prev_hist is not None:
                    # Compare with previous frame
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    frame.visual_change_score = diff

                    if diff > self.visual_change_threshold:
                        change_frames.append(frame)

                prev_hist = hist

            except Exception as e:
                logger.debug(f"Error processing frame {frame.path}: {e}")
                continue

        return change_frames

    def select_all(
        self,
        teaching_units: List[TeachingUnit]
    ) -> Tuple[List[FrameInfo], Dict[str, Any]]:
        """
        Select frames for all teaching units.

        Returns:
            - List of unique selected frames
            - Statistics dictionary
        """
        all_selected = []
        seen_timestamps = set()

        for unit in teaching_units:
            unit_frames = self.select_frames_for_unit(unit)
            for frame in unit_frames:
                if frame.timestamp not in seen_timestamps:
                    all_selected.append(frame)
                    seen_timestamps.add(frame.timestamp)

        # Sort by timestamp
        all_selected.sort(key=lambda f: f.timestamp)

        stats = {
            "total_frames_available": len(self.frame_pool),
            "frames_selected": len(all_selected),
            "selection_ratio": len(all_selected) / max(1, len(self.frame_pool)),
            "avg_frames_per_unit": len(all_selected) / max(1, len(teaching_units)),
        }

        logger.info(
            f"Selected {stats['frames_selected']} frames from "
            f"{stats['total_frames_available']} available "
            f"({stats['selection_ratio']:.1%} selection ratio)"
        )

        return all_selected, stats


# =============================================================================
# Context-Enriched Vision Prompts
# =============================================================================

class ContextEnrichedPromptBuilder:
    """
    Builds vision prompts that include audio context.

    This is the key innovation - instead of asking "what do you see?",
    we tell the model what ICT is teaching and ask it to find visual evidence.
    """

    def __init__(self):
        self.base_prompt_template = """You are analyzing a trading chart screenshot from an ICT (Inner Circle Trader) educational video.

TEACHING CONTEXT:
{teaching_context}

ICT TRANSCRIPT AT THIS MOMENT:
"{transcript_excerpt}"

DETECTED CONCEPTS: {concepts}
TEACHING TYPE: {teaching_type}

YOUR TASK:
1. Identify the specific chart patterns ICT is referencing
2. Extract precise price levels mentioned or visible
3. Note any annotations, markings, or highlighted areas
4. Describe how the visual supports the audio teaching
5. Identify the timeframe and trading pair if visible

FOCUS ESPECIALLY ON: {focus_concepts}

Provide your analysis in a structured format."""

    def build_prompt(
        self,
        unit: TeachingUnit,
        frame_timestamp: float
    ) -> str:
        """Build a context-enriched prompt for vision analysis"""

        # Get relevant transcript excerpt (around the frame timestamp)
        excerpt = self._get_transcript_excerpt(unit, frame_timestamp)

        # Format concepts
        concepts_str = ", ".join(unit.detected_concepts) if unit.detected_concepts else "general price action"

        # Determine teaching context
        teaching_context = self._get_teaching_context(unit)

        return self.base_prompt_template.format(
            teaching_context=teaching_context,
            transcript_excerpt=excerpt[:500],  # Limit length
            concepts=concepts_str,
            teaching_type=unit.teaching_type,
            focus_concepts=concepts_str,
        )

    def _get_transcript_excerpt(
        self,
        unit: TeachingUnit,
        frame_timestamp: float,
        window_seconds: float = 10.0
    ) -> str:
        """Get transcript excerpt around the frame timestamp"""

        # Find segments near the frame timestamp
        relevant_text = []
        for segment in unit.segments:
            if (segment.start_time <= frame_timestamp + window_seconds and
                segment.end_time >= frame_timestamp - window_seconds):
                relevant_text.append(segment.text)

        if relevant_text:
            return " ".join(relevant_text)
        return unit.text[:500]

    def _get_teaching_context(self, unit: TeachingUnit) -> str:
        """Generate teaching context description"""

        context_parts = []

        if unit.teaching_type == "definition":
            context_parts.append("ICT is DEFINING a concept")
        elif unit.teaching_type == "example":
            context_parts.append("ICT is showing a PRACTICAL EXAMPLE")
        elif unit.teaching_type == "annotation":
            context_parts.append("ICT is ANNOTATING the chart to highlight key areas")
        elif unit.teaching_type == "summary":
            context_parts.append("ICT is SUMMARIZING key points")
        else:
            context_parts.append("ICT is explaining a trading concept")

        if unit.detected_concepts:
            main_concept = unit.detected_concepts[0]
            context_parts.append(f"The main topic is: {main_concept.upper()}")

        if unit.deictic_references:
            context_parts.append("ICT is pointing to specific areas on the chart")

        return ". ".join(context_parts)


# =============================================================================
# Audio-First Training Pipeline
# =============================================================================

class AudioFirstTrainer:
    """
    Main orchestrator for Audio-First training pipeline.

    Pipeline:
    1. Load/transcribe audio
    2. Detect teaching segments
    3. Smart frame selection
    4. Context-enriched vision analysis
    5. Knowledge synthesis
    """

    def __init__(
        self,
        data_dir: str = "data",
        use_faster_whisper: bool = True,
        whisper_model: str = "small",
    ):
        self.data_dir = Path(data_dir)
        self.use_faster_whisper = use_faster_whisper
        self.whisper_model = whisper_model

        # Initialize components
        self.segment_detector = TeachingSegmentDetector()
        self.prompt_builder = ContextEnrichedPromptBuilder()

        # Paths
        self.audio_dir = self.data_dir / "audio"
        self.frames_dir = self.data_dir / "video_frames"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.output_dir = self.data_dir / "audio_first_training"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, video_id: str, force_retranscribe: bool = False) -> AudioFirstResult:
        """
        Run the complete Audio-First training pipeline.

        Args:
            video_id: YouTube video ID
            force_retranscribe: Force re-transcription even if exists

        Returns:
            AudioFirstResult with all training data
        """
        import time
        start_time = time.time()

        logger.info(f"Starting Audio-First training for: {video_id}")

        # Phase 1: Get transcript
        logger.info("Phase 1: Loading/generating transcript...")
        segments = self._get_transcript(video_id, force_retranscribe)
        logger.info(f"  → {len(segments)} transcript segments")

        # Phase 2: Detect teaching units
        logger.info("Phase 2: Detecting teaching units...")
        teaching_units = self.segment_detector.detect_units(segments)
        logger.info(f"  → {len(teaching_units)} teaching units detected")

        # Phase 3: Smart frame selection
        logger.info("Phase 3: Smart frame selection...")
        frame_selector = SmartFrameSelector(
            frames_dir=str(self.frames_dir / video_id)
        )
        selected_frames, selection_stats = frame_selector.select_all(teaching_units)
        logger.info(f"  → {len(selected_frames)} frames selected from {selection_stats['total_frames_available']}")

        # Calculate statistics
        total_concepts = sum(len(u.detected_concepts) for u in teaching_units)
        total_deictic = sum(len(u.deictic_references) for u in teaching_units)

        # Calculate audio coverage
        total_audio_time = segments[-1].end_time - segments[0].start_time if segments else 0
        covered_time = sum(u.duration for u in teaching_units)
        audio_coverage = (covered_time / total_audio_time * 100) if total_audio_time > 0 else 0

        processing_time = time.time() - start_time

        result = AudioFirstResult(
            video_id=video_id,
            teaching_units=teaching_units,
            selected_frames=selected_frames,
            total_frames_available=selection_stats["total_frames_available"],
            frames_selected=len(selected_frames),
            audio_coverage_percent=audio_coverage,
            processing_time_seconds=processing_time,
            total_concepts_detected=total_concepts,
            total_deictic_references=total_deictic,
        )

        # Save results
        self._save_results(video_id, result, teaching_units)

        logger.info(f"\n{'='*60}")
        logger.info("AUDIO-FIRST TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"  Video ID: {video_id}")
        logger.info(f"  Teaching Units: {len(teaching_units)}")
        logger.info(f"  Frames Selected: {len(selected_frames)} / {selection_stats['total_frames_available']}")
        logger.info(f"  Concepts Detected: {total_concepts}")
        logger.info(f"  Deictic References: {total_deictic}")
        logger.info(f"  Audio Coverage: {audio_coverage:.1f}%")
        logger.info(f"  Processing Time: {processing_time:.1f}s")
        logger.info(f"{'='*60}")

        return result

    def _get_transcript(
        self,
        video_id: str,
        force_retranscribe: bool = False
    ) -> List[TranscriptSegment]:
        """Get transcript, either from cache or by transcribing"""

        # Check for existing transcript
        transcript_path = self.transcripts_dir / f"{video_id}.json"

        if transcript_path.exists() and not force_retranscribe:
            logger.info(f"Loading existing transcript: {transcript_path}")
            return self._load_transcript(transcript_path)

        # Transcribe with faster-whisper
        audio_path = self.audio_dir / f"{video_id}.mp3"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing with faster-whisper: {audio_path}")
        return self._transcribe_audio(str(audio_path))

    def _load_transcript(self, path: Path) -> List[TranscriptSegment]:
        """Load transcript from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        segments = []
        for seg in data.get('segments', []):
            segments.append(TranscriptSegment(
                text=seg.get('text', ''),
                start_time=seg.get('start_time', 0),
                end_time=seg.get('end_time', seg.get('start_time', 0) + seg.get('duration', 0)),
            ))

        return segments

    def _transcribe_audio(self, audio_path: str) -> List[TranscriptSegment]:
        """Transcribe audio using faster-whisper"""
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
            segments_gen, info = model.transcribe(
                audio_path,
                language="en",
                word_timestamps=True,
                vad_filter=True,
            )

            segments = []
            for seg in segments_gen:
                segments.append(TranscriptSegment(
                    text=seg.text.strip(),
                    start_time=seg.start,
                    end_time=seg.end,
                ))

            return segments

        except ImportError:
            logger.error("faster-whisper not installed")
            raise

    def _save_results(
        self,
        video_id: str,
        result: AudioFirstResult,
        teaching_units: List[TeachingUnit]
    ):
        """Save training results to files"""

        # Save summary
        summary_path = self.output_dir / f"{video_id}_summary.json"
        summary = {
            "video_id": video_id,
            "teaching_units_count": len(teaching_units),
            "frames_selected": result.frames_selected,
            "total_frames_available": result.total_frames_available,
            "audio_coverage_percent": result.audio_coverage_percent,
            "processing_time_seconds": result.processing_time_seconds,
            "total_concepts_detected": result.total_concepts_detected,
            "total_deictic_references": result.total_deictic_references,
            "generated_at": datetime.now().isoformat(),
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save teaching units
        units_path = self.output_dir / f"{video_id}_teaching_units.json"
        units_data = []
        for unit in teaching_units:
            units_data.append({
                "id": unit.id,
                "start_time": unit.start_time,
                "end_time": unit.end_time,
                "duration": unit.duration,
                "text": unit.text,
                "detected_concepts": unit.detected_concepts,
                "teaching_type": unit.teaching_type,
                "deictic_references": unit.deictic_references,
                "confidence": unit.confidence,
                "selected_frames": unit.selected_frames,
                "frame_timestamps": unit.frame_timestamps,
                "word_count": unit.word_count,
            })

        with open(units_path, 'w') as f:
            json.dump(units_data, f, indent=2)

        # Save selected frames list
        frames_path = self.output_dir / f"{video_id}_selected_frames.json"
        frames_data = [
            {"path": f.path, "timestamp": f.timestamp}
            for f in result.selected_frames
        ]

        with open(frames_path, 'w') as f:
            json.dump(frames_data, f, indent=2)

        logger.info(f"Results saved to: {self.output_dir}")

    def get_vision_prompts(
        self,
        video_id: str
    ) -> List[Tuple[str, str, TeachingUnit]]:
        """
        Generate context-enriched vision prompts for selected frames.

        Returns list of (frame_path, prompt, teaching_unit) tuples.
        """
        # Load teaching units
        units_path = self.output_dir / f"{video_id}_teaching_units.json"
        if not units_path.exists():
            raise FileNotFoundError(f"Run train() first: {units_path}")

        with open(units_path, 'r') as f:
            units_data = json.load(f)

        prompts = []
        for unit_data in units_data:
            unit = TeachingUnit(
                id=unit_data["id"],
                start_time=unit_data["start_time"],
                end_time=unit_data["end_time"],
                text=unit_data["text"],
                segments=[],
                detected_concepts=unit_data["detected_concepts"],
                teaching_type=unit_data["teaching_type"],
                deictic_references=[tuple(d) for d in unit_data["deictic_references"]],
                confidence=unit_data["confidence"],
                selected_frames=unit_data["selected_frames"],
                frame_timestamps=unit_data["frame_timestamps"],
            )

            for frame_path, timestamp in zip(unit.selected_frames, unit.frame_timestamps):
                prompt = self.prompt_builder.build_prompt(unit, timestamp)
                prompts.append((frame_path, prompt, unit))

        return prompts

    def run_vision_analysis(
        self,
        video_id: str,
        max_frames: Optional[int] = None,
        use_mlx: bool = True,
        save_progress: bool = True,
        progress_callback=None,
        return_model: bool = False,
    ) -> Dict[str, Any]:
        """
        Run vision analysis on selected frames with context-enriched prompts.

        This is Phase 4 of Audio-First training - analyzing frames with
        full teaching context.

        Args:
            video_id: YouTube video ID
            max_frames: Limit frames to analyze (for testing)
            use_mlx: Use MLX-VLM (faster) or Ollama fallback
            save_progress: Save progress after each frame
            progress_callback: Optional callback(current, total, result)
            return_model: If True, return the MLX model for reuse in Phase 5

        Returns:
            Dictionary with analysis results (and optionally the model)
        """
        import time
        from .video_vision_analyzer import VisionAnalyzer

        logger.info(f"Starting Audio-First vision analysis for: {video_id}")

        # Get vision prompts (frame_path, prompt, teaching_unit)
        prompts = self.get_vision_prompts(video_id)
        if max_frames:
            prompts = prompts[:max_frames]

        logger.info(f"Analyzing {len(prompts)} frames with context-enriched prompts...")

        # Initialize vision analyzer
        vision_analyzer = VisionAnalyzer(use_mlx=use_mlx)

        # Get the actual model to use
        if use_mlx and vision_analyzer.mlx_model:
            model = vision_analyzer.mlx_model
            logger.info("Using MLX-VLM model for analysis")
        elif vision_analyzer.ollama_model:
            model = vision_analyzer.ollama_model
            logger.info("Using Ollama model for analysis")
        else:
            raise RuntimeError("No vision model available. Ensure MLX-VLM or Ollama is installed.")

        # Track results
        results = []
        start_time = time.time()
        successful = 0
        failed = 0

        # Load existing progress if any
        progress_path = self.output_dir / f"{video_id}_vision_progress.json"
        analyzed_frames = set()
        if progress_path.exists() and save_progress:
            with open(progress_path, 'r') as f:
                existing = json.load(f)
                results = existing.get('results', [])
                analyzed_frames = {r['frame_path'] for r in results}
                logger.info(f"Resuming from {len(analyzed_frames)} previously analyzed frames")

        for i, (frame_path, prompt, unit) in enumerate(prompts):
            # Skip if already analyzed
            if frame_path in analyzed_frames:
                continue

            frame_start = time.time()

            try:
                # Run vision analysis with context-enriched prompt
                analysis = model.analyze(
                    image_path=frame_path,
                    prompt=prompt,
                    max_tokens=1500
                )

                frame_time = time.time() - frame_start

                result = {
                    'frame_path': frame_path,
                    'timestamp': unit.frame_timestamps[0] if unit.frame_timestamps else 0,
                    'teaching_unit_id': unit.id,
                    'concepts': unit.detected_concepts,
                    'teaching_type': unit.teaching_type,
                    'analysis': analysis,
                    'processing_time': frame_time,
                    'success': True,
                }

                results.append(result)
                successful += 1

                logger.info(
                    f"[{i+1}/{len(prompts)}] Analyzed {frame_path.split('/')[-1]} "
                    f"({frame_time:.1f}s) - Concepts: {unit.detected_concepts}"
                )

                if progress_callback:
                    progress_callback(i + 1, len(prompts), result)

            except Exception as e:
                logger.error(f"Failed to analyze {frame_path}: {e}")
                results.append({
                    'frame_path': frame_path,
                    'teaching_unit_id': unit.id,
                    'success': False,
                    'error': str(e),
                })
                failed += 1

            # Save progress periodically
            if save_progress and (i + 1) % 5 == 0:
                self._save_vision_progress(video_id, results)

        # Final save
        total_time = time.time() - start_time

        final_results = {
            'video_id': video_id,
            'total_frames': len(prompts),
            'successful': successful,
            'failed': failed,
            'total_time_seconds': total_time,
            'avg_time_per_frame': total_time / max(1, successful),
            'results': results,
            'completed_at': datetime.now().isoformat(),
        }

        # Save final results
        self._save_vision_results(video_id, final_results)

        logger.info(f"\n{'='*60}")
        logger.info("AUDIO-FIRST VISION ANALYSIS COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"  Frames Analyzed: {successful}/{len(prompts)}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total Time: {total_time/60:.1f} minutes")
        logger.info(f"  Avg Time/Frame: {total_time/max(1,successful):.1f}s")
        logger.info(f"{'='*60}")

        # Return model for reuse in Phase 5 if requested
        if return_model:
            return final_results, model
        return final_results

    def _save_vision_progress(self, video_id: str, results: List[Dict]):
        """Save vision analysis progress"""
        progress_path = self.output_dir / f"{video_id}_vision_progress.json"
        with open(progress_path, 'w') as f:
            json.dump({'results': results}, f, indent=2)

    def _save_vision_results(self, video_id: str, results: Dict):
        """Save final vision analysis results"""
        results_path = self.output_dir / f"{video_id}_vision_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Vision results saved to: {results_path}")

    def synthesize_knowledge(
        self,
        video_id: str,
        use_llm: bool = True,
        mlx_model=None,
    ) -> Dict[str, Any]:
        """
        Synthesize knowledge from teaching units and vision analysis.

        Always uses LLM generation for rich, actionable summaries.

        Args:
            video_id: YouTube video ID
            use_llm: Always True - kept for API compatibility
            mlx_model: Optional pre-initialized MLX model (to avoid reloading)

        Args:
            video_id: YouTube video ID
            use_llm: Always True - kept for API compatibility

        Returns:
            Knowledge base dictionary with LLM-generated concept summaries
        """
        import time
        from collections import defaultdict

        logger.info(f"Starting knowledge synthesis for: {video_id}")
        start_time = time.time()

        # Load teaching units and vision analysis
        units_path = self.output_dir / f"{video_id}_teaching_units.json"
        vision_path = self.output_dir / f"{video_id}_vision_analysis.json"

        if not units_path.exists():
            raise FileNotFoundError(f"Run train() first: {units_path}")
        if not vision_path.exists():
            raise FileNotFoundError(f"Run run_vision_analysis() first: {vision_path}")

        with open(units_path, 'r') as f:
            teaching_units = json.load(f)

        with open(vision_path, 'r') as f:
            vision_data = json.load(f)

        # Organize data by concept
        logger.info("Organizing data by ICT concept...")
        concept_data = defaultdict(lambda: {
            'teaching_units': [],
            'total_duration': 0,
            'word_count': 0,
            'transcript_texts': [],
            'vision_texts': [],
            'frame_count': 0,
            'deictic_count': 0,
            'teaching_types': defaultdict(int),
        })

        # Map vision results by teaching unit ID
        vision_by_unit = defaultdict(list)
        for result in vision_data.get('results', []):
            if result.get('success'):
                vision_by_unit[result['teaching_unit_id']].append(result)

        # Process each teaching unit
        for unit in teaching_units:
            concepts = unit.get('detected_concepts', [])
            if not concepts:
                concepts = ['general_trading']

            for concept in concepts:
                concept_data[concept]['teaching_units'].append(unit['id'])
                concept_data[concept]['total_duration'] += unit['duration']
                concept_data[concept]['word_count'] += unit['word_count']
                concept_data[concept]['deictic_count'] += len(unit.get('deictic_references', []))
                concept_data[concept]['teaching_types'][unit['teaching_type']] += 1
                concept_data[concept]['transcript_texts'].append(unit['text'])

                for vision in vision_by_unit.get(unit['id'], []):
                    concept_data[concept]['frame_count'] += 1
                    concept_data[concept]['vision_texts'].append(vision.get('analysis', ''))

        # Memory cleanup before Phase 5 (only if no model provided)
        import gc

        if mlx_model is None:
            # Full cleanup needed - we need to load a new model
            logger.info("Clearing memory before Phase 5 (loading new model)...")
            gc.collect()

            # Try to clear MLX cache if available
            try:
                import mlx.core as mx
                mx.metal.clear_cache() if hasattr(mx.metal, 'clear_cache') else None
            except Exception:
                pass

            # Delay to allow memory to settle before loading model
            time.sleep(2)

            # Initialize new MLX model
            logger.info("Initializing MLX-VLM for concept summarization...")
            from .video_vision_analyzer import MLXVisionModel
            mlx_model = MLXVisionModel()
        else:
            # Model provided - just light cleanup, keep the model
            logger.info("Reusing MLX model from Phase 4 (no reload needed)")
            gc.collect()  # Light cleanup only

        # Use first frame as context image for MLX-VLM
        frames_dir = self.frames_dir / video_id
        first_frame = None
        if frames_dir.exists():
            frames = list(frames_dir.glob("frame_*.jpg"))
            if frames:
                first_frame = str(sorted(frames)[0])

        if not first_frame:
            raise FileNotFoundError(f"No frames found in: {frames_dir}")

        # Generate LLM summaries for each concept
        concepts_to_process = [c for c in concept_data.keys() if c != 'general_trading']
        logger.info(f"Generating LLM summaries for {len(concepts_to_process)} concepts...")

        concept_summaries = {}

        for i, concept in enumerate(concepts_to_process):
            data = concept_data[concept]
            concept_start = time.time()

            # Build rich context
            transcript_context = "\n\n".join(data['transcript_texts'][:3])[:1500]
            vision_context = "\n\n".join(data['vision_texts'][:2])[:800]

            prompt = f"""You are an expert on ICT (Inner Circle Trader) methodology. Based on the teaching transcript and chart analysis below, provide a comprehensive summary of the "{concept.upper()}" concept.

=== ICT TEACHING TRANSCRIPT ===
{transcript_context}

=== CHART ANALYSIS ===
{vision_context}

=== YOUR TASK ===
Write a trading-focused summary of {concept.upper()} that includes:

1. **Definition**: What is {concept}? (2-3 sentences)
2. **Identification**: How to spot it on a chart? (key visual characteristics)
3. **Trading Application**: How does ICT recommend trading this pattern?
4. **Key Price Levels**: What levels are important?

Keep the summary practical and actionable. Use ICT terminology."""

            # Retry mechanism for resilience
            max_retries = 3
            summary = None

            for retry in range(max_retries):
                try:
                    logger.info(f"[{i+1}/{len(concepts_to_process)}] Generating summary for: {concept}..." +
                               (f" (retry {retry+1})" if retry > 0 else ""))

                    # If retry, reinitialize the model
                    if retry > 0:
                        logger.info("    Reinitializing MLX model after failure...")
                        gc.collect()
                        time.sleep(2)
                        mlx_model = MLXVisionModel()

                    summary = mlx_model.analyze(
                        image_path=first_frame,
                        prompt=prompt,
                        max_tokens=800
                    )

                    concept_time = time.time() - concept_start

                    concept_summaries[concept] = {
                        'llm_summary': summary,
                        'generation_time_seconds': concept_time,
                        'statistics': {
                            'teaching_duration_seconds': data['total_duration'],
                            'word_count': data['word_count'],
                            'teaching_units': len(data['teaching_units']),
                            'frames_analyzed': data['frame_count'],
                            'deictic_references': data['deictic_count'],
                        },
                        'teaching_types': dict(data['teaching_types']),
                    }

                    logger.info(f"    Done in {concept_time:.1f}s")
                    break  # Success, exit retry loop

                except Exception as e:
                    logger.warning(f"    Attempt {retry+1} failed for {concept}: {e}")
                    if retry == max_retries - 1:
                        # Final failure
                        logger.error(f"    All retries failed for {concept}")
                        concept_summaries[concept] = {
                            'llm_summary': f"Error generating summary after {max_retries} attempts: {e}",
                            'generation_time_seconds': 0,
                            'statistics': {
                                'teaching_duration_seconds': data['total_duration'],
                                'word_count': data['word_count'],
                                'teaching_units': len(data['teaching_units']),
                                'frames_analyzed': data['frame_count'],
                            },
                        }

            # Garbage collection after each concept to prevent memory buildup
            gc.collect()

        total_time = time.time() - start_time

        # Build final knowledge base
        knowledge_base = {
            'video_id': video_id,
            'generated_at': datetime.now().isoformat(),
            'generation_method': 'LLM per concept (MLX-VLM)',
            'total_generation_time_seconds': total_time,
            'processing_stats': {
                'teaching_units': len(teaching_units),
                'vision_analyses': len(vision_data.get('results', [])),
                'concepts_extracted': len(concepts_to_process),
                'total_audio_duration': sum(u['duration'] for u in teaching_units),
                'total_words': sum(u['word_count'] for u in teaching_units),
            },
            'concepts': concept_summaries,
        }

        # Save knowledge base
        kb_path = self.output_dir / f"{video_id}_knowledge_base.json"
        with open(kb_path, 'w') as f:
            json.dump(knowledge_base, f, indent=2)

        # Save markdown summary
        md_path = self.output_dir / f"{video_id}_knowledge_summary.md"
        with open(md_path, 'w') as f:
            f.write("# ICT Knowledge Base (LLM-Generated)\n\n")
            f.write(f"**Video ID**: {video_id}\n")
            f.write(f"**Generated**: {knowledge_base['generated_at']}\n")
            f.write(f"**Method**: LLM generation per concept using MLX-VLM\n")
            f.write(f"**Total Time**: {total_time:.1f} seconds ({total_time/60:.1f} minutes)\n\n")
            f.write("## Processing Statistics\n\n")
            f.write(f"- Teaching Units: {knowledge_base['processing_stats']['teaching_units']}\n")
            f.write(f"- Vision Analyses: {knowledge_base['processing_stats']['vision_analyses']}\n")
            f.write(f"- Concepts Extracted: {knowledge_base['processing_stats']['concepts_extracted']}\n")
            f.write(f"- Total Audio: {knowledge_base['processing_stats']['total_audio_duration']/60:.1f} minutes\n\n")
            f.write("---\n\n")

            for concept in sorted(concept_summaries.keys()):
                data = concept_summaries[concept]
                stats = data.get('statistics', {})
                f.write(f"## {concept.upper().replace('_', ' ')}\n\n")
                f.write(f"*Teaching time: {stats.get('teaching_duration_seconds', 0)/60:.1f} min | ")
                f.write(f"Words: {stats.get('word_count', 0)} | ")
                f.write(f"Frames: {stats.get('frames_analyzed', 0)} | ")
                f.write(f"Generation time: {data.get('generation_time_seconds', 0):.1f}s*\n\n")
                f.write(f"{data.get('llm_summary', 'No summary available')}\n\n")
                f.write("---\n\n")

        logger.info(f"\n{'='*60}")
        logger.info("KNOWLEDGE SYNTHESIS COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"  Concepts: {len(concepts_to_process)}")
        logger.info(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"  Knowledge Base: {kb_path}")
        logger.info(f"  Markdown: {md_path}")
        logger.info(f"{'='*60}")

        return knowledge_base

    def run_full_pipeline(
        self,
        video_id: str,
        force_retranscribe: bool = False,
        skip_vision_if_exists: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete Audio-First training pipeline:
        1. Teaching unit detection
        2. Smart frame selection
        3. Vision analysis
        4. Knowledge synthesis (with LLM)

        Args:
            video_id: YouTube video ID
            force_retranscribe: Force re-transcription
            skip_vision_if_exists: Skip vision analysis if results exist

        Returns:
            Complete pipeline results
        """
        import time
        pipeline_start = time.time()

        logger.info("="*60)
        logger.info("AUDIO-FIRST FULL PIPELINE")
        logger.info("="*60)

        # Phase 1-3: Training (teaching units + frame selection)
        logger.info("\n[PHASE 1-3] Training: Teaching Units + Frame Selection")
        training_result = self.train(video_id, force_retranscribe)

        # Phase 4: Vision Analysis
        vision_path = self.output_dir / f"{video_id}_vision_analysis.json"
        mlx_model = None  # Will hold the model for reuse in Phase 5

        if skip_vision_if_exists and vision_path.exists():
            logger.info("\n[PHASE 4] Vision Analysis: Using existing results")
            with open(vision_path, 'r') as f:
                vision_result = json.load(f)
        else:
            logger.info("\n[PHASE 4] Vision Analysis: Running...")
            # Request model to be returned for reuse in Phase 5
            vision_result, mlx_model = self.run_vision_analysis(video_id, return_model=True)

        # Light memory cleanup (but keep the model!)
        import gc
        logger.info("\n[MEMORY CLEANUP] Light cleanup before Phase 5 (keeping MLX model)...")
        gc.collect()
        time.sleep(1)  # Brief pause

        # Phase 5: Knowledge Synthesis (reuse MLX model from Phase 4!)
        logger.info("\n[PHASE 5] Knowledge Synthesis: Generating LLM summaries...")
        if mlx_model:
            logger.info("  → Reusing MLX model from Phase 4 (no reload needed)")
        knowledge_base = self.synthesize_knowledge(video_id, mlx_model=mlx_model)

        total_time = time.time() - pipeline_start

        result = {
            'video_id': video_id,
            'pipeline_time_seconds': total_time,
            'training': {
                'teaching_units': len(training_result.teaching_units),
                'frames_selected': training_result.frames_selected,
                'audio_coverage': training_result.audio_coverage_percent,
            },
            'vision': {
                'frames_analyzed': vision_result.get('successful', 0),
                'failed': vision_result.get('failed', 0),
            },
            'knowledge': {
                'concepts': len(knowledge_base.get('concepts', {})),
                'generation_time': knowledge_base.get('total_generation_time_seconds', 0),
            },
        }

        logger.info("\n" + "="*60)
        logger.info("FULL PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"  Total Time: {total_time/60:.1f} minutes")
        logger.info(f"  Teaching Units: {result['training']['teaching_units']}")
        logger.info(f"  Frames Analyzed: {result['vision']['frames_analyzed']}")
        logger.info(f"  Concepts Synthesized: {result['knowledge']['concepts']}")
        logger.info("="*60)

        return result

    def train_from_url(
        self,
        url: str,
        force_preprocess: bool = False,
        skip_vision_if_exists: bool = True,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Complete end-to-end training from a YouTube URL.

        This is the main entry point for training. Just provide a URL
        and everything happens automatically:

        Phase 0: Prerequisites (download, extract frames, transcribe)
        Phase 1-3: Training (teaching units, frame selection)
        Phase 4: Vision Analysis
        Phase 5: Knowledge Synthesis

        Args:
            url: YouTube video URL or video ID
            force_preprocess: Force re-download and re-process
            skip_vision_if_exists: Skip vision if results exist
            progress_callback: Optional callback(phase, message)

        Returns:
            Complete pipeline results with all phases

        Usage:
            trainer = AudioFirstTrainer()
            result = trainer.train_from_url('https://youtube.com/watch?v=VIDEO_ID')
        """
        import time
        from .video_preprocessor import VideoPreprocessor

        total_start = time.time()

        logger.info("\n" + "="*70)
        logger.info("       COMPLETE END-TO-END ML TRAINING PIPELINE")
        logger.info("="*70)

        # Initialize preprocessor
        preprocessor = VideoPreprocessor(data_dir=str(self.data_dir))

        # Extract video ID for logging
        video_id = preprocessor.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from: {url}")

        logger.info(f"\nVideo ID: {video_id}")
        logger.info(f"URL: {url}")

        # =================================================================
        # PHASE 0: Prerequisites
        # =================================================================
        if progress_callback:
            progress_callback("phase0", "Preparing prerequisites...")

        logger.info("\n" + "-"*60)
        logger.info("[PHASE 0] PREREQUISITES")
        logger.info("-"*60)

        preprocess_result = preprocessor.prepare(
            url,
            force=force_preprocess,
            progress_callback=progress_callback,
        )

        if not preprocess_result.success:
            raise RuntimeError(f"Phase 0 failed: {preprocess_result.error}")

        phase0_time = preprocess_result.total_time

        logger.info(f"\nPhase 0 Complete:")
        logger.info(f"  Audio: {preprocess_result.audio_path}")
        logger.info(f"  Frames: {preprocess_result.frame_count}")
        logger.info(f"  Transcript: {preprocess_result.transcript_segments} segments")
        logger.info(f"  Time: {phase0_time:.1f}s")

        # =================================================================
        # PHASE 1-5: Audio-First Training Pipeline
        # =================================================================
        if progress_callback:
            progress_callback("training", "Running Audio-First training...")

        logger.info("\n" + "-"*60)
        logger.info("[PHASE 1-5] AUDIO-FIRST TRAINING")
        logger.info("-"*60)

        pipeline_result = self.run_full_pipeline(
            video_id,
            skip_vision_if_exists=skip_vision_if_exists,
        )

        total_time = time.time() - total_start

        # Combine results
        result = {
            'video_id': video_id,
            'url': url,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,

            # Phase 0 results
            'preprocessing': {
                'audio_path': preprocess_result.audio_path,
                'frames_dir': preprocess_result.frames_dir,
                'transcript_path': preprocess_result.transcript_path,
                'duration_seconds': preprocess_result.duration_seconds,
                'frame_count': preprocess_result.frame_count,
                'transcript_segments': preprocess_result.transcript_segments,
                'time_seconds': phase0_time,
            },

            # Phase 1-5 results
            'training': pipeline_result['training'],
            'vision': pipeline_result['vision'],
            'knowledge': pipeline_result['knowledge'],

            # Metadata
            'title': preprocess_result.title,
            'channel': preprocess_result.channel,
            'completed_at': datetime.now().isoformat(),
        }

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("       END-TO-END TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"  Video: {preprocess_result.title}")
        logger.info(f"  Video ID: {video_id}")
        logger.info(f"  Duration: {preprocess_result.duration_seconds/60:.1f} minutes")
        logger.info(f"  ")
        logger.info(f"  Phase 0 (Preprocess): {phase0_time:.1f}s")
        logger.info(f"    - Frames: {preprocess_result.frame_count}")
        logger.info(f"    - Transcript: {preprocess_result.transcript_segments} segments")
        logger.info(f"  ")
        logger.info(f"  Phase 1-3 (Training): {pipeline_result['training']['teaching_units']} teaching units")
        logger.info(f"    - Frames Selected: {pipeline_result['training']['frames_selected']}")
        logger.info(f"    - Audio Coverage: {pipeline_result['training']['audio_coverage']:.1f}%")
        logger.info(f"  ")
        logger.info(f"  Phase 4 (Vision): {pipeline_result['vision']['frames_analyzed']} frames analyzed")
        logger.info(f"  ")
        logger.info(f"  Phase 5 (Knowledge): {pipeline_result['knowledge']['concepts']} concepts synthesized")
        logger.info(f"  ")
        logger.info(f"  TOTAL TIME: {total_time/60:.1f} minutes")
        logger.info("="*70)

        return result

    def train_playlist(
        self,
        playlist_url: str,
        max_videos: int = 0,
        force_preprocess: bool = False,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Train on all videos in a YouTube playlist - ONE VIDEO AT A TIME.

        IMPORTANT: This method processes videos SEQUENTIALLY, waiting for
        each video to complete entirely before starting the next. This
        prevents memory issues on systems with limited RAM.

        Args:
            playlist_url: YouTube playlist URL
            max_videos: Max videos to process (0 = all)
            force_preprocess: Force re-download and re-process
            progress_callback: Optional callback(video_index, total, video_id, phase)

        Returns:
            List of training results for each video
        """
        import gc
        import time
        from .video_preprocessor import VideoPreprocessor

        logger.info("\n" + "="*70)
        logger.info("       PLAYLIST SEQUENTIAL TRAINING")
        logger.info("       (One video at a time - memory safe)")
        logger.info("="*70)

        preprocessor = VideoPreprocessor(data_dir=str(self.data_dir))
        video_ids = preprocessor._get_playlist_videos(playlist_url)

        if not video_ids:
            raise ValueError(f"Could not get videos from playlist: {playlist_url}")

        if max_videos > 0:
            video_ids = video_ids[:max_videos]

        total_videos = len(video_ids)
        logger.info(f"Found {total_videos} videos in playlist")
        logger.info("Processing ONE video at a time to prevent memory issues...")

        results = []
        playlist_start_time = time.time()

        for i, video_id in enumerate(video_ids):
            video_num = i + 1

            logger.info("\n" + "#"*70)
            logger.info(f"#  VIDEO {video_num}/{total_videos}: {video_id}")
            logger.info("#"*70)

            if progress_callback:
                progress_callback(i, total_videos, video_id, "starting")

            # Force garbage collection before starting each video
            gc.collect()

            video_start_time = time.time()

            try:
                # Train this video completely before moving to next
                result = self.train_from_url(
                    video_id,
                    force_preprocess=force_preprocess,
                    progress_callback=lambda phase, msg: progress_callback(i, total_videos, video_id, phase) if progress_callback else None
                )

                video_time = time.time() - video_start_time
                result['video_training_time_minutes'] = video_time / 60
                results.append(result)

                concepts = result.get('knowledge', {}).get('concepts', 0)
                logger.info(f"\n>>> VIDEO {video_num}/{total_videos} COMPLETE <<<")
                logger.info(f"    Concepts extracted: {concepts}")
                logger.info(f"    Time: {video_time/60:.1f} minutes")

            except Exception as e:
                logger.error(f"\n>>> VIDEO {video_num}/{total_videos} FAILED <<<")
                logger.error(f"    Error: {e}")
                results.append({
                    'video_id': video_id,
                    'success': False,
                    'error': str(e),
                })

            # Save progress after each video (in case of crash)
            self._save_playlist_progress(playlist_url, results, video_ids)

            # Force garbage collection after each video
            gc.collect()

            # Small delay between videos to let system stabilize
            if i < total_videos - 1:
                logger.info("\nCleaning up memory before next video...")
                time.sleep(3)

        # Final Summary
        playlist_time = time.time() - playlist_start_time
        successful = sum(1 for r in results if r.get('knowledge'))
        failed = len(results) - successful
        total_concepts = sum(r.get('knowledge', {}).get('concepts', 0) for r in results)

        logger.info("\n" + "="*70)
        logger.info("       PLAYLIST TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"  Videos Processed: {len(results)}/{total_videos}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total Concepts Extracted: {total_concepts}")
        logger.info(f"  Total Time: {playlist_time/60:.1f} minutes")
        logger.info("="*70)

        # Log individual results
        logger.info("\nIndividual Results:")
        for i, r in enumerate(results):
            vid = r.get('video_id', 'unknown')
            if r.get('knowledge'):
                concepts = r['knowledge']['concepts']
                time_min = r.get('video_training_time_minutes', 0)
                logger.info(f"  {i+1}. {vid}: ✅ {concepts} concepts ({time_min:.1f} min)")
            else:
                error = r.get('error', 'Unknown error')[:50]
                logger.info(f"  {i+1}. {vid}: ❌ {error}")

        return results

    def _save_playlist_progress(
        self,
        playlist_url: str,
        results: List[Dict[str, Any]],
        all_video_ids: List[str]
    ):
        """Save playlist training progress to a file for recovery."""
        progress_file = self.output_dir / "playlist_progress.json"

        progress_data = {
            'playlist_url': playlist_url,
            'total_videos': len(all_video_ids),
            'completed': len(results),
            'video_ids': all_video_ids,
            'results': [
                {
                    'video_id': r.get('video_id'),
                    'success': bool(r.get('knowledge')),
                    'concepts': r.get('knowledge', {}).get('concepts', 0) if r.get('knowledge') else 0,
                    'error': r.get('error'),
                }
                for r in results
            ],
            'last_updated': datetime.now().isoformat(),
        }

        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

        logger.info(f"Progress saved: {len(results)}/{len(all_video_ids)} videos")


# =============================================================================
# Convenience Functions
# =============================================================================

def train_from_url(url: str, data_dir: str = "data", **kwargs) -> Dict[str, Any]:
    """
    Complete end-to-end training from a YouTube URL.

    This is the simplest way to train - just provide a URL.

    Args:
        url: YouTube video URL or video ID
        data_dir: Base data directory
        **kwargs: Additional arguments

    Returns:
        Complete training results

    Usage:
        from backend.app.ml.audio_first_learning import train_from_url
        result = train_from_url('https://youtube.com/watch?v=VIDEO_ID')
    """
    trainer = AudioFirstTrainer(data_dir=data_dir)
    return trainer.train_from_url(url, **kwargs)


def train_playlist(playlist_url: str, data_dir: str = "data", **kwargs) -> List[Dict[str, Any]]:
    """
    Train on all videos in a YouTube playlist.

    Args:
        playlist_url: YouTube playlist URL
        data_dir: Base data directory
        **kwargs: Additional arguments

    Returns:
        List of training results

    Usage:
        from backend.app.ml.audio_first_learning import train_playlist
        results = train_playlist('https://youtube.com/playlist?list=PLxxx')
    """
    trainer = AudioFirstTrainer(data_dir=data_dir)
    return trainer.train_playlist(playlist_url, **kwargs)


def run_audio_first_training(video_id: str, data_dir: str = "data") -> AudioFirstResult:
    """
    Convenience function to run Audio-First training.

    Args:
        video_id: YouTube video ID
        data_dir: Base data directory

    Returns:
        AudioFirstResult with training data
    """
    trainer = AudioFirstTrainer(data_dir=data_dir)
    return trainer.train(video_id)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    def print_usage():
        print("Audio-First ML Training Pipeline")
        print("="*50)
        print("\nUsage:")
        print("  python audio_first_learning.py <url_or_id>              # Train single video")
        print("  python audio_first_learning.py <playlist_url> --playlist  # Train playlist")
        print("  python audio_first_learning.py <video_id> --phases-only   # Phases 1-5 only (no download)")
        print("\nExamples:")
        print("  python audio_first_learning.py 'https://youtube.com/watch?v=FgacYSN9QEo'")
        print("  python audio_first_learning.py FgacYSN9QEo")
        print("  python audio_first_learning.py 'https://youtube.com/playlist?list=PLxxx' --playlist")
        print("\nOptions:")
        print("  --playlist      Process as YouTube playlist")
        print("  --phases-only   Skip Phase 0 (assume prerequisites exist)")
        print("  --force         Force re-download and re-process")

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    url_or_id = sys.argv[1]
    is_playlist = '--playlist' in sys.argv
    phases_only = '--phases-only' in sys.argv
    force = '--force' in sys.argv

    trainer = AudioFirstTrainer()

    if is_playlist:
        # Train entire playlist
        print(f"\nTraining playlist: {url_or_id}")
        results = trainer.train_playlist(url_or_id, force_preprocess=force)

        successful = sum(1 for r in results if r.get('knowledge'))
        print(f"\nPlaylist Training Complete!")
        print(f"  Videos: {len(results)}")
        print(f"  Successful: {successful}")

    elif phases_only:
        # Run only phases 1-5 (prerequisites must exist)
        from .video_preprocessor import VideoPreprocessor
        preprocessor = VideoPreprocessor()
        video_id = preprocessor.extract_video_id(url_or_id) or url_or_id

        print(f"\nRunning phases 1-5 for: {video_id}")
        result = trainer.run_full_pipeline(video_id)

        print(f"\nTraining Complete!")
        print(f"  Teaching Units: {result['training']['teaching_units']}")
        print(f"  Frames Analyzed: {result['vision']['frames_analyzed']}")
        print(f"  Concepts: {result['knowledge']['concepts']}")

    else:
        # Full end-to-end training
        print(f"\nFull end-to-end training: {url_or_id}")
        result = trainer.train_from_url(url_or_id, force_preprocess=force)

        print(f"\nTraining Complete!")
        print(f"  Video: {result.get('title', result['video_id'])}")
        print(f"  Duration: {result['preprocessing']['duration_seconds']/60:.1f} minutes")
        print(f"  Teaching Units: {result['training']['teaching_units']}")
        print(f"  Frames Analyzed: {result['vision']['frames_analyzed']}")
        print(f"  Concepts Extracted: {result['knowledge']['concepts']}")
        print(f"  Total Time: {result['total_time_minutes']:.1f} minutes")
