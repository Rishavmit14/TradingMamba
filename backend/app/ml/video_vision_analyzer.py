"""
Video Vision Analyzer - Multimodal Understanding for Trading Videos

This module extracts and analyzes visual content from trading tutorial videos,
enabling the ML to understand:
1. Chart patterns being shown (FVGs, Order Blocks, etc.)
2. Price action context when tutor says "this" or "here"
3. Visual annotations and drawings
4. Comparative analysis ("this one is stronger than that one")

Uses vision models (Claude/GPT-4V) to analyze extracted frames.
"""

import os
import json
import base64
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class FrameDeduplicator:
    """
    Smart frame deduplication using perceptual hashing and histogram comparison.
    Reduces ~120 frames to ~25-40 unique frames without losing important content.
    """

    def __init__(self, similarity_threshold: float = 0.92):
        """
        Args:
            similarity_threshold: Frames more similar than this are considered duplicates (0-1).
                                  Higher = more aggressive deduplication.
                                  0.92 is good for trading videos (same chart = duplicate)
        """
        self.similarity_threshold = similarity_threshold
        self._has_cv2 = None

    def _check_cv2(self) -> bool:
        """Check if OpenCV is available"""
        if self._has_cv2 is None:
            try:
                import cv2
                self._has_cv2 = True
            except ImportError:
                self._has_cv2 = False
                logger.warning("OpenCV not installed. Using basic deduplication. Install with: pip install opencv-python")
        return self._has_cv2

    def compute_image_hash(self, image_path: str) -> Optional[str]:
        """
        Compute perceptual hash (pHash) for an image.
        Similar images will have similar hashes.
        """
        if not self._check_cv2():
            return self._compute_basic_hash(image_path)

        import cv2
        import numpy as np

        try:
            # Read and resize to 32x32 for consistent hashing
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            # Resize to 32x32
            resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

            # Compute DCT (Discrete Cosine Transform)
            dct = cv2.dct(np.float32(resized))

            # Use top-left 8x8 of DCT (low frequencies = structure)
            dct_low = dct[:8, :8]

            # Compute median and create hash
            median = np.median(dct_low)
            hash_bits = (dct_low > median).flatten()

            # Convert to hex string
            hash_int = sum(bit << i for i, bit in enumerate(hash_bits))
            return format(hash_int, '016x')

        except Exception as e:
            logger.warning(f"Failed to compute hash for {image_path}: {e}")
            return self._compute_basic_hash(image_path)

    def _compute_basic_hash(self, image_path: str) -> Optional[str]:
        """Fallback: use file content hash (less smart but works)"""
        try:
            with open(image_path, 'rb') as f:
                # Read chunks to create a simple hash
                content = f.read(50000)  # First 50KB
                return hashlib.md5(content).hexdigest()[:16]
        except:
            return None

    def compute_histogram_similarity(self, img_path1: str, img_path2: str) -> float:
        """
        Compare two images using histogram correlation.
        Returns similarity score 0-1 (1 = identical).
        """
        if not self._check_cv2():
            # Fallback: compare file sizes as rough similarity
            try:
                size1 = os.path.getsize(img_path1)
                size2 = os.path.getsize(img_path2)
                return 1.0 - abs(size1 - size2) / max(size1, size2)
            except:
                return 0.0

        import cv2

        try:
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)

            if img1 is None or img2 is None:
                return 0.0

            # Convert to HSV for better color comparison
            hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

            # Compute histograms
            hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])

            # Normalize
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)

            # Compare using correlation
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            return max(0.0, similarity)  # Correlation can be negative

        except Exception as e:
            logger.warning(f"Histogram comparison failed: {e}")
            return 0.0

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hashes"""
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 64  # Max distance

        try:
            int1 = int(hash1, 16)
            int2 = int(hash2, 16)
            xor = int1 ^ int2
            return bin(xor).count('1')
        except:
            return 64

    def deduplicate_frames(
        self,
        frames: List[Tuple[float, str, str]],  # (timestamp, path, context)
        progress_callback=None,
        max_gap_seconds: float = 0  # 0 = no max gap enforcement
    ) -> List[Tuple[float, str, str]]:
        """
        Remove duplicate/similar frames from a list.

        Returns filtered list keeping only unique frames.
        Prioritizes frames with richer transcript context.

        Args:
            frames: List of (timestamp, path, context) tuples
            progress_callback: Optional callback for progress updates
            max_gap_seconds: Maximum allowed gap between kept frames (0 = no limit).
                             If > 0, forces frame retention even if similar to prevent
                             skipping too much teaching content.
        """
        if not frames:
            return []

        if len(frames) <= 5:
            return frames  # Not worth deduplicating tiny lists

        logger.info(f"Deduplicating {len(frames)} frames (max_gap={max_gap_seconds}s)...")

        # Compute hashes for all frames
        frame_hashes = []
        for i, (ts, path, context) in enumerate(frames):
            if progress_callback:
                progress_callback(i, len(frames), f"Computing hash for frame {i+1}/{len(frames)}")

            img_hash = self.compute_image_hash(path)
            frame_hashes.append((ts, path, context, img_hash))

        # Keep track of unique frames
        unique_frames = []

        for ts, path, context, img_hash in frame_hashes:
            is_duplicate = False

            # Get the timestamp of the last kept frame
            last_kept_ts = unique_frames[-1][0] if unique_frames else -999
            time_since_last = ts - last_kept_ts

            # SINCERE STUDENT RULE: If max_gap is set and we've gone too long
            # without keeping a frame, force keep this one regardless of similarity
            if max_gap_seconds > 0 and time_since_last >= max_gap_seconds:
                # Force keep - don't skip more than max_gap_seconds of teaching
                unique_frames.append((ts, path, context, img_hash))
                continue

            # Check against all kept frames for duplicates
            for kept_ts, kept_path, kept_context, kept_hash in unique_frames:
                # Method 1: Hash similarity (fast)
                if img_hash and kept_hash:
                    hamming = self.hamming_distance(img_hash, kept_hash)
                    hash_similarity = 1.0 - (hamming / 64.0)

                    if hash_similarity > self.similarity_threshold:
                        is_duplicate = True
                        break

                # Method 2: Histogram similarity (more accurate, for close calls)
                if not is_duplicate and abs(ts - kept_ts) < 15:  # Only compare nearby frames
                    hist_sim = self.compute_histogram_similarity(path, kept_path)
                    if hist_sim > self.similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_frames.append((ts, path, context, img_hash))

        # Return without hash
        result = [(ts, path, context) for ts, path, context, _ in unique_frames]

        reduction = len(frames) - len(result)
        reduction_pct = (reduction / len(frames)) * 100 if frames else 0
        logger.info(f"Deduplication complete: {len(frames)} → {len(result)} frames ({reduction_pct:.1f}% reduction)")

        return result

    def detect_scene_changes(
        self,
        video_path: str,
        threshold: float = 30.0,
        min_scene_length: float = 2.0
    ) -> List[float]:
        """
        Detect scene changes in a video using frame differencing.
        Returns list of timestamps where significant visual changes occur.

        This is MUCH faster than extracting all frames - only marks where to look.
        """
        if not self._check_cv2():
            logger.warning("OpenCV required for scene detection")
            return []

        import cv2
        import numpy as np

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                logger.warning("Could not get video FPS")
                return []

            scene_changes = [0.0]  # Always include start
            last_scene_time = 0.0

            prev_frame = None
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = frame_idx / fps
                frame_idx += 1

                # Only check every 0.5 seconds for speed
                if frame_idx % int(fps * 0.5) != 0:
                    continue

                # Convert to grayscale and resize for speed
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (160, 90))

                if prev_frame is not None:
                    # Compute absolute difference
                    diff = cv2.absdiff(small, prev_frame)
                    mean_diff = np.mean(diff)

                    # Scene change detected
                    if mean_diff > threshold and (current_time - last_scene_time) >= min_scene_length:
                        scene_changes.append(current_time)
                        last_scene_time = current_time

                prev_frame = small

            cap.release()

            logger.info(f"Detected {len(scene_changes)} scene changes in video")
            return scene_changes

        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return []


@dataclass
class FrameAnalysis:
    """Analysis result for a single video frame"""
    timestamp: float  # seconds into video
    frame_path: str
    transcript_context: str  # surrounding transcript text

    # Visual analysis results
    chart_detected: bool
    chart_type: Optional[str]  # candlestick, line, etc.
    timeframe_visible: Optional[str]  # if visible on chart
    symbol_visible: Optional[str]  # if visible on chart

    # Smart Money patterns detected visually
    patterns_detected: List[Dict]  # [{type: "FVG", location: "...", description: "..."}]
    annotations_detected: List[Dict]  # drawings, arrows, boxes, zones

    # Price levels and zones
    price_levels: List[Dict]  # support/resistance, entry/SL/TP
    market_structure: Optional[str]  # bullish/bearish bias visible

    # Contextual understanding
    visual_description: str  # Natural language description
    teaching_point: str  # What is being taught at this moment
    visual_text: List[str]  # Any text visible in the frame (labels, etc.)

    # Confidence scores
    chart_confidence: float
    pattern_confidence: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VideoVisualSummary:
    """Complete visual analysis summary for a video"""
    video_id: str
    title: str
    total_frames_analyzed: int
    chart_frames: int

    # Aggregated patterns from all frames
    all_patterns: List[Dict]
    pattern_frequency: Dict[str, int]  # pattern_type -> count

    # Key teaching moments
    key_moments: List[Dict]  # timestamp + what's being taught visually

    # Visual concepts learned
    visual_concepts: List[Dict]  # concepts with visual examples

    # Comparison examples (when tutor compares patterns)
    comparison_examples: List[Dict]

    analyzed_at: str

    def to_dict(self) -> Dict:
        return asdict(self)


class VideoFrameExtractor:
    """Extract frames from YouTube videos at key timestamps"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / "video_frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.deduplicator = FrameDeduplicator(similarity_threshold=0.92)

    def extract_frames_at_timestamps(
        self,
        video_id: str,
        timestamps: List[float],
        max_frames: int = 0  # 0 = no limit (recommended for smart mode)
    ) -> List[Tuple[float, str]]:
        """
        Extract frames at specific timestamps.
        Returns list of (timestamp, frame_path) tuples.

        Args:
            max_frames: Maximum frames to extract. 0 = no limit (let smart deduplication handle it)
        """
        import yt_dlp
        import subprocess

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_dir = self.output_dir / video_id
        video_dir.mkdir(exist_ok=True)

        # Limit number of frames only if specified
        if max_frames > 0:
            timestamps = timestamps[:max_frames]

        # First, download the video temporarily
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, f"{video_id}.mp4")

            ydl_opts = {
                'format': 'best[height<=720]',  # 720p for reasonable file size
                'outtmpl': video_path,
                'quiet': True,
                'no_warnings': True,
                'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
            except Exception as e:
                logger.error(f"Failed to download video {video_id}: {e}")
                return []

            if not os.path.exists(video_path):
                logger.error(f"Video file not found after download: {video_path}")
                return []

            # Extract frames using ffmpeg
            extracted = []
            for ts in timestamps:
                frame_name = f"frame_{ts:.2f}.jpg"
                frame_path = video_dir / frame_name

                if frame_path.exists():
                    extracted.append((ts, str(frame_path)))
                    continue

                try:
                    # Use ffmpeg to extract frame at timestamp
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(ts),
                        '-i', video_path,
                        '-vframes', '1',
                        '-q:v', '2',
                        str(frame_path)
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=30)

                    if frame_path.exists():
                        extracted.append((ts, str(frame_path)))
                except Exception as e:
                    logger.warning(f"Failed to extract frame at {ts}s: {e}")

            return extracted

    def extract_key_frames(
        self,
        video_id: str,
        transcript_segments: List[Dict],
        interval_seconds: float = 10.0,
        keyword_boost: bool = True,
        extraction_mode: str = "sincere_student",
        progress_callback=None
    ) -> List[Tuple[float, str, str]]:
        """
        Extract frames based on the specified extraction mode.
        Returns list of (timestamp, frame_path, transcript_context) tuples.

        Extraction modes:
        - "sincere_student" (RECOMMENDED): Like a dedicated student who never misses
          more than 15 seconds of teaching. Extracts every 3s, deduplicates similar
          frames, but ENFORCES max 15s gap. Best for ICT videos where important
          concepts are taught on static charts. ~60-75 min for 57-min video.
        - "smart": Extracts at short intervals then uses AI deduplication
          to remove similar frames. Fast but may skip long teaching segments on
          static charts. ~22 min for 57-min video.
        - "comprehensive": Extract frames at very short intervals (3-5 seconds)
          to capture EVERYTHING. Misses nothing but VERY SLOW (~6+ hrs for 57-min video).
        - "thorough": Extract frames at moderate intervals (5-8 seconds) with
          extra frames at key moments. Good balance of coverage and efficiency.
        - "balanced": Extract frames at regular intervals (10-15 seconds) with
          keyword boosting. Original behavior.
        - "selective": Extract frames only at demonstrative moments ("this", "here").
          Fastest but may miss important content.
        """

        # Keywords that suggest visual demonstration
        VISUAL_KEYWORDS = [
            "this", "here", "look", "see", "notice", "observe",
            "drawing", "mark", "highlight", "zone", "level",
            "fvg", "fair value gap", "order block", "liquidity",
            "breaker", "mitigation", "displacement", "imbalance",
            "structure", "swing", "high", "low", "premium", "discount"
        ]

        DEMONSTRATIVE_PHRASES = [
            "right here", "this one", "look at this", "see how",
            "this is", "that's a", "here we have", "notice the",
            "this shows", "as you can see", "look at the",
            "this fvg", "this order block", "the zone here"
        ]

        key_timestamps = []

        # Get video duration from transcript
        max_time = 0
        if transcript_segments:
            max_time = max(s.get('end_time', s.get('start_time', 0)) for s in transcript_segments)

        # Configure based on extraction mode
        if extraction_mode == "sincere_student":
            # SINCERE STUDENT MODE: Learn like a dedicated student
            # Never skip more than 15 seconds of teaching, even on static charts
            interval = 3.0  # Extract every 3 seconds
            min_gap = 2.5
            use_keywords = True
            use_deduplication = True
            max_gap_seconds = 15.0  # KEY: Never skip more than 15s of teaching
            logger.info(f"Sincere Student mode: Extracting every {interval}s, max {max_gap_seconds}s gap allowed")

        elif extraction_mode == "smart":
            # SMART MODE: Extract comprehensively, then deduplicate
            # This gives the quality of comprehensive with ~70% fewer frames to analyze
            interval = 3.0  # Start with comprehensive extraction
            min_gap = 2.5
            use_keywords = True
            use_deduplication = True  # Key difference: dedupe after extraction
            max_gap_seconds = 0  # No max gap enforcement
            logger.info(f"Smart mode: Extracting every {interval}s then deduplicating similar frames")

        elif extraction_mode == "comprehensive":
            # Like a dedicated student - capture EVERYTHING
            # Extract frame every 3 seconds to miss nothing
            interval = 3.0
            min_gap = 2.5
            use_keywords = True  # Still mark keywords but extract everything
            use_deduplication = False
            max_gap_seconds = 0
            logger.info(f"Comprehensive mode: Extracting frame every {interval}s for full coverage")

        elif extraction_mode == "thorough":
            # Good coverage with extra attention to key moments
            interval = 5.0
            min_gap = 3.0
            use_keywords = True
            use_deduplication = False
            max_gap_seconds = 0
            logger.info(f"Thorough mode: Extracting frame every {interval}s with keyword boosting")

        elif extraction_mode == "selective":
            # Only extract at demonstrative moments - fastest but may miss content
            interval = 30.0  # Very sparse base intervals
            min_gap = 2.0
            use_keywords = True
            use_deduplication = False
            max_gap_seconds = 0
            logger.info("Selective mode: Extracting only at key demonstrative moments")

        else:  # "balanced" - default
            interval = interval_seconds
            min_gap = 3.0
            use_keywords = keyword_boost
            use_deduplication = False
            max_gap_seconds = 0

        # Add regular interval timestamps
        if max_time > 0:
            current = 0
            while current < max_time:
                # Get transcript context for this timestamp
                context = self._get_transcript_context_at_time(transcript_segments, current)
                key_timestamps.append((current, "interval", context))
                current += interval

        # Add keyword-triggered timestamps (extra frames at important moments)
        if use_keywords and transcript_segments:
            for seg in transcript_segments:
                text_lower = seg.get('text', '').lower()
                start_time = seg.get('start_time', 0)

                # Check for demonstrative phrases (highest priority)
                for phrase in DEMONSTRATIVE_PHRASES:
                    if phrase in text_lower:
                        key_timestamps.append((start_time, "demonstrative", seg.get('text', '')))
                        break
                else:
                    # Check for visual keywords
                    for keyword in VISUAL_KEYWORDS:
                        if keyword in text_lower:
                            key_timestamps.append((start_time, "keyword", seg.get('text', '')))
                            break

        # Deduplicate timestamps that are too close
        key_timestamps.sort(key=lambda x: x[0])
        deduped = []
        last_ts = -min_gap
        for ts, reason, context in key_timestamps:
            if ts - last_ts >= min_gap:
                deduped.append((ts, reason, context))
                last_ts = ts

        logger.info(f"Extraction mode '{extraction_mode}': {len(deduped)} frames to extract from {max_time:.0f}s video")

        if progress_callback:
            progress_callback(0, 3, f"Extracting {len(deduped)} frames from video...")

        # Extract frames
        timestamps_only = [ts for ts, _, _ in deduped]
        extracted = self.extract_frames_at_timestamps(video_id, timestamps_only)

        if progress_callback:
            progress_callback(1, 3, f"Extracted {len(extracted)} frames")

        # Combine with context
        result = []
        ts_to_context = {ts: (reason, context) for ts, reason, context in deduped}
        for ts, path in extracted:
            # Find closest original timestamp
            closest = min(deduped, key=lambda x: abs(x[0] - ts))
            reason, context = ts_to_context.get(closest[0], ("interval", ""))
            result.append((ts, path, context))

        # Apply smart deduplication if enabled
        if use_deduplication and len(result) > 5:
            if progress_callback:
                gap_msg = f" (max gap: {max_gap_seconds}s)" if max_gap_seconds > 0 else ""
                progress_callback(2, 3, f"Deduplicating {len(result)} frames{gap_msg}...")

            original_count = len(result)
            result = self.deduplicator.deduplicate_frames(
                result,
                progress_callback=None,
                max_gap_seconds=max_gap_seconds
            )

            reduction_pct = ((original_count - len(result)) / original_count) * 100
            logger.info(f"Smart deduplication: {original_count} → {len(result)} frames ({reduction_pct:.1f}% reduction)")

            if progress_callback:
                progress_callback(3, 3, f"Optimized: {original_count} → {len(result)} unique frames")

        return result

    def _get_transcript_context_at_time(
        self,
        segments: List[Dict],
        timestamp: float,
        window_seconds: float = 5.0
    ) -> str:
        """Get transcript text around a specific timestamp"""
        context_parts = []
        for seg in segments:
            seg_start = seg.get('start_time', 0)
            seg_end = seg.get('end_time', seg_start + 2)

            # Check if segment overlaps with our window
            if seg_start <= timestamp + window_seconds and seg_end >= timestamp - window_seconds:
                context_parts.append(seg.get('text', ''))

        return ' '.join(context_parts)[:500]  # Limit context length


class LocalVisionModel:
    """
    Local vision model using Ollama (optimized for Apple Silicon M1/M2/M3).
    Runs completely FREE on your Mac - no API costs!

    Requires Ollama to be installed: https://ollama.ai
    Then run: ollama pull llava
    """

    def __init__(self, model_name: str = "llava"):
        """
        Initialize local vision model.

        Recommended models for Ollama:
        - "llava" - LLaVA 7B, good balance (default)
        - "llava:13b" - LLaVA 13B, better quality
        - "bakllava" - BakLLaVA, alternative
        """
        self.model_name = model_name
        self._checked = False
        self.ollama_url = "http://localhost:11434"

    def _ensure_available(self):
        """Check if Ollama is running and model is available"""
        if self._checked:
            return

        import requests

        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise RuntimeError("Ollama is not responding")

            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]

            if self.model_name.split(":")[0] not in model_names:
                logger.warning(f"Model '{self.model_name}' not found. Pulling it now...")
                self._pull_model()

            self._checked = True
            logger.info(f"Ollama model '{self.model_name}' is ready!")

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Ollama is not running!\n\n"
                "To use FREE local vision analysis:\n"
                "1. Install Ollama from https://ollama.ai\n"
                "2. Start Ollama (it runs in the background)\n"
                "3. Run: ollama pull llava\n"
                "4. Try vision training again"
            )

    def _pull_model(self):
        """Pull the model using Ollama"""
        import requests

        logger.info(f"Pulling model '{self.model_name}'... This may take a few minutes.")
        response = requests.post(
            f"{self.ollama_url}/api/pull",
            json={"name": self.model_name},
            stream=True,
            timeout=600
        )

        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "status" in data:
                    logger.info(f"  {data['status']}")

    def analyze(self, image_path: str, prompt: str, max_tokens: int = 1500) -> str:
        """Analyze an image with a prompt using Ollama"""
        import requests

        self._ensure_available()

        # Encode image to base64
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Call Ollama API
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            },
            timeout=120
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.text}")

        result = response.json()
        return result.get("response", "")


class VisionAnalyzer:
    """Analyze video frames using vision AI models"""

    def __init__(self, provider: str = "local"):
        """
        Initialize vision analyzer.

        Args:
            provider:
                - "local" - FREE! Uses mlx-vlm on Apple Silicon (recommended)
                - "anthropic" - Claude API (costs money)
                - "openai" - GPT-4V API (costs money)
        """
        self.provider = provider

        if provider == "local":
            # Free local model for Apple Silicon
            self.local_model = LocalVisionModel()
            self.client = None
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
            self.local_model = None
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI()
            self.local_model = None
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'local', 'anthropic', or 'openai'")

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _get_image_media_type(self, image_path: str) -> str:
        """Get MIME type for image"""
        ext = Path(image_path).suffix.lower()
        return {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')

    def analyze_frame(
        self,
        image_path: str,
        transcript_context: str = "",
        timestamp: float = 0
    ) -> FrameAnalysis:
        """
        Analyze a single video frame for trading content.
        """

        prompt = f"""Analyze this trading video screenshot. The tutor was saying: "{transcript_context}"

Please identify and describe:

1. **Chart Detection**: Is there a trading chart visible? What type (candlestick, line, etc.)?

2. **Chart Details**: If a chart is present, identify:
   - Timeframe (if visible)
   - Symbol/Pair (if visible)
   - Market structure (bullish/bearish trend)

3. **Smart Money Patterns**: Identify any ICT/Smart Money patterns visible:
   - Fair Value Gaps (FVG) - price imbalances
   - Order Blocks - institutional zones
   - Breaker Blocks
   - Liquidity pools (equal highs/lows)
   - Market structure shifts (BOS/CHoCH)
   - Premium/Discount zones

   For each pattern, describe its location and characteristics.

4. **Annotations/Drawings**: Identify any visual annotations:
   - Rectangles/boxes marking zones
   - Lines (support/resistance, trendlines)
   - Arrows or pointers
   - Text labels
   - Highlighted areas

5. **Price Levels**: Note any significant price levels:
   - Support/Resistance levels
   - Entry zones
   - Stop loss levels
   - Take profit targets

6. **Teaching Point**: Based on the visual + transcript context, what specific concept is being taught at this moment?

7. **Visual Text**: List any text visible in the screenshot (labels, prices, etc.)

Respond in JSON format:
{{
    "chart_detected": true/false,
    "chart_type": "candlestick/line/none",
    "timeframe": "M15/H1/H4/D1/null",
    "symbol": "EURUSD/BTCUSD/null",
    "market_structure": "bullish/bearish/ranging/null",
    "patterns": [
        {{"type": "FVG", "location": "description", "characteristic": "bullish/bearish", "significance": "high/medium/low"}}
    ],
    "annotations": [
        {{"type": "rectangle/line/arrow/text", "description": "what it marks", "color": "if visible"}}
    ],
    "price_levels": [
        {{"type": "support/resistance/entry/sl/tp", "approximate_location": "description"}}
    ],
    "teaching_point": "What the tutor is demonstrating visually",
    "visual_text": ["list", "of", "visible", "text"],
    "visual_description": "Overall description of what's shown",
    "chart_confidence": 0.0-1.0,
    "pattern_confidence": 0.0-1.0
}}"""

        try:
            if self.provider == "local":
                # FREE local model on Apple Silicon
                result_text = self.local_model.analyze(image_path, prompt)

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": self._get_image_media_type(image_path),
                                        "data": self._encode_image(image_path)
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )
                result_text = response.content[0].text

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=2000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{self._get_image_media_type(image_path)};base64,{self._encode_image(image_path)}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )
                result_text = response.choices[0].message.content

            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            # Parse JSON from response
            # Handle markdown code blocks
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text.strip())

            return FrameAnalysis(
                timestamp=timestamp,
                frame_path=image_path,
                transcript_context=transcript_context,
                chart_detected=result.get("chart_detected", False),
                chart_type=result.get("chart_type"),
                timeframe_visible=result.get("timeframe"),
                symbol_visible=result.get("symbol"),
                patterns_detected=result.get("patterns", []),
                annotations_detected=result.get("annotations", []),
                price_levels=result.get("price_levels", []),
                market_structure=result.get("market_structure"),
                visual_description=result.get("visual_description", ""),
                teaching_point=result.get("teaching_point", ""),
                visual_text=result.get("visual_text", []),
                chart_confidence=result.get("chart_confidence", 0.0),
                pattern_confidence=result.get("pattern_confidence", 0.0)
            )

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return FrameAnalysis(
                timestamp=timestamp,
                frame_path=image_path,
                transcript_context=transcript_context,
                chart_detected=False,
                chart_type=None,
                timeframe_visible=None,
                symbol_visible=None,
                patterns_detected=[],
                annotations_detected=[],
                price_levels=[],
                market_structure=None,
                visual_description=f"Analysis failed: {str(e)}",
                teaching_point="",
                visual_text=[],
                chart_confidence=0.0,
                pattern_confidence=0.0
            )

    def analyze_video_frames(
        self,
        video_id: str,
        frames: List[Tuple[float, str, str]],  # timestamp, path, context
        progress_callback=None
    ) -> VideoVisualSummary:
        """
        Analyze all frames from a video and create a summary.
        """
        analyses = []
        total = len(frames)

        for i, (timestamp, frame_path, context) in enumerate(frames):
            if progress_callback:
                progress_callback(i + 1, total, f"Analyzing frame at {timestamp:.1f}s")

            analysis = self.analyze_frame(frame_path, context, timestamp)
            analyses.append(analysis)

        # Aggregate results
        chart_frames = sum(1 for a in analyses if a.chart_detected)

        # Collect all patterns
        all_patterns = []
        pattern_freq = {}
        for a in analyses:
            for p in a.patterns_detected:
                all_patterns.append({
                    "timestamp": a.timestamp,
                    **p
                })
                ptype = p.get("type", "unknown")
                pattern_freq[ptype] = pattern_freq.get(ptype, 0) + 1

        # Identify key teaching moments (high confidence frames with clear teaching points)
        key_moments = []
        for a in analyses:
            if a.teaching_point and a.pattern_confidence > 0.7:
                key_moments.append({
                    "timestamp": a.timestamp,
                    "teaching_point": a.teaching_point,
                    "patterns": a.patterns_detected,
                    "transcript": a.transcript_context
                })

        # Extract visual concepts (patterns with visual examples)
        visual_concepts = []
        seen_concepts = set()
        for a in analyses:
            for p in a.patterns_detected:
                ptype = p.get("type", "")
                if ptype and ptype not in seen_concepts and a.chart_confidence > 0.6:
                    seen_concepts.add(ptype)
                    visual_concepts.append({
                        "concept": ptype,
                        "visual_example_timestamp": a.timestamp,
                        "frame_path": a.frame_path,
                        "description": p.get("characteristic", ""),
                        "context": a.transcript_context
                    })

        return VideoVisualSummary(
            video_id=video_id,
            title="",  # Will be filled by caller
            total_frames_analyzed=len(analyses),
            chart_frames=chart_frames,
            all_patterns=all_patterns,
            pattern_frequency=pattern_freq,
            key_moments=key_moments,
            visual_concepts=visual_concepts,
            comparison_examples=[],  # TODO: detect comparison moments
            analyzed_at=datetime.utcnow().isoformat()
        )


class VideoVisionTrainer:
    """
    Integrate video vision analysis into the ML training pipeline.
    Combines transcript understanding with visual pattern recognition.
    """

    def __init__(self, data_dir: str, vision_provider: str = "anthropic"):
        self.data_dir = Path(data_dir)
        self.transcripts_dir = self.data_dir / "transcripts"
        self.vision_dir = self.data_dir / "video_vision"
        self.vision_dir.mkdir(parents=True, exist_ok=True)

        self.frame_extractor = VideoFrameExtractor(str(self.data_dir))
        self.vision_analyzer = VisionAnalyzer(provider=vision_provider)

    def process_video(
        self,
        video_id: str,
        max_frames: int = 0,  # 0 = no limit (let deduplication handle it)
        extraction_mode: str = "sincere_student",  # Sincere student learns everything, never skips >15s
        progress_callback=None,
        force_reprocess: bool = False
    ) -> Optional[VideoVisualSummary]:
        """
        Process a video: extract frames and analyze visually.

        Args:
            video_id: YouTube video ID
            max_frames: Maximum frames to analyze (0 = no limit, recommended for smart mode)
            extraction_mode: How thoroughly to extract frames:
                - "sincere_student" (RECOMMENDED): Like a dedicated student - never skips
                  more than 15s of teaching. Best for ICT videos. ~60-75 min for 57-min video.
                - "smart": Fast deduplication but may skip long teaching segments.
                  ~22 min for 57-min video.
                - "comprehensive": Every 3s, no deduplication. SLOW (~6+ hrs for 57-min video)
                - "thorough": Every 5s with keyword boosting
                - "balanced": Every 10-15s with keyword boosting
                - "selective": Only at demonstrative moments (fastest, may miss content)
            progress_callback: Optional callback(current, total, message)
            force_reprocess: Re-analyze even if already processed
        """
        # Check if already processed
        summary_path = self.vision_dir / f"{video_id}_vision.json"
        if summary_path.exists() and not force_reprocess:
            with open(summary_path) as f:
                data = json.load(f)
                return VideoVisualSummary(**data)

        # Load transcript
        transcript_path = self.transcripts_dir / f"{video_id}.json"
        if not transcript_path.exists():
            logger.warning(f"No transcript found for video {video_id}")
            return None

        with open(transcript_path) as f:
            transcript_data = json.load(f)

        segments = transcript_data.get("segments", [])
        title = transcript_data.get("title", video_id)

        if progress_callback:
            progress_callback(0, 3, f"Extracting frames ({extraction_mode} mode)...")

        # Extract frames based on mode
        frames = self.frame_extractor.extract_key_frames(
            video_id,
            segments,
            extraction_mode=extraction_mode
        )

        # Limit frames if specified (0 = no limit for comprehensive learning)
        if max_frames > 0:
            frames = frames[:max_frames]

        if not frames:
            logger.warning(f"No frames extracted for video {video_id}")
            return None

        logger.info(f"Processing video {video_id} with {len(frames)} frames ({extraction_mode} mode)")

        if progress_callback:
            progress_callback(1, 3, f"Analyzing {len(frames)} frames with AI vision...")

        # Analyze frames
        def frame_progress(current, total, msg):
            if progress_callback:
                progress_callback(1 + current/total, 3, msg)

        summary = self.vision_analyzer.analyze_video_frames(
            video_id,
            frames,
            progress_callback=frame_progress
        )
        summary.title = title

        # Save summary
        with open(summary_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)

        if progress_callback:
            progress_callback(3, 3, "Vision analysis complete")

        return summary

    def get_visual_knowledge(self, video_ids: List[str] = None) -> Dict:
        """
        Get aggregated visual knowledge from processed videos.
        """
        if video_ids is None:
            # Load all
            video_ids = [
                f.stem.replace("_vision", "")
                for f in self.vision_dir.glob("*_vision.json")
            ]

        all_patterns = []
        all_concepts = []
        all_moments = []
        pattern_counts = {}

        for vid in video_ids:
            summary_path = self.vision_dir / f"{vid}_vision.json"
            if not summary_path.exists():
                continue

            with open(summary_path) as f:
                data = json.load(f)

            all_patterns.extend(data.get("all_patterns", []))
            all_concepts.extend(data.get("visual_concepts", []))
            all_moments.extend(data.get("key_moments", []))

            for ptype, count in data.get("pattern_frequency", {}).items():
                pattern_counts[ptype] = pattern_counts.get(ptype, 0) + count

        return {
            "videos_analyzed": len(video_ids),
            "total_patterns_detected": len(all_patterns),
            "pattern_frequency": pattern_counts,
            "visual_concepts": all_concepts,
            "key_teaching_moments": all_moments,
            "patterns_by_type": self._group_patterns_by_type(all_patterns)
        }

    def _group_patterns_by_type(self, patterns: List[Dict]) -> Dict[str, List[Dict]]:
        """Group patterns by their type"""
        grouped = {}
        for p in patterns:
            ptype = p.get("type", "unknown")
            if ptype not in grouped:
                grouped[ptype] = []
            grouped[ptype].append(p)
        return grouped

    def enhance_transcript_with_vision(
        self,
        video_id: str,
        transcript_data: Dict
    ) -> Dict:
        """
        Enhance a transcript with visual analysis data.
        Adds visual context to each segment.
        """
        summary_path = self.vision_dir / f"{video_id}_vision.json"
        if not summary_path.exists():
            return transcript_data

        with open(summary_path) as f:
            vision_data = json.load(f)

        # Create timestamp -> visual info mapping
        visual_by_ts = {}
        for moment in vision_data.get("key_moments", []):
            ts = moment.get("timestamp", 0)
            visual_by_ts[ts] = moment

        # Enhance segments
        enhanced_segments = []
        for seg in transcript_data.get("segments", []):
            seg_start = seg.get("start_time", 0)

            # Find closest visual analysis
            closest_ts = min(visual_by_ts.keys(), key=lambda x: abs(x - seg_start), default=None)

            enhanced_seg = dict(seg)
            if closest_ts is not None and abs(closest_ts - seg_start) < 5:
                enhanced_seg["visual_context"] = visual_by_ts[closest_ts]

            enhanced_segments.append(enhanced_seg)

        return {
            **transcript_data,
            "segments": enhanced_segments,
            "vision_summary": {
                "patterns_detected": vision_data.get("pattern_frequency", {}),
                "visual_concepts": vision_data.get("visual_concepts", []),
                "frames_analyzed": vision_data.get("total_frames_analyzed", 0)
            }
        }
