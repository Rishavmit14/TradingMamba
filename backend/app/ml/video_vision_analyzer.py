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

    def extract_frames_at_timestamps(
        self,
        video_id: str,
        timestamps: List[float],
        max_frames: int = 50
    ) -> List[Tuple[float, str]]:
        """
        Extract frames at specific timestamps.
        Returns list of (timestamp, frame_path) tuples.
        """
        import yt_dlp
        import subprocess

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_dir = self.output_dir / video_id
        video_dir.mkdir(exist_ok=True)

        # Limit number of frames
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
        extraction_mode: str = "balanced"
    ) -> List[Tuple[float, str, str]]:
        """
        Extract frames based on the specified extraction mode.
        Returns list of (timestamp, frame_path, transcript_context) tuples.

        Extraction modes:
        - "comprehensive": Extract frames at very short intervals (3-5 seconds)
          to capture EVERYTHING like a dedicated student. Misses nothing.
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
        if extraction_mode == "comprehensive":
            # Like a dedicated student - capture EVERYTHING
            # Extract frame every 3 seconds to miss nothing
            interval = 3.0
            min_gap = 2.5
            use_keywords = True  # Still mark keywords but extract everything
            logger.info(f"Comprehensive mode: Extracting frame every {interval}s for full coverage")

        elif extraction_mode == "thorough":
            # Good coverage with extra attention to key moments
            interval = 5.0
            min_gap = 3.0
            use_keywords = True
            logger.info(f"Thorough mode: Extracting frame every {interval}s with keyword boosting")

        elif extraction_mode == "selective":
            # Only extract at demonstrative moments - fastest but may miss content
            interval = 30.0  # Very sparse base intervals
            min_gap = 2.0
            use_keywords = True
            logger.info("Selective mode: Extracting only at key demonstrative moments")

        else:  # "balanced" - default
            interval = interval_seconds
            min_gap = 3.0
            use_keywords = keyword_boost

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

        # Extract frames
        timestamps_only = [ts for ts, _, _ in deduped]
        extracted = self.extract_frames_at_timestamps(video_id, timestamps_only)

        # Combine with context
        result = []
        ts_to_context = {ts: (reason, context) for ts, reason, context in deduped}
        for ts, path in extracted:
            # Find closest original timestamp
            closest = min(deduped, key=lambda x: abs(x[0] - ts))
            reason, context = ts_to_context.get(closest[0], ("interval", ""))
            result.append((ts, path, context))

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
        max_frames: int = 30,
        extraction_mode: str = "comprehensive",
        progress_callback=None,
        force_reprocess: bool = False
    ) -> Optional[VideoVisualSummary]:
        """
        Process a video: extract frames and analyze visually.

        Args:
            video_id: YouTube video ID
            max_frames: Maximum frames to analyze (0 = no limit)
            extraction_mode: How thoroughly to extract frames:
                - "comprehensive": Every 3s, learns EVERYTHING like a student
                - "thorough": Every 5s with keyword boosting
                - "balanced": Every 10-15s with keyword boosting
                - "selective": Only at demonstrative moments
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
