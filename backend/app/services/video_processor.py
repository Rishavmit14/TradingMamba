"""
Video Processing Pipeline for ICT YouTube Videos

This module handles:
1. Downloading videos from YouTube playlists
2. Extracting audio for transcription
3. Transcribing using Whisper
4. Storing results for concept extraction
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    import whisper
except ImportError:
    whisper = None

from ..models.video import Video, Transcript, TranscriptSegment, Playlist, VideoStatus
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class PlaylistInfo:
    """Information about a YouTube playlist"""
    playlist_id: str
    title: str
    description: str
    video_count: int
    videos: List[Dict]


@dataclass
class ProcessingResult:
    """Result of processing a video"""
    video_id: str
    success: bool
    transcript: Optional[Transcript] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class VideoProcessor:
    """
    Process ICT YouTube videos for the learning pipeline

    Usage:
        processor = VideoProcessor()
        result = processor.process_video("dQw4w9WgXcQ")
    """

    def __init__(
        self,
        whisper_model: str = "large-v3",
        download_path: str = "/tmp/videos",
        transcript_path: str = "/tmp/transcripts"
    ):
        self.whisper_model_name = whisper_model
        self.download_path = Path(download_path)
        self.transcript_path = Path(transcript_path)

        # Create directories
        self.download_path.mkdir(parents=True, exist_ok=True)
        self.transcript_path.mkdir(parents=True, exist_ok=True)

        # Lazy load whisper model
        self._whisper_model = None

        # yt-dlp options
        self.ydl_opts_info = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        self.ydl_opts_audio = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }

    @property
    def whisper_model(self):
        """Lazy load Whisper model"""
        if self._whisper_model is None and whisper is not None:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            self._whisper_model = whisper.load_model(self.whisper_model_name)
        return self._whisper_model

    def get_video_info(self, video_id: str) -> Optional[Video]:
        """Get video information from YouTube"""
        if yt_dlp is None:
            logger.error("yt-dlp not installed")
            return None

        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            with yt_dlp.YoutubeDL(self.ydl_opts_info) as ydl:
                info = ydl.extract_info(url, download=False)

                return Video(
                    youtube_id=video_id,
                    title=info.get('title', ''),
                    description=info.get('description', ''),
                    duration_seconds=info.get('duration', 0),
                    published_at=datetime.strptime(
                        info.get('upload_date', '20000101'),
                        '%Y%m%d'
                    ) if info.get('upload_date') else None,
                )
        except Exception as e:
            logger.error(f"Error getting video info for {video_id}: {e}")
            return None

    def download_audio(self, video_id: str) -> Optional[str]:
        """Download audio from YouTube video"""
        if yt_dlp is None:
            logger.error("yt-dlp not installed")
            return None

        url = f"https://www.youtube.com/watch?v={video_id}"
        output_path = self.download_path / f"{video_id}.mp3"

        # Check if already downloaded
        if output_path.exists():
            logger.info(f"Audio already exists: {output_path}")
            return str(output_path)

        opts = self.ydl_opts_audio.copy()
        opts['outtmpl'] = str(self.download_path / f"{video_id}.%(ext)s")

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
                logger.info(f"Downloaded audio: {output_path}")
                return str(output_path)
        except Exception as e:
            logger.error(f"Error downloading audio for {video_id}: {e}")
            return None

    def transcribe_audio(self, audio_path: str) -> Optional[Transcript]:
        """Transcribe audio file using Whisper"""
        if self.whisper_model is None:
            logger.error("Whisper not available")
            return None

        try:
            logger.info(f"Transcribing: {audio_path}")

            result = self.whisper_model.transcribe(
                audio_path,
                verbose=False,
                word_timestamps=True,
                language='en'
            )

            # Create transcript segments
            segments = []
            for seg in result.get('segments', []):
                segment = TranscriptSegment(
                    start_time=seg['start'],
                    end_time=seg['end'],
                    text=seg['text'].strip(),
                    confidence=seg.get('avg_logprob', 0),
                    words=seg.get('words', [])
                )
                segments.append(segment)

            # Create full transcript
            transcript = Transcript(
                segments=segments,
                full_text=result.get('text', ''),
                language=result.get('language', 'en'),
                avg_confidence=sum(s.confidence for s in segments) / len(segments) if segments else 0,
                word_count=len(result.get('text', '').split())
            )

            logger.info(f"Transcription complete: {len(segments)} segments, {transcript.word_count} words")
            return transcript

        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return None

    def save_transcript(self, video_id: str, transcript: Transcript) -> str:
        """Save transcript to file"""
        output_path = self.transcript_path / f"{video_id}.json"

        data = {
            'id': transcript.id,
            'video_id': video_id,
            'full_text': transcript.full_text,
            'language': transcript.language,
            'avg_confidence': transcript.avg_confidence,
            'word_count': transcript.word_count,
            'segments': [
                {
                    'id': seg.id,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'text': seg.text,
                    'confidence': seg.confidence,
                }
                for seg in transcript.segments
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved transcript: {output_path}")
        return str(output_path)

    def load_transcript(self, video_id: str) -> Optional[Transcript]:
        """Load transcript from file"""
        input_path = self.transcript_path / f"{video_id}.json"

        if not input_path.exists():
            return None

        try:
            with open(input_path, 'r') as f:
                data = json.load(f)

            segments = [
                TranscriptSegment(
                    id=seg['id'],
                    video_id=video_id,
                    start_time=seg['start_time'],
                    end_time=seg['end_time'],
                    text=seg['text'],
                    confidence=seg['confidence'],
                )
                for seg in data.get('segments', [])
            ]

            return Transcript(
                id=data['id'],
                video_id=video_id,
                segments=segments,
                full_text=data['full_text'],
                language=data['language'],
                avg_confidence=data['avg_confidence'],
                word_count=data['word_count'],
            )
        except Exception as e:
            logger.error(f"Error loading transcript for {video_id}: {e}")
            return None

    def process_video(self, video_id: str, force: bool = False) -> ProcessingResult:
        """
        Complete processing pipeline for a single video

        1. Get video info
        2. Download audio
        3. Transcribe
        4. Save results

        Args:
            video_id: YouTube video ID
            force: Force reprocessing even if transcript exists

        Returns:
            ProcessingResult with transcript or error
        """
        import time
        start_time = time.time()

        # Check for existing transcript
        if not force:
            existing = self.load_transcript(video_id)
            if existing:
                logger.info(f"Using existing transcript for {video_id}")
                return ProcessingResult(
                    video_id=video_id,
                    success=True,
                    transcript=existing,
                    processing_time=time.time() - start_time
                )

        # Get video info
        video = self.get_video_info(video_id)
        if not video:
            return ProcessingResult(
                video_id=video_id,
                success=False,
                error="Failed to get video info",
                processing_time=time.time() - start_time
            )

        # Download audio
        audio_path = self.download_audio(video_id)
        if not audio_path:
            return ProcessingResult(
                video_id=video_id,
                success=False,
                error="Failed to download audio",
                processing_time=time.time() - start_time
            )

        # Transcribe
        transcript = self.transcribe_audio(audio_path)
        if not transcript:
            return ProcessingResult(
                video_id=video_id,
                success=False,
                error="Failed to transcribe audio",
                processing_time=time.time() - start_time
            )

        transcript.video_id = video_id

        # Save transcript
        self.save_transcript(video_id, transcript)

        # Clean up audio file to save space (optional)
        # os.remove(audio_path)

        return ProcessingResult(
            video_id=video_id,
            success=True,
            transcript=transcript,
            processing_time=time.time() - start_time
        )


class PlaylistProcessor:
    """
    Process entire YouTube playlists

    Handles the systematic processing of ICT playlists in order,
    ensuring progressive learning from basics to advanced.
    """

    def __init__(self, video_processor: Optional[VideoProcessor] = None):
        self.video_processor = video_processor or VideoProcessor()
        self.playlists_data_path = Path("data/playlists")
        self.playlists_data_path.mkdir(parents=True, exist_ok=True)

    def get_playlist_info(self, playlist_url: str) -> Optional[PlaylistInfo]:
        """Get information about a playlist including all videos"""
        if yt_dlp is None:
            logger.error("yt-dlp not installed")
            return None

        opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'dump_single_json': True,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)

                videos = []
                for i, entry in enumerate(info.get('entries', [])):
                    if entry:
                        videos.append({
                            'video_id': entry.get('id'),
                            'title': entry.get('title'),
                            'duration': entry.get('duration'),
                            'order': i + 1,
                        })

                return PlaylistInfo(
                    playlist_id=info.get('id', ''),
                    title=info.get('title', ''),
                    description=info.get('description', ''),
                    video_count=len(videos),
                    videos=videos
                )

        except Exception as e:
            logger.error(f"Error getting playlist info: {e}")
            return None

    def save_playlist_info(self, playlist_info: PlaylistInfo, learning_tier: int = 1) -> str:
        """Save playlist information to file"""
        output_path = self.playlists_data_path / f"{playlist_info.playlist_id}.json"

        data = {
            'playlist_id': playlist_info.playlist_id,
            'title': playlist_info.title,
            'description': playlist_info.description,
            'video_count': playlist_info.video_count,
            'learning_tier': learning_tier,
            'videos': playlist_info.videos,
            'fetched_at': datetime.utcnow().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved playlist info: {output_path}")
        return str(output_path)

    def process_playlist(
        self,
        playlist_url: str,
        learning_tier: int = 1,
        max_videos: Optional[int] = None,
        start_index: int = 0
    ) -> List[ProcessingResult]:
        """
        Process all videos in a playlist

        Args:
            playlist_url: YouTube playlist URL
            learning_tier: Learning tier (1=Foundation, 2=Core, 3=Advanced, 4=Mastery)
            max_videos: Maximum number of videos to process (None for all)
            start_index: Index to start processing from

        Returns:
            List of ProcessingResult for each video
        """
        # Get playlist info
        playlist_info = self.get_playlist_info(playlist_url)
        if not playlist_info:
            logger.error(f"Failed to get playlist info: {playlist_url}")
            return []

        logger.info(f"Processing playlist: {playlist_info.title}")
        logger.info(f"Total videos: {playlist_info.video_count}")

        # Save playlist info
        self.save_playlist_info(playlist_info, learning_tier)

        # Process videos
        results = []
        videos_to_process = playlist_info.videos[start_index:]

        if max_videos:
            videos_to_process = videos_to_process[:max_videos]

        for i, video_data in enumerate(videos_to_process):
            video_id = video_data['video_id']
            logger.info(f"Processing video {i + 1}/{len(videos_to_process)}: {video_data['title']}")

            result = self.video_processor.process_video(video_id)
            results.append(result)

            if result.success:
                logger.info(f"✓ Completed: {video_data['title']}")
            else:
                logger.error(f"✗ Failed: {video_data['title']} - {result.error}")

        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Playlist processing complete: {successful}/{len(results)} videos successful")

        return results


# Playlist URLs with metadata
ICT_PLAYLISTS = [
    {
        "name": "ICT Market Maker Forex Series",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paZ0R9gMaq0y2fM_2vyNJadp",
        "learning_tier": 1,
        "order": 1,
        "description": "Foundation series on market maker concepts"
    },
    {
        "name": "ICT OTE Pattern Recognition Series",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paaRnabpBl38GoMkxF1FiXCF",
        "learning_tier": 1,
        "order": 2,
        "description": "Optimal Trade Entry pattern recognition"
    },
    {
        "name": "ICT Forex Market Maker Primer Course",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paah1dHyad1OMJQJdm6iP2Yn",
        "learning_tier": 1,
        "order": 3,
        "description": "2016 primer course on market maker concepts"
    },
    {
        "name": "ICT Private Mentorship Core Content Month 01",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paYzh3KwUFX0UHQUf31CAEXk",
        "learning_tier": 2,
        "order": 4,
        "description": "2016 Private Mentorship Month 1"
    },
    {
        "name": "ICT Private Mentorship Core Content Month 02",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paZvjqerfbn320myZ06L1MOB",
        "learning_tier": 2,
        "order": 5,
        "description": "2016 Private Mentorship Month 2"
    },
    {
        "name": "ICT Private Mentorship Core Content Month 03",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paaY69GotBJyZ7KN_U09ra2o",
        "learning_tier": 2,
        "order": 6,
        "description": "2016 Private Mentorship Month 3"
    },
    {
        "name": "ICT Private Mentorship Core Content Month 04",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63pabb9rl1nyG58TG8PG8yzuao",
        "learning_tier": 2,
        "order": 7,
        "description": "2016 Private Mentorship Month 4"
    },
    {
        "name": "ICT Private Mentorship Core Content Month 05",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paYBN404Q2QZ7D4mOJz1IHAk",
        "learning_tier": 2,
        "order": 8,
        "description": "2017 Private Mentorship Month 5"
    },
    {
        "name": "ICT Private Mentorship Core Content Month 06",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paaG-26YEf2svQ_EsdGXjws1",
        "learning_tier": 2,
        "order": 9,
        "description": "2017 Private Mentorship Month 6"
    },
    {
        "name": "ICT Private Mentorship Core Content Month 07",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paYWV_3PDkYajv_oNznvK2aR",
        "learning_tier": 2,
        "order": 10,
        "description": "2017 Private Mentorship Month 7"
    },
]
