"""
Video Preprocessor - Phase 0 of Audio-First Training Pipeline

This module handles all prerequisite preparation for ML training:
1. Download video/audio from YouTube (single video or playlist)
2. Extract frames at regular intervals
3. Generate transcript using faster-whisper

After Phase 0, all data is ready for the Audio-First training pipeline.

Usage:
    from backend.app.ml.video_preprocessor import VideoPreprocessor

    # Single video
    preprocessor = VideoPreprocessor()
    result = preprocessor.prepare('https://youtube.com/watch?v=VIDEO_ID')

    # Playlist
    results = preprocessor.prepare_playlist('https://youtube.com/playlist?list=PLAYLIST_ID')
"""

import os
import re
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Result of video preprocessing (Phase 0)"""
    video_id: str
    success: bool

    # Paths
    audio_path: Optional[str] = None
    frames_dir: Optional[str] = None
    transcript_path: Optional[str] = None

    # Statistics
    duration_seconds: float = 0
    frame_count: int = 0
    transcript_segments: int = 0

    # Metadata
    title: Optional[str] = None
    channel: Optional[str] = None

    # Timing
    download_time: float = 0
    frame_extraction_time: float = 0
    transcription_time: float = 0
    total_time: float = 0

    # Errors
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class VideoPreprocessor:
    """
    Phase 0: Prepare all prerequisites for Audio-First training.

    This handles:
    - Video/Audio download from YouTube
    - Frame extraction at configurable intervals
    - Audio transcription with faster-whisper

    Supports both single videos and playlists.
    """

    def __init__(
        self,
        data_dir: str = "data",
        frame_interval: float = 3.0,  # Extract frame every N seconds
        whisper_model: str = "small",  # faster-whisper model size
        audio_format: str = "mp3",
        video_quality: str = "720p",
    ):
        """
        Initialize the preprocessor.

        Args:
            data_dir: Base directory for all data
            frame_interval: Seconds between frame extractions
            whisper_model: Model size for faster-whisper (tiny, base, small, medium, large)
            audio_format: Audio format (mp3, wav)
            video_quality: Video quality for frame extraction
        """
        self.data_dir = Path(data_dir)
        self.frame_interval = frame_interval
        self.whisper_model = whisper_model
        self.audio_format = audio_format
        self.video_quality = video_quality

        # Create directories
        self.audio_dir = self.data_dir / "audio"
        self.frames_dir = self.data_dir / "video_frames"
        self.transcripts_dir = self.data_dir / "transcripts"

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'(?:embed/)([a-zA-Z0-9_-]{11})',
            r'^([a-zA-Z0-9_-]{11})$',  # Just the ID
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def extract_playlist_id(self, url: str) -> Optional[str]:
        """Extract playlist ID from YouTube URL"""
        match = re.search(r'list=([a-zA-Z0-9_-]+)', url)
        return match.group(1) if match else None

    def is_playlist_url(self, url: str) -> bool:
        """Check if URL is a playlist"""
        return 'list=' in url and 'watch?v=' not in url

    def prepare(
        self,
        url_or_id: str,
        force: bool = False,
        skip_frames: bool = False,
        skip_transcript: bool = False,
        progress_callback=None,
    ) -> PreprocessResult:
        """
        Prepare a single video for training (Phase 0).

        Args:
            url_or_id: YouTube URL or video ID
            force: Force re-download/re-process even if exists
            skip_frames: Skip frame extraction (if frames already exist)
            skip_transcript: Skip transcription (if transcript already exists)
            progress_callback: Optional callback(phase, message)

        Returns:
            PreprocessResult with paths to all generated files
        """
        import time
        total_start = time.time()

        # Extract video ID
        video_id = self.extract_video_id(url_or_id)
        if not video_id:
            return PreprocessResult(
                video_id=url_or_id,
                success=False,
                error=f"Could not extract video ID from: {url_or_id}"
            )

        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 0: PREPROCESSING VIDEO {video_id}")
        logger.info(f"{'='*60}")

        result = PreprocessResult(video_id=video_id, success=True)
        warnings = []

        # Phase 0.1: Download Audio
        if progress_callback:
            progress_callback("download", f"Downloading audio for {video_id}...")

        logger.info("\n[0.1] Downloading audio...")
        download_start = time.time()
        audio_result = self._download_audio(video_id, force)
        result.download_time = time.time() - download_start

        if audio_result['success']:
            result.audio_path = audio_result['path']
            result.duration_seconds = audio_result.get('duration', 0)
            result.title = audio_result.get('title')
            result.channel = audio_result.get('channel')
            logger.info(f"    Audio: {result.audio_path}")
            logger.info(f"    Duration: {result.duration_seconds/60:.1f} minutes")
        else:
            result.success = False
            result.error = f"Audio download failed: {audio_result.get('error')}"
            return result

        # Phase 0.2: Extract Frames
        frames_exist = (self.frames_dir / video_id).exists() and \
                       len(list((self.frames_dir / video_id).glob("frame_*.jpg"))) > 10

        if skip_frames and frames_exist:
            logger.info("\n[0.2] Skipping frame extraction (frames exist)")
            result.frames_dir = str(self.frames_dir / video_id)
            result.frame_count = len(list((self.frames_dir / video_id).glob("frame_*.jpg")))
        else:
            if progress_callback:
                progress_callback("frames", f"Extracting frames from {video_id}...")

            logger.info("\n[0.2] Extracting frames...")
            frames_start = time.time()
            frames_result = self._extract_frames(video_id, force or not frames_exist)
            result.frame_extraction_time = time.time() - frames_start

            if frames_result['success']:
                result.frames_dir = frames_result['dir']
                result.frame_count = frames_result['count']
                logger.info(f"    Frames: {result.frame_count} extracted")
                logger.info(f"    Directory: {result.frames_dir}")
            else:
                warnings.append(f"Frame extraction failed: {frames_result.get('error')}")
                logger.warning(f"    Frame extraction failed: {frames_result.get('error')}")

        # Phase 0.3: Transcribe Audio
        transcript_path = self.transcripts_dir / f"{video_id}.json"
        if skip_transcript and transcript_path.exists():
            logger.info("\n[0.3] Skipping transcription (transcript exists)")
            result.transcript_path = str(transcript_path)
            # Count segments
            import json
            with open(transcript_path, 'r') as f:
                data = json.load(f)
                result.transcript_segments = len(data.get('segments', []))
        else:
            if progress_callback:
                progress_callback("transcribe", f"Transcribing audio for {video_id}...")

            logger.info("\n[0.3] Transcribing audio...")
            transcribe_start = time.time()
            transcript_result = self._transcribe_audio(video_id, result.audio_path, force)
            result.transcription_time = time.time() - transcribe_start

            if transcript_result['success']:
                result.transcript_path = transcript_result['path']
                result.transcript_segments = transcript_result['segments']
                logger.info(f"    Transcript: {result.transcript_segments} segments")
                logger.info(f"    Path: {result.transcript_path}")
            else:
                warnings.append(f"Transcription failed: {transcript_result.get('error')}")
                logger.warning(f"    Transcription failed: {transcript_result.get('error')}")

        result.total_time = time.time() - total_start
        result.warnings = warnings

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 0 COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"  Video ID: {video_id}")
        logger.info(f"  Title: {result.title}")
        logger.info(f"  Duration: {result.duration_seconds/60:.1f} minutes")
        logger.info(f"  Frames: {result.frame_count}")
        logger.info(f"  Transcript Segments: {result.transcript_segments}")
        logger.info(f"  Total Time: {result.total_time:.1f}s")
        if warnings:
            logger.warning(f"  Warnings: {len(warnings)}")
        logger.info(f"{'='*60}")

        return result

    def prepare_playlist(
        self,
        playlist_url: str,
        max_videos: int = 0,  # 0 = all
        force: bool = False,
        progress_callback=None,
    ) -> List[PreprocessResult]:
        """
        Prepare all videos in a playlist for training.

        Args:
            playlist_url: YouTube playlist URL
            max_videos: Maximum videos to process (0 = all)
            force: Force re-download/re-process
            progress_callback: Optional callback(video_index, total, video_id, phase)

        Returns:
            List of PreprocessResult for each video
        """
        logger.info(f"\n{'='*60}")
        logger.info("PHASE 0: PREPROCESSING PLAYLIST")
        logger.info(f"{'='*60}")

        # Get video IDs from playlist
        video_ids = self._get_playlist_videos(playlist_url)

        if not video_ids:
            logger.error("Could not get videos from playlist")
            return []

        if max_videos > 0:
            video_ids = video_ids[:max_videos]

        logger.info(f"Found {len(video_ids)} videos in playlist")

        results = []
        for i, video_id in enumerate(video_ids):
            logger.info(f"\n[{i+1}/{len(video_ids)}] Processing: {video_id}")

            if progress_callback:
                progress_callback(i, len(video_ids), video_id, "starting")

            result = self.prepare(
                video_id,
                force=force,
                progress_callback=lambda phase, msg: progress_callback(i, len(video_ids), video_id, phase) if progress_callback else None
            )
            results.append(result)

            if result.success:
                logger.info(f"    Success: {result.frame_count} frames, {result.transcript_segments} segments")
            else:
                logger.error(f"    Failed: {result.error}")

        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"\n{'='*60}")
        logger.info("PLAYLIST PREPROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"  Total Videos: {len(results)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {len(results) - successful}")
        logger.info(f"{'='*60}")

        return results

    def _download_audio(self, video_id: str, force: bool = False) -> Dict[str, Any]:
        """Download audio using pytubefix (handles YouTube auth better)"""
        output_path = self.audio_dir / f"{video_id}.{self.audio_format}"

        # Check if already exists
        if output_path.exists() and not force:
            logger.info(f"    Audio already exists: {output_path}")
            # Get duration from existing file
            duration = self._get_audio_duration(str(output_path))
            return {
                'success': True,
                'path': str(output_path),
                'duration': duration,
            }

        # Try pytubefix first (most reliable for auth issues)
        try:
            from pytubefix import YouTube

            url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"    Downloading with pytubefix: {video_id}")

            yt = YouTube(url)
            title = yt.title
            channel = yt.author
            duration = yt.length

            # Get best audio stream
            audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()

            if not audio_stream:
                return {'success': False, 'error': "No audio stream found"}

            # Download to temp file
            temp_path = audio_stream.download(
                output_path=str(self.audio_dir),
                filename=f"{video_id}_temp"
            )

            # Convert to desired format if needed
            if self.audio_format == "mp3" and not temp_path.endswith('.mp3'):
                self._convert_to_mp3(temp_path, output_path)
                os.remove(temp_path)
            else:
                os.rename(temp_path, output_path)

            return {
                'success': True,
                'path': str(output_path),
                'duration': duration,
                'title': title,
                'channel': channel,
            }

        except ImportError:
            logger.warning("pytubefix not installed, trying yt-dlp fallback")
        except Exception as e:
            logger.warning(f"pytubefix failed: {e}, trying yt-dlp fallback")

        # Fallback to yt-dlp
        try:
            import yt_dlp

            url = f"https://www.youtube.com/watch?v={video_id}"

            opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': self.audio_format,
                    'preferredquality': '192',
                }],
                'outtmpl': str(self.audio_dir / f"{video_id}.%(ext)s"),
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return {
                    'success': True,
                    'path': str(output_path),
                    'duration': info.get('duration', 0),
                    'title': info.get('title'),
                    'channel': info.get('uploader'),
                }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _extract_frames(self, video_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Extract frames from video at regular intervals using ffmpeg.

        Uses ADAPTIVE streams first (more reliable), then falls back to progressive.
        This fixes issues with some videos that have corrupted progressive streams.
        """
        output_dir = self.frames_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if frames already exist
        existing_frames = list(output_dir.glob("frame_*.jpg"))
        if existing_frames and not force:
            logger.info(f"    Frames already exist: {len(existing_frames)}")
            return {
                'success': True,
                'dir': str(output_dir),
                'count': len(existing_frames),
            }

        # Need to download video temporarily for frame extraction
        try:
            from pytubefix import YouTube
            import tempfile

            url = f"https://www.youtube.com/watch?v={video_id}"
            yt = YouTube(url)

            # Try ADAPTIVE streams first (720p/480p video-only) - more reliable
            video_stream = None

            # Priority 1: 720p adaptive mp4
            video_stream = yt.streams.filter(
                res="720p",
                file_extension='mp4',
                only_video=True
            ).first()

            # Priority 2: 480p adaptive mp4
            if not video_stream:
                video_stream = yt.streams.filter(
                    res="480p",
                    file_extension='mp4',
                    only_video=True
                ).first()

            # Priority 3: Any adaptive mp4
            if not video_stream:
                video_stream = yt.streams.filter(
                    file_extension='mp4',
                    only_video=True
                ).order_by('resolution').desc().first()

            # Priority 4: Progressive stream (fallback - may be corrupted for some videos)
            if not video_stream:
                video_stream = yt.streams.filter(
                    progressive=True,
                    file_extension='mp4'
                ).order_by('resolution').desc().first()

            if not video_stream:
                return {'success': False, 'error': "No suitable video stream found"}

            logger.info(f"    Using stream: {video_stream.resolution} ({video_stream.mime_type})")

            # Download to temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"    Downloading video for frame extraction...")
                video_path = video_stream.download(
                    output_path=temp_dir,
                    filename=f"{video_id}.mp4"
                )

                file_size = os.path.getsize(video_path) / 1024 / 1024
                logger.info(f"    Downloaded: {file_size:.1f} MB")

                # Extract frames with ffmpeg
                logger.info(f"    Extracting frames every {self.frame_interval}s...")

                # Use ffmpeg to extract frames (with error suppression for cleaner logs)
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-vf', f'fps=1/{self.frame_interval}',  # 1 frame per N seconds
                    '-q:v', '2',  # High quality JPEG
                    str(output_dir / 'frame_%06d.jpg'),
                    '-y',  # Overwrite
                    '-loglevel', 'error'  # Suppress verbose output
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    # Clean up any partial frames
                    for f in output_dir.glob("frame_*.jpg"):
                        f.unlink()
                    return {'success': False, 'error': f"ffmpeg failed: {result.stderr[:200]}"}

                # Rename frames to include timestamp
                frames = sorted(output_dir.glob("frame_*.jpg"))
                for i, frame in enumerate(frames):
                    timestamp = i * self.frame_interval
                    new_name = output_dir / f"frame_{timestamp:08.2f}s.jpg"
                    frame.rename(new_name)

                final_frames = list(output_dir.glob("frame_*.jpg"))
                logger.info(f"    Extracted {len(final_frames)} frames")

                return {
                    'success': True,
                    'dir': str(output_dir),
                    'count': len(final_frames),
                }

        except ImportError:
            return {'success': False, 'error': "pytubefix not installed"}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _transcribe_audio(
        self,
        video_id: str,
        audio_path: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """Transcribe audio using faster-whisper"""
        import json

        output_path = self.transcripts_dir / f"{video_id}.json"

        # Check if transcript already exists
        if output_path.exists() and not force:
            logger.info(f"    Transcript already exists: {output_path}")
            with open(output_path, 'r') as f:
                data = json.load(f)
                return {
                    'success': True,
                    'path': str(output_path),
                    'segments': len(data.get('segments', [])),
                }

        try:
            from faster_whisper import WhisperModel

            logger.info(f"    Loading faster-whisper model: {self.whisper_model}")
            model = WhisperModel(
                self.whisper_model,
                device="auto",  # Use GPU if available
                compute_type="auto"
            )

            logger.info(f"    Transcribing audio...")
            segments_gen, info = model.transcribe(
                audio_path,
                word_timestamps=True,
                language="en"
            )

            # Convert generator to list
            segments = []
            for segment in segments_gen:
                seg_data = {
                    'start_time': segment.start,
                    'end_time': segment.end,
                    'text': segment.text.strip(),
                }

                # Add word-level timestamps if available
                if segment.words:
                    seg_data['words'] = [
                        {
                            'word': w.word,
                            'start': w.start,
                            'end': w.end,
                            'probability': w.probability
                        }
                        for w in segment.words
                    ]

                segments.append(seg_data)

            # Save transcript
            transcript_data = {
                'video_id': video_id,
                'language': info.language,
                'duration': info.duration,
                'transcribed_at': datetime.now().isoformat(),
                'model': self.whisper_model,
                'segments': segments,
            }

            with open(output_path, 'w') as f:
                json.dump(transcript_data, f, indent=2)

            return {
                'success': True,
                'path': str(output_path),
                'segments': len(segments),
            }

        except ImportError:
            return {'success': False, 'error': "faster-whisper not installed"}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _get_playlist_videos(self, playlist_url: str) -> List[str]:
        """Get all video IDs from a playlist"""
        try:
            from pytubefix import Playlist

            playlist = Playlist(playlist_url)
            video_ids = []

            for url in playlist.video_urls:
                video_id = self.extract_video_id(url)
                if video_id:
                    video_ids.append(video_id)

            return video_ids

        except ImportError:
            logger.warning("pytubefix not installed, trying yt-dlp")
        except Exception as e:
            logger.warning(f"pytubefix playlist failed: {e}")

        # Fallback to yt-dlp
        try:
            import yt_dlp

            opts = {
                'extract_flat': True,
                'quiet': True,
            }

            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)

                if 'entries' in info:
                    return [entry['id'] for entry in info['entries'] if entry.get('id')]

            return []

        except Exception as e:
            logger.error(f"Failed to get playlist videos: {e}")
            return []

    def _convert_to_mp3(self, input_path: str, output_path: Path) -> bool:
        """Convert audio file to MP3 using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vn',  # No video
                '-acodec', 'libmp3lame',
                '-q:a', '2',  # High quality
                str(output_path),
                '-y'  # Overwrite
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"MP3 conversion failed: {e}")
            return False

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except Exception:
            return 0


# =============================================================================
# Convenience Functions
# =============================================================================

def prepare_video(url_or_id: str, data_dir: str = "data", **kwargs) -> PreprocessResult:
    """
    Convenience function to prepare a single video.

    Args:
        url_or_id: YouTube URL or video ID
        data_dir: Base data directory
        **kwargs: Additional arguments passed to VideoPreprocessor.prepare()

    Returns:
        PreprocessResult
    """
    preprocessor = VideoPreprocessor(data_dir=data_dir)
    return preprocessor.prepare(url_or_id, **kwargs)


def prepare_playlist(playlist_url: str, data_dir: str = "data", **kwargs) -> List[PreprocessResult]:
    """
    Convenience function to prepare a playlist.

    Args:
        playlist_url: YouTube playlist URL
        data_dir: Base data directory
        **kwargs: Additional arguments passed to VideoPreprocessor.prepare_playlist()

    Returns:
        List of PreprocessResult
    """
    preprocessor = VideoPreprocessor(data_dir=data_dir)
    return preprocessor.prepare_playlist(playlist_url, **kwargs)


# CLI usage
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python video_preprocessor.py <youtube_url_or_id>")
        print("       python video_preprocessor.py <playlist_url> --playlist")
        print("\nExamples:")
        print("  python video_preprocessor.py FgacYSN9QEo")
        print("  python video_preprocessor.py 'https://youtube.com/watch?v=FgacYSN9QEo'")
        print("  python video_preprocessor.py 'https://youtube.com/playlist?list=PLxxx' --playlist")
        sys.exit(1)

    url = sys.argv[1]
    is_playlist = '--playlist' in sys.argv

    preprocessor = VideoPreprocessor()

    if is_playlist:
        results = preprocessor.prepare_playlist(url)
        successful = sum(1 for r in results if r.success)
        print(f"\nProcessed {len(results)} videos, {successful} successful")
    else:
        result = preprocessor.prepare(url)
        if result.success:
            print(f"\nSuccess! Ready for training:")
            print(f"  Audio: {result.audio_path}")
            print(f"  Frames: {result.frames_dir} ({result.frame_count} frames)")
            print(f"  Transcript: {result.transcript_path} ({result.transcript_segments} segments)")
        else:
            print(f"\nFailed: {result.error}")
            sys.exit(1)
