"""
Audio Downloader Utility using pytubefix.

pytubefix is a maintained fork of pytube that handles YouTube's
authentication requirements better than yt-dlp in many cases.

This module provides reliable audio extraction for YouTube videos,
essential for the Audio-First learning approach.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AudioDownloadResult:
    """Result of an audio download operation"""
    success: bool
    video_id: str
    audio_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    file_size_mb: Optional[float] = None
    error: Optional[str] = None
    title: Optional[str] = None


class AudioDownloader:
    """
    Download audio from YouTube videos using pytubefix.

    pytubefix handles YouTube's authentication better than yt-dlp
    for many videos that get 403 errors.

    Usage:
        downloader = AudioDownloader(output_dir="data/audio")
        result = downloader.download("FgacYSN9QEo")
        if result.success:
            print(f"Audio saved to: {result.audio_path}")
    """

    def __init__(
        self,
        output_dir: str = "data/audio",
        preferred_format: str = "mp3",
        quality: str = "high"  # "high", "medium", "low"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preferred_format = preferred_format
        self.quality = quality

    def download(
        self,
        video_id: str,
        force: bool = False
    ) -> AudioDownloadResult:
        """
        Download audio from a YouTube video.

        Args:
            video_id: YouTube video ID
            force: Force re-download even if file exists

        Returns:
            AudioDownloadResult with success status and file path
        """
        output_path = self.output_dir / f"{video_id}.{self.preferred_format}"

        # Check if already downloaded
        if output_path.exists() and not force:
            logger.info(f"Audio already exists: {output_path}")
            return AudioDownloadResult(
                success=True,
                video_id=video_id,
                audio_path=str(output_path),
                file_size_mb=output_path.stat().st_size / (1024 * 1024)
            )

        # Try pytubefix first (most reliable for auth issues)
        result = self._download_with_pytubefix(video_id, output_path)
        if result.success:
            return result

        # Fallback to yt-dlp with various strategies
        logger.info("pytubefix failed, trying yt-dlp fallback...")
        result = self._download_with_ytdlp(video_id, output_path)
        if result.success:
            return result

        return AudioDownloadResult(
            success=False,
            video_id=video_id,
            error="All download methods failed"
        )

    def _download_with_pytubefix(
        self,
        video_id: str,
        output_path: Path
    ) -> AudioDownloadResult:
        """Download using pytubefix library"""
        try:
            from pytubefix import YouTube

            url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Downloading with pytubefix: {video_id}")

            yt = YouTube(url)
            title = yt.title
            duration = yt.length

            # Get audio stream based on quality preference
            if self.quality == "high":
                audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
            elif self.quality == "low":
                audio_stream = yt.streams.filter(only_audio=True).order_by('abr').asc().first()
            else:
                audio_stream = yt.streams.filter(only_audio=True).first()

            if not audio_stream:
                return AudioDownloadResult(
                    success=False,
                    video_id=video_id,
                    error="No audio stream found"
                )

            logger.info(f"Found audio stream: {audio_stream}")

            # Download to temp file
            temp_path = audio_stream.download(
                output_path=str(self.output_dir),
                filename=f"{video_id}_temp"
            )

            # Convert to preferred format if needed
            if self.preferred_format == "mp3" and not temp_path.endswith('.mp3'):
                final_path = self._convert_to_mp3(temp_path, output_path)
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                final_path = output_path
                if temp_path != str(final_path):
                    os.rename(temp_path, final_path)

            if os.path.exists(final_path):
                file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
                logger.info(f"Successfully downloaded: {final_path} ({file_size_mb:.2f} MB)")

                return AudioDownloadResult(
                    success=True,
                    video_id=video_id,
                    audio_path=str(final_path),
                    duration_seconds=duration,
                    file_size_mb=file_size_mb,
                    title=title
                )
            else:
                return AudioDownloadResult(
                    success=False,
                    video_id=video_id,
                    error="File not created after download"
                )

        except ImportError:
            logger.warning("pytubefix not installed. Install with: pip install pytubefix")
            return AudioDownloadResult(
                success=False,
                video_id=video_id,
                error="pytubefix not installed"
            )
        except Exception as e:
            logger.error(f"pytubefix download failed: {e}")
            return AudioDownloadResult(
                success=False,
                video_id=video_id,
                error=str(e)
            )

    def _download_with_ytdlp(
        self,
        video_id: str,
        output_path: Path
    ) -> AudioDownloadResult:
        """Fallback download using yt-dlp with cookies"""
        try:
            import yt_dlp

            url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Downloading with yt-dlp: {video_id}")

            opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': self.preferred_format,
                    'preferredquality': '192' if self.quality == 'high' else '128',
                }],
                'outtmpl': str(self.output_dir / f"{video_id}.%(ext)s"),
                'quiet': True,
                'no_warnings': True,
            }

            # Try with browser cookies if available
            try:
                opts['cookiesfrombrowser'] = ('chrome',)
            except Exception:
                pass

            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', '')
                duration = info.get('duration', 0)

            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                return AudioDownloadResult(
                    success=True,
                    video_id=video_id,
                    audio_path=str(output_path),
                    duration_seconds=duration,
                    file_size_mb=file_size_mb,
                    title=title
                )
            else:
                return AudioDownloadResult(
                    success=False,
                    video_id=video_id,
                    error="File not created after yt-dlp download"
                )

        except Exception as e:
            logger.error(f"yt-dlp download failed: {e}")
            return AudioDownloadResult(
                success=False,
                video_id=video_id,
                error=str(e)
            )

    def _convert_to_mp3(self, input_path: str, output_path: Path) -> Path:
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

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"Converted to MP3: {output_path}")
                return output_path
            else:
                logger.error(f"ffmpeg conversion failed: {result.stderr}")
                # Return input path if conversion fails
                return Path(input_path)

        except FileNotFoundError:
            logger.warning("ffmpeg not found, keeping original format")
            return Path(input_path)

    def get_audio_info(self, video_id: str) -> Optional[dict]:
        """Get audio/video info without downloading"""
        try:
            from pytubefix import YouTube

            url = f"https://www.youtube.com/watch?v={video_id}"
            yt = YouTube(url)

            return {
                'video_id': video_id,
                'title': yt.title,
                'duration_seconds': yt.length,
                'author': yt.author,
                'publish_date': str(yt.publish_date) if yt.publish_date else None,
                'views': yt.views,
                'available_audio_streams': [
                    {
                        'itag': s.itag,
                        'abr': s.abr,
                        'codec': s.audio_codec,
                        'filesize_mb': s.filesize / (1024 * 1024) if s.filesize else None
                    }
                    for s in yt.streams.filter(only_audio=True)
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return None


def download_audio(
    video_id: str,
    output_dir: str = "data/audio",
    force: bool = False
) -> AudioDownloadResult:
    """
    Convenience function to download audio from a YouTube video.

    Args:
        video_id: YouTube video ID
        output_dir: Directory to save audio files
        force: Force re-download even if file exists

    Returns:
        AudioDownloadResult with success status and file path
    """
    downloader = AudioDownloader(output_dir=output_dir)
    return downloader.download(video_id, force=force)


# CLI usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_downloader.py <video_id>")
        print("Example: python audio_downloader.py FgacYSN9QEo")
        sys.exit(1)

    video_id = sys.argv[1]

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    result = download_audio(video_id)

    if result.success:
        print(f"\n✅ Success!")
        print(f"   Audio: {result.audio_path}")
        print(f"   Size: {result.file_size_mb:.2f} MB")
        if result.duration_seconds:
            print(f"   Duration: {result.duration_seconds / 60:.1f} minutes")
    else:
        print(f"\n❌ Failed: {result.error}")
        sys.exit(1)
