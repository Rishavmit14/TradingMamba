#!/usr/bin/env python3
"""
Process ICT YouTube videos - Download audio and transcribe.

This script:
1. Downloads audio from YouTube videos
2. Transcribes using Whisper (or alternative)
3. Saves transcripts for concept extraction
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import tempfile

# Add yt-dlp to path
sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')

import yt_dlp

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLAYLISTS_DIR = DATA_DIR / "playlists"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

# Create directories
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def download_audio(video_id: str, output_dir: Path) -> str:
    """Download audio from a YouTube video"""

    url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = output_dir / f"{video_id}.mp3"

    # Check if already downloaded
    if output_path.exists():
        print(f"      Audio already exists: {output_path.name}")
        return str(output_path)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / f"{video_id}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"      ✓ Downloaded audio: {output_path.name}")
        return str(output_path)
    except Exception as e:
        print(f"      ✗ Failed to download: {e}")
        return None


def transcribe_with_whisper_api(audio_path: str, video_id: str) -> dict:
    """
    Transcribe using OpenAI Whisper API (if available)
    Falls back to local processing
    """
    # Check for OpenAI API key
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("      ℹ No OpenAI API key - using alternative transcription")
        return transcribe_basic(audio_path, video_id)

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        with open(audio_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        # Convert to our format
        segments = []
        for seg in transcript.segments:
            segments.append({
                'start_time': seg.start,
                'end_time': seg.end,
                'text': seg.text.strip(),
            })

        return {
            'video_id': video_id,
            'full_text': transcript.text,
            'segments': segments,
            'language': transcript.language,
            'duration': transcript.duration,
            'transcribed_at': datetime.utcnow().isoformat(),
            'method': 'whisper_api'
        }

    except Exception as e:
        print(f"      ℹ Whisper API failed: {e}")
        return transcribe_basic(audio_path, video_id)


def transcribe_basic(audio_path: str, video_id: str) -> dict:
    """
    Basic transcription placeholder - creates structure for manual transcription
    or uses YouTube's auto-captions if available
    """

    # Try to get YouTube captions
    captions = get_youtube_captions(video_id)

    if captions:
        return captions

    # Create placeholder for manual processing
    return {
        'video_id': video_id,
        'full_text': '[Transcription pending - requires Whisper or manual processing]',
        'segments': [],
        'language': 'en',
        'duration': 0,
        'transcribed_at': datetime.utcnow().isoformat(),
        'method': 'pending',
        'audio_path': audio_path
    }


def get_youtube_captions(video_id: str) -> dict:
    """Try to get captions directly from YouTube"""

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'skip_download': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Check for subtitles
            subtitles = info.get('subtitles', {})
            auto_captions = info.get('automatic_captions', {})

            # Prefer manual subtitles, fall back to auto
            if 'en' in subtitles:
                caption_info = subtitles['en']
            elif 'en' in auto_captions:
                caption_info = auto_captions['en']
            else:
                return None

            # Get the caption URL (prefer vtt or json3)
            caption_url = None
            for fmt in caption_info:
                if fmt.get('ext') in ['vtt', 'json3', 'srv1']:
                    caption_url = fmt.get('url')
                    break

            if not caption_url:
                return None

            # Download and parse captions
            import urllib.request
            with urllib.request.urlopen(caption_url) as response:
                caption_data = response.read().decode('utf-8')

            # Parse VTT format (simplified)
            segments = parse_vtt_captions(caption_data)
            full_text = ' '.join([s['text'] for s in segments])

            return {
                'video_id': video_id,
                'full_text': full_text,
                'segments': segments,
                'language': 'en',
                'duration': info.get('duration', 0),
                'transcribed_at': datetime.utcnow().isoformat(),
                'method': 'youtube_captions'
            }

    except Exception as e:
        print(f"      ℹ Could not get YouTube captions: {e}")
        return None


def parse_vtt_captions(vtt_content: str) -> list:
    """Parse VTT caption format into segments"""

    segments = []
    lines = vtt_content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for timestamp line (00:00:00.000 --> 00:00:05.000)
        if '-->' in line:
            try:
                times = line.split('-->')
                start_time = parse_vtt_time(times[0].strip())
                end_time = parse_vtt_time(times[1].strip().split()[0])

                # Get text lines until empty line
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    # Remove VTT formatting tags
                    text = lines[i].strip()
                    text = text.replace('<c>', '').replace('</c>', '')
                    # Remove other tags
                    import re
                    text = re.sub(r'<[^>]+>', '', text)
                    if text:
                        text_lines.append(text)
                    i += 1

                if text_lines:
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': ' '.join(text_lines)
                    })
            except:
                pass
        i += 1

    # Deduplicate overlapping segments
    unique_segments = []
    seen_texts = set()
    for seg in segments:
        text = seg['text'].strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_segments.append(seg)

    return unique_segments


def parse_vtt_time(time_str: str) -> float:
    """Parse VTT timestamp to seconds"""
    parts = time_str.replace(',', '.').split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return 0.0


def process_video(video_id: str, title: str) -> dict:
    """Process a single video - download and transcribe"""

    # Check if already processed
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"
    if transcript_path.exists():
        print(f"      ℹ Already processed: {video_id}")
        with open(transcript_path) as f:
            return json.load(f)

    # Download audio
    audio_path = download_audio(video_id, AUDIO_DIR)

    if not audio_path:
        return None

    # Transcribe
    print(f"      Transcribing...")
    transcript = transcribe_with_whisper_api(audio_path, video_id)

    if transcript:
        transcript['title'] = title

        # Save transcript
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f, indent=2)
        print(f"      ✓ Saved transcript: {transcript_path.name}")
        print(f"      📝 Method: {transcript.get('method', 'unknown')}")
        if transcript.get('segments'):
            print(f"      📊 Segments: {len(transcript['segments'])}")

    return transcript


def process_playlist(playlist_id: str, max_videos: int = None):
    """Process videos from a playlist"""

    # Load playlist info
    playlist_file = PLAYLISTS_DIR / f"{playlist_id}.json"

    if not playlist_file.exists():
        print(f"Playlist file not found: {playlist_file}")
        return

    with open(playlist_file) as f:
        playlist = json.load(f)

    print(f"\n{'='*70}")
    print(f"Processing Playlist: {playlist['title']}")
    print(f"Total Videos: {playlist['video_count']}")
    print(f"{'='*70}\n")

    videos = playlist['videos']
    if max_videos:
        videos = videos[:max_videos]

    results = []

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video['title'][:60]}...")
        print(f"    Video ID: {video['video_id']}")

        result = process_video(video['video_id'], video['title'])

        if result:
            results.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'success': True,
                'method': result.get('method'),
                'segments': len(result.get('segments', []))
            })
        else:
            results.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'success': False
            })

    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Processed: {successful}/{len(results)} videos")

    # Save processing report
    report_path = DATA_DIR / f"processing_report_{playlist_id}.json"
    with open(report_path, 'w') as f:
        json.dump({
            'playlist_id': playlist_id,
            'playlist_title': playlist['title'],
            'processed_at': datetime.utcnow().isoformat(),
            'results': results
        }, f, indent=2)
    print(f"📋 Report saved: {report_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process ICT YouTube videos')
    parser.add_argument('--playlist', type=int, default=1, help='Playlist number (1-3 for now)')
    parser.add_argument('--max-videos', type=int, default=None, help='Max videos to process')
    parser.add_argument('--video', type=str, help='Process single video by ID')

    args = parser.parse_args()

    if args.video:
        # Process single video
        result = process_video(args.video, "Single Video")
        if result:
            print(f"\n✓ Processed video: {args.video}")
        return

    # Get playlist ID based on number
    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))

    if not playlist_files:
        print("No playlist files found. Run fetch_playlist.py first.")
        return

    if args.playlist < 1 or args.playlist > len(playlist_files):
        print(f"Invalid playlist number. Choose 1-{len(playlist_files)}")
        return

    playlist_file = playlist_files[args.playlist - 1]
    playlist_id = playlist_file.stem

    process_playlist(playlist_id, args.max_videos)


if __name__ == "__main__":
    main()
