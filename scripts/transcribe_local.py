#!/usr/bin/env python3
"""
Local Whisper Transcription Script

Uses OpenAI's Whisper model locally (FREE but slower).
Requires: pip install openai-whisper torch

First time will download the model (~1.5GB for 'base', ~3GB for 'medium').
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLAYLISTS_DIR = DATA_DIR / "playlists"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
AUDIO_DIR = DATA_DIR / "audio"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def check_whisper_installed():
    """Check if Whisper is installed"""
    try:
        import whisper
        return True
    except ImportError:
        return False


def install_whisper():
    """Install Whisper and dependencies"""
    import subprocess
    print("Installing Whisper (this may take a few minutes)...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--user',
        'openai-whisper', 'torch', 'torchaudio'
    ], check=True)
    print("Whisper installed successfully!")


def download_audio(video_id: str) -> str:
    """Download audio from YouTube video using yt-dlp"""
    import yt_dlp

    output_path = AUDIO_DIR / f"{video_id}.mp3"

    if output_path.exists():
        print(f"    Audio exists: {output_path.name}")
        return str(output_path)

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(AUDIO_DIR / f"{video_id}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',  # Lower quality = faster download
        }],
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
        'retries': 3,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"    ✓ Downloaded audio")
        return str(output_path)
    except Exception as e:
        print(f"    ✗ Download failed: {e}")
        return None


def transcribe_with_whisper(audio_path: str, video_id: str, model_name: str = "base") -> dict:
    """Transcribe audio using local Whisper"""
    import whisper

    print(f"    Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"    Transcribing (this may take a while)...")
    result = model.transcribe(
        audio_path,
        verbose=False,
        language='en',
        fp16=False  # Use FP32 for CPU compatibility
    )

    # Convert to our format
    segments = []
    for seg in result.get('segments', []):
        segments.append({
            'start_time': seg['start'],
            'end_time': seg['end'],
            'text': seg['text'].strip(),
        })

    full_text = result.get('text', '')

    return {
        'video_id': video_id,
        'full_text': full_text,
        'segments': segments,
        'language': result.get('language', 'en'),
        'duration': segments[-1]['end_time'] if segments else 0,
        'transcribed_at': datetime.utcnow().isoformat(),
        'method': f'whisper_local_{model_name}',
        'word_count': len(full_text.split())
    }


def process_video(video_id: str, title: str, model_name: str = "base") -> dict:
    """Process a single video"""

    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"

    # Check if already done
    if transcript_path.exists():
        print(f"    ℹ Already transcribed")
        with open(transcript_path) as f:
            return json.load(f)

    # Download audio
    audio_path = download_audio(video_id)
    if not audio_path:
        return None

    # Transcribe
    try:
        transcript = transcribe_with_whisper(audio_path, video_id, model_name)
        transcript['title'] = title

        # Save
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        print(f"    ✓ Transcribed: {transcript['word_count']:,} words")

        # Optionally remove audio to save space
        # os.remove(audio_path)

        return transcript

    except Exception as e:
        print(f"    ✗ Transcription failed: {e}")
        return None


def process_playlist(playlist_num: int, max_videos: int = None, model_name: str = "base"):
    """Process videos from a playlist"""

    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))

    if playlist_num < 1 or playlist_num > len(playlist_files):
        print(f"Invalid playlist. Choose 1-{len(playlist_files)}")
        return

    playlist_file = playlist_files[playlist_num - 1]

    with open(playlist_file) as f:
        playlist = json.load(f)

    print(f"\n{'='*70}")
    print(f"📺 {playlist['title']}")
    print(f"📊 Videos: {playlist['video_count']}")
    print(f"🤖 Model: {model_name}")
    print(f"{'='*70}\n")

    videos = playlist['videos']
    if max_videos:
        videos = videos[:max_videos]

    results = []
    total_words = 0

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video['title'][:50]}...")

        result = process_video(video['video_id'], video['title'], model_name)

        if result:
            results.append({'success': True, 'words': result.get('word_count', 0)})
            total_words += result.get('word_count', 0)
        else:
            results.append({'success': False})

    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n{'='*70}")
    print(f"✓ Processed: {successful}/{len(results)}")
    print(f"📝 Total words: {total_words:,}")
    print(f"{'='*70}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Local Whisper Transcription')
    parser.add_argument('--install', action='store_true', help='Install Whisper')
    parser.add_argument('--playlist', type=int, default=1, help='Playlist number')
    parser.add_argument('--max-videos', type=int, default=None, help='Max videos')
    parser.add_argument('--model', type=str, default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (tiny=fastest, large=best)')
    parser.add_argument('--list', action='store_true', help='List playlists')

    args = parser.parse_args()

    if args.install:
        install_whisper()
        return

    if args.list:
        playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))
        print("\n📚 Playlists:\n")
        for i, pf in enumerate(playlist_files, 1):
            with open(pf) as f:
                p = json.load(f)
            print(f"  {i}. {p['title']} ({p['video_count']} videos)")
        return

    if not check_whisper_installed():
        print("Whisper not installed. Run with --install first.")
        print("  python transcribe_local.py --install")
        return

    process_playlist(args.playlist, args.max_videos, args.model)


if __name__ == "__main__":
    main()
