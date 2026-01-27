#!/usr/bin/env python3
"""
Free Local Whisper Transcription

Uses OpenAI's Whisper model locally - completely FREE.
Uses 'tiny' or 'base' model for reasonable speed on CPU.

Estimated times per 30-min video:
- tiny: ~10-15 minutes on CPU
- base: ~20-30 minutes on CPU
- small: ~45-60 minutes on CPU
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Add packages to path
sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')

import whisper
import yt_dlp

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLAYLISTS_DIR = DATA_DIR / "playlists"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
AUDIO_DIR = DATA_DIR / "audio"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Global model cache
_whisper_model = None


def get_whisper_model(model_name: str = "tiny"):
    """Load Whisper model (cached)"""
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper '{model_name}' model (first time downloads ~75MB)...")
        _whisper_model = whisper.load_model(model_name)
        print("Model loaded!")
    return _whisper_model


def download_audio(video_id: str, title: str) -> str:
    """Download audio from YouTube"""
    output_path = AUDIO_DIR / f"{video_id}.mp3"

    if output_path.exists():
        print(f"  Audio cached: {output_path.name}")
        return str(output_path)

    print(f"  Downloading audio...")
    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        'format': 'worstaudio/worst',  # Smallest audio = fastest download
        'outtmpl': str(AUDIO_DIR / f"{video_id}.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '64',  # Lower quality = smaller file = faster
        }],
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 60,
        'retries': 5,
        'ignoreerrors': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Downloaded: {size_mb:.1f} MB")
            return str(output_path)
        else:
            # Check for other extensions
            for ext in ['m4a', 'webm', 'opus']:
                alt_path = AUDIO_DIR / f"{video_id}.{ext}"
                if alt_path.exists():
                    return str(alt_path)
            print(f"  Download completed but file not found")
            return None
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def transcribe_audio(audio_path: str, video_id: str, model_name: str = "tiny") -> dict:
    """Transcribe audio using Whisper"""
    model = get_whisper_model(model_name)

    print(f"  Transcribing with Whisper '{model_name}'...")
    start_time = time.time()

    result = model.transcribe(
        audio_path,
        language='en',
        fp16=False,  # Use FP32 for CPU
        verbose=False
    )

    elapsed = time.time() - start_time

    # Convert to our format
    segments = []
    for seg in result.get('segments', []):
        segments.append({
            'start_time': round(seg['start'], 2),
            'end_time': round(seg['end'], 2),
            'text': seg['text'].strip(),
        })

    full_text = result.get('text', '').strip()
    word_count = len(full_text.split())

    print(f"  Transcribed in {elapsed/60:.1f} minutes")
    print(f"  Words: {word_count:,} | Segments: {len(segments)}")

    return {
        'video_id': video_id,
        'full_text': full_text,
        'segments': segments,
        'language': 'en',
        'word_count': word_count,
        'duration_seconds': segments[-1]['end_time'] if segments else 0,
        'transcribed_at': datetime.utcnow().isoformat(),
        'method': f'whisper_local_{model_name}',
        'processing_time_seconds': round(elapsed, 1)
    }


def process_video(video_id: str, title: str, model_name: str = "tiny") -> dict:
    """Process a single video"""
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"

    # Check if already done
    if transcript_path.exists():
        print(f"  Already transcribed, skipping")
        with open(transcript_path) as f:
            return json.load(f)

    # Download audio
    audio_path = download_audio(video_id, title)
    if not audio_path:
        return None

    # Transcribe
    try:
        transcript = transcribe_audio(audio_path, video_id, model_name)
        transcript['title'] = title

        # Save
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {transcript_path.name}")

        # Delete audio to save disk space
        try:
            os.remove(audio_path)
            print(f"  Cleaned up audio file")
        except:
            pass

        return transcript

    except Exception as e:
        print(f"  Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_playlist(playlist_num: int, max_videos: int = None, model_name: str = "tiny"):
    """Process videos from a playlist"""
    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))

    if playlist_num < 1 or playlist_num > len(playlist_files):
        print(f"Invalid playlist. Choose 1-{len(playlist_files)}")
        return

    playlist_file = playlist_files[playlist_num - 1]

    with open(playlist_file) as f:
        playlist = json.load(f)

    total_videos = playlist['video_count']
    videos = playlist['videos']
    if max_videos:
        videos = videos[:max_videos]

    print(f"\n{'='*60}")
    print(f"TRANSCRIPTION SESSION")
    print(f"{'='*60}")
    print(f"Playlist: {playlist['title']}")
    print(f"Videos: {len(videos)}/{total_videos}")
    print(f"Model: Whisper '{model_name}'")
    print(f"{'='*60}\n")

    results = []
    total_words = 0
    total_time = 0
    start_session = time.time()

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video['title']}")
        print(f"  ID: {video['video_id']}")

        video_start = time.time()
        result = process_video(video['video_id'], video['title'], model_name)
        video_time = time.time() - video_start

        if result:
            results.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'success': True,
                'word_count': result.get('word_count', 0),
                'processing_time': round(video_time, 1)
            })
            total_words += result.get('word_count', 0)
            total_time += video_time
        else:
            results.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'success': False
            })

        # Progress estimate
        if i < len(videos):
            avg_time = total_time / i
            remaining = (len(videos) - i) * avg_time
            print(f"  Estimated remaining: {remaining/60:.0f} minutes")

    # Summary
    session_time = time.time() - start_session
    successful = sum(1 for r in results if r['success'])

    print(f"\n{'='*60}")
    print(f"SESSION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {successful}/{len(results)} videos")
    print(f"Total words: {total_words:,}")
    print(f"Total time: {session_time/60:.1f} minutes")
    print(f"{'='*60}")

    # Save report
    report_path = DATA_DIR / f"transcription_report_{playlist['playlist_id']}.json"
    with open(report_path, 'w') as f:
        json.dump({
            'playlist_id': playlist['playlist_id'],
            'playlist_title': playlist['title'],
            'model': model_name,
            'completed_at': datetime.utcnow().isoformat(),
            'total_words': total_words,
            'total_time_minutes': round(session_time/60, 1),
            'results': results
        }, f, indent=2)
    print(f"Report: {report_path}")

    return results


def list_playlists():
    """List available playlists"""
    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))

    print("\n📚 Available Playlists:\n")

    for i, pf in enumerate(playlist_files, 1):
        with open(pf) as f:
            p = json.load(f)

        # Check transcription progress
        done = 0
        for v in p.get('videos', []):
            if (TRANSCRIPTS_DIR / f"{v['video_id']}.json").exists():
                done += 1

        status = f"[{done}/{p['video_count']}]" if done > 0 else ""
        tier = p.get('tier', 1)

        print(f"  {i}. [Tier {tier}] {p['title']}")
        print(f"     {p['video_count']} videos {status}")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Free Whisper Transcription for ICT Videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_free.py --list              # List playlists
  python transcribe_free.py --playlist 1        # Process playlist 1
  python transcribe_free.py --playlist 1 --max 2  # First 2 videos only
  python transcribe_free.py --video Vh0NtdPPj1M  # Single video
        """
    )
    parser.add_argument('--list', action='store_true', help='List playlists')
    parser.add_argument('--playlist', type=int, help='Playlist number to process')
    parser.add_argument('--max', type=int, help='Max videos to process')
    parser.add_argument('--video', type=str, help='Process single video by ID')
    parser.add_argument('--model', type=str, default='tiny',
                       choices=['tiny', 'base', 'small'],
                       help='Whisper model (tiny=fastest, small=best)')

    args = parser.parse_args()

    if args.list:
        list_playlists()
        return

    if args.video:
        print(f"\nProcessing single video: {args.video}")
        result = process_video(args.video, "Single Video", args.model)
        if result:
            print(f"\nSuccess! Transcript saved.")
        return

    if args.playlist:
        process_playlist(args.playlist, args.max, args.model)
        return

    # Default: show help
    parser.print_help()
    list_playlists()


if __name__ == "__main__":
    main()
