#!/usr/bin/env python3
"""
Free Local Whisper Transcription v2

Uses cookies and updated extraction methods to bypass YouTube restrictions.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import time
import subprocess
import tempfile

sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')

import whisper

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
        print(f"  Loading Whisper '{model_name}' model...")
        _whisper_model = whisper.load_model(model_name)
        print(f"  Model loaded!")
    return _whisper_model


def download_audio_v2(video_id: str) -> str:
    """Download audio using yt-dlp with various workarounds"""
    output_path = AUDIO_DIR / f"{video_id}.mp3"

    if output_path.exists():
        print(f"  Audio cached")
        return str(output_path)

    print(f"  Downloading audio...")
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Try different methods
    methods = [
        # Method 1: Use Android client
        {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'extractor_args': {'youtube': {'player_client': ['android']}},
        },
        # Method 2: Use iOS client
        {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'extractor_args': {'youtube': {'player_client': ['ios']}},
        },
        # Method 3: Use web client with age gate bypass
        {
            'format': 'bestaudio/best',
            'extractor_args': {'youtube': {'player_client': ['web']}},
        },
        # Method 4: Standard
        {
            'format': 'worstaudio/worst',
        },
    ]

    for i, method_opts in enumerate(methods, 1):
        print(f"    Trying method {i}...")

        ydl_opts = {
            **method_opts,
            'outtmpl': str(AUDIO_DIR / f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '64',
            }],
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': 3,
            'ignoreerrors': False,
            'nocheckcertificate': True,
        }

        try:
            import yt_dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  Downloaded: {size_mb:.1f} MB")
                return str(output_path)

            # Check for other extensions
            for ext in ['m4a', 'webm', 'opus', 'mp4']:
                alt_path = AUDIO_DIR / f"{video_id}.{ext}"
                if alt_path.exists():
                    # Convert to mp3
                    print(f"    Converting {ext} to mp3...")
                    subprocess.run([
                        'ffmpeg', '-i', str(alt_path),
                        '-vn', '-acodec', 'libmp3lame', '-q:a', '6',
                        str(output_path), '-y'
                    ], capture_output=True)
                    if output_path.exists():
                        os.remove(alt_path)
                        return str(output_path)

        except Exception as e:
            print(f"    Method {i} failed: {str(e)[:50]}")
            continue

    print(f"  All download methods failed")
    return None


def try_get_existing_transcript(video_id: str) -> dict:
    """Try to get transcript from YouTube's existing captions"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        print(f"  Checking for YouTube captions...")

        # Try different transcript types
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try manual English first
            for transcript in transcript_list:
                if transcript.language_code.startswith('en'):
                    data = transcript.fetch()
                    segments = [{
                        'start_time': item['start'],
                        'end_time': item['start'] + item['duration'],
                        'text': item['text']
                    } for item in data]

                    full_text = ' '.join([s['text'] for s in segments])
                    print(f"  Found YouTube captions! ({len(full_text.split())} words)")

                    return {
                        'video_id': video_id,
                        'full_text': full_text,
                        'segments': segments,
                        'word_count': len(full_text.split()),
                        'method': 'youtube_captions',
                        'transcribed_at': datetime.utcnow().isoformat()
                    }
        except Exception as e:
            print(f"  No captions available: {e}")

    except ImportError:
        pass

    return None


def transcribe_audio(audio_path: str, video_id: str, model_name: str = "tiny") -> dict:
    """Transcribe audio using Whisper"""
    model = get_whisper_model(model_name)

    print(f"  Transcribing...")
    start_time = time.time()

    result = model.transcribe(
        audio_path,
        language='en',
        fp16=False,
        verbose=False
    )

    elapsed = time.time() - start_time

    segments = [{
        'start_time': round(seg['start'], 2),
        'end_time': round(seg['end'], 2),
        'text': seg['text'].strip(),
    } for seg in result.get('segments', [])]

    full_text = result.get('text', '').strip()

    print(f"  Done in {elapsed/60:.1f} min ({len(full_text.split())} words)")

    return {
        'video_id': video_id,
        'full_text': full_text,
        'segments': segments,
        'word_count': len(full_text.split()),
        'duration_seconds': segments[-1]['end_time'] if segments else 0,
        'method': f'whisper_{model_name}',
        'processing_time_seconds': round(elapsed, 1),
        'transcribed_at': datetime.utcnow().isoformat()
    }


def process_video(video_id: str, title: str, model_name: str = "tiny") -> dict:
    """Process a single video"""
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"

    if transcript_path.exists():
        print(f"  Already done, skipping")
        with open(transcript_path) as f:
            return json.load(f)

    # First try YouTube captions (fastest, no download needed)
    transcript = try_get_existing_transcript(video_id)

    if not transcript:
        # Download and transcribe with Whisper
        audio_path = download_audio_v2(video_id)

        if audio_path:
            try:
                transcript = transcribe_audio(audio_path, video_id, model_name)
                # Clean up audio
                try:
                    os.remove(audio_path)
                except:
                    pass
            except Exception as e:
                print(f"  Transcription error: {e}")
                return None
        else:
            return None

    if transcript:
        transcript['title'] = title
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        print(f"  Saved transcript")

    return transcript


def process_playlist(playlist_num: int, max_videos: int = None, model_name: str = "tiny"):
    """Process videos from a playlist"""
    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))

    if playlist_num < 1 or playlist_num > len(playlist_files):
        print(f"Invalid playlist number")
        return

    with open(playlist_files[playlist_num - 1]) as f:
        playlist = json.load(f)

    videos = playlist['videos'][:max_videos] if max_videos else playlist['videos']

    print(f"\n{'='*60}")
    print(f"Processing: {playlist['title']}")
    print(f"Videos: {len(videos)}")
    print(f"{'='*60}\n")

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

    print(f"\n{'='*60}")
    print(f"Complete: {sum(1 for r in results if r['success'])}/{len(results)}")
    print(f"Total words: {total_words:,}")
    print(f"{'='*60}")


def list_playlists():
    """List playlists with progress"""
    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))

    print("\n📚 Playlists:\n")
    for i, pf in enumerate(playlist_files, 1):
        with open(pf) as f:
            p = json.load(f)

        done = sum(1 for v in p.get('videos', [])
                   if (TRANSCRIPTS_DIR / f"{v['video_id']}.json").exists())

        progress = f"[{done}/{p['video_count']}]" if done else ""
        print(f"  {i}. {p['title'][:45]} {progress}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Free Whisper Transcription')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--playlist', type=int)
    parser.add_argument('--max', type=int)
    parser.add_argument('--video', type=str)
    parser.add_argument('--model', default='tiny', choices=['tiny', 'base', 'small'])

    args = parser.parse_args()

    if args.list:
        list_playlists()
    elif args.video:
        process_video(args.video, "Video", args.model)
    elif args.playlist:
        process_playlist(args.playlist, args.max, args.model)
    else:
        parser.print_help()
        list_playlists()


if __name__ == "__main__":
    main()
