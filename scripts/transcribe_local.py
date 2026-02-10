#!/usr/bin/env python3
"""
Local Transcription Script using whisper.cpp

Uses whisper.cpp with Metal GPU acceleration for fast local transcription.
Optimized for Hindi audio with English ICT/SMC terms on Apple Silicon.

Default model: large-v3-turbo (Metal GPU, ~0.85x real-time on M1 8GB)

Usage:
  python transcribe_local.py --video VIDEO_ID              # Single video
  python transcribe_local.py --playlist 1                  # Full playlist
  python transcribe_local.py --playlist 1 --only-missing   # Only videos without transcripts
  python transcribe_local.py --list                        # List playlists
"""

import json
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLAYLISTS_DIR = DATA_DIR / "playlists"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
AUDIO_DIR = DATA_DIR / "audio"
MODELS_DIR = BASE_DIR / "models"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# whisper.cpp binary (installed via brew)
WHISPER_CLI = "/opt/homebrew/bin/whisper-cli"

# GGML model paths
GGML_MODELS = {
    "large-v3-turbo": MODELS_DIR / "ggml-large-v3-turbo.bin",
    "medium": MODELS_DIR / "ggml-medium.bin",
    "small": MODELS_DIR / "ggml-small.bin",
}

DEFAULT_MODEL = "large-v3-turbo"


def download_audio(video_id: str) -> str:
    """Download audio from YouTube video using yt-dlp"""
    import yt_dlp

    output_path = AUDIO_DIR / f"{video_id}.mp3"

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Audio cached: {output_path.name} ({size_mb:.1f} MB)")
        return str(output_path)

    print(f"  Downloading audio...")
    url = f"https://www.youtube.com/watch?v={video_id}"

    methods = [
        {'format': 'bestaudio/best'},
        {'format': 'bestaudio[ext=m4a]/bestaudio/best',
         'extractor_args': {'youtube': {'player_client': ['android']}}},
        {'format': 'worstaudio/worst'},
    ]

    for i, method_opts in enumerate(methods, 1):
        ydl_opts = {
            **method_opts,
            'outtmpl': str(AUDIO_DIR / f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
            }],
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 30,
            'retries': 3,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  Downloaded: {size_mb:.1f} MB")
                return str(output_path)

            for ext in ['m4a', 'webm', 'opus', 'mp4']:
                alt_path = AUDIO_DIR / f"{video_id}.{ext}"
                if alt_path.exists():
                    subprocess.run([
                        'ffmpeg', '-i', str(alt_path),
                        '-vn', '-acodec', 'libmp3lame', '-q:a', '4',
                        str(output_path), '-y'
                    ], capture_output=True)
                    if output_path.exists():
                        os.remove(alt_path)
                        size_mb = output_path.stat().st_size / (1024 * 1024)
                        print(f"  Downloaded + converted: {size_mb:.1f} MB")
                        return str(output_path)

        except Exception as e:
            if i < len(methods):
                continue
            print(f"  Download failed: {e}")
            return None

    print(f"  All download methods failed")
    return None


def convert_to_wav(audio_path: str) -> str:
    """Convert audio to 16kHz mono WAV (required by whisper.cpp)"""
    wav_path = audio_path.rsplit('.', 1)[0] + '.wav'

    if os.path.exists(wav_path):
        return wav_path

    print(f"  Converting to WAV (16kHz mono)...")
    result = subprocess.run([
        'ffmpeg', '-i', audio_path,
        '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
        wav_path, '-y'
    ], capture_output=True, text=True)

    if result.returncode == 0 and os.path.exists(wav_path):
        return wav_path

    print(f"  WAV conversion failed: {result.stderr[:200]}")
    return None


def transcribe_with_whisper_cpp(wav_path: str, video_id: str, model_name: str = DEFAULT_MODEL) -> dict:
    """Transcribe audio using whisper.cpp with Metal GPU"""
    model_path = GGML_MODELS.get(model_name)
    if not model_path or not model_path.exists():
        print(f"  GGML model not found: {model_path}")
        print(f"  Download it from: https://huggingface.co/ggerganov/whisper.cpp/tree/main")
        return None

    output_prefix = wav_path.rsplit('.', 1)[0] + '_whisper'
    output_json = output_prefix + '.json'

    print(f"  Transcribing with whisper.cpp '{model_name}' (Metal GPU)...")
    start_time = time.time()

    result = subprocess.run([
        WHISPER_CLI,
        '-m', str(model_path),
        '-f', wav_path,
        '-l', 'hi',
        '-oj',
        '-of', output_prefix,
    ], capture_output=True, timeout=3600)

    elapsed = time.time() - start_time

    if result.returncode != 0 or not os.path.exists(output_json):
        stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
        print(f"  whisper.cpp failed: {stderr[:300]}")
        return None

    # Parse whisper.cpp JSON output
    with open(output_json, 'rb') as f:
        content = f.read().decode('utf-8', errors='replace')
    raw = json.loads(content)

    segments = []
    full_text_parts = []
    for item in raw.get('transcription', []):
        start_ms = item['offsets']['from']
        end_ms = item['offsets']['to']
        text = item['text'].strip()
        if text:
            segments.append({
                'start_time': round(start_ms / 1000, 2),
                'end_time': round(end_ms / 1000, 2),
                'text': text
            })
            full_text_parts.append(text)

    full_text = ' '.join(full_text_parts)
    word_count = len(full_text.split())

    print(f"  Transcribed in {elapsed/60:.1f} min | {word_count:,} words | {len(segments)} segments")

    # Clean up whisper.cpp output file
    try:
        os.remove(output_json)
    except Exception:
        pass

    return {
        'video_id': video_id,
        'full_text': full_text,
        'segments': segments,
        'language': 'hi',
        'duration': segments[-1]['end_time'] if segments else 0,
        'transcribed_at': datetime.utcnow().isoformat(),
        'method': f'whisper_cpp_{model_name}',
        'model': f'whisper.cpp/{model_name} (Metal GPU)',
        'word_count': word_count,
        'processing_time_seconds': round(elapsed, 1),
    }


def process_video(video_id: str, title: str = "Video", model_name: str = DEFAULT_MODEL,
                  force: bool = False) -> dict:
    """
    Process a single video using whisper.cpp with Metal GPU.
    Always uses whisper.cpp — no YouTube captions (they have poor quality).
    """
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"

    if transcript_path.exists() and not force:
        print(f"  Already transcribed, skipping")
        with open(transcript_path) as f:
            return json.load(f)

    # Download audio, convert to WAV, transcribe with whisper.cpp
    audio_path = download_audio(video_id)
    if not audio_path:
        print(f"  FAILED: Could not download audio")
        return None

    wav_path = convert_to_wav(audio_path)
    if not wav_path:
        print(f"  FAILED: Could not convert to WAV")
        return None

    try:
        transcript = transcribe_with_whisper_cpp(wav_path, video_id, model_name)
    except Exception as e:
        print(f"  Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Clean up audio files
    for path in [audio_path, wav_path]:
        try:
            os.remove(path)
        except Exception:
            pass
    print(f"  Cleaned up audio files")

    if transcript:
        transcript['title'] = title

        with open(transcript_path, 'w') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {transcript_path.name}")

    return transcript


def process_playlist(playlist_num: int = None, playlist_id: str = None,
                     max_videos: int = None, model_name: str = DEFAULT_MODEL,
                     only_missing: bool = False):
    """Process videos from a playlist"""
    if playlist_id:
        playlist_file = PLAYLISTS_DIR / f"{playlist_id}.json"
    else:
        playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))
        if not playlist_num or playlist_num < 1 or playlist_num > len(playlist_files):
            print(f"Invalid playlist number. Choose 1-{len(playlist_files)}")
            return
        playlist_file = playlist_files[playlist_num - 1]

    with open(playlist_file) as f:
        playlist = json.load(f)

    videos = playlist['videos']
    if max_videos:
        videos = videos[:max_videos]

    if only_missing:
        videos = [v for v in videos
                  if not (TRANSCRIPTS_DIR / f"{v['video_id']}.json").exists()]
        print(f"  {len(videos)} videos missing transcripts")

    if not videos:
        print("  All videos already transcribed!")
        return

    print(f"\n{'='*60}")
    print(f"  Playlist: {playlist['title']}")
    print(f"  Videos: {len(videos)}")
    print(f"  Engine: whisper.cpp (Metal GPU)")
    print(f"  Model: {model_name}")
    print(f"{'='*60}\n")

    results = []
    total_words = 0
    session_start = time.time()

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video['title']}")
        print(f"  ID: {video['video_id']}")

        video_start = time.time()
        result = process_video(video['video_id'], video['title'], model_name)
        video_time = time.time() - video_start

        if result:
            words = result.get('word_count', 0)
            results.append({'success': True, 'words': words, 'time': video_time})
            total_words += words
        else:
            results.append({'success': False, 'time': video_time})

        successful_so_far = [r for r in results if r['success']]
        if successful_so_far and i < len(videos):
            avg_time = sum(r['time'] for r in successful_so_far) / len(successful_so_far)
            remaining = (len(videos) - i) * avg_time
            print(f"  Estimated remaining: {remaining/60:.0f} min")

    session_time = time.time() - session_start
    successful = sum(1 for r in results if r['success'])

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"  Processed: {successful}/{len(results)} videos")
    print(f"  Total words: {total_words:,}")
    print(f"  Total time: {session_time/60:.1f} min")
    print(f"{'='*60}")

    return results


def list_playlists():
    """List available playlists with transcription progress"""
    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))

    print(f"\nPlaylists:\n")
    for i, pf in enumerate(playlist_files, 1):
        with open(pf) as f:
            p = json.load(f)

        total = len(p.get('videos', []))
        done = sum(1 for v in p.get('videos', [])
                   if (TRANSCRIPTS_DIR / f"{v['video_id']}.json").exists())

        status = f"[{done}/{total}]"
        print(f"  {i}. {p['title']} {status}")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Local transcription using whisper.cpp (Metal GPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe_local.py --list                         # List playlists
  python transcribe_local.py --video Ovd5QzZutsw            # Single video
  python transcribe_local.py --playlist 1 --only-missing    # Only untranscribed videos
  python transcribe_local.py --playlist 1 --max 3           # First 3 videos
  python transcribe_local.py --video XYZ --model medium     # Use medium model
        """
    )
    parser.add_argument('--list', action='store_true', help='List playlists')
    parser.add_argument('--video', type=str, help='Process single video by ID')
    parser.add_argument('--title', type=str, default='Video', help='Video title (with --video)')
    parser.add_argument('--playlist', type=int, help='Playlist number')
    parser.add_argument('--playlist-id', type=str, help='Playlist ID directly')
    parser.add_argument('--max', type=int, help='Max videos to process')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help=f'Whisper model (default: {DEFAULT_MODEL})')
    parser.add_argument('--only-missing', action='store_true',
                       help='Only process videos without existing transcripts')
    parser.add_argument('--force', action='store_true',
                       help='Re-transcribe even if transcript exists')

    args = parser.parse_args()

    if args.list:
        list_playlists()
        return

    if args.video:
        print(f"\nProcessing: {args.video}")
        result = process_video(args.video, args.title, args.model, args.force)
        if result:
            print(f"\nDone! {result.get('word_count', 0):,} words")
        else:
            print(f"\nFailed to transcribe video")
        return

    if args.playlist or args.playlist_id:
        process_playlist(
            playlist_num=args.playlist,
            playlist_id=args.playlist_id,
            max_videos=args.max,
            model_name=args.model,
            only_missing=args.only_missing,
        )
        return

    parser.print_help()
    list_playlists()


if __name__ == "__main__":
    main()
