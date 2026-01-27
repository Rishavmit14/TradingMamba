#!/usr/bin/env python3
"""
Add New Playlist Script

Easily add new ICT playlists to the TradingMamba system.
Automatically fetches video info and queues for transcription.

Usage:
  python add_playlist.py "https://youtube.com/playlist?list=PLxxxxxx" --tier 3
  python add_playlist.py --video "https://youtube.com/watch?v=xxxxx"
"""

import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')

try:
    import yt_dlp
except ImportError:
    print("Please install yt-dlp: pip install yt-dlp")
    sys.exit(1)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLAYLISTS_DIR = DATA_DIR / "playlists"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

PLAYLISTS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_playlist_id(url: str) -> str:
    """Extract playlist ID from URL"""
    patterns = [
        r'list=([a-zA-Z0-9_-]+)',
        r'playlist\?list=([a-zA-Z0-9_-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def extract_video_id(url: str) -> str:
    """Extract video ID from URL"""
    patterns = [
        r'watch\?v=([a-zA-Z0-9_-]+)',
        r'youtu\.be/([a-zA-Z0-9_-]+)',
        r'embed/([a-zA-Z0-9_-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url  # Assume it's already an ID


def fetch_playlist_info(playlist_url: str) -> dict:
    """Fetch playlist information from YouTube"""
    print(f"Fetching playlist info...")

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'skip_download': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

        videos = []
        for entry in info.get('entries', []):
            if entry:
                videos.append({
                    'video_id': entry.get('id'),
                    'title': entry.get('title', 'Unknown'),
                    'duration': entry.get('duration'),
                    'url': f"https://www.youtube.com/watch?v={entry.get('id')}"
                })

        return {
            'playlist_id': info.get('id'),
            'title': info.get('title', 'Unknown Playlist'),
            'channel': info.get('channel', info.get('uploader', 'Unknown')),
            'video_count': len(videos),
            'videos': videos,
            'fetched_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        print(f"Error fetching playlist: {e}")
        return None


def fetch_video_info(video_url: str) -> dict:
    """Fetch single video information"""
    print(f"Fetching video info...")

    video_id = extract_video_id(video_url)

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

        return {
            'video_id': info.get('id'),
            'title': info.get('title', 'Unknown'),
            'channel': info.get('channel', info.get('uploader', 'Unknown')),
            'duration': info.get('duration'),
            'description': info.get('description', '')[:500],
            'url': f"https://www.youtube.com/watch?v={info.get('id')}"
        }

    except Exception as e:
        print(f"Error fetching video: {e}")
        return None


def get_transcript(video_id: str) -> dict:
    """Try to get transcript for a video"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt_api = YouTubeTranscriptApi()
        result = ytt_api.fetch(video_id)

        segments = []
        for item in result:
            segments.append({
                'start_time': round(item.start, 2),
                'end_time': round(item.start + item.duration, 2),
                'text': item.text.strip()
            })

        full_text = ' '.join([s['text'] for s in segments])

        return {
            'video_id': video_id,
            'full_text': full_text,
            'segments': segments,
            'word_count': len(full_text.split()),
            'segment_count': len(segments),
            'method': 'youtube_transcript_api',
            'transcribed_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {'error': str(e)}


def add_playlist(url: str, tier: int = 3, description: str = None):
    """Add a new playlist to the system"""
    playlist_info = fetch_playlist_info(url)

    if not playlist_info:
        print("Failed to fetch playlist")
        return

    # Add metadata
    playlist_info['tier'] = tier
    playlist_info['description'] = description or f"ICT Playlist - Tier {tier}"
    playlist_info['added_at'] = datetime.utcnow().isoformat()

    # Save playlist
    save_path = PLAYLISTS_DIR / f"{playlist_info['playlist_id']}.json"
    with open(save_path, 'w') as f:
        json.dump(playlist_info, f, indent=2)

    print(f"\n✓ Playlist saved: {save_path}")
    print(f"  Title: {playlist_info['title']}")
    print(f"  Videos: {playlist_info['video_count']}")
    print(f"  Tier: {tier}")

    # Try to get transcripts
    print(f"\n🎬 Fetching transcripts...")

    success = 0
    failed = 0

    for i, video in enumerate(playlist_info['videos'], 1):
        video_id = video['video_id']
        transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"

        if transcript_path.exists():
            print(f"  [{i}/{playlist_info['video_count']}] {video['title'][:40]}... (already done)")
            success += 1
            continue

        print(f"  [{i}/{playlist_info['video_count']}] {video['title'][:40]}...", end=' ')

        transcript = get_transcript(video_id)

        if 'error' in transcript:
            print(f"✗ {transcript['error'][:30]}")
            failed += 1
        else:
            transcript['title'] = video['title']
            with open(transcript_path, 'w') as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            print(f"✓ {transcript['word_count']} words")
            success += 1

    print(f"\n📊 Summary:")
    print(f"   Success: {success}")
    print(f"   Failed: {failed}")
    print(f"\n💡 Run 'python run_pipeline.py --train' to retrain with new data")


def add_video(url: str, title: str = None):
    """Add a single video"""
    video_info = fetch_video_info(url)

    if not video_info:
        print("Failed to fetch video")
        return

    video_id = video_info['video_id']
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"

    if transcript_path.exists():
        print(f"Video already transcribed: {video_id}")
        return

    print(f"\n🎬 Getting transcript for: {video_info['title'][:50]}...")

    transcript = get_transcript(video_id)

    if 'error' in transcript:
        print(f"✗ Failed: {transcript['error']}")
        print("\n💡 This video may need local Whisper transcription")
        print(f"   Run: python transcribe_v2.py --video {video_id}")
    else:
        transcript['title'] = title or video_info['title']
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved: {transcript['word_count']} words")
        print(f"\n💡 Run 'python run_pipeline.py --train' to include in training")


def list_playlists():
    """List all playlists with status"""
    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))

    if not playlist_files:
        print("No playlists found")
        return

    print(f"\n📚 Playlists ({len(playlist_files)} total):\n")

    total_videos = 0
    total_transcribed = 0

    for i, pf in enumerate(playlist_files, 1):
        with open(pf) as f:
            p = json.load(f)

        # Count transcribed
        done = sum(1 for v in p.get('videos', [])
                   if (TRANSCRIPTS_DIR / f"{v['video_id']}.json").exists())

        total_videos += p.get('video_count', 0)
        total_transcribed += done

        tier = p.get('tier', '?')
        progress = f"[{done}/{p.get('video_count', 0)}]"
        status = "✓" if done == p.get('video_count', 0) else "○"

        print(f"  {i:2}. {status} [T{tier}] {p.get('title', 'Unknown')[:40]:<40} {progress}")

    print(f"\n  Total: {total_transcribed}/{total_videos} videos transcribed")


def main():
    parser = argparse.ArgumentParser(
        description='Add new ICT playlists/videos to TradingMamba',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_playlist.py "https://youtube.com/playlist?list=PLxxx" --tier 2
  python add_playlist.py --video "https://youtube.com/watch?v=xxx"
  python add_playlist.py --list
        """
    )

    parser.add_argument('url', nargs='?', help='Playlist URL to add')
    parser.add_argument('--tier', type=int, default=3, help='Learning tier (1-5)')
    parser.add_argument('--description', type=str, help='Playlist description')
    parser.add_argument('--video', type=str, help='Add single video URL')
    parser.add_argument('--list', action='store_true', help='List all playlists')

    args = parser.parse_args()

    if args.list:
        list_playlists()
    elif args.video:
        add_video(args.video)
    elif args.url:
        add_playlist(args.url, args.tier, args.description)
    else:
        parser.print_help()
        print("\n")
        list_playlists()


if __name__ == "__main__":
    main()
