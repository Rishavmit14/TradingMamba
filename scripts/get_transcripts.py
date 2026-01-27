#!/usr/bin/env python3
"""
Get transcripts from YouTube videos (FREE)

Uses youtube-transcript-api to fetch auto-generated or manual captions.
No download required - completely free and fast!
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')

from youtube_transcript_api import YouTubeTranscriptApi

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLAYLISTS_DIR = DATA_DIR / "playlists"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize API
ytt_api = YouTubeTranscriptApi()


def get_transcript(video_id: str) -> dict:
    """Fetch transcript for a video"""
    try:
        # Fetch transcript
        result = ytt_api.fetch(video_id)

        # Convert to segments
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
            'duration_seconds': segments[-1]['end_time'] if segments else 0,
            'method': 'youtube_transcript_api',
            'transcribed_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {'error': str(e)}


def process_video(video_id: str, title: str) -> dict:
    """Process a single video"""
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"

    # Check if already done
    if transcript_path.exists():
        print(f"  ✓ Already done")
        with open(transcript_path) as f:
            data = json.load(f)
            return data

    # Get transcript
    result = get_transcript(video_id)

    if 'error' in result:
        print(f"  ✗ {result['error'][:60]}")
        return None

    # Add title and save
    result['title'] = title

    with open(transcript_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  ✓ {result['word_count']:,} words | {result['segment_count']} segments")
    return result


def process_playlist(playlist_num: int, max_videos: int = None):
    """Process all videos in a playlist"""
    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))

    if playlist_num < 1 or playlist_num > len(playlist_files):
        print(f"Invalid playlist number. Choose 1-{len(playlist_files)}")
        return

    with open(playlist_files[playlist_num - 1]) as f:
        playlist = json.load(f)

    videos = playlist['videos']
    if max_videos:
        videos = videos[:max_videos]

    print(f"\n{'='*60}")
    print(f"📺 {playlist['title']}")
    print(f"📊 Processing {len(videos)} videos")
    print(f"{'='*60}\n")

    results = {'success': 0, 'failed': 0, 'skipped': 0, 'total_words': 0}

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video['title'][:50]}...")

        transcript_path = TRANSCRIPTS_DIR / f"{video['video_id']}.json"
        if transcript_path.exists():
            results['skipped'] += 1
            with open(transcript_path) as f:
                data = json.load(f)
                results['total_words'] += data.get('word_count', 0)
            print(f"  ✓ Already done")
            continue

        result = process_video(video['video_id'], video['title'])

        if result:
            results['success'] += 1
            results['total_words'] += result.get('word_count', 0)
        else:
            results['failed'] += 1

    print(f"\n{'='*60}")
    print(f"📋 SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Success: {results['success']}")
    print(f"↷ Skipped (already done): {results['skipped']}")
    print(f"✗ Failed: {results['failed']}")
    print(f"📝 Total words: {results['total_words']:,}")
    print(f"{'='*60}\n")

    return results


def process_all_playlists(max_per_playlist: int = None):
    """Process all playlists"""
    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))

    print(f"\n🚀 Processing ALL {len(playlist_files)} playlists\n")

    grand_total = {'success': 0, 'failed': 0, 'skipped': 0, 'total_words': 0}

    for i, pf in enumerate(playlist_files, 1):
        result = process_playlist(i, max_per_playlist)
        if result:
            grand_total['success'] += result['success']
            grand_total['failed'] += result['failed']
            grand_total['skipped'] += result['skipped']
            grand_total['total_words'] += result['total_words']

    print(f"\n{'='*60}")
    print(f"🎉 ALL PLAYLISTS COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Total transcribed: {grand_total['success']}")
    print(f"↷ Already had: {grand_total['skipped']}")
    print(f"✗ Failed: {grand_total['failed']}")
    print(f"📝 Total words: {grand_total['total_words']:,}")
    print(f"{'='*60}\n")


def list_playlists():
    """List all playlists with progress"""
    playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))

    print(f"\n📚 Available Playlists ({len(playlist_files)} total):\n")

    total_videos = 0
    total_done = 0

    for i, pf in enumerate(playlist_files, 1):
        with open(pf) as f:
            p = json.load(f)

        # Count completed
        done = sum(1 for v in p.get('videos', [])
                   if (TRANSCRIPTS_DIR / f"{v['video_id']}.json").exists())

        total_videos += p['video_count']
        total_done += done

        progress = f"[{done}/{p['video_count']}]"
        status = "✓" if done == p['video_count'] else "○"

        print(f"  {i:2}. {status} {p['title'][:42]:<42} {progress}")

    print(f"\n  Total Progress: {total_done}/{total_videos} videos transcribed")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Get YouTube transcripts for ICT videos (FREE)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_transcripts.py --list                    # Show playlists
  python get_transcripts.py --playlist 4              # Process playlist 4
  python get_transcripts.py --playlist 4 --max 3      # First 3 videos only
  python get_transcripts.py --all                     # Process ALL playlists
  python get_transcripts.py --video Vh0NtdPPj1M       # Single video
        """
    )
    parser.add_argument('--list', action='store_true', help='List playlists')
    parser.add_argument('--playlist', type=int, help='Playlist number')
    parser.add_argument('--max', type=int, help='Max videos per playlist')
    parser.add_argument('--video', type=str, help='Single video ID')
    parser.add_argument('--all', action='store_true', help='Process all playlists')

    args = parser.parse_args()

    if args.list:
        list_playlists()
    elif args.video:
        result = process_video(args.video, "Single Video")
        if result:
            print(f"\n✓ Transcript saved: {result['word_count']} words")
    elif args.all:
        process_all_playlists(args.max)
    elif args.playlist:
        process_playlist(args.playlist, args.max)
    else:
        parser.print_help()
        list_playlists()


if __name__ == "__main__":
    main()
