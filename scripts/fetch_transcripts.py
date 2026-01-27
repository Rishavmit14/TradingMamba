#!/usr/bin/env python3
"""
Fetch transcripts/captions directly from YouTube videos.

This script gets the auto-generated or manual captions from YouTube
without downloading the video/audio files.
"""

import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.parse

# Add packages to path
sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')

import yt_dlp

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PLAYLISTS_DIR = DATA_DIR / "playlists"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

# Create directories
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def get_video_transcript(video_id: str) -> dict:
    """
    Get transcript/captions from a YouTube video

    Tries multiple methods:
    1. YouTube Data API subtitles
    2. yt-dlp subtitle extraction
    3. Third-party transcript services
    """

    # Method 1: Try yt-dlp for subtitles (doesn't download video)
    transcript = get_subtitles_ytdlp(video_id)
    if transcript and transcript.get('segments'):
        return transcript

    # Method 2: Try YouTube Transcript API
    transcript = get_transcript_api(video_id)
    if transcript and transcript.get('segments'):
        return transcript

    return None


def get_subtitles_ytdlp(video_id: str) -> dict:
    """Get subtitles using yt-dlp"""

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'en-US', 'en-GB'],
        'subtitlesformat': 'json3',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            title = info.get('title', '')
            duration = info.get('duration', 0)

            # Check for subtitles
            subtitles = info.get('subtitles', {})
            auto_captions = info.get('automatic_captions', {})

            # Try to get English captions
            caption_data = None
            caption_type = None

            for lang in ['en', 'en-US', 'en-GB']:
                if lang in subtitles:
                    caption_data = subtitles[lang]
                    caption_type = 'manual'
                    break
                elif lang in auto_captions:
                    caption_data = auto_captions[lang]
                    caption_type = 'auto'
                    break

            if not caption_data:
                print(f"      No English captions available")
                return None

            # Find json3 or vtt format
            caption_url = None
            for fmt in caption_data:
                if fmt.get('ext') == 'json3':
                    caption_url = fmt.get('url')
                    break
                elif fmt.get('ext') == 'vtt':
                    caption_url = fmt.get('url')

            if not caption_url:
                print(f"      No usable caption format found")
                return None

            # Download caption data
            req = urllib.request.Request(
                caption_url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response:
                caption_content = response.read().decode('utf-8')

            # Parse based on format
            if 'json3' in caption_url or caption_content.strip().startswith('{'):
                segments = parse_json3_captions(caption_content)
            else:
                segments = parse_vtt_captions(caption_content)

            if not segments:
                print(f"      Failed to parse captions")
                return None

            full_text = ' '.join([s['text'] for s in segments if s.get('text')])

            return {
                'video_id': video_id,
                'title': title,
                'full_text': full_text,
                'segments': segments,
                'language': 'en',
                'duration': duration,
                'transcribed_at': datetime.utcnow().isoformat(),
                'method': f'youtube_{caption_type}_captions',
                'word_count': len(full_text.split())
            }

    except Exception as e:
        print(f"      yt-dlp error: {e}")
        return None


def parse_json3_captions(json_content: str) -> list:
    """Parse JSON3 caption format"""
    try:
        data = json.loads(json_content)
        segments = []

        events = data.get('events', [])
        for event in events:
            if 'segs' not in event:
                continue

            start_time = event.get('tStartMs', 0) / 1000
            duration = event.get('dDurationMs', 0) / 1000
            end_time = start_time + duration

            text_parts = []
            for seg in event.get('segs', []):
                text = seg.get('utf8', '').strip()
                if text and text != '\n':
                    text_parts.append(text)

            if text_parts:
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': ' '.join(text_parts)
                })

        return segments

    except json.JSONDecodeError:
        return []


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
                    text = lines[i].strip()
                    # Remove VTT formatting tags
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

    # Deduplicate
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


def get_transcript_api(video_id: str) -> dict:
    """
    Try to get transcript using youtube-transcript-api library
    (if installed)
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to get English transcript
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        except:
            # Try auto-generated
            transcript = transcript_list.find_generated_transcript(['en'])

        data = transcript.fetch()

        segments = []
        for item in data:
            segments.append({
                'start_time': item['start'],
                'end_time': item['start'] + item['duration'],
                'text': item['text']
            })

        full_text = ' '.join([s['text'] for s in segments])

        return {
            'video_id': video_id,
            'full_text': full_text,
            'segments': segments,
            'language': 'en',
            'transcribed_at': datetime.utcnow().isoformat(),
            'method': 'youtube_transcript_api',
            'word_count': len(full_text.split())
        }

    except ImportError:
        return None
    except Exception as e:
        print(f"      Transcript API error: {e}")
        return None


def process_video(video_id: str, title: str) -> dict:
    """Process a single video - get transcript"""

    # Check if already processed
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"
    if transcript_path.exists():
        print(f"      ℹ Already processed")
        with open(transcript_path) as f:
            return json.load(f)

    # Get transcript
    transcript = get_video_transcript(video_id)

    if transcript:
        transcript['title'] = title

        # Save transcript
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        print(f"      ✓ Got transcript ({transcript.get('method', 'unknown')})")
        print(f"      📝 Words: {transcript.get('word_count', 0)}")
        print(f"      📊 Segments: {len(transcript.get('segments', []))}")

        return transcript

    print(f"      ✗ Could not get transcript")
    return None


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
    print(f"📺 Processing: {playlist['title']}")
    print(f"📊 Total Videos: {playlist['video_count']}")
    print(f"{'='*70}\n")

    videos = playlist['videos']
    if max_videos:
        videos = videos[:max_videos]

    results = []
    total_words = 0
    total_segments = 0

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video['title']}")
        print(f"    ID: {video['video_id']}")

        result = process_video(video['video_id'], video['title'])

        if result:
            results.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'success': True,
                'method': result.get('method'),
                'word_count': result.get('word_count', 0),
                'segments': len(result.get('segments', []))
            })
            total_words += result.get('word_count', 0)
            total_segments += len(result.get('segments', []))
        else:
            results.append({
                'video_id': video['video_id'],
                'title': video['title'],
                'success': False
            })

    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n{'='*70}")
    print(f"📋 SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Processed: {successful}/{len(results)} videos")
    print(f"📝 Total Words: {total_words:,}")
    print(f"📊 Total Segments: {total_segments:,}")

    # Save processing report
    report_path = DATA_DIR / f"transcript_report_{playlist_id}.json"
    with open(report_path, 'w') as f:
        json.dump({
            'playlist_id': playlist_id,
            'playlist_title': playlist['title'],
            'processed_at': datetime.utcnow().isoformat(),
            'total_words': total_words,
            'total_segments': total_segments,
            'results': results
        }, f, indent=2)
    print(f"💾 Report: {report_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fetch ICT YouTube transcripts')
    parser.add_argument('--playlist', type=int, default=1, help='Playlist number')
    parser.add_argument('--max-videos', type=int, default=None, help='Max videos')
    parser.add_argument('--video', type=str, help='Single video ID')
    parser.add_argument('--list', action='store_true', help='List playlists')

    args = parser.parse_args()

    if args.list:
        playlist_files = sorted(PLAYLISTS_DIR.glob("*.json"))
        print("\n📚 Available Playlists:\n")
        for i, pf in enumerate(playlist_files, 1):
            with open(pf) as f:
                p = json.load(f)
            print(f"  {i}. {p['title']} ({p['video_count']} videos)")
        return

    if args.video:
        result = process_video(args.video, "Single Video")
        return

    # Get playlist
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
