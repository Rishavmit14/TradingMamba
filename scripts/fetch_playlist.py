#!/usr/bin/env python3
"""
Fetch and display playlist information from ICT YouTube playlists.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add yt-dlp to path
sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')

import yt_dlp

def get_playlist_info(playlist_url: str) -> dict:
    """Get information about a playlist including all videos"""

    opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'dump_single_json': True,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

            videos = []
            for i, entry in enumerate(info.get('entries', [])):
                if entry:
                    videos.append({
                        'video_id': entry.get('id'),
                        'title': entry.get('title'),
                        'duration': entry.get('duration'),
                        'order': i + 1,
                    })

            return {
                'playlist_id': info.get('id', ''),
                'title': info.get('title', ''),
                'description': info.get('description', ''),
                'video_count': len(videos),
                'videos': videos,
                'fetched_at': datetime.utcnow().isoformat()
            }

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    # ICT Playlists
    playlists = [
        {
            "name": "ICT Market Maker Forex Series",
            "url": "https://youtube.com/playlist?list=PLVgHx4Z63paZ0R9gMaq0y2fM_2vyNJadp",
            "tier": 1
        },
        {
            "name": "ICT OTE Pattern Recognition Series",
            "url": "https://youtube.com/playlist?list=PLVgHx4Z63paaRnabpBl38GoMkxF1FiXCF",
            "tier": 1
        },
        {
            "name": "ICT Forex Market Maker Primer Course",
            "url": "https://youtube.com/playlist?list=PLVgHx4Z63paah1dHyad1OMJQJdm6iP2Yn",
            "tier": 1
        },
    ]

    # Create data directory
    data_dir = Path(__file__).parent.parent / "data" / "playlists"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ICT YouTube Playlist Fetcher")
    print("=" * 70)

    for i, playlist in enumerate(playlists, 1):
        print(f"\n[{i}/{len(playlists)}] Fetching: {playlist['name']}...")

        info = get_playlist_info(playlist['url'])

        if info:
            print(f"    ✓ Found {info['video_count']} videos")
            print(f"    📋 Title: {info['title']}")

            # Save to file
            output_file = data_dir / f"{info['playlist_id']}.json"
            with open(output_file, 'w') as f:
                json.dump(info, f, indent=2)
            print(f"    💾 Saved to: {output_file}")

            # Show first 5 videos
            print(f"\n    First 5 videos:")
            for video in info['videos'][:5]:
                duration_min = video['duration'] // 60 if video['duration'] else 0
                print(f"      {video['order']}. {video['title'][:60]}... ({duration_min} min)")
        else:
            print(f"    ✗ Failed to fetch playlist")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
