#!/usr/bin/env python3
"""
Fetch all ICT playlists information.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')
import yt_dlp

# All playlists provided by user
ICT_PLAYLISTS = [
    {
        "name": "ICT Market Maker Forex Series",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paZ0R9gMaq0y2fM_2vyNJadp",
        "tier": 1,
    },
    {
        "name": "ICT OTE Pattern Recognition Series",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paaRnabpBl38GoMkxF1FiXCF",
        "tier": 1,
    },
    {
        "name": "ICT Forex Market Maker Primer Course",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paah1dHyad1OMJQJdm6iP2Yn",
        "tier": 1,
    },
    {
        "name": "ICT Private Mentorship Month 01",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paYzh3KwUFX0UHQUf31CAEXk",
        "tier": 2,
    },
    {
        "name": "ICT Private Mentorship Month 02",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paZvjqerfbn320myZ06L1MOB",
        "tier": 2,
    },
    {
        "name": "ICT Private Mentorship Month 03",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paaY69GotBJyZ7KN_U09ra2o",
        "tier": 2,
    },
    {
        "name": "ICT Private Mentorship Month 04",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63pabb9rl1nyG58TG8PG8yzuao",
        "tier": 2,
    },
    {
        "name": "ICT Private Mentorship Month 05",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paYBN404Q2QZ7D4mOJz1IHAk",
        "tier": 2,
    },
    {
        "name": "ICT Private Mentorship Month 06",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paaG-26YEf2svQ_EsdGXjws1",
        "tier": 2,
    },
    {
        "name": "ICT Private Mentorship Month 07",
        "url": "https://youtube.com/playlist?list=PLVgHx4Z63paYWV_3PDkYajv_oNznvK2aR",
        "tier": 2,
    },
]

def get_playlist_info(playlist_url: str) -> dict:
    opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
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
                'video_count': len(videos),
                'videos': videos,
                'fetched_at': datetime.utcnow().isoformat()
            }

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    data_dir = Path(__file__).parent.parent / "data" / "playlists"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Fetching All ICT Playlists")
    print("=" * 70)

    total_videos = 0

    for i, playlist in enumerate(ICT_PLAYLISTS, 1):
        print(f"\n[{i}/{len(ICT_PLAYLISTS)}] {playlist['name']}...")

        info = get_playlist_info(playlist['url'])

        if info:
            info['tier'] = playlist['tier']
            total_videos += info['video_count']

            print(f"    ✓ {info['video_count']} videos")

            # Save
            output_file = data_dir / f"{info['playlist_id']}.json"
            with open(output_file, 'w') as f:
                json.dump(info, f, indent=2)
        else:
            print(f"    ✗ Failed")

    print(f"\n{'='*70}")
    print(f"Total Videos Across All Playlists: {total_videos}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
