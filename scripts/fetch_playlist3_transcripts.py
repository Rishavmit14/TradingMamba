import sys
sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')
import json
import os

from youtube_transcript_api import YouTubeTranscriptApi

# Videos from playlist 3 that need transcripts
videos = [
    {"video_id": "B7_cjybYQ0g", "title": "ICT Mentorship Core Content - Month 1 - What To Focus On Right Now"},
    {"video_id": "qC0LogyIk2I", "title": "ICT Mentorship Core Content - Month 1 - Equilibrium Vs. Discount"},
    {"video_id": "YuefjnUKQdM", "title": "ICT Mentorship Core Content - Month 1 - Equilibrium Vs. Premium"},
    {"video_id": "SiVmoeyOWZE", "title": "ICT Mentorship Core Content - Month 1 - Fair Valuation"},
    {"video_id": "22XkhpJR5eA", "title": "ICT Mentorship Core Content - Month 1 - Liquidity Runs"},
    {"video_id": "K4LtfujVpJs", "title": "ICT Mentorship Core Content - Month 1 - Impulse Price Swings & Market Protraction"},
]

transcripts_dir = "/tmp/TradingMamba/data/transcripts"
ytt_api = YouTubeTranscriptApi()

success_count = 0
failed_count = 0
failed_videos = []

for video in videos:
    video_id = video["video_id"]
    title = video["title"]
    output_path = os.path.join(transcripts_dir, f"{video_id}.json")

    # Check if transcript already exists
    if os.path.exists(output_path):
        print(f"SKIP: {video_id} - Transcript already exists")
        success_count += 1
        continue

    try:
        print(f"Fetching transcript for {video_id}...")
        result = ytt_api.fetch(video_id)

        # Build full text from segments
        segments = []
        full_text_parts = []
        for item in result:
            segments.append({
                "text": item.text,
                "start": item.start,
                "duration": item.duration
            })
            full_text_parts.append(item.text)

        full_text = " ".join(full_text_parts)
        word_count = len(full_text.split())

        transcript_data = {
            "video_id": video_id,
            "title": title,
            "full_text": full_text,
            "segments": segments,
            "word_count": word_count,
            "method": "youtube_transcript_api"
        }

        with open(output_path, 'w') as f:
            json.dump(transcript_data, f, indent=2)

        print(f"SUCCESS: {video_id} - {title} ({word_count} words)")
        success_count += 1

    except Exception as e:
        print(f"FAILED: {video_id} - {title}")
        print(f"  Error: {str(e)}")
        failed_count += 1
        failed_videos.append({"video_id": video_id, "title": title, "error": str(e)})

print("\n" + "="*60)
print(f"SUMMARY FOR PLAYLIST 3 (PLVgHx4Z63paYzh3KwUFX0UHQUf31CAEXk)")
print(f"Total videos in playlist: 8")
print(f"Successfully transcribed: {success_count}")
print(f"Failed: {failed_count}")
if failed_videos:
    print("\nFailed videos:")
    for v in failed_videos:
        print(f"  - {v['video_id']}: {v['title']}")
        print(f"    Error: {v['error']}")
